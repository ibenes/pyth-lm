#!/usr/bin/env python
import argparse
import time
import math
import random

import torch

import smm_lstm_models
import vocab
import language_model
import split_corpus_dataset
import ivec_appenders
import smm_ivec_extractor

from runtime_utils import CudaStream, init_seeds, filelist_to_tokenized_splits
from runtime_multifile import train, evaluate, BatchFilter

from loggers import InfinityLogger
import numpy as np


def length(a):
    return a.pow(2).sum(dim=-1).pow(0.5)

def cosine_similarity(a, b):
    return (a*b).sum(dim=-1) / (length(a) * length(b))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM Language Model')
    parser.add_argument('--file-list', type=str, required=True,
                        help='file with paths to training documents')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=35,
                        help='sequence length')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--concat-articles', action='store_true',
                        help='pass hidden states over article boundaries')
    parser.add_argument('--shuffle-articles', action='store_true',
                        help='shuffle the order of articles (at the start of the training)')
    parser.add_argument('--keep-shuffling', action='store_true',
                        help='shuffle the order of articles for each epoch')
    parser.add_argument('--min-batch-size', type=int, default=1,
                        help='stop, once batch is smaller than given size')
    parser.add_argument('--ivec-extractor', type=str, required=True,
                        help='where to load a ivector extractor from')
    parser.add_argument('--load', type=str, required=True,
                        help='where to load a model from')
    args = parser.parse_args()
    print(args)

    init_seeds(args.seed, args.cuda)

    print("loading LSTM model (we need the vocab from there) ...")
    with open(args.load, 'rb') as f:
        lm = language_model.load(f)
    vocab = lm.vocab
    model = lm.model
    print(model)

    print("loading SMM iVector extractor ...")
    with open(args.ivec_extractor, 'rb') as f:
        ivec_extractor = smm_ivec_extractor.load(f)
    print(ivec_extractor)
    translator = ivec_extractor.build_translator(vocab)

    print("preparing data...")
    ivec_app_creator = lambda ts: ivec_appenders.HistoryIvecAppender(ts, ivec_extractor)

    tss = filelist_to_tokenized_splits(args.file_list, vocab, args.bptt)
    data_old = split_corpus_dataset.BatchBuilder([ivec_app_creator(ts) for ts in tss], args.batch_size,
                                                   discard_h=not args.concat_articles)
    if args.cuda:
        data_old = CudaStream(data_old)

    data_new = split_corpus_dataset.BatchBuilder(tss, args.batch_size, discard_h=not args.concat_articles)
    if args.cuda:
        data_new = CudaStream(data_new)
    data_new = ivec_appenders.ParalelIvecAppender(data_new, ivec_extractor, translator)

    for old, new in zip(data_old, data_new):
        ivec_diff = (old[2] - new[2]).abs().mean()
        ivec_cos_sim = cosine_similarity(old[2], new[2])
        print (
            "bsz", old[0].size(0),
            "mean absolute diff: {:.9f}".format(ivec_diff),
            "mean cs {:.5f}".format(ivec_cos_sim.mean()),
            "min cs {:.5f}".format(ivec_cos_sim.min()),
        )
