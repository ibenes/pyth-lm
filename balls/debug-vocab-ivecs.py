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
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--ivec-extractor', type=str, required=True,
                        help='where to load a ivector extractor from')
    parser.add_argument('--load', type=str, required=True,
                        help='where to load a model from')
    args = parser.parse_args()

    init_seeds(args.seed, args.cuda)
    with open(args.load, 'rb') as f:
        lm = language_model.load(f)
    vocab = lm.vocab

    with open(args.ivec_extractor, 'rb') as f:
        ivec_extractor = smm_ivec_extractor.load(f)
    translator = ivec_extractor.build_translator(vocab)

    for w in vocab:
        ivec_old = ivec_extractor(w)

        w_index = torch.LongTensor([vocab[w]])
        if args.cuda:
            w_index = w_index.cuda()
        bow = translator(w_index)
        ivec_new = ivec_extractor(bow.view(1,-1))
        print("{}\t{:.5f}\t{:.5f}".format(w, cosine_similarity(ivec_old, ivec_new)[0], length(ivec_old - ivec_new)[0]))
