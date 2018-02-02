import argparse
import math
import torch
import torch.nn as nn
from torch.autograd import Variable

import model
import lstm_model
import vocab
import language_model
import split_corpus_dataset
from hidden_state_reorganization import HiddenStateReorganizer

from runtime_utils import CudaStream, repackage_hidden, filelist_to_tokenized_splits

import numpy as np


def variablilize_targets(targets):
    return Variable(targets.contiguous().view(-1))

def evaluate(data_source):
    model.eval()

    total_loss = 0.0
    total_timesteps = 0

    hs_reorganizer = HiddenStateReorganizer(model)
    hidden = model.init_hidden(args.batch_size)

    if args.cuda:
        model.cuda()
        hidden = tuple(h.cuda() for h in hidden)

    for X, targets, ivecs, mask in data_source:
        hidden = hs_reorganizer(hidden, mask, X.size(1))
        hidden = repackage_hidden(hidden)

        output, hidden = model(Variable(X), hidden, Variable(ivecs))
        output_flat = output.view(-1, len(vocab))
        curr_loss = len(X) * criterion(output_flat, variablilize_targets(targets)).data
        total_loss += curr_loss
        total_timesteps += len(X)

    return total_loss[0] / total_timesteps


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
    parser.add_argument('--load', type=str, required=True,
                        help='where to load a model from')
    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)
    if args.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    print("loading model...")
    with open(args.load, 'rb') as f:
        lm = language_model.load(f)
    vocab = lm.vocab
    model = lm.model
    print(model)

    print("preparing data...")
    ivec_eetor = lambda x: np.asarray([float(sum(x) % 1337 - 668)/1337]*2, dtype=np.float32)
    ivec_app_creator = lambda ts: split_corpus_dataset.CheatingIvecAppender(ts, ivec_eetor)

    tss = filelist_to_tokenized_splits(args.file_list, vocab, args.bptt)
    data = split_corpus_dataset.BatchBuilder(tss, ivec_app_creator, args.batch_size,
                                               discard_h=not args.concat_articles)
    if args.cuda:
        data = CudaStream(data)

    criterion = nn.NLLLoss()

    loss = evaluate(data)
    print('loss {:5.2f} | ppl {:8.2f}'.format( loss, math.exp(loss)))
