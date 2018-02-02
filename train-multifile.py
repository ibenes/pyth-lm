import argparse
import time
import math
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
import sys

import model
import lstm_model
import vocab
import language_model
import split_corpus_dataset
from hidden_state_reorganization import HiddenStateReorganizer

from runtime_utils import CudaStream, repackage_hidden, filelist_to_tokenized_splits

import pickle
from loggers import InfinityLogger
import numpy as np

import IPython


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

        output, hidden = model(Variable(X), hidden)
        output_flat = output.view(-1, len(vocab))
        curr_loss = len(X) * criterion(output_flat, variablilize_targets(targets)).data
        total_loss += curr_loss
        total_timesteps += len(X)

    return total_loss[0] / total_timesteps



def train(logger, data):
    model.train()
    hs_reorganizer = HiddenStateReorganizer(model)
    hidden = model.init_hidden(args.batch_size)

    if args.cuda:
        model.cuda()
        hidden = tuple(h.cuda() for h in hidden)

    optim = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.beta)
    
    skipping = False
    nb_skipped_updates = 0
    nb_skipped_words = 0
    nb_skipped_seqs = 0 # accumulates size of skipped batches


    for batch, (X, targets, ivecs, mask) in enumerate(data):
        if X.size(1) < args.min_batch_size:
            skipping = True

        if skipping:
            nb_skipped_updates += 1
            nb_skipped_words += X.size(0) * X.size(1)
            nb_skipped_seqs += X.size(1)
            continue

        hidden = hs_reorganizer(hidden, mask, X.size(1))
        hidden = repackage_hidden(hidden)

        output, hidden = model(Variable(X), hidden)
        loss = criterion(output.view(-1, len(vocab)), variablilize_targets(targets))

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)

        optim.step()

        logger.log(loss.data)

    if skipping:
        sys.stderr.write(
            "WARNING: due to skipping, a total of {} updates was skipped,"
            " containing {} words. Avg batch size {}. Equal to {} full batches"
            "\n".format(nb_skipped_updates, nb_skipped_words, nb_skipped_seqs/nb_skipped_updates,
                        nb_skipped_words/(args.batch_size*args.bptt))
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM Language Model')
    parser.add_argument('--train-list', type=str, required=True,
                        help='file with paths to training documents')
    parser.add_argument('--valid-list', type=str, required=True,
                        help='file with paths to validation documents')
    parser.add_argument('--test-list', type=str, required=True,
                        help='file with paths to testin documents')
    parser.add_argument('--lr', type=float, default=20,
                        help='initial learning rate')
    parser.add_argument('--beta', type=float, default=0,
                        help='L2 regularization penalty')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=40,
                        help='upper epoch limit')
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
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--load', type=str, required=True,
                        help='where to load a model from')
    parser.add_argument('--save', type=str,  required=True,
                        help='path to save the final model')
    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
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
    ivec_eetor = lambda x: np.asarray([sum(x) % 1337])
    ivec_app_creator = lambda ts: split_corpus_dataset.CheatingIvecAppender(ts, ivec_eetor)

    print("\ttraining...")
    train_tss = filelist_to_tokenized_splits(args.train_list, vocab, args.bptt)
    train_data = split_corpus_dataset.BatchBuilder(train_tss, ivec_app_creator, args.batch_size,
                                                   discard_h=not args.concat_articles)
    if args.cuda:
        train_data = CudaStream(train_data)

    print("\tvalidation...")
    valid_tss = filelist_to_tokenized_splits(args.valid_list, vocab, args.bptt)
    valid_data = split_corpus_dataset.BatchBuilder(valid_tss, ivec_app_creator, args.batch_size,
                                                   discard_h=not args.concat_articles)
    if args.cuda:
        valid_data = CudaStream(valid_data)

    print("\ttesting...")
    test_tss = filelist_to_tokenized_splits(args.test_list, vocab, args.bptt)
    test_data = split_corpus_dataset.BatchBuilder(test_tss, ivec_app_creator, args.batch_size,
                                                   discard_h=not args.concat_articles)
    if args.cuda:
        test_data = CudaStream(test_data)

    criterion = nn.NLLLoss()

    print("training...")
    lr = args.lr
    best_val_loss = None

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, args.epochs+1):
            if args.keep_shuffling:
                random.shuffle(train_tss)
                train_data = split_corpus_dataset.BatchBuilder(train_tss, ivec_app_creator, args.batch_size,
                                                               discard_h=not args.concat_articles)
                if args.cuda:
                    train_data = CudaStream(train_data)
                

            epoch_start_time = time.time()

            logger = InfinityLogger(epoch, args.log_interval, lr)
            train(logger, train_data)
            val_loss = evaluate(valid_data)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | # updates: {} | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, logger.time_since_creation(), logger.nb_updates(),
                                               val_loss, math.exp(val_loss)))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(args.save, 'wb') as f:
                    lm.save(f)
                best_val_loss = val_loss
            else:
                lr /= 2.0
                pass

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open(args.save, 'rb') as f:
        lm = language_model.load(f)
    vocab = lm.vocab
    model = lm.model

    # Run on test data.
    test_loss = evaluate(test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)
