import argparse
import math
import random

import torch

from language_models import language_model
import split_corpus_dataset
import ivec_appenders
import smm_ivec_extractor
from data_pipeline.multistream import BatchBuilder

from runtime_utils import CudaStream, init_seeds, filelist_to_tokenized_splits, BatchFilter
from runtime_multifile import evaluate_no_transpose, train_no_transpose

from loggers import InfinityLogger

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM Language Model')
    parser.add_argument('--train-list', type=str, required=True,
                        help='file with paths to training documents')
    parser.add_argument('--valid-list', type=str, required=True,
                        help='file with paths to validation documents')
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
    parser.add_argument('--target-seq-len', type=int, default=35, metavar='N',
                        help='number of words to take from every sequence in a single step')
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
    parser.add_argument('--ivec-extractor', type=str, required=True,
                        help='where to load a ivector extractor from')
    parser.add_argument('--ivec-nb-iters', type=int,
                        help='override the number of iterations when extracting ivectors')
    parser.add_argument('--load', type=str, required=True,
                        help='where to load a model from')
    parser.add_argument('--save', type=str, required=True,
                        help='path to save the final model')
    args = parser.parse_args()
    print(args)

    init_seeds(args.seed, args.cuda)

    print("loading model...")
    with open(args.load, 'rb') as f:
        lm = language_model.load(f)
    if args.cuda:
        lm.model.cuda()
    print(lm.model)

    print("loading SMM iVector extractor ...")
    with open(args.ivec_extractor, 'rb') as f:
        ivec_extractor = smm_ivec_extractor.load(f)
    if args.ivec_nb_iters:
        ivec_extractor._nb_iters = args.ivec_nb_iters
    print(ivec_extractor)

    print("preparing data...")
    ivec_app_creator = lambda ts: ivec_appenders.CheatingIvecAppender(ts, ivec_extractor)

    print("\ttraining...")
    ts_constructor = lambda *x: split_corpus_dataset.TokenizedSplitFFMultiTarget(*x, args.target_seq_len)

    train_tss = filelist_to_tokenized_splits(args.train_list, lm.vocab, lm.model.in_len, ts_constructor)
    train_data = BatchBuilder([ivec_app_creator(ts) for ts in train_tss], args.batch_size,
                              discard_h=not args.concat_articles)
    if args.cuda:
        train_data = CudaStream(train_data)

    print("\tvalidation...")
    valid_tss = filelist_to_tokenized_splits(args.valid_list, lm.vocab, lm.model.in_len, ts_constructor)
    valid_data = BatchBuilder([ivec_app_creator(ts) for ts in valid_tss], args.batch_size,
                              discard_h=not args.concat_articles)
    if args.cuda:
        valid_data = CudaStream(valid_data)

    print("training...")
    lr = args.lr
    best_val_loss = None

    for epoch in range(1, args.epochs+1):
        if args.keep_shuffling:
            random.shuffle(train_tss)
            train_data = BatchBuilder([ivec_app_creator(ts) for ts in train_tss], args.batch_size,
                                      discard_h=not args.concat_articles)
            if args.cuda:
                train_data = CudaStream(train_data)

        logger = InfinityLogger(epoch, args.log_interval, lr)
        train_data_filtered = BatchFilter(
            train_data, args.batch_size, lm.model.in_len, args.min_batch_size
        )
        optim = torch.optim.SGD(lm.model.parameters(), lr=lr, weight_decay=args.beta)

        train_no_transpose(
            lm.model, train_data_filtered, optim, logger,
            clip=args.clip,
            use_ivecs=True
        )
        train_data_filtered.report()

        val_loss = evaluate_no_transpose(lm.model, valid_data, use_ivecs=True)
        print('-' * 89)
        print(
            '| end of epoch {:3d} | time: {:5.2f}s | # updates: {} | valid loss {:5.2f} | '
            'valid ppl {:8.2f}'.format(
                epoch, logger.time_since_creation(), logger.nb_updates(),
                val_loss, math.exp(val_loss)
            )
        )
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                lm.save(f)
            best_val_loss = val_loss
        else:
            lr /= 2.0
            pass