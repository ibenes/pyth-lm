#!/usr/bin/env python

import argparse
import math
import torch

from balls.data_pipeline.data import tokens_from_fn
from balls.data_pipeline.multistream import batchify
from balls.data_pipeline.temporal_splitting import TemporalSplits

from balls.runtime.runtime_utils import TransposeWrapper, init_seeds, epoch_summary
from balls.runtime.runtime_multifile import evaluate_, repackage_hidden

from balls.runtime.loggers import ProgressLogger


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, required=True,
                        help='location of the train corpus')
    parser.add_argument('--valid', type=str, required=True,
                        help='location of the valid corpus')
    parser.add_argument('--characters', action='store_true',
                        help='work on character level, whitespace is significant')
    parser.add_argument('--shuffle-lines', action='store_true',
                        help='shuffle lines before every epoch')

    parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--target-seq-len', type=int, default=35,
                        help='sequence length')

    parser.add_argument('--lr', type=float, default=20,
                        help='initial learning rate')
    parser.add_argument('--beta', type=float, default=0,
                        help='L2 regularization penalty')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=40,
                        help='upper epoch limit')

    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--load', type=str, required=True,
                        help='where to load a model from')
    parser.add_argument('--save', type=str, required=True,
                        help='path to save the final model')
    args = parser.parse_args()
    print(args)

    init_seeds(args.seed, args.cuda)

    print("loading model...")
    lm = torch.load(args.load)
    if args.cuda:
        lm.cuda()
    print(lm.model)

    print("preparing data...")
    tokenize_regime = 'words'
    if args.characters:
        tokenize_regime = 'chars'

    train_ids = tokens_from_fn(args.train, lm.vocab, randomize=False, regime=tokenize_regime)
    train_batched = batchify(train_ids, args.batch_size, args.cuda)
    train_data_tb = TemporalSplits(
        train_batched,
        nb_inputs_necessary=lm.model.in_len,
        nb_targets_parallel=args.target_seq_len
    )
    train_data = TransposeWrapper(train_data_tb)

    valid_ids = tokens_from_fn(args.valid, lm.vocab, randomize=False, regime=tokenize_regime)
    valid_batched = batchify(valid_ids, 10, args.cuda)
    valid_data_tb = TemporalSplits(
        valid_batched,
        nb_inputs_necessary=lm.model.in_len,
        nb_targets_parallel=args.target_seq_len
    )
    valid_data = TransposeWrapper(valid_data_tb)

    print('Initial perplexity {:.2f}'.format(math.exp(evaluate_(lm, valid_data, use_ivecs=False, custom_batches=False))))

    print("training...")
    lr = args.lr
    best_val_loss = None

    running_loss = 0.0
    nb_updates = 0

    for epoch in range(1, args.epochs + 1):
        logger = ProgressLogger(epoch, args.log_interval, lr, len(train_batched) // args.target_seq_len)
        optim = torch.optim.SGD(lm.parameters(), lr, weight_decay=args.beta)

        lm.train()

        hidden = None
        do_transpose = not lm.model.batch_first

        for X, targets in train_data:
            X = X.t()
            targets = targets.t().contiguous()

            if hidden is None:
                hidden = lm.model.init_hidden(args.batch_size)

            hidden = repackage_hidden(hidden)

            output, hidden = lm.model(X, hidden)
            loss, nb_words = lm.decoder.neg_log_prob(output, targets)
            loss /= nb_words

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(lm.parameters(), args.clip)

            optim.step()
            nb_updates += 1
            running_loss += loss.data
            if nb_updates % args.log_interval == 0:
                train_ppl = math.exp(running_loss / args.log_interval)
                val_ppl = math.exp(evaluate_(
                    lm, valid_data,
                    use_ivecs=False,
                    custom_batches=False,
                ))
                print('{}: {:.1f} {:.1f} {:.2f}'.format(nb_updates, train_ppl, val_ppl, val_ppl - train_ppl))
                running_loss = 0.0
                lm.train()

            logger.log(loss.data)

        val_loss = evaluate_(
            lm, valid_data,
            use_ivecs=False,
            custom_batches=False,
        )
        print(epoch_summary(epoch, logger.nb_updates(), logger.time_since_creation(), val_loss))

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            torch.save(lm, args.save)
            best_val_loss = val_loss
        else:
            lr /= 2.0
            pass
