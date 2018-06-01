import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import sys
import os

import data
import model
import lstm_model
import vocab
import language_model

import pickle
from loggers import ProgressLogger

import IPython


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def padded_loss(output, targets, lengths):
    loss = Variable(torch.FloatTensor([0.0]))
    if args.cuda:
        loss = loss.cuda()

    for o, t, l in zip(output, targets, lengths):
        loss += criterion(o[:l], t[:l])

    return loss

def evaluate(data_source, eval_batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    total_timesteps = 0
    for X, targets in data_source:
        hidden = model.init_hidden(eval_batch_size)
        output, _ = model(X, hidden)
        output_flat = output.view(-1, len(vocab))
        total_loss += len(X) * criterion(output_flat, targets).data
        total_timesteps += len(X)
        hidden = repackage_hidden(hidden)
    return total_loss[0] / total_timesteps


# @profile
def train(logger):
    model.train()

    optim = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.beta)
    for batch, (X, targets, lengths) in enumerate(train_data):
        X = X.cuda()        
        targets = targets.cuda()
        lenghts = lengths.cuda()
        hidden = model.init_hidden(X.size(1))

        output, _ = model(X, hidden)
        # loss = criterion(output.view(-1, len(vocab)), targets)
        loss = padded_loss(output, targets, lengths)

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)

        optim.step()

        logger.log(loss.data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM Language Model')
    parser.add_argument('--data', type=str, required=True,
                        help='location of the data corpus')
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
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--shuffle-lines', action='store_true',
                        help='shuffle lines before every epoch')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--load', type=str, required=True,
                        help='where to load a model from')
    parser.add_argument('--save', type=str,  required=True,
                        help='path to save the final model')
    parser.add_argument('--cpu-save', action='store_true',
                        help='save a model to the path "<SAVE>.cpu"')
    args = parser.parse_args()

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if args.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    print("loading model...")
    with open(args.load, 'rb') as f:
        lm = language_model.load(f)
    vocab = lm.vocab
    model = lm.model
    if args.cuda:
        model.cuda()
    print(model)

    print("preparing data...")
    train_data = data.LineOrientedCorpus(os.path.join(args.data, 'train.txt'), vocab, args.cuda)
    train_data = DataLoader(train_data, args.batch_size, collate_fn=data.packing_collate)

    valid_data = data.LineOrientedCorpus(os.path.join(args.data, 'valid.txt'), vocab, args.cuda)
    test_data = data.LineOrientedCorpus(os.path.join(args.data, 'test.txt'), vocab, args.cuda)

    criterion = nn.NLLLoss()

    print("training...")
    # Loop over epochs.
    lr = args.lr
    best_val_loss = None

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()

            logger = ProgressLogger(epoch, args.log_interval, lr, len(train_data))
            train(logger)
            val_loss = evaluate(val_gen.iterable_data())
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, logger.time_since_creation(),
                                               val_loss, math.exp(val_loss)))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                if args.cpu_save:
                    model.cpu()
                    with open(args.save+".cpu", 'wb') as f:
                        torch.save(model, f)
                if args.cuda:
                    model.cuda()
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                # lr /= 4.0
                pass

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open(args.save, 'rb') as f:
        model = torch.load(f)

    # Run on test data.
    test_loss = evaluate(test_gen.iterable_data())
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)

    if args.cpu_save:
        model.cpu()
        with open(args.save + ".cpu", 'wb') as f:
            torch.save(model, f)
