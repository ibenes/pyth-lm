import argparse
import math
import torch
import torch.nn as nn
from torch.autograd import Variable

import data
import lstm_model
import language_model

from runtime_singlefile import evaluate, format_data

 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM Language Model')
    parser.add_argument('--data', type=str, required=True,
                        help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=35,
                        help='sequence length')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--shuffle-lines', action='store_true',
                        help='shuffle lines before every epoch')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--load', type=str, required=True,
                        help='where to load a model from')
    args = parser.parse_args()
    print(args)

    # Set the random seed manually for reproducibility.
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
    train_data, val_data, test_data = format_data(
        args.data, 
        vocab, 
        args.batch_size,
        args.batch_size,
        args.cuda,
        shuffle_lines=False
    )
    train_gen = data.DataIteratorBuilder(train_data, args.bptt)
    val_gen = data.DataIteratorBuilder(val_data, args.bptt)
    test_gen = data.DataIteratorBuilder(test_data, args.bptt)

    criterion = nn.NLLLoss()

    # Run on test data.
    for name, gen in zip("train val test".split(), [train_gen, val_gen, test_gen]):
        loss = evaluate(lm, gen.iterable_data(), args.cuda, args.batch_size)
        print('{} loss {:5.2f} | {} ppl {:8.2f}'.format(name, loss, name, math.exp(loss)))
