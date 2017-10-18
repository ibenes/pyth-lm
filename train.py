import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable

import data
import model
import vocab

import IPython
import pickle


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target


def evaluate(data_source, eval_batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    hidden = model.init_hidden(eval_batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, evaluation=True)
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, len(vocab))
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)


# @profile
def train(heavy_logging, epoch_no):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    hidden = model.init_hidden(args.batch_size)

    optim = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.beta)
    # optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.beta)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        optim.zero_grad()

        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, len(vocab)), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optim.step()

        total_loss += loss.data

        # if batch == args.log_interval:
        #     print("storing grads...")
        #     with open("grads-" + str(batch) + ".pkl", 'wb') as f:
        #         pickle.dump(all_grads, f)
        #     print("stored")

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:.3e} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

            if heavy_logging:
                with open(heavy_logging + "/" + str(epoch_no) + "-" + str(batch) + ".grads.pkl", 'wb') as f:
                    pickle.dump([c.weight.grad.data.cpu().numpy() for c in model._cs], f)
                with open(heavy_logging + "/" + str(epoch_no) + "-" + str(batch) + ".hs.pkl", 'wb') as f:
                    pickle.dump(model._hs, f)



def format_data(path, vocab, eval_batch_size):
    corpus = data.Corpus(path, vocab, args.shuffle_lines)
    train = data.batchify(corpus.train, args.batch_size, args.cuda)
    valid = data.batchify(corpus.valid, eval_batch_size, args.cuda)
    test = data.batchify(corpus.test, eval_batch_size, args.cuda)

    return train, valid, test
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM Language Model')
    parser.add_argument('--data', type=str, required=True,
                        help='location of the data corpus')
    parser.add_argument('--wordlist', type=str, required=True,
                        help='word -> int map; Kaldi style "words.txt"')
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
    parser.add_argument('--cpu-save', type=str,
                        help='path to save a CPU model')
    parser.add_argument('--heavy-logging', type=str,
                        help='path to a directory to store saved dumps of gradients and weights')
    args = parser.parse_args()

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if args.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    print("preparing data...")
    with open(args.wordlist, 'r') as f:
        vocab = vocab.vocab_from_kaldi_wordlist(f)

    train_data, val_data, test_data = format_data(args.data, vocab, eval_batch_size=10)

    print("building model...")

    with open(args.load, 'rb') as f:
        model = torch.load(f)

    if args.cuda:
        model.cuda()

    criterion = nn.NLLLoss()

    print("training...")
    # Loop over epochs.
    lr = args.lr
    best_val_loss = None

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()

            train(args.heavy_logging, epoch)
            val_loss = evaluate(val_data)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               val_loss, math.exp(val_loss)))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                if args.cpu_save:
                    model.cpu()
                    with open(args.cpu_save, 'wb') as f:
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
    test_loss = evaluate(test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)

    if args.cpu_save:
        model.cpu()
        with open(args.cpu_save, 'wb') as f:
            torch.save(model, f)
