from torch import nn
from runtime_utils import repackage_hidden

import data

def format_data(path, vocab, train_batch_size, eval_batch_size, cuda, shuffle_lines):
    corpus = data.Corpus(path, vocab, shuffle_lines)

    train = data.batchify(corpus.train, train_batch_size, cuda)
    valid = data.batchify(corpus.valid, eval_batch_size, cuda)
    test = data.batchify(corpus.test, eval_batch_size, cuda)

    return train, valid, test

def evaluate(lm, data_source, cuda, eval_batch_size=10):
    model = lm.model
    vocab = lm.vocab

    if cuda:
        model.cuda()
    # Turn on evaluation mode which disables dropout.
    model.eval()
    criterion = nn.NLLLoss()

    total_loss = 0
    total_timesteps = 0
    hidden = model.init_hidden(eval_batch_size)
    for X, targets in data_source:
        output, hidden = model(X, hidden)
        output_flat = output.view(-1, len(vocab))
        total_loss += len(X) * criterion(output_flat, targets).data
        total_timesteps += len(X)
        hidden = repackage_hidden(hidden)
    return total_loss[0] / total_timesteps
