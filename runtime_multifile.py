import sys

import torch
from torch.autograd import Variable
import torch.nn as nn

from runtime_utils import repackage_hidden

from tensor_reorganization import TensorReorganizer

from loggers import NoneLogger

def evaluate_(model, data_source, use_ivecs, rnn_mode):
    model.eval()

    total_loss = 0.0
    total_timesteps = 0

    hs_reorganizer = TensorReorganizer(model.init_hidden)
    hidden = None

    for inputs in data_source:
        X = inputs[0] 
        if rnn_mode:
            X = X.t()
        X = Variable(X)
        inputs = (X,) + inputs[1:]

        X = inputs[0]
        targets = inputs[1]
        ivecs = inputs[2]
        mask = inputs[-1] # 3

        if hidden is None:
            if rnn_mode:
                hidden = model.init_hidden(X.size(1))
            else:
                hidden = model.init_hidden(X.size(0))

        if rnn_mode:
            hidden = hs_reorganizer(hidden, Variable(mask), X.size(1))

        hidden = repackage_hidden(hidden)

        criterion = nn.NLLLoss()
        if use_ivecs:
            output, hidden = model(X, hidden, Variable(ivecs))
        else:
            output, hidden = model(X, hidden)
        output_flat = output.view(-1, output.size(-1))
        if rnn_mode:
            targets_flat = variablilize_targets(targets)
        else:
            targets_flat = variablilize_targets_no_transpose(targets)
        curr_loss = len(X) * criterion(output_flat, targets_flat).data
        total_loss += curr_loss
        total_timesteps += len(X)

    return total_loss[0] / total_timesteps
    

def evaluate(model, data_source, use_ivecs):
    return evaluate_(model, data_source, use_ivecs, rnn_mode=True)


def evaluate_no_transpose(model, data_source, use_ivecs):
    return evaluate_(model, data_source, use_ivecs, rnn_mode=False)


class BatchFilter:
    def __init__(self, data, batch_size, bptt, min_batch_size): 
        self._data = data
        self._batch_size = batch_size
        self._bptt = bptt
        self._min_batch_size = min_batch_size

        self._nb_skipped_updates = 0
        self._nb_skipped_words = 0
        self._nb_skipped_seqs = 0 # accumulates size of skipped batches

    def __iter__(self):
        for batch in self._data:
            X = batch[0]
            if X.size(0) >= self._min_batch_size:
                yield batch
            else:
                self._nb_skipped_updates += 1
                self._nb_skipped_words += X.size(0) * X.size(1)
                self._nb_skipped_seqs += X.size(0)

    def report(self):
        if self._nb_skipped_updates > 0:
            sys.stderr.write(
                "WARNING: due to skipping, a total of {} updates was skipped,"
                " containing {} words. Avg batch size {}. Equal to {} full batches"
                "\n".format(
                    self._nb_skipped_updates,
                    self._nb_skipped_words,
                    self._nb_skipped_seqs/self._nb_skipped_updates,
                    self._nb_skipped_words/(self._batch_size*self._bptt)
                )
            )

# TODO time X batch or vice-versa?

def train_(model, data, optim, logger, clip, use_ivecs, rnn_mode):
    model.train()

    hs_reorganizer = TensorReorganizer(model.init_hidden)
    hidden = None

    for batch, inputs in enumerate(data):
        X = inputs[0]
        if rnn_mode:
            X = X.t()
        X = Variable(X)
        inputs = (X,) + inputs[1:]

        X = inputs[0]
        targets = inputs[1]
        ivecs = inputs[2]
        mask = inputs[-1] # 3

        if hidden is None:
            hidden = model.init_hidden(X.size(1))

        if rnn_mode:
            hidden = hs_reorganizer(hidden, Variable(mask), X.size(1))
        hidden = repackage_hidden(hidden)

        criterion = nn.NLLLoss()
        # print("[debug]", X, hidden, ivecs)
        if use_ivecs:
            output, hidden = model(X, hidden, Variable(ivecs))
        else:
            output, hidden = model(X, hidden)
        output_flat = output.view(-1, output.size(-1))

        if rnn_mode:
            targets_flat = variablilize_targets(targets)
        else:
            targets_flat = variablilize_targets_no_transpose(targets)
        loss = criterion(output_flat, targets_flat)

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), clip)

        optim.step()
        logger.log(loss.data)

def train(model, data, optim, logger, clip, use_ivecs):
    train_(model, data, optim, logger, clip, use_ivecs, rnn_mode=True)

def train_no_transpose(model, data, optim, logger, clip, use_ivecs):
    train_(model, data, optim, logger, clip, use_ivecs, rnn_mode=False)

def train_debug(lm, data, optim, logger, batch_size, clip, cuda, use_ivecs=True, grad_logger=NoneLogger()):
    model = lm.model
    model.train()
    hs_reorganizer = TensorReorganizer(model.init_hidden)
    hidden = model.init_hidden(batch_size)

    if cuda:
        model.cuda()
        hidden = tuple(h.cuda() for h in hidden)


    for batch, (X, targets, ivecs, mask) in enumerate(data):
        X = Variable(X.t())
        hidden = hs_reorganizer(hidden, Variable(mask), X.size(1))
        hidden = repackage_hidden(hidden)

        criterion = nn.NLLLoss()
        # print("[debug]", X, hidden, ivecs)
        if use_ivecs:
            output, hidden = model(X, hidden, Variable(ivecs))
        else:
            output, hidden = model(X, hidden)
        loss = criterion(output.view(-1, len(lm.vocab)), variablilize_targets(targets))

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), clip)
        grad_logger.log()

        optim.step()
        logger.log(loss.data)


def variablilize_targets_no_transpose(targets):
    return Variable(targets.contiguous().view(-1))

def variablilize_targets(targets):
    return Variable(targets.t().contiguous().view(-1))
