import sys


import torch
from torch.autograd import Variable
import torch.nn as nn

from runtime_utils import repackage_hidden

from hidden_state_reorganization import HiddenStateReorganizer

def evaluate(lm, data_source, batch_size, cuda, use_ivecs=True):
    model = lm.model

    model.eval()

    total_loss = 0.0
    total_timesteps = 0

    hs_reorganizer = HiddenStateReorganizer(model)
    hidden = model.init_hidden(batch_size)

    if cuda:
        model.cuda()
        hidden = tuple(h.cuda() for h in hidden)

    for X, targets, ivecs, mask in data_source:
        hidden = hs_reorganizer(hidden, mask, X.size(1))
        hidden = repackage_hidden(hidden)

        criterion = nn.NLLLoss()
        if use_ivecs:
            output, hidden = model(Variable(X), hidden, Variable(ivecs))
        else:
            output, hidden = model(Variable(X), hidden)
        output_flat = output.view(-1, len(lm.vocab))
        curr_loss = len(X) * criterion(output_flat, variablilize_targets(targets)).data
        total_loss += curr_loss
        total_timesteps += len(X)

    return total_loss[0] / total_timesteps


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
            if X.size(1) >= self._min_batch_size:
                yield batch
            else:
                self._nb_skipped_updates += 1
                self._nb_skipped_words += X.size(0) * X.size(1)
                self._nb_skipped_seqs += X.size(1)

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


def train(lm, data, optim, logger, batch_size, bptt, min_batch_size, clip, cuda, use_ivecs=True):
    model = lm.model
    model.train()
    hs_reorganizer = HiddenStateReorganizer(model)
    hidden = model.init_hidden(batch_size)

    if cuda:
        model.cuda()
        hidden = tuple(h.cuda() for h in hidden)

    data_filter = BatchFilter(data, batch_size, bptt, min_batch_size)

    for batch, (X, targets, ivecs, mask) in enumerate(data_filter):
        hidden = hs_reorganizer(hidden, mask, X.size(1))
        hidden = repackage_hidden(hidden)

        criterion = nn.NLLLoss()
        if use_ivecs:
            output, hidden = model(Variable(X), hidden, Variable(ivecs))
        else:
            output, hidden = model(Variable(X), hidden)
        loss = criterion(output.view(-1, len(lm.vocab)), variablilize_targets(targets))

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), clip)

        optim.step()
        logger.log(loss.data)

    data_filter.report()


def variablilize_targets(targets):
    return Variable(targets.contiguous().view(-1))
