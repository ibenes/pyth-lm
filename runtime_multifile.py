import sys


import torch
from torch.autograd import Variable
import torch.nn as nn

from runtime_utils import repackage_hidden

from hidden_state_reorganization import HiddenStateReorganizer

def evaluate(lm, data_source, batch_size, cuda):
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
        output, hidden = model(Variable(X), hidden, Variable(ivecs))
        output_flat = output.view(-1, len(lm.vocab))
        curr_loss = len(X) * criterion(output_flat, variablilize_targets(targets)).data
        total_loss += curr_loss
        total_timesteps += len(X)

    return total_loss[0] / total_timesteps


def train(lm, data, optim, logger, batch_size, bptt, min_batch_size, clip, cuda):
    model = lm.model
    model.train()
    hs_reorganizer = HiddenStateReorganizer(model)
    hidden = model.init_hidden(batch_size)

    if cuda:
        model.cuda()
        hidden = tuple(h.cuda() for h in hidden)

    skipping = False
    nb_skipped_updates = 0
    nb_skipped_words = 0
    nb_skipped_seqs = 0 # accumulates size of skipped batches

    for batch, (X, targets, ivecs, mask) in enumerate(data):
        if X.size(1) < min_batch_size:
            skipping = True

        if skipping:
            nb_skipped_updates += 1
            nb_skipped_words += X.size(0) * X.size(1)
            nb_skipped_seqs += X.size(1)
            continue

        hidden = hs_reorganizer(hidden, mask, X.size(1))
        hidden = repackage_hidden(hidden)

        criterion = nn.NLLLoss()
        output, hidden = model(Variable(X), hidden, Variable(ivecs))
        loss = criterion(output.view(-1, len(lm.vocab)), variablilize_targets(targets))

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), clip)

        optim.step()
        logger.log(loss.data)

    if skipping:
        sys.stderr.write(
            "WARNING: due to skipping, a total of {} updates was skipped,"
            " containing {} words. Avg batch size {}. Equal to {} full batches"
            "\n".format(nb_skipped_updates, nb_skipped_words, nb_skipped_seqs/nb_skipped_updates,
                        nb_skipped_words/(batch_size*bptt))
        )


def variablilize_targets(targets):
    return Variable(targets.contiguous().view(-1))
