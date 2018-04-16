import sys

import torch
from torch.autograd import Variable
import torch.nn as nn

from runtime_utils import repackage_hidden

from tensor_reorganization import TensorReorganizer

from loggers import NoneLogger

def evaluate_(model, data_source, use_ivecs, do_transpose, custom_batches, batch_first):
    model.eval()
    criterion = nn.NLLLoss()

    total_loss = 0.0
    total_timesteps = 0

    if custom_batches:
        hs_reorganizer = TensorReorganizer(model.init_hidden)

    hidden = None

    for inputs in data_source:
        X = inputs[0] 
        if batch_first:
            batch_size = X.size(0)
        else:
            batch_size = X.size(1) 
        if do_transpose:
            X = X.t()
        X = Variable(X)
        inputs = (X,) + inputs[1:]

        X = inputs[0]
        targets = inputs[1]
        if use_ivecs:
            ivecs = inputs[2]
        if custom_batches:
            mask = inputs[-1] # 3

        if hidden is None:
            hidden = model.init_hidden(batch_size)

        if custom_batches:
            hidden = hs_reorganizer(hidden, Variable(mask), batch_size)

        hidden = repackage_hidden(hidden)

        if use_ivecs:
            output, hidden = model(X, hidden, Variable(ivecs))
        else:
            output, hidden = model(X, hidden)
        output_flat = output.view(-1, output.size(-1))

        if do_transpose:
            targets = targets.t().contiguous()
        targets_flat = Variable(targets.view(-1))

        curr_loss = len(X) * criterion(output_flat, targets_flat).data
        total_loss += curr_loss
        total_timesteps += len(X)

    return total_loss[0] / total_timesteps
    

def evaluate(model, data_source, use_ivecs):
    return evaluate_(model, data_source, use_ivecs, do_transpose=True, custom_batches=True, batch_first=True)

def evaluate_no_transpose(model, data_source, use_ivecs):
    return evaluate_(model, data_source, use_ivecs, do_transpose=False, custom_batches=False, batch_first=True)

def evaluate_uniform_stream(model, data_source):
    return evaluate_(model, data_source, use_ivecs=False, do_transpose=False, custom_batches=False, batch_first=False)


# TODO time X batch or vice-versa?

def train_(model, data, optim, logger, clip, use_ivecs, do_transpose, custom_batches, batch_first):
    model.train()
    criterion = nn.NLLLoss()

    if custom_batches:
        hs_reorganizer = TensorReorganizer(model.init_hidden)

    hidden = None

    for batch, inputs in enumerate(data):
        X = inputs[0]
        if batch_first:
            batch_size = X.size(0)
        else:
            batch_size = X.size(1)
            
        if do_transpose:
            X = X.t()
        X = Variable(X)
        inputs = (X,) + inputs[1:]

        X = inputs[0]
        targets = inputs[1]
        if use_ivecs:
            ivecs = inputs[2]
        if custom_batches:
            mask = inputs[-1] # 3

        if hidden is None:
            hidden = model.init_hidden(batch_size)

        if custom_batches:
            hidden = hs_reorganizer(hidden, Variable(mask), batch_size)
        hidden = repackage_hidden(hidden)

        if use_ivecs:
            output, hidden = model(X, hidden, Variable(ivecs))
        else:
            output, hidden = model(X, hidden)
        output_flat = output.view(-1, output.size(-1))

        if do_transpose:
            targets = targets.t().contiguous()
        targets_flat = Variable(targets.view(-1))

        loss = criterion(output_flat, targets_flat)

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), clip)

        optim.step()
        logger.log(loss.data)

def train(model, data, optim, logger, clip, use_ivecs):
    train_(model, data, optim, logger, clip, use_ivecs, do_transpose=True, custom_batches=True, batch_first=True)

def train_no_transpose(model, data, optim, logger, clip, use_ivecs):
    train_(model, data, optim, logger, clip, use_ivecs, do_transpose=False, custom_batches=False, batch_first=True)

def train_uniform_stream(model, data, logger, optim, clip):
    train_(model, data, optim, logger, clip, use_ivecs=False, do_transpose=False, custom_batches=False, batch_first=False)


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
