import torch
from torch.autograd import Variable
import torch.nn as nn

from .runtime_utils import repackage_hidden
from .tensor_reorganization import TensorReorganizer


def prepare_inputs(inputs, do_transpose, use_ivecs, custom_batches):
    X = inputs[0]
    batch_size = X.size(0)
    if do_transpose:
        X = X.t()
    X = Variable(X)

    targets = inputs[1]
    if do_transpose:
        targets = targets.t().contiguous()
    targets_flat = Variable(targets.view(-1))

    if use_ivecs:
        ivecs = Variable(inputs[2])
    else:
        ivecs = None

    if custom_batches:
        mask = Variable(inputs[-1])  # 3
    else:
        mask = None

    return X, targets_flat, ivecs, mask, batch_size


def evaluate_(lm, data_source, use_ivecs, custom_batches):
    lm.eval()

    total_loss = 0.0
    total_timesteps = 0

    if custom_batches:
        hs_reorganizer = TensorReorganizer(lm.model.init_hidden)

    hidden = None
    do_transpose = not lm.model.batch_first

    for inputs in data_source:
        X, targets_flat, ivecs, mask, batch_size = prepare_inputs(
            inputs,
            do_transpose, use_ivecs, custom_batches
        )

        if hidden is None:
            hidden = lm.model.init_hidden(batch_size)

        if custom_batches:
            hidden = hs_reorganizer(hidden, mask, batch_size)

        hidden = repackage_hidden(hidden)

        if use_ivecs:
            output, hidden = lm.model(X, hidden, ivecs)
        else:
            output, hidden = lm.model(X, hidden)
        output = lm.decoder(output)
        output_flat = output.view(-1, output.size(-1))

        total_loss += lm.criterion(output_flat, targets_flat).data
        total_timesteps += len(targets_flat)

    return total_loss[0] / total_timesteps


def evaluate(model, data_source, use_ivecs):
    return evaluate_(
        model, data_source,
        use_ivecs, custom_batches=True
    )


def evaluate_no_transpose(model, data_source, use_ivecs):
    return evaluate_(
        model, data_source,
        use_ivecs, custom_batches=False
    )


# TODO time X batch or vice-versa?

def train_(lm, data, optim, logger, clip, use_ivecs, custom_batches):
    lm.train()

    if custom_batches:
        hs_reorganizer = TensorReorganizer(lm.model.init_hidden)

    hidden = None
    do_transpose = not lm.model.batch_first

    for inputs in data:
        X, targets_flat, ivecs, mask, batch_size = prepare_inputs(
            inputs,
            do_transpose, use_ivecs, custom_batches
        )

        if hidden is None:
            hidden = lm.model.init_hidden(batch_size)

        if custom_batches:
            hidden = hs_reorganizer(hidden, mask, batch_size)
        hidden = repackage_hidden(hidden)

        if use_ivecs:
            output, hidden = lm.model(X, hidden, ivecs)
        else:
            output, hidden = lm.model(X, hidden)
        output = lm.decoder(output)
        output_flat = output.view(-1, output.size(-1))

        loss = lm.criterion(output_flat, targets_flat) / output_flat.size(0)

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(lm.parameters(), clip)

        optim.step()
        logger.log(loss.data)


def train(model, data, optim, logger, clip, use_ivecs):
    train_(
        model, data, optim, logger, clip,
        use_ivecs, custom_batches=True
    )


def train_no_transpose(model, data, optim, logger, clip, use_ivecs):
    train_(
        lm, data, optim, logger, clip,
        use_ivecs, custom_batches=False
    )
