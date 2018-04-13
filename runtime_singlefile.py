import torch
from torch import nn

from runtime_utils import repackage_hidden

def train(lm, data, batch_size, logger, optim, clip):
    model = lm.model
    vocab = lm.vocab

    model.train()
    hidden = model.init_hidden(batch_size)

    criterion = nn.NLLLoss()
    for batch, (X, targets) in enumerate(data):
        hidden = repackage_hidden(hidden)

        output, hidden = model(X, hidden)
        loss = criterion(output.view(-1, len(vocab)), targets)

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), clip)

        optim.step()

        logger.log(loss.data)


def evaluate(lm, data_source, eval_batch_size=10):
    model = lm.model
    vocab = lm.vocab

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
