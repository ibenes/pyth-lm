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


def variablilize_targets(targets):
    return Variable(targets.contiguous().view(-1))
