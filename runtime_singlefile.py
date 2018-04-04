from torch import nn
from runtime_utils import repackage_hidden

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
