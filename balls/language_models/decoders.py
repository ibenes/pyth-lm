import torch


class FullSoftmaxDecoder(torch.nn.Module):
    def __init__(self, nb_hidden, nb_output, init_range=0.1):
        super().__init__()

        self.projection = torch.nn.Linear(nb_hidden, nb_output)
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)

        self.projection.weight.data.uniform_(-init_range, init_range)
        self.projection.bias.data.fill_(0)

    def forward(self, X):
        a = self.projection(X)
        return self.log_softmax(a)
