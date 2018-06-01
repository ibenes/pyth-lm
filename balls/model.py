import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import math

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = nn.LogSoftmax()(self.decoder(output.view(output.size(0)*output.size(1), output.size(2))))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())


class ResidualMemoryModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(ResidualMemoryModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)

        self._nb_layers = nlayers
        self._residals_f = 3

        self._cs = nn.ModuleList()
        self._ps = nn.ModuleList()

        for i in range(self._nb_layers):
            self._cs.append(nn.Linear(nhid, nhid))
            self._ps.append(nn.Linear(nhid, nhid))

        self.decoder = nn.Linear(nhid, ntoken)

        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.nhid = nhid
        self.nlayers = nlayers

        self._hs = [0] * self._nb_layers # for inspection purpose only

    def init_weights(self):
        # TODO this HAS to be changed :-)
        initrange = math.sqrt(6.0/(10000+100))
        self.encoder.weight.data.uniform_(-initrange, initrange)

        weight_range = math.sqrt(6.0/(100+100))
        for i in range(self._nb_layers):
            self._cs[i].weight.data.uniform_(-weight_range, weight_range)
            self._ps[i].weight.data.uniform_(-weight_range, weight_range)
            self._cs[i].bias.data.uniform_(0,0.1)
            self._ps[i].bias.data.uniform_(0,0.1)

        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    # @profile
    def forward(self, input, hidden):
        nb_steps = input.size()[0]

        emb = self.drop(self.encoder(input))
        top_hiddens = []

        for step in range(nb_steps):
            curr_hidden = []
            curr_hidden.append(emb[step])
            for i in range(self._nb_layers):
                curr_proj = self._cs[i](curr_hidden[i]) 
                hist_h_projected = self._cs[i](hidden[i][i])
                hist_proj = self._ps[i](hist_h_projected)
                h_i = curr_proj + hist_proj
                if i % self._residals_f == self._residals_f-1:
                    h_i += curr_hidden[i-self._residals_f]
                h_i = F.relu(h_i)
                self._hs[i] = h_i.data.cpu().numpy()
                curr_hidden.append(h_i)

            top_hiddens.append(curr_hidden[-1])

            # add these hidden representation in the current timestep to the "history"
            # while also cutting anything too old
            hidden = list(hidden)
            for i, old_hid in enumerate(hidden):
                hidden[i] = (curr_hidden[i], ) + old_hid[:-1]


        output = torch.stack(top_hiddens)

        output = self.drop(output)
        decoded = nn.LogSoftmax()(self.decoder(output.view(output.size(0)*output.size(1), output.size(2))))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden


    def grads(self):
        return [self._cs[i].weight.grad.data for i in range(self.nlayers)]


    def init_hidden(self, bsz):
        weight = next(self.parameters()).data

        initial = []
        for i in range(self._nb_layers):
            hist_len = i + 1
            layer_initial = [Variable(weight.new(bsz, self.nhid).zero_()) for i in range(hist_len)]
            initial.append(layer_initial)

        return tuple([tuple(layer_initial) for layer_initial in initial])