import torch
import torch.nn as nn
from torch.autograd import Variable


class OutputEnhancedLM(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers, ivec_dim, dropout=0.5, dropout_ivec=0.0, tie_weights=False):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.drop_ivec = nn.Dropout(dropout_ivec)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)
        self.ivec_proj = nn.Linear(ivec_dim, ntoken)

        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.ivec_proj.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, ivec):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        ivec = self.drop_ivec(ivec)
        decoded = nn.LogSoftmax(dim=2)(self.decoder(output) + self.ivec_proj(ivec))

        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))

class OutputBottleneckLM(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers, ivec_dim, dropout=0.5, dropout_ivec=0.0, tie_weights=False):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.drop_ivec = nn.Dropout(dropout_ivec)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)
        self.ivec_proj = nn.Linear(ivec_dim, nhid)

        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.ivec_proj.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, ivec):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        ivec = self.drop_ivec(ivec)
        combined = output + self.ivec_proj(ivec)
        decoded = nn.LogSoftmax(dim=2)(self.decoder(combined))

        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))


class GatedLinearUnit(nn.Module):   
    def __init__(self, in_size, out_size):
        self.transfer = nn.Linear(in_size, out_size)
        self.gate = nn.Linear(in_size, out_size)

    def forward(self, input):
        t = self.transfer(input) 
        g = F.sigmoid(self.gate(input))
        return t * g


class OutputGLULM(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers, ivec_dim, dropout=0.5, dropout_ivec=0.0, tie_weights=False):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.drop_ivec = nn.Dropout(dropout_ivec)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)
        self.glu = GatedLinearUnit(nhid+ivec_dim, ntoken)

        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.ivec_proj.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, ivec):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        ivec = self.drop_ivec(ivec)
        combined = output + self.ivec_proj(ivec)
        decoded = nn.LogSoftmax(dim=2)(self.decoder(combined))

        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))


class OutputMultiplicativeLM(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers, ivec_dim, dropout=0.5, tie_weights=False):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)
        self.ivec_proj = nn.Linear(ivec_dim, ntoken)

        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.ivec_proj.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, ivec):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        ivec = self.drop(ivec)
        decoded = nn.LogSoftmax(dim=2)(self.decoder(output) * self.ivec_proj(ivec))

        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))


class InputEnhancedLM(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers, ivec_dim, dropout=0.5, tie_weights=False):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)
        self.ivec_proj = nn.Linear(ivec_dim, nhid)

        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.ivec_proj.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, ivec):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb + self.ivec_proj(ivec), hidden)
        output = self.drop(output)
        ivec = self.drop(ivec)
        decoded = nn.LogSoftmax(dim=2)(self.decoder(output))

        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))


class IvecOnlyLM(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ivec_dim, dropout=0.5, tie_weights=False):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.decoder = nn.Linear(ivec_dim, ntoken)

        self.ivec_dim = ivec_dim

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, ivec):
        decoded = nn.LogSoftmax(dim=2)(torch.stack([self.decoder(ivec)] * input.size(0)))

        return decoded, hidden

    def init_hidden(self, bsz):
        # not used, but to fit into the framework of other ivec-LMs
        weight = next(self.parameters()).data
        return (Variable(weight.new(1, bsz, self.ivec_dim).zero_())) 
