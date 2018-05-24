import torch.nn as nn
from torch.autograd import Variable


class LSTMLanguageModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(LSTMLanguageModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.nhid = nhid
        self.nlayers = nlayers

        self.batch_first = False

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = nn.LogSoftmax(dim=2)(self.decoder(output))
        return decoded, hidden

    def output_expected_embs(self, input):
        assert (len(input.size()) == 2)  # time X batch index
        assert (input.size()[1] == 1)

        hidden = self.init_hidden(1)
        emb = self.drop(self.encoder(input))
        outputs, _ = self.rnn(emb, hidden)
        return outputs

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
