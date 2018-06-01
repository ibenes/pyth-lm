import torch
import torch.nn as nn
from torch.autograd import Variable


class IvecLSTMLanguageModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers, ivec_dim, dropout=0.5, tie_weights=False):
        super(IvecLSTMLanguageModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
         
        self.decoder = nn.Linear(nhid + ivec_dim, ntoken)

        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.nhid = nhid
        self.nlayers = nlayers
        self._ivec_dim = ivec_dim

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, ivecs):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        # TODO appending ivecs to the outputs
        decoded = nn.LogSoftmax()(self.decoder(output.view(output.size(0)*output.size(1), output.size(2))))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def output_expected_embs(self, input):
        raise NotImplementedError("Attempt to call an un-maintained method! Will be subject to a refactor.")
        assert (len(input.size()) == 2) # time X batch index
        assert (input.size()[1] == 1)

        hidden = self.init_hidden(1)
        emb = self.drop(self.encoder(input))
        outputs, _ = self.rnn(emb, hidden)
        return outputs
        

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
