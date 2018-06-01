#!/usr/bin/env python

import language_model

import torch
from torch.autograd import Variable
import numpy as np


class EmbsExpectator():
    def __init__(self, lm):
        self._model = lm.model
        self._vocab = lm.vocab

    def __call__(self, sentence):
        if len(sentence) == 0:
            return np.zeros(self._model.nhid)

        if isinstance(sentence[0], str):
            return self._expected_embedding(sentence) 
        elif isinstance(sentence[0], int):
            return self._expected_embedding_inds(sentence)
        else:
            raise TypeError("First element of the sentence is of unsupported type " + str(type(sentence[0])))

    def _expected_embedding(self, sentence):
        seq = [self._vocab.w2i(w) for w in sentence]
        return self._expected_embedding_inds(seq)

    def _expected_embedding_inds(self, inds):
        tensored_seq, _ = seqs_to_tensor([inds])
        last_emb = self._model.output_expected_embs(Variable(tensored_seq)).data[-1]
        return np.squeeze(last_emb.numpy())
        


def seqs_to_tensor(seqs):
    batch_size = len(seqs)
    maxlen = max([len(s) for s in seqs])

    ids = torch.LongTensor(batch_size, maxlen).zero_()
    for seq_n, seq in enumerate(seqs):
        for word_n, word in enumerate(seq):
            ids[seq_n, word_n] = word

    # indexing should be X[time][batch], thus we transpose
    data = ids.t().contiguous()
    return data, batch_size


if __name__ == '__main__':
    with open('/mnt/matylda5/ibenes/projects/katja-embs/ls-init.lm', 'rb') as f:
        lm = language_model.load(f)

    eetor = EmbsExpectator(lm)

    sentences = [
        "there is a ",
        "there are",
        "and then he",
        "suddenly the cat",
    ]

    exp_embs = [eetor(sent.split()) for sent in sentences]

    exp_embs = np.vstack(exp_embs)
    print(exp_embs @ exp_embs.T)
