import os
import torch
from torch.autograd import Variable
import torch.utils.data

class DataIteratorBuilder():
    def __init__(self, data, seq_len):
        self._data = data
        self._seq_len = seq_len

    def iterable_data(self):
        """
            The data is expected to be arranged as [time][batch], 
            slices along the time dimension will be provided
        """

        for i in range(0, self._data.size(0) -1, self._seq_len):
            yield self._get_batch(self._data, i)

    def _get_batch(self, source, i, evaluation=False):
        act_seq_len = min(self._seq_len, len(source) - 1 - i)
        data = source[i:i+act_seq_len]
        target = source[i+1:i+1+act_seq_len].view(-1)
        return data, target


def batchify(data, bsz, cuda):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if cuda:
        data = data.cuda()
    return data

def nb_words(f):
    n = 0
    for line in f:
        words = line.split()
        n += len(words)

    return n

def tokens_from_fn(fn, vocab, randomize):
    with open(fn, 'r') as f:
        nb_tokens = nb_words(f)

    # Tokenize file content
    with open(fn, 'r') as f:
        ids = torch.LongTensor(nb_tokens)
        token = 0

        lines = f.read().split('\n')

        if randomize:
            import random
            random.shuffle(lines)

        for line in lines:
            words = line.split()
            for word in words:
                ids[token] = vocab.w2i(word)
                token += 1

    return ids
