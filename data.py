import os
import torch
from torch.autograd import Variable
import torch.utils.data

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


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
        data = Variable(source[i:i+act_seq_len], volatile=evaluation)
        target = Variable(source[i+1:i+1+act_seq_len].view(-1))
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

class LineOrientedCorpus(torch.utils.data.Dataset):
    def __init__(self, path, vocab, cuda):
        self._vocab = vocab
        self._cuda = cuda

        self.tokenize(path)

    def __len__(self):
        return len(self._sentences)

    def __getitem__(self, i):
        return self._sentences[i][:-1], self._sentences[i][1:]

    def tokenize(self, path):
        self._sentences = []

        with open(path, 'r') as f:
            for line in f:
                words = line.split()
                ids = torch.LongTensor(len(words))
                for i, w in enumerate(words):
                    ids[i] = self._vocab.w2i(w)

                if self._cuda:
                    ids.cuda()

                self._sentences.append(ids)

def pad_to_length(x, length):
    assert x.size(0) <= length 

    if x.size(0) < length:
        data_size = x.size()[1:]
        app_length = length - x.size(0)
        appendix = torch.zeros((app_length, )).long()
        return torch.cat([x, appendix])
    else:
        return x

def packing_collate(batch):
    batch_x, batch_t = zip(*batch)

    lengts = torch.LongTensor([len(x) for x in batch_x])
    max_len = lengts.max()

    padded_xs = torch.stack([pad_to_length(x, max_len) for x in batch_x])
    padded_ts = torch.stack([pad_to_length(t, max_len) for t in batch_t])

    lengts, perm_idx = lengts.sort(0, descending=True)
    padded_xs = padded_xs[perm_idx]
    padded_ts = padded_ts[perm_idx]

    # transpose to get TxB order
    return  Variable(padded_xs.t()), Variable(padded_ts.t()), lengts
