import os
import torch

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

class Corpus(object):
    def __init__(self, path, vocab, randomize=False):
        self.dictionary = vocab
        self._randomize = randomize
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = nb_words(f)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0

            lines = f.read().split('\n')

            if self._randomize:
                import random
                random.shuffle(lines)

            for line in lines:
                words = line.split()
                for word in words:
                    ids[token] = self.dictionary.w2i(word)
                    token += 1

        return ids

def _batchify(data, batch_size, randomize):
    sent_ids = range(len(data))

    if randomize:
        random.shuffle(sent_ids)

    batches = []
    i = 0
    while i + batch_size < len(data):
        batch_ids = sent_ids[i:i+batch_size]
        batches.append(batch_ids)
        i += batch_size



class LineOrientedCorpus:
    def __init__(self, path, vocab, randomize=True):
        self._vocab = vocab
        self._randomize = randomize
        self._train_ids = self.tokenize(os.path.join(path, 'train.txt'))
        self._valid_ids = self.tokenize(os.path.join(path, 'valid.txt'))
        self._test_ids = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        sentences = []

        with open(path, 'r') as f:
            for line in f:
                words = line.split()
                ids = torch.LongTensor(len(words))
                for i, w in enumerate(words):
                    ids[i] = self._vocab.w2i(w)

                sentences.append(ids)

    def batched_train(self, batch_size, randomize=True):
        pass

    def batched_valid(self, batch_size=10, randomize=False):
        pass

    def batched_test(self, batch_size=10, randomize=False):
        pass
