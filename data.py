import os
import torch

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
