import os
import torch


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
