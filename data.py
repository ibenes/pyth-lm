import os
import torch


def tokens_from_fn(fn, vocab, randomize):
    word_ids = []
    with open(fn, 'r') as f:
        lines = f.read().split('\n')

        if randomize:
            import random
            random.shuffle(lines)

        for line in lines:
            words = line.split()
            word_ids.extend([vocab[w] for w in words])

    return torch.LongTensor(word_ids)
