import os
import torch


def tokens_from_fn(fn, vocab, randomize, regime='words'):
    ids = []
    with open(fn, 'r') as f:
        lines = f.read().split('\n')

        if randomize:
            import random
            random.shuffle(lines)

        for line in lines:
            if regime == 'words':
                elements = line.split()
            elif regime == 'chars':
                elements = line
            else:
                raise ValueError("unsupported regime {}".format(regime))

            ids.extend([vocab[e] for e in elements])

    return torch.LongTensor(ids)
