import numpy as np


def emb_line_iterator(f):
    for line in f:
        fields = line.split()
        key = fields[0]
        embedding = np.asarray([float(e) for e in fields[1:]])

        yield key, embedding


def all_embs_from_file(f):
    embs = []
    keys = []

    for key, emb in emb_line_iterator(f):
        embs.append(emb)
        keys.append(key)

    return keys, np.stack(embs)


def str_from_embedding(emb):
    return " ".join(["{:.4f}".format(e) for e in emb])
