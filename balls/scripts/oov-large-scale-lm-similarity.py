#!/usr/bin/env python

import argparse
import sys

from embeddings_io import all_embs_from_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    keys, embs = all_embs_from_file(sys.stdin)

    for i, emb_a in enumerate(embs[:-1]):
        others = embs[(i+1):]
        similarities = emb_a @ others.T

        similarities_str = " ".join(["{:.3e}".format(x) for x in similarities])
        sys.stdout.write("{}\n".format(similarities_str))
