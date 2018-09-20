#!/usr/bin/env python

import argparse
import sys

import numpy as np

from embeddings_io import emb_line_iterator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--transform', required=True)
    args = parser.parse_args()

    transform = np.loadtxt(args.transform)

    for key, embedding in emb_line_iterator(sys.stdin):
        projected = embedding @ transform

        emb_str = " ".join(["{:.4f}".format(e) for e in projected])
        sys.stdout.write("{} {}\n".format(key, emb_str))
