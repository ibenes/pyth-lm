#!/usr/bin/env python

import argparse
import sys

import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--transform', required=True)
    args = parser.parse_args()

    transform = np.loadtxt(args.transform)

    for line in sys.stdin:
        fields = line.split()
        key = fields[0]
        embedding = np.asarray([float(e) for e in fields[1:]])
        projected = embedding @ transform

        emb_str = " ".join(["{:.4f}".format(e) for e in projected])
        sys.stdout.write("{} {}\n".format(key, emb_str))
