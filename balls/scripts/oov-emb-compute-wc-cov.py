#!/usr/bin/env python

import argparse
import sys
from typing import Dict, List

import numpy as np
from scipy.linalg import fractional_matrix_power

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--show-cov', action='store_true')
    args = parser.parse_args()

    collection: Dict[str, List[np.ndarray]] = {}

    for line in sys.stdin:
        fields = line.split()
        word = fields[0]
        emb = np.asarray([float(f) for f in fields[1:]])

        if word in collection:
            collection[word].append(emb)
        else:
            collection[word] = [emb]

    for w in collection:
        collection[w] = np.stack(collection[w])

    centered = []
    for w in collection:
        w_vectors = collection[w]
        mean = w_vectors.mean(axis=0)
        centered.append(w_vectors - mean)

    all_centered = np.concatenate(centered)
    covariance = np.cov(all_centered, rowvar=False)

    whitener = fractional_matrix_power(covariance, -0.5)
    np.savetxt(sys.stdout, whitener)

    if args.show_cov:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.imshow(covariance)
        plt.colorbar()
        plt.show()
