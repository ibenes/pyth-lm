#!/usr/bin/env python

import argparse
import sys

import numpy as np
from scipy.spatial.distance import pdist, squareform

from embeddings_io import all_embs_from_file
from det import DETCurve


def trial_scores_list(keys, similarities):
    score_tg = []
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            a = keys[i].split(':')[0]
            b = keys[j].split(':')[0]

            score = similarities[i, j]

            if a == b:
                score_tg.append((score, 1))
            else:
                score_tg.append((score, 0))

    return score_tg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--length-norm', action='store_true')
    parser.add_argument('--log-det', action='store_true')
    parser.add_argument('--eps', type=float, default=1e-3, help='to prevent log of zero')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--free-axis', action='store_true')
    parser.add_argument('--metric', default='inner_prod', choices=['inner_prod', 'l2_dist'])
    args = parser.parse_args()

    keys, embs = all_embs_from_file(sys.stdin)
    if args.length_norm:
        embs /= np.linalg.norm(embs, axis=1)[:, None]

    if args.metric == 'inner_prod':
        similarities = embs @ embs.T
    elif args.metric == 'l2_dist':
        similarities = -squareform(pdist(embs))

    if args.plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(similarities)
        plt.colorbar()

    score_tg = trial_scores_list(keys, similarities)

    det = DETCurve(score_tg, args.baseline)
    sys.stdout.write(det.textual_report())
    if args.plot:
        det.plot(args.log_det, not args.free_axis)
