#!/usr/bin/env python

import argparse
import copy
import sys

import numpy as np
from scipy.spatial.distance import pdist, squareform

from embeddings_io import all_embs_from_file


def area_under_curve(xs_in, ys_in):
    assert(len(xs_in) == len(ys_in))

    xs = list(copy.deepcopy(xs_in))
    ys = list(copy.deepcopy(ys_in))

    if xs[0] > 0.0:
        xs.insert(0, 0.0)
        ys.insert(0, 1.0)

    if ys[-1] > 0.0:
        xs.append(1.0)
        ys.append(0.0)

    running_sum = 0.0

    for i in range(len(xs)-1):
        x_len = xs[i+1] - xs[i]
        avg_y = (ys[i] + ys[i+1])/2
        running_sum += x_len * avg_y

    return running_sum


def eer(xs, ys):
    assert(len(xs) == len(ys))

    eer = float('nan')
    for i in range(len(xs)-1):
        if xs[i] < ys[i] and xs[i+1] >= ys[i+1]:
            d_i = abs(xs[i] - ys[i])
            d_ip1 = xs[i+1] - ys[i+1]
            lambda_i = d_i/(d_i + d_ip1)
            eer = lambda_i * xs[i] + (1.0-lambda_i)*xs[i+1]

    return eer


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

    if args.plot:
        import matplotlib.pyplot as plt

    keys, embs = all_embs_from_file(sys.stdin)
    if args.length_norm:
        embs /= np.linalg.norm(embs, axis=1)[:, None]

    if args.metric == 'inner_prod':
        similarities = embs @ embs.T
    elif args.metric == 'l2_dist':
        similarities = -squareform(pdist(embs))

    if args.plot:
        plt.figure()
        plt.imshow(similarities)
        plt.colorbar()

    score_tg = trial_scores_list(keys, similarities)

    nb_trials = len(score_tg)
    nb_same = sum(s[1] for s in score_tg)
    nb_different = nb_trials - nb_same

    max_miss_rate = nb_same / nb_trials
    max_fa_rate = nb_different / nb_trials

    print("# positive trials: {} ({:.1f} %)".format(nb_same, 100.0*nb_same/nb_trials))
    print("# negative trials: {} ({:.1f} %)".format(nb_different, 100.0*nb_different/nb_trials))

    score_tg = sorted(score_tg, key=lambda s: s[0])

    mis_fas = []
    nb_correct_same = nb_same
    nb_correct_different = 0
    nb_false_alarms = nb_different
    nb_misses = 0

    for s in score_tg:
        if s[1] == 1:
            nb_misses += 1
            nb_correct_same -= 1
        else:
            nb_false_alarms -= 1
            nb_correct_different += 1

        mis_fas.append([(nb_misses+args.eps)/nb_trials, (nb_false_alarms+args.eps)/nb_trials])

    mis_fas = np.asarray(mis_fas)
    miss_rate = mis_fas[:, 0]
    fa_rate = mis_fas[:, 1]

    area_line_fmt = "Area under DET curve (in linspace): {:.5f}"
    eer_line_fmt = "EER: {:.2f} %"

    system_au_det = area_under_curve(miss_rate, fa_rate)
    system_eer = eer(miss_rate, fa_rate)

    if args.baseline:
        area_line_fmt += " / {:.5f} / {:.2f} %"
        eer_line_fmt += " / {:.2f} % / {:.2f} %"

        baseline_au_det = max_miss_rate * max_fa_rate / 2.0
        baseline_eer = max_miss_rate * max_fa_rate / (max_miss_rate + max_fa_rate)

        print(area_line_fmt.format(
            system_au_det,
            baseline_eer,
            100.0 * (1.0 - system_au_det/baseline_au_det)
        ))
        print(eer_line_fmt.format(
            100.0*system_eer,
            100.0*baseline_eer,
            100.0 * (1.0 - system_eer/baseline_eer)
        ))

    else:
        print(area_line_fmt.format(system_au_det))
        print(eer_line_fmt.format(100.0*system_eer))

    if args.plot:
        plt.figure()

        if args.log_det:
            plt_func = plt.loglog
        else:
            plt_func = plt.plot
        det_handle = plt_func(miss_rate, fa_rate, label='System')

        if args.baseline:
            xs = np.linspace(0, max_miss_rate)
            ys = np.linspace(max_fa_rate, 0)
            baseline_handle = plt_func(xs, ys, label='Baseline')

            plt.legend()

        if not args.free_axis:
            plt.axis('scaled')
        plt.xlim(left=0.0)
        plt.ylim(bottom=0.0)
        plt.xlabel('miss rate')
        plt.ylabel('FA rate')

        plt.show()
