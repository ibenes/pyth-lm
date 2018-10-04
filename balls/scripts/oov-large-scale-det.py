#!/usr/bin/env python

import argparse
from det import DETCurve
from typing import List, Tuple
import pickle
import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', required=True,
                        help='reference matrix, text, triangular')
    parser.add_argument('--scores', required=True,
                        help='reference matrix, text, triangular')
    parser.add_argument('--sampling-rate', type=float, default=0.1,
                        help='reference matrix, text, triangular')
    parser.add_argument('--det-file',
                        help='where to put the pickled DETCurve object')
    args = parser.parse_args()

    score_tg: List[Tuple[float, float]] = []
    with open(args.ref) as ref_f, open(args.scores) as scores_f:
        for i, (ref_line, score_line) in enumerate(zip(ref_f, scores_f)):
            ref_fields = [float(x) for x in ref_line.split()]
            score_fields = [float(x) for x in score_line.split()]

            line_score_tg = list(zip(score_fields, ref_fields))

            score_tg.extend(random.sample(
                line_score_tg,
                int(len(line_score_tg)*args.sampling_rate)
            ))

    det = DETCurve(score_tg, baseline=True, max_det_points=500)

    with open(args.det_file, 'wb') as f:
        pickle.dump(det, f)
