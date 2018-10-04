#!/usr/bin/env python

import argparse
import sys

import numpy as np

from oov_alignment_lib import align


def parse_oov_id(oov_id):
    return tuple(oov_id.split('_'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text-references', required=True)
    args = parser.parse_args()

    np.set_printoptions(threshold=2000, linewidth=np.inf)

    references = {}
    with open(args.text_references) as f:
        for line in f:
            fields = line.split()
            references[fields[0]] = fields[1:]

    for line in sys.stdin:
        fields = line.split()
        _, utt_id, _, _, _ = parse_oov_id(fields[0])

        candidate_line = fields[1:]
        reference_line = references[utt_id]
        alignment = align(reference_line, candidate_line)

        print('------' * 25)
        print(candidate_line)
        print(reference_line)
        print(alignment)
