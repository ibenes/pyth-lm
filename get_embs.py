#!/usr/bin/env python

import argparse
import sys

import embedding_lib
import language_model

def key_words(line):
    fields = line.split()

    key = fields[0]
    words = [int(w) for w in fields[1:]]

    return key, words

def collapse_oovs(words, oov_start_idx=6, oov_end_idx=4, oov_repre=6):
    collapsed = []

    # state == 0 : out of OOV
    # state == 1 : inside OOV
    state = 0

    i = 0
    while i < len(words):
        if state == 0:
            if words[i] == oov_start_idx:
                collapsed.append(oov_repre)
                state = 1
            else:
                collapsed.append(words[i])
        elif state == 1:
            if words[i] == oov_end_idx:
                state = 0
            else:
                pass
        else:
            raise ValueError("An unexpected state {} has arisen".format(state))

        i += 1

    if state != 0:
        raise ValueError("Ended in an unacceptable state {}. Was there an un-finished OOV?".format(state))

    return collapsed


def prefix(words, oov_repre=6):
    oov_positions = [i for i, w in enumerate(words) if w == oov_repre]

    if len(oov_positions) == 0:
        raise ValueError("No OOVs found in the sequence.")
    elif len(oov_positions) > 1:
        raise NotImplementedError("prefix() currently provides prefix only for utterances with exactly 1 OOV.")

    prefix = words[:oov_positions[0]]

    return prefix

def prefix_fallback(words, oov_repre=6):
    oov_positions = [i for i, w in enumerate(words) if w == oov_repre]
    prefix = words[:oov_positions[0]]

    return prefix


def katja_print_emb(emb):
    return " ".join([str(x) for x in emb])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--faulty-lines", type=str,
        help="file to store linenumbers of faulty lines into")
    parser.add_argument("lm")
    args = parser.parse_args()

    with open(args.lm, 'rb') as f:
        lm = language_model.load(f)

    if args.faulty_lines:
        with open(args.faulty_lines, 'w'):
            pass # erases the file

    eetor = embedding_lib.EmbsExpectator(lm)

    nb_multi_oov_lines = 0

    for line_no, line in enumerate(sys.stdin):
        key, words = key_words(line)

        if len(words) > 0:
            cleaned = collapse_oovs(words)
            try:
                left_context = prefix(cleaned)
            except NotImplementedError as e:
                nb_multi_oov_lines += 1
                left_context = prefix_fallback(cleaned)

                if args.faulty_lines:
                    with open(args.faulty_lines, 'a') as f:
                        f.write("{} : multiple OOVs\n".format(line_no))

            emb = eetor(left_context)
            emb_str = katja_print_emb(emb)
        else:
            emb_str = ""

        sys.stdout.write("{} {}\n".format(key, emb_str))

    if nb_multi_oov_lines:
        sys.stderr.write("There were {} lines with multiple OOVs.\n".format(nb_multi_oov_lines))
