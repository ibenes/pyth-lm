#!/usr/bin/env python
import argparse
import sys

import torch

from language_models.vocab import vocab_from_kaldi_wordlist

ST_WORDS = 0
ST_OOV_INTEREST = 1
ST_OOV_OTHER = 2


OOV_OI_WORD = '________'


def words_from_idx(idx_list):
    transcript = []
    state = ST_WORDS
    for idx in idxes:
        if state == ST_WORDS:
            if idx == oov_start_idx + args.interest_constant:
                state = ST_OOV_INTEREST
            elif idx == oov_start_idx:
                state = ST_OOV_OTHER
            elif idx == oov_end_idx + args.interest_constant:
                raise ValueError("Unacceptable end of OOV-OI within WORDS ({}, key {})".format(line_no, key))
            elif idx == oov_end_idx:
                raise ValueError("Unacceptable end of OOV-NI within WORDS ({}, key {})".format(line_no, key))
            else:
                transcript.append(decoder_vocabulary.i2w(idx))
        elif state == ST_OOV_INTEREST:
            if idx == oov_end_idx + args.interest_constant:
                transcript.append(OOV_OI_WORD)
                state = ST_WORDS
            elif idx == oov_end_idx:
                raise ValueError("Unacceptable end of OOV-NI within OOV-OI ({}, key {})".format(line_no, key))
            elif idx == oov_start_idx + args.interest_constant:
                raise ValueError("Unacceptable start of OOV-OI within OOV-OI ({}, key {})".format(line_no, key))
            elif idx == oov_start_idx:
                raise ValueError("Unacceptable start of OOV-NI within OOV-OI ({}, key {})".format(line_no, key))
            else:
                pass
        elif state == ST_OOV_OTHER:
            if idx == oov_end_idx:
                transcript.append(args.unk)
                state = ST_WORDS
            elif idx == oov_end_idx + args.interest_constant:
                raise ValueError("Unacceptable end of OOV-OI within OOV-NI ({}, key {})".format(line_no, key))
            elif idx == oov_start_idx + args.interest_constant:
                raise ValueError("Unacceptable start of OOV-OI within OOV-NI ({}, key {})".format(line_no, key))
            elif idx == oov_start_idx:
                raise ValueError("Unacceptable start of OOV-NI within OOV-NI ({}, key {})".format(line_no, key))
            else:
                pass
        else:
            raise RuntimeError("got into an impossible state {}".format(state))

    return transcript


def relevant_prefix(transcript):
    first_oov_oi_loc = transcript.index(OOV_OI_WORD)
    if OOV_OI_WORD in transcript[first_oov_oi_loc+1:]:
        raise ValueError("there are multiple OOVs of interest!")

    return transcript[:first_oov_oi_loc]


def tensor_from_words(words, lm):
    tensor = torch.LongTensor([lm.vocab[w] for w in words]).view(1, -1)

    return torch.autograd.Variable(tensor)


BATCH_SIZE = 1


def emb_from_string(transcript, lm):
    prefix = relevant_prefix(transcript)
    prefix.insert(0, "</s>")

    th_data = tensor_from_words(prefix, lm)
    h0 = lm.model.init_hidden(th_data.size(0))
    emb, h = lm.model(th_data, h0)
    out_emb = emb[0][-1].data

    return out_emb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--unk', default="<UNK>")
    parser.add_argument('--oov-start', required=True)
    parser.add_argument('--oov-end', required=True)
    parser.add_argument('--interest-constant', type=int, required=True)
    parser.add_argument('--decoder-wordlist', required=True)
    parser.add_argument('--fwd-lm')
    parser.add_argument('--bwd-lm')
    args = parser.parse_args()

    if not args.fwd_lm and not args.bwd_lm:
        sys.stderr.write("At least one of '--fwd-lm' and '--bwd-lm' needs to be specified\n")
        sys.exit(1)

    with open(args.decoder_wordlist) as f:
        decoder_vocabulary = vocab_from_kaldi_wordlist(f, unk_word=args.unk)

    if args.fwd_lm:
        fwd_lm = torch.load(args.fwd_lm, map_location=lambda storage, location: storage)
        fwd_lm.eval()
    if args.bwd_lm:
        bwd_lm = torch.load(args.bwd_lm, map_location=lambda storage, location: storage)
        bwd_lm.eval()

    oov_start_idx = decoder_vocabulary[args.oov_start]
    oov_end_idx = decoder_vocabulary[args.oov_end]

    for line_no, line in enumerate(sys.stdin):
        fields = line.split()
        key = fields[0]
        idxes = [int(idx) for idx in fields[1:]]
        transcript = words_from_idx(idxes)

        output = key
        if args.fwd_lm:
            fwd_emb = emb_from_string(transcript, fwd_lm)
            fwd_emb_str = " ".join("{:.4f}".format(e) for e in fwd_emb)
            output += " " + fwd_emb_str
        if args.bwd_lm:
            bwd_emb = emb_from_string(list(reversed(transcript)), bwd_lm)
            bwd_emb_str = " ".join("{:.4f}".format(e) for e in bwd_emb)
            output += " " + bwd_emb_str
        output += '\n'

        sys.stdout.write(output)
