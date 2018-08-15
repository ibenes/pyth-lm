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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--unk', default="<UNK>")
    parser.add_argument('--oov-start', required=True)
    parser.add_argument('--oov-end', required=True)
    parser.add_argument('--interest-constant', type=int, required=True)
    parser.add_argument('--decoder-wordlist', required=True)
    parser.add_argument('--lm', required=True)
    args = parser.parse_args()

    with open(args.decoder_wordlist) as f:
        decoder_vocabulary = vocab_from_kaldi_wordlist(f, unk_word=args.unk)

    lm = torch.load(args.lm, map_location=lambda storage, location: storage)
    lm.eval()

    oov_start_idx = decoder_vocabulary[args.oov_start]
    oov_end_idx = decoder_vocabulary[args.oov_end]

    for line_no, line in enumerate(sys.stdin):
        fields = line.split()
        key = fields[0]
        idxes = [int(idx) for idx in fields[1:]]

        transcript = words_from_idx(idxes)
        prefix = relevant_prefix(transcript)
        if len(prefix) > 0:
            th_data = tensor_from_words(prefix, lm)
            h0 = lm.model.init_hidden(th_data.size(0))
            emb, h = lm.model(th_data, h0)
            out_emb = emb[0][-1].data
        else:
            out_emb = lm.model.init_hidden(BATCH_SIZE)[0][0, 0].data

        print(key, " ".join("{:.4f}".format(e) for e in out_emb))
