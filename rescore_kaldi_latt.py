import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import data
import model
import vocab

import IPython

###############################################################################
# Load data
###############################################################################
def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data

def max_len(seqs):
    return max([len(seq) for seq in seqs])

def pick_ys(y, seq_x):
    seqs_ys = []
    for seq_n, seq in enumerate(seq_x):
        seq_ys = [0] # we WLOG assume that the <s> at the begining has proba 1
        for w_n, w in enumerate(seq[1:]): # starting at position one
            seq_ys.append(y[w_n, seq_n, w])
        seqs_ys.append(seq_ys)

    return seqs_ys

def seqs_logprob(seqs, model):
    ''' Sequence as a list of integers
    '''
    # ids are indexed as ids[batch][time]
    batch_size = len(seqs)
    maxlen = max_len(seqs)
    ids = torch.LongTensor(batch_size, maxlen).zero_()
    for seq_n, seq in enumerate(seqs):
        for word_n, word in enumerate(seq):
            ids[seq_n, word_n] = word

    # indexing is X[time][batch], thus we transpose
    data = ids.t().contiguous()
    if args.cuda:
        data = data.cuda()

    X = Variable(data)
    hidden = model.init_hidden(batch_size)
    # proba of first (0-th) word is not a problem -- first word is always the '<s>', so no-one cares

    y, _ = model(X, hidden)
    y = y.data # we do not care about the Variable wrapping

    word_log_scores = pick_ys(y, seqs)

    seq_log_scores = [sum(seq) for seq in word_log_scores]

    return seq_log_scores

def string_to_pythlm(seq, vocab):
    tokens = seq.split()
    tokens = ['<s>'] + tokens + ['</s>']

    return [vocab.w2i(tok) for tok in tokens]

def dict_to_list(utts_map):
    list_of_lists = []
    rev_map = {}
    for key in utts_map:
        rev_map[len(list_of_lists)] = key
        list_of_lists.append(utts_map[key])

    return list_of_lists, rev_map

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM Language Model')
    parser.add_argument('--latt-vocab', type=str, required=True,
                        help='word -> int map; Kaldi style "words.txt"')
    parser.add_argument('--model-vocab', type=str, required=True,
                        help='word -> int map; Kaldi style "words.txt"')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                        help='batch size')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--model-from', type=str, required=True,
                        help='where to load the model from')
    parser.add_argument('in_filename', help='second output of nbest-to-linear, textual')
    parser.add_argument('out_filename', help='where to put the LM scores')
    args = parser.parse_args()

    print(args)

    print("reading vocabs...")
    with open(args.latt_vocab, 'r') as f:
        latt_vocab = vocab.vocab_from_kaldi_wordlist(f, unk_word='<UNK>')

    with open(args.model_vocab, 'r') as f:
        model_vocab = vocab.vocab_from_kaldi_wordlist(f)

    print("reading model...")
    with open(args.model_from, 'rb') as f:
        model = torch.load(f)
    if args.cuda:
        model.cuda()
    model.eval()

    criterion = nn.NLLLoss()

    curr_seg = None
    segment_utts = {}

    with open(args.in_filename) as in_f, open(args.out_filename, 'w') as out_f:
        for line in in_f:
            fields = line.split()
            utt_id = fields[0]
            word_ids = [int(wi) for wi in fields[1:]]

            words = [latt_vocab.i2w(i) for i in word_ids]
            ids = string_to_pythlm(" ".join(words), model_vocab)
            print(utt_id, " UNKs:", words.count("<UNK>"), "\t<unk>s ", ids.count(model_vocab.unk_index_), ' / ', len(words))

            fields = utt_id.split('-')
            segment = "-".join(fields[:-1])
            trans_id = fields[-1]

            if not curr_seg:
                curr_seg = segment

            if segment != curr_seg:
                X, rev_map = dict_to_list(segment_utts) # reform the word sequences
                y = seqs_logprob(X, model) # score

                # write
                for i, log_p in enumerate(y):
                    out_f.write(curr_seg + '-' + rev_map[i] + ' ' + str(-log_p) + '\n')

                curr_seg = segment
                segment_utts = {}

            segment_utts[trans_id] = ids

        # Last segment:
        X, rev_map = dict_to_list(segment_utts) # reform the word sequences
        y = seqs_logprob(X, model) # score

        # write
        for i, log_p in enumerate(y):
            out_f.write(curr_seg + '-' + rev_map[i] + ' ' + str(-log_p) + '\n')
