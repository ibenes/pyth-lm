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
    print(data)
    
    X = Variable(data)
    hidden = model.init_hidden(batch_size) 
    # proba of first (0-th) word is not a problem -- first word is always the '<s>', so no-one cares

    y, _ = model(X, hidden)
    y = y.data # we do not care about the Variable wrapping
    print(y.size())

    word_log_scores = pick_ys(y, seqs)
    for line in word_log_scores:
        print(np.around(line, decimals=2))

    seq_log_scores = [sum(seq) for seq in word_log_scores]
    print(seq_log_scores)

def string_to_pythlm(seq, vocab):
    tokens = seq.split()
    tokens = ['<s>'] + tokens + ['</s>']
    
    return [vocab.w2i(tok) for tok in tokens]
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM Language Model')
    parser.add_argument('--wordlist', type=str, required=True,
                        help='word -> int map; Kaldi style "words.txt"')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                        help='batch size')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--model-from', type=str, required=True,
                        help='where to load the model from')
    args = parser.parse_args()

    print("reading vocab...")
    with open(args.wordlist, 'r') as f:
        vocab = vocab.vocab_from_kaldi_wordlist(f)

    print("reading model...")
    with open(args.model_from, 'rb') as f:
        model = torch.load(f)
    if args.cuda:
        model.cuda()
    model.eval()

    criterion = nn.NLLLoss()

    strings = []
    strings.append("ښه د کور مصروفیت دي کار بس کار کښې تېر شي او سبق وایې")
    strings.append("سکول سبق وایې مکتب")
    strings.append("څو هم کښې یې")
    strings.append("دوولسم کښې یې کوم مضمون دي ډېر خوښ دې")
    strings.append("انګلش دي خوښ دي")
    strings.append("ښه نه انګلش خو ښه مضمون دې او چې ستا خوښ دې نو بس")

    X = [string_to_pythlm(string, vocab) for string in strings] 
    for x in X:
        print(x)

    print(max_len(X))

    seqs_logprob(X, model)
