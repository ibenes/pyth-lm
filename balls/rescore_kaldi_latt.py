import argparse
import torch

import vocab

import kaldi_itf


def max_len(seqs):
    return max([len(seq) for seq in seqs])


def pick_ys(y, seq_x):
    seqs_ys = []
    for seq_n, seq in enumerate(seq_x):
        seq_ys = [1.0]  # hard 1.0 for the 'sure' <s>
        for w_n, w in enumerate(seq[1:]):  # skipping the initial element ^^^
            seq_ys.append(y[w_n, seq_n, w])
        seqs_ys.append(seq_ys)

    return seqs_ys


def seqs_to_tensor(seqs):
    batch_size = len(seqs)
    maxlen = max_len(seqs)
    ids = torch.LongTensor(batch_size, maxlen).zero_()
    for seq_n, seq in enumerate(seqs):
        for word_n, word in enumerate(seq):
            ids[seq_n, word_n] = word

    # indexing is X[time][batch], thus we transpose
    data = ids.t().contiguous()
    return data, batch_size


def seqs_logprob(seqs, model):
    ''' Sequence as a list of integers
    '''
    data, batch_size = seqs_to_tensor(seqs)

    if args.cuda:
        data = data.cuda()

    X = data
    h0 = model.init_hidden(batch_size)

    y, _ = model(X, h0)

    word_log_scores = pick_ys(y, seqs)
    seq_log_scores = [sum(seq) for seq in word_log_scores]

    return seq_log_scores


def tokens_to_pythlm(toks, vocab):
    return [vocab.w2i('<s>')] + [vocab.w2i(tok) for tok in toks] + [vocab.w2i("</s>")]


def dict_to_list(utts_map):
    list_of_lists = []
    rev_map = {}
    for key in utts_map:
        rev_map[len(list_of_lists)] = key
        list_of_lists.append(utts_map[key])

    return list_of_lists, rev_map


def translate_latt_to_model(words, latt_vocab, model_vocab):
    words = [latt_vocab.i2w(i) for i in word_ids]
    return tokens_to_pythlm(words, model_vocab)


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
        latt_vocab = vocab.vocab_from_kaldi_wordlist(f, unk_word='<unk>')

    with open(args.model_vocab, 'r') as f:
        model_vocab = vocab.vocab_from_kaldi_wordlist(f)

    print("reading model...")
    with open(args.model_from, 'rb') as f:
        model = torch.load(f)
    if args.cuda:
        model.cuda()
    model.eval()

    print("scoring...")
    curr_seg = None
    segment_utts = {}

    with open(args.in_filename) as in_f, open(args.out_filename, 'w') as out_f:
        for line in in_f:
            fields = line.split()
            segment, trans_id = kaldi_itf.split_nbest_key(fields[0])

            word_ids = [int(wi) for wi in fields[1:]]
            ids = translate_latt_to_model(word_ids, latt_vocab, model_vocab)

            if not curr_seg:
                curr_seg = segment

            if segment != curr_seg:
                X, rev_map = dict_to_list(segment_utts)  # reform the word sequences
                y = seqs_logprob(X, model)  # score

                # write
                for i, log_p in enumerate(y):
                    out_f.write(curr_seg + '-' + rev_map[i] + ' ' + str(-log_p) + '\n')

                curr_seg = segment
                segment_utts = {}

            segment_utts[trans_id] = ids

        # Last segment:
        X, rev_map = dict_to_list(segment_utts)  # reform the word sequences
        y = seqs_logprob(X, model)  # score

        # write
        for i, log_p in enumerate(y):
            out_f.write(curr_seg + '-' + rev_map[i] + ' ' + str(-log_p) + '\n')
