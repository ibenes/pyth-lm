import argparse
import math

import data

import language_model
import split_corpus_dataset
import multistream

from runtime_utils import init_seeds
from runtime_multifile import evaluate_


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM Language Model')
    parser.add_argument('--data', type=str, required=True,
                        help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--target-seq-len', type=int, default=35, metavar='N',
                        help='number of words to take from every sequence in a single step')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--concat-articles', action='store_true',
                        help='pass hidden states over article boundaries')
    parser.add_argument('--load', type=str, required=True,
                        help='where to load a model from')
    args = parser.parse_args()
    print(args)

    init_seeds(args.seed, args.cuda)

    print("loading model...")
    with open(args.load, 'rb') as f:
        lm = language_model.load(f)
    if args.cuda:
        lm.model.cuda()
    print(lm.model)

    print("preparing data...")

    ids = data.tokens_from_fn(args.data, lm.vocab, randomize=False)
    batched = multistream.batchify(ids, 10, args.cuda)
    data = split_corpus_dataset.TemporalSplits(
        batched,
        nb_inputs_necessary=lm.model.in_len,
        nb_targets_parallel=args.target_seq_len
    )
    data = split_corpus_dataset.TransposeWrapper(data)

    loss = evaluate_(
        lm.model, data,
        use_ivecs=False,
        custom_batches=False,
    )
    print('loss {:5.2f} | ppl {:8.2f}'.format(loss, math.exp(loss)))
