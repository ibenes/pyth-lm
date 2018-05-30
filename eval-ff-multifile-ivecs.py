import argparse
import math

import language_model
import ivec_appenders
import split_corpus_dataset
import smm_ivec_extractor
import multistream

from runtime_utils import CudaStream, filelist_to_tokenized_splits, init_seeds
from runtime_multifile import evaluate_no_transpose


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM Language Model')
    parser.add_argument('--file-list', type=str, required=True,
                        help='file with paths to training documents')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--target-seq-len', type=int, default=35,
                        help='sequence length')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--concat-articles', action='store_true',
                        help='pass hidden states over article boundaries')
    parser.add_argument('--load', type=str, required=True,
                        help='where to load a model from')
    parser.add_argument('--ivec-extractor', type=str, required=True,
                        help='where to load a ivector extractor from')
    parser.add_argument('--ivec-nb-iters', type=int,
                        help='override the number of iterations when extracting ivectors')
    args = parser.parse_args()
    print(args)

    init_seeds(args.seed, args.cuda)

    print("loading LM...")
    with open(args.load, 'rb') as f:
        lm = language_model.load(f)
    if args.cuda:
        lm.model.cuda()
    print(lm.model)

    print("loading SMM iVector extractor ...")
    with open(args.ivec_extractor, 'rb') as f:
        ivec_extractor = smm_ivec_extractor.load(f)
    if args.ivec_nb_iters is not None:
        ivec_extractor._nb_iters = args.ivec_nb_iters
    print(ivec_extractor)

    print("preparing data...")
    ts_constructor = lambda *x: split_corpus_dataset.TokenizedSplitFFMultiTarget(*x, args.target_seq_len)
    tss = filelist_to_tokenized_splits(args.file_list, lm.vocab, lm.model.in_len, ts_constructor)

    ivec_app_creator = lambda ts: ivec_appenders.CheatingIvecAppender(ts, ivec_extractor)
    tss_ivecs = [ivec_app_creator(ts) for ts in tss]
    data = multistream.BatchBuilder(tss_ivecs, args.batch_size, discard_h=not args.concat_articles)

    if args.cuda:
        data = CudaStream(data)

    print("evaluating...")
    loss = evaluate_no_transpose(lm.model, data, use_ivecs=True)
    print('loss {:5.2f} | ppl {:8.2f}'.format(loss, math.exp(loss)))
