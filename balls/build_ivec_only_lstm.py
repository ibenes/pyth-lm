import argparse
import torch

import smm_lstm_models
import vocab
import language_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch LSTM Language Model')
    parser.add_argument('--wordlist', type=str, required=True,
                        help='word -> int map; Kaldi style "words.txt"')
    parser.add_argument('--unk', type=str, default="<unk>",
                        help='expected form of "unk" word. Most likely a <UNK> or <unk>')
    parser.add_argument('--ivec-size', type=int, required=True,
                        help='number of hidden units per layer')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--save', type=str,  required=True,
                        help='path to save the final model')
    args = parser.parse_args()

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)

    print("loading vocabulary...")
    with open(args.wordlist, 'r') as f:
        vocab = vocab.vocab_from_kaldi_wordlist(f, args.unk)

    print("building model...")

    model = smm_lstm_models.IvecOnlyLM(len(vocab), args.ivec_size, args.dropout)

    lm = language_model.LanguageModel(model, vocab)
    with open(args.save, 'wb') as f:
        lm.save(f)
