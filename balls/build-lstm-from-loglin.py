#!/usr/bin/env python
import argparse
import torch

import lstm_model
import vocab
import language_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch LSTM Language Model')
    parser.add_argument('--load', type=str,  required=True,
                        help='path to a (pretrained) OutputEnhancedLM() RNN LM')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--save', type=str,  required=True,
                        help='path to save the final LSTM only model')
    args = parser.parse_args()

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)

    print("loading LSTM model...")
    with open(args.load, 'rb') as f:
        loglin_lm = language_model.load(f)
    print(loglin_lm.model)

    print("building model...")

    model = lstm_model.LSTMLanguageModel(
        len(loglin_lm.vocab), loglin_lm.model.nhid, loglin_lm.model.nhid, 
        loglin_lm.model.nlayers, loglin_lm.model.drop.p, 
    )

    model.encoder = loglin_lm.model.encoder
    model.rnn = loglin_lm.model.rnn
    model.decoder = loglin_lm.model.decoder
    model.decoder.bias.data += loglin_lm.model.ivec_proj.bias.data

    lm = language_model.LanguageModel(model, loglin_lm.vocab)
    with open(args.save, 'wb') as f:
        lm.save(f)
