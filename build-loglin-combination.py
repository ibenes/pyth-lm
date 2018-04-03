#!/usr/bin/env python
import argparse
import torch

import smm_lstm_models
import vocab
import language_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch LSTM Language Model')
    parser.add_argument('--load-rnnlm', type=str,  required=True,
                        help='path to a (pretrained) RNN LM')
    parser.add_argument('--load-iveclm', type=str,  required=True,
                        help='path to a (pretrained) ivec-only LM')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--ivec-weight', type=float, default=1.0,
                        help='coefficient to multiply all ivec model weights with')
    parser.add_argument('--save', type=str,  required=True,
                        help='path to save the final model')
    args = parser.parse_args()

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)

    print("loading LSTM model...")
    with open(args.load_rnnlm, 'rb') as f:
        lstm_lm = language_model.load(f)
    print(lstm_lm.model)

    print("loading ivec-only model...")
    with open(args.load_iveclm, 'rb') as f:
        ivec_lm = language_model.load(f)
    print(ivec_lm.model)

    print("building model...")

    model = smm_lstm_models.OutputEnhancedLM(
        len(lstm_lm.vocab), lstm_lm.model.nhid, lstm_lm.model.nhid, 
        lstm_lm.model.nlayers, ivec_lm.model.ivec_dim, lstm_lm.model.drop.p, 
        tie_weights=True,
        dropout_ivec=ivec_lm.model.drop.p,
    )

    model.encoder = lstm_lm.model.encoder
    model.rnn = lstm_lm.model.rnn
    model.decoder = lstm_lm.model.decoder
    model.ivec_proj = ivec_lm.model.decoder
    model.ivec_proj.weight.data *= args.ivec_weight
    model.ivec_proj.bias.data *= args.ivec_weight

    lm = language_model.LanguageModel(model, lstm_lm.vocab)
    with open(args.save, 'wb') as f:
        lm.save(f)
