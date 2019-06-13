import torch


class LanguageModel(torch.nn.Module):
    def __init__(self, model, decoder, vocab):
        super().__init__()

        self.model = model
        self.decoder = decoder
        self.vocab = vocab

        self.forward = model.forward

    def single_sentence_nll(self, sentence, prefix):
        prefix_id = self.vocab[prefix]
        sentence_ids = [self.vocab[c] for c in sentence]

        device = next(self.parameters()).device
        tensor = torch.tensor([prefix_id] + sentence_ids).view(-1, 1).to(device)
        o, _ = self.model(tensor[:-1], self.model.init_hidden(1))
        nll, _ = self.decoder.neg_log_prob(o, tensor[1:])
        return nll.item()
