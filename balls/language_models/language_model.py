import torch


class LanguageModel(torch.nn.Module):
    def __init__(self, model, vocab):
        super().__init__()

        self.model = model
        self.vocab = vocab
