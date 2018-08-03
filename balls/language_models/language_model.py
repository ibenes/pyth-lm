import torch


class LanguageModel(torch.nn.Module):
    def __init__(self, model, decoder, vocab):
        super().__init__()

        self.model = model
        self.decoder = decoder
        self.vocab = vocab

        self.criterion = torch.nn.NLLLoss(size_average=False)

        self.forward = model.forward
