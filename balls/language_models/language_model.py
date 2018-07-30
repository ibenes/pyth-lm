import torch


class LanguageModel(torch.nn.Module):
    def __init__(self, model, vocab):
        super().__init__()

        self.model = model
        self.vocab = vocab

        self.criterion = torch.nn.NLLLoss(size_average=False)

        self.forward = model.forward
