import torch


def tensor_from_words(words, vocab):
    tensor = torch.LongTensor([vocab[w] for w in words]).view(1, -1)
    return torch.autograd.Variable(tensor)
