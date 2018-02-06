import torch


def categorical_entropy(p, eps=1e-100):
    zeros = p <= eps

    log_p = p.log()
    log_p.masked_fill_(zeros, 0.0) # eliminates -inf for p[x] = 0.0

    H_p = - torch.sum(p*log_p, dim=-1)
    return H_p / torch.log(torch.FloatTensor([2]))
    

def categorical_cross_entropy(p, q, eps=1e-100):
    non_zeros = p > eps

    log_q = q.log()
    Xent = - torch.sum((p*log_q)[non_zeros])
    return Xent / torch.log(torch.FloatTensor([2]))


def categorical_kld(p, q):
    return categorical_cross_entropy(p, q) - categorical_entropy(p)
