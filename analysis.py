import torch


def categorical_entropy(p, eps=1e-100):
    non_zeros = p > eps

    log_p = p.log()
    H_p = - torch.sum(p*log_p[non_zeros])
    return H_p / torch.log(torch.FloatTensor([2]))
    

def categorical_cross_entropy(p, q, eps=1e-100):
    non_zeros = p > eps

    log_q = q.log()
    Xent = - torch.sum((p*log_q)[non_zeros])
    return Xent / torch.log(torch.FloatTensor([2]))

def categorical_kld(p, q):
    return categorical_cross_entropy(p, q) - categorical_entropy(p)
