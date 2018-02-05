import torch

def categorical_entropy(p, eps=1e-100):
    non_zeros = p > eps

    log_p = p.log()
    H_p = - torch.sum(p*log_p[non_zeros])
    return H_p / torch.log(torch.FloatTensor([2]))
    
