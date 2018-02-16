import random
import torch

from torch.autograd import Variable
import split_corpus_dataset


class CudaStream():
    def __init__(self, source):
        self._source = source

    def __iter__(self):
        for x, target, ivecs, mask in self._source:
            cuda_batch = x.cuda(), target.cuda(), ivecs.cuda(), mask.cuda()
            yield cuda_batch


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def filelist_to_tokenized_splits(filelist_filename, vocab, bptt):
    filenames = filenames_file_to_filenames(filelist_filename)
    tss = []
    for filename in filenames:
        with open(filename, 'r') as f:
            tss.append(split_corpus_dataset.TokenizedSplit(f, vocab, bptt)) 

    return tss

def filenames_file_to_filenames(filelist_filename):
    with open(filelist_filename) as filelist: 
        filenames = filelist.read().split()
        
    return filenames

def init_seeds(seed, cuda):
    random.seed(seed)
    torch.manual_seed(seed)
    if cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
