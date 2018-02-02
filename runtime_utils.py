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
    with open(filelist_filename) as filelist: 
        files = filelist.read().split()
        tss = []
        for filename in files:
            with open(filename, 'r') as f:
                tss.append(split_corpus_dataset.TokenizedSplit(f, vocab, bptt)) 

        return tss

