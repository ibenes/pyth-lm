from torch.autograd import Variable

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
