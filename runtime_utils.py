import random
import torch

from torch.autograd import Variable
import split_corpus_dataset


class CudaStream():
    def __init__(self, source):
        self._source = source

    def __iter__(self):
        for batch in self._source:
            yield tuple(x.cuda() for x in batch)


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def filelist_to_tokenized_splits(filelist_filename, vocab, bptt, wrapper=split_corpus_dataset.TokenizedSplit):
    filenames = filenames_file_to_filenames(filelist_filename)
    tss = []
    for filename in filenames:
        with open(filename, 'r') as f:
            tss.append(wrapper(f, vocab, bptt)) 

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


class BatchFilter:
    def __init__(self, data, batch_size, bptt, min_batch_size): 
        self._data = data
        self._batch_size = batch_size
        self._bptt = bptt
        self._min_batch_size = min_batch_size

        self._nb_skipped_updates = 0
        self._nb_skipped_words = 0
        self._nb_skipped_seqs = 0 # accumulates size of skipped batches

    def __iter__(self):
        for batch in self._data:
            X = batch[0]
            if X.size(0) >= self._min_batch_size:
                yield batch
            else:
                self._nb_skipped_updates += 1
                self._nb_skipped_words += X.size(0) * X.size(1)
                self._nb_skipped_seqs += X.size(0)

    def report(self):
        if self._nb_skipped_updates > 0:
            sys.stderr.write(
                "WARNING: due to skipping, a total of {} updates was skipped,"
                " containing {} words. Avg batch size {}. Equal to {} full batches"
                "\n".format(
                    self._nb_skipped_updates,
                    self._nb_skipped_words,
                    self._nb_skipped_seqs/self._nb_skipped_updates,
                    self._nb_skipped_words/(self._batch_size*self._bptt)
                )
            )
