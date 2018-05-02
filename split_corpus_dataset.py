import vocab
import torch

class BatchBuilder():
    def __init__(self, streams, max_batch_size, discard_h=True):
        """
            Args:
                fs ([file]): List of opened files to construct batches from
        """
        self._streams = streams

        if max_batch_size <= 0:
            raise ValueError("BatchBuilder must be constructed"
                "with a positive batch size, (got {})".format(max_batch_size)
            )
        self._max_bsz = max_batch_size
        self._discard_h = discard_h

    def __iter__(self):
        streams = [iter(s) for s in self._streams]
        active_streams = []
        reserve_streams = streams

        while True:
            batch = []
            streams_continued = []
            streams_ended = []
            for i, s in enumerate(active_streams):
                try:
                    batch.append(next(s))
                    streams_continued.append(i)
                except StopIteration:
                    streams_ended.append(i)

            active_streams = [active_streams[i] for i in streams_continued]

            # refill the batch (of active streams)
            while len(reserve_streams) > 0:
                if len(batch) == self._max_bsz:
                    break

                stream = reserve_streams[0]
                del reserve_streams[0]
                try:
                    batch.append(next(stream))
                    active_streams.append(stream)
                except StopIteration:
                    pass

            if len(batch) == 0:
                raise StopIteration

            if self._discard_h:
                hs_passed_on = streams_continued
            else:
                hs_passed_on = (streams_continued + streams_ended)[:len(batch)]

            parts = zip(*batch)
            parts = [torch.stack(part) for part in parts]
            yield tuple(parts) + (torch.LongTensor(hs_passed_on), )


class TemporalSplits():
    def __init__(self, seq, nb_inputs_necessary, nb_targets_parallel):
        self._seq = seq
        self._nb_inputs_necessary = nb_inputs_necessary
        self._nb_target_parallel = nb_targets_parallel


    def __iter__(self):
        for lend, rend in self.ranges():
            yield (
                self._seq[lend:rend],
                self._seq[lend+self._nb_inputs_necessary:rend+1]
            )

    def __len__(self):
        return max(len(self._seq) - self._nb_inputs_necessary - self._nb_target_parallel + 1, 0)
        

    def ranges(self):
        for i in range(0, len(self), self._nb_target_parallel):
            lend = i
            rend = i + self._nb_inputs_necessary + self._nb_target_parallel - 1
            yield lend, rend


class TokenizedSplitFFBase():
    def __init__(self):
        """
            Args:
                f (file): File with a document.
                vocab (Vocabulary): Vocabulary for translation word -> index
        """
        self._temp_splitter = TemporalSplits(self._tokens, self._hist_len, self._nb_target_parallel)

    def __iter__(self):
        for x,t in self._temp_splitter:
            yield x,t

    def __len__(self):
        return len(self._temp_splitter)


    def input_words(self):
        for lend, rend in self._temp_splitter.ranges():
            yield " ".join(self._words[lend:rend])


class TokenizedSplit(TokenizedSplitFFBase):
    def __init__(self, f, vocab, unroll_length):
        """
            Args:
                f (file): File with a document.
                vocab (Vocabulary): Vocabulary for translation word -> index
        """
        sentence = f.read()
        self._words = sentence.split()
        self._tokens = torch.LongTensor([vocab[w] for w in self._words])
        self._nb_target_parallel = unroll_length
        self._hist_len = 1
        super().__init__()


class TokenizedSplitSingleTarget(TokenizedSplitFFBase):
    def __init__(self, f, vocab, unroll_length):
        """
            Args:
                f (file): File with a document.
                vocab (Vocabulary): Vocabulary for translation word -> index
        """
        sentence = f.read()
        self._words = sentence.split()
        self._tokens = torch.LongTensor([vocab[w] for w in self._words])
        self._hist_len = unroll_length
        self._nb_target_parallel = 1
        super().__init__()


class TokenizedSplitFFMultiTarget(TokenizedSplitFFBase):
    def __init__(self, f, vocab, hist_len, nb_targets_parallel):
        """
            Args:
                f (file): File with a document.
                vocab (Vocabulary): Vocabulary for translation word -> index
        """
        sentence = f.read()
        self._words = sentence.split()
        self._tokens = torch.LongTensor([vocab[w] for w in self._words])
        self._hist_len = hist_len
        self._nb_target_parallel = nb_targets_parallel
        super().__init__()


class DomainAdaptationSplitFFBase:
    def __init__(self):
        self._temp_splitter = TemporalSplits(self._tokens, self._hist_len, self._nb_target_parallel)

    def __iter__(self):
        for x,t in self._temp_splitter:
            yield x,t

    def __len__(self):
        return len(self._temp_splitter)

    def input_words(self):
        return [self._domain_string]


class DomainAdaptationSplitFFMultiTarget(DomainAdaptationSplitFFBase):
    def __init__(self, f, vocab, hist_len, nb_targets_parallel, end_portion):
        """
            Args:
                f (file): File with a document.
                vocab (Vocabulary): Vocabulary for translation word -> index
        """
        sentence = f.read()
        words = sentence.split()

        nb_domain_words = int(len(words)*end_portion-0.01)

        self._tokens = torch.LongTensor([vocab[w] for w in words[:len(words)-nb_domain_words]])
        self._domain_string = " ".join(words[len(words)-nb_domain_words:])

        self._hist_len = hist_len
        self._nb_target_parallel = nb_targets_parallel
        self._end_portion = end_portion
        super().__init__()


class DomainAdaptationSplit(DomainAdaptationSplitFFBase):
    def __init__(self, f, vocab, unroll_length, end_portion):
        """
            Args:
                f (file): File with a document.
                vocab (Vocabulary): Vocabulary for translation word -> index
        """
        sentence = f.read()
        words = sentence.split()

        nb_domain_words = int(len(words)*end_portion-0.01)

        self._tokens = torch.LongTensor([vocab[w] for w in words[:-nb_domain_words]])
        self._domain_string = " ".join(words[-nb_domain_words:])

        self._hist_len = unroll_length
        self._nb_target_parallel = 1
        self._end_portion = end_portion
        super().__init__()
