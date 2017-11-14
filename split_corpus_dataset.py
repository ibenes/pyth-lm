import vocab
import torch

class BatchBuilder():
    def __init__(self, token_streams, ivec_app_ctor, max_batch_size):
        """
            Args:
                fs ([file]): List of opened files to construct batches from
        """
        pass
        self._token_streams = token_streams
        self._ivec_app_ctor = ivec_app_ctor

        if max_batch_size <= 0:
            raise ValueError("BatchBuilder must be constructed"
                "with a positive batch size, (got {})".format(max_batch_size)
            )
        self._max_bsz = max_batch_size

    def __iter__(self):
        streams = [iter(self._ivec_app_ctor(ts)) for ts in self._token_streams]
        while True:
            batch = []
            for s in streams:
                if len(batch) == self._max_bsz:
                    break
                try:
                    batch.append(next(s))
                except StopIteration:
                    pass

            if len(batch) == 0:
                raise StopIteration
            else:
                yield (
                    torch.LongTensor([x for x,t,i in batch]).t(),
                    torch.LongTensor([t for x,t,i in batch]).t(),
                    torch.stack([torch.from_numpy(i) for x,t,i in batch])
                )


class CheatingIvecAppender():
    def __init__(self, tokens, ivec_eetor):
        """
            Args:
                tokens (TokenizedSplit): Source of tokens, represents single 'document'.
        """
        self.tokens = tokens
        self.ivec_eetor = ivec_eetor
        self._ivec = ivec_eetor(self.tokens._tokens)


    def __iter__(self):
        for x, t in self.tokens:
            yield (x, t, self._ivec)


class TokenizedSplit():
    def __init__(self, f, vocab, unroll_length):
        """
            Args:
                f (file): File with a document.
                vocab (Vocabulary): Vocabulary for translation word -> index
        """
        pass
        sentence = f.read()
        words = sentence.split()
        self._tokens = [vocab[w] for w in words]
        self._unroll_length = unroll_length

    def __iter__(self):
        for i in range(0, len(self), self._unroll_length):
            lend = i
            rend = i + self._unroll_length
            yield(self._tokens[lend:rend], self._tokens[lend+1:rend+1])

    def __len__(self):
        return len(self._tokens) - self._unroll_length
