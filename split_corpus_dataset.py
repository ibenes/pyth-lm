import vocab
import torch

class BatchBuilder():
    def __init__(self, token_streams, ivec_app_ctor, max_batch_size, discard_h=True):
        """
            Args:
                fs ([file]): List of opened files to construct batches from
        """
        self._token_streams = token_streams
        self._ivec_app_ctor = ivec_app_ctor

        if max_batch_size <= 0:
            raise ValueError("BatchBuilder must be constructed"
                "with a positive batch size, (got {})".format(max_batch_size)
            )
        self._max_bsz = max_batch_size
        self._discard_h = discard_h

    def __iter__(self):
        streams = [iter(self._ivec_app_ctor(ts)) for ts in self._token_streams]
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

            yield (
                torch.LongTensor([x for x,t,i in batch]).t(),
                torch.LongTensor([t for x,t,i in batch]).t(),
                torch.stack([torch.from_numpy(i) for x,t,i in batch]),
                torch.LongTensor(hs_passed_on)
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
        return max(len(self._tokens) - self._unroll_length, 0)
