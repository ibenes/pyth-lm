import vocab

class BatchBuilder():
    def __init__(self, fs, max_batch_size, unroll):
        """
            Args:
                fs ([file]): List of opened files to construct batches from
        """
        pass
        # Construct TokenizedSplit()s for all files in the file list
        # Construct as many IvecAppenders as needed

    def __iter__(self):
        pass
        # 

    def batches(self):
        pass 
        # Ask every IvecAppenders for a chunk
        # If any fails:
        #       Delete it
        #       Construct new
        #       Get a chunk from it
        # Stack words
        # Stack i-vectors
        # Return both as a batch

    def _new_chunker(self):
        pass
        # Pick a non-used TokenizedSplit
        # Build a Chunker around



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
