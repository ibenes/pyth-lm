class CheatingIvecAppender():
    def __init__(self, tokens, ivec_eetor):
        """
            Args:
                tokens (TokenizedSplit): Source of tokens, represents single 'document'.
        """
        self.tokens = tokens
        all_words = " ".join(self.tokens.input_words())
        self._ivec = ivec_eetor(all_words)


    def __iter__(self):
        for x, t in self.tokens:
            yield (x, t, self._ivec)
