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


class HistoryIvecAppender():
    def __init__(self, tokens, ivec_eetor):
        """
            Args:
                tokens (TokenizedSplit): Source of tokens, represents single 'document'.
        """
        self.tokens = tokens
        self._ivec_eetor = ivec_eetor


    def __iter__(self):
        history_words = []
        for (x, t), words in zip(self.tokens, self.tokens.input_words()):
            ivec = self._ivec_eetor(" ".join(history_words))
            history_words += words.split()
            yield (x, t, ivec)
