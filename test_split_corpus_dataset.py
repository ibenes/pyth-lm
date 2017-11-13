import split_corpus_dataset
import unittest

import io
import numpy as np


class CheatingIvecAppenderTests(unittest.TestCase):
    def setUp(self):
        self.ivec_eetor = lambda x: np.asarray([hash(tuple(x)) % 1337])
        self.test_words_short = "a b c a".split()
        self.test_words_long = "a b c a a".split()
        self.vocab = {
            "a": 0,
            "b": 1,
            "c": 2
        }

    def test_single_data(self):
        data_source = getStream(self.test_words_short)
        ts = split_corpus_dataset.TokenizedSplit(data_source, self.vocab, 1)
        appender = split_corpus_dataset.CheatingIvecAppender(ts, self.ivec_eetor)

         # cannot acces ts._tokens, it's an implementation 
        tokens = [self.vocab[w] for w in self.test_words_short]

        expectation = ([0], [1], self.ivec_eetor(tokens))
        seqs = list(iter(appender))
        first = seqs[0]

        self.assertEqual(first, expectation)

    def test_whole_seq(self):
        data_source = getStream(self.test_words_short)
        ts = split_corpus_dataset.TokenizedSplit(data_source, self.vocab, 1)
        appender = split_corpus_dataset.CheatingIvecAppender(ts, self.ivec_eetor)

         # cannot acces ts._tokens, it's an implementation 
        tokens = [self.vocab[w] for w in self.test_words_short]

        expectation = [
            ([0], [1], self.ivec_eetor(tokens)),
            ([1], [2], self.ivec_eetor(tokens)),
            ([2], [0], self.ivec_eetor(tokens))
        ]

        seqs = list(iter(appender))
        self.assertEqual(seqs, expectation)

    def test_whole_seq_with_next(self):
        data_source = getStream(self.test_words_short)
        ts = split_corpus_dataset.TokenizedSplit(data_source, self.vocab, 1)
        appender = split_corpus_dataset.CheatingIvecAppender(ts, self.ivec_eetor)
        appender = iter(appender)

         # cannot acces ts._tokens, it's an implementation 
        tokens = [self.vocab[w] for w in self.test_words_short]
        expectation = [
            ([0], [1], self.ivec_eetor(tokens)),
            ([1], [2], self.ivec_eetor(tokens)),
            ([2], [0], self.ivec_eetor(tokens))
        ]

        seq0 = next(appender)
        self.assertEqual(seq0, expectation[0])

        seq1 = next(appender)
        self.assertEqual(seq1, expectation[1])

        seq2 = next(appender)
        self.assertEqual(seq2, expectation[2])

    def test_iter_ends(self):
        data_source = getStream(self.test_words_short)
        ts = split_corpus_dataset.TokenizedSplit(data_source, self.vocab, 1)
        appender = split_corpus_dataset.CheatingIvecAppender(ts, self.ivec_eetor)
        appender = iter(appender)

        next(appender)
        next(appender)
        next(appender)

        self.assertRaises(StopIteration, next, appender)


def getStream(words):
    data_source = io.StringIO()
    data_source.write(" ".join(words))
    data_source.seek(0)

    return data_source


class TokenizedSplitTests(unittest.TestCase):
    def setUp(self):
        self.test_words_short = "a b c a".split()
        self.test_words_long = "a b c a a".split()

        self.vocab = {
            "a": 0,
            "b": 1,
            "c": 2
        }
        

    def test_single_word(self):
        data_source = getStream(self.test_words_short)
        ts = split_corpus_dataset.TokenizedSplit(data_source, self.vocab, 1)
        tokens_strings = list(iter(ts))
        expectation = ([0], [1])
        self.assertEqual(tokens_strings[0], expectation)

    def test_single_word_seq(self):
        data_source = getStream(self.test_words_short)
        ts = split_corpus_dataset.TokenizedSplit(data_source, self.vocab, 1)
        tokens_strings = list(iter(ts))
        expectation = [([0], [1]), ([1], [2]), ([2], [0])]
        self.assertEqual(tokens_strings, expectation)

    def test_single_word_len(self):
        data_source = getStream(self.test_words_short)
        ts = split_corpus_dataset.TokenizedSplit(data_source, self.vocab, 1)
        self.assertEqual(len(ts), len(self.test_words_short)-1)

    def test_two_word_seq(self):
        data_source = getStream(self.test_words_short)
        ts = split_corpus_dataset.TokenizedSplit(data_source, self.vocab, 2)
        tokens_strings = list(iter(ts))
        expectation = [([0, 1], [1, 2])]
        self.assertEqual(tokens_strings, expectation)

    def test_two_word_seq_long(self):
        data_source = getStream(self.test_words_long)
        ts = split_corpus_dataset.TokenizedSplit(data_source, self.vocab, 2)
        tokens_strings = list(iter(ts))
        expectation = [([0, 1], [1, 2]), ([2, 0], [0, 0])]
        self.assertEqual(tokens_strings, expectation)
