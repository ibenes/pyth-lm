import unittest

import split_corpus_dataset
import ivec_appenders
import numpy as np
import common

from utils import getStream

class CheatingIvecAppenderTests(common.TestCase):
    def setUp(self):
        self.ivec_eetor = lambda x: np.asarray([hash(x) % 1337])
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
        appender = ivec_appenders.CheatingIvecAppender(ts, self.ivec_eetor)

         # cannot acces ts._tokens, it's an implementation 
        tokens = [self.vocab[w] for w in self.test_words_short]

        expectation = self.ivec_eetor(" ".join(self.test_words_short[:-1]))
        seqs = next(iter(appender))
        first = seqs[2]

        self.assertEqual(first, expectation)

    def test_whole_seq(self):
        data_source = getStream(self.test_words_short)
        ts = split_corpus_dataset.TokenizedSplit(data_source, self.vocab, 1)
        appender = ivec_appenders.CheatingIvecAppender(ts, self.ivec_eetor)

         # cannot acces ts._tokens, it's an implementation 
        tokens = [self.vocab[w] for w in self.test_words_short]

        expectation = [
            self.ivec_eetor(" ".join(self.test_words_short[:-1])),
            self.ivec_eetor(" ".join(self.test_words_short[:-1])),
            self.ivec_eetor(" ".join(self.test_words_short[:-1]))
        ]

        seqs = [x[2] for x in (iter(appender))]
        self.assertEqual(seqs, expectation)

    def test_whole_seq_with_next(self):
        data_source = getStream(self.test_words_short)
        ts = split_corpus_dataset.TokenizedSplit(data_source, self.vocab, 1)
        appender = ivec_appenders.CheatingIvecAppender(ts, self.ivec_eetor)
        appender = iter(appender)

         # cannot acces ts._tokens, it's an implementation 
        tokens = [self.vocab[w] for w in self.test_words_short]
        expectation = [
            self.ivec_eetor(" ".join(self.test_words_short[:-1])),
            self.ivec_eetor(" ".join(self.test_words_short[:-1])),
            self.ivec_eetor(" ".join(self.test_words_short[:-1]))
        ]

        seq0 = next(appender)[2]
        self.assertEqual(seq0, expectation[0])

        seq1 = next(appender)[2]
        self.assertEqual(seq1, expectation[1])

        seq2 = next(appender)[2]
        self.assertEqual(seq2, expectation[2])

    def test_iter_ends(self):
        data_source = getStream(self.test_words_short)
        ts = split_corpus_dataset.TokenizedSplit(data_source, self.vocab, 1)
        appender = ivec_appenders.CheatingIvecAppender(ts, self.ivec_eetor)
        appender = iter(appender)

        next(appender)
        next(appender)
        next(appender)

        self.assertRaises(StopIteration, next, appender)


class HistoryIvecAppenderTests(common.TestCase):
    def setUp(self):
        self.ivec_eetor = lambda x: np.asarray([hash(x) % 1337])
        self.test_words_short = "a b c a".split()
        self.test_words_long = "ab bb cd a b".split()
        self.vocab = {
            "a": 0,
            "b": 1,
            "c": 2,
            "ab": 3,
            "bb": 4,
            "cd": 5,
        }

    def test_single_data(self):
        data_source = getStream(self.test_words_short)
        ts = split_corpus_dataset.TokenizedSplit(data_source, self.vocab, 1)
        appender = ivec_appenders.HistoryIvecAppender(ts, self.ivec_eetor)

         # cannot acces ts._tokens, it's an implementation 
        tokens = [self.vocab[w] for w in self.test_words_short]

        expectation = self.ivec_eetor(" ".join([]))
        seqs = next(iter(appender))
        first = seqs[2]

        self.assertEqual(first, expectation)

    def test_whole_seq(self):
        data_source = getStream(self.test_words_short)
        ts = split_corpus_dataset.TokenizedSplit(data_source, self.vocab, 1)
        appender = ivec_appenders.HistoryIvecAppender(ts, self.ivec_eetor)

         # cannot acces ts._tokens, it's an implementation 
        tokens = [self.vocab[w] for w in self.test_words_short]

        expectation = [
            self.ivec_eetor(" ".join(self.test_words_short[:0])),
            self.ivec_eetor(" ".join(self.test_words_short[:1])),
            self.ivec_eetor(" ".join(self.test_words_short[:2])),
        ]
        seqs = [x[2] for x in (iter(appender))]

        self.assertEqual(seqs, expectation)

    def test_multiletter_words(self):
        data_source = getStream(self.test_words_long)
        ts = split_corpus_dataset.TokenizedSplit(data_source, self.vocab, 1)
        appender = ivec_appenders.HistoryIvecAppender(ts, self.ivec_eetor)

         # cannot acces ts._tokens, it's an implementation 
        tokens = [self.vocab[w] for w in self.test_words_short]

        expectation = [
            self.ivec_eetor(" ".join(self.test_words_long[:0])),
            self.ivec_eetor(" ".join(self.test_words_long[:1])),
            self.ivec_eetor(" ".join(self.test_words_long[:2])),
            self.ivec_eetor(" ".join(self.test_words_long[:3])),
        ]
        seqs = [x[2] for x in iter(appender)]

        self.assertEqual(seqs, expectation)
