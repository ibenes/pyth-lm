import common
import smm_ivec_extractor

import torch
import torch.nn as nn
from torch.autograd import Variable

from sklearn.feature_extraction.text import CountVectorizer

class DummySMM(nn.Module):
    def __init__(self, ivec_dim):
        self.W = Variable(torch.zeros(ivec_dim, 1), requires_grad=True)


class IvecExtractorTests(common.TestCase):
    def setUp(self):
        smm = DummySMM(ivec_dim=4)
        documents = ["text consisting of SIX different words", "text"]
        self.cvect = CountVectorizer(documents, strip_accents='ascii', analyzer='word')
        self.cvect.fit(documents)
        self.extractor = smm_ivec_extractor.IvecExtractor(smm, nb_iters=10, lr=0.1, tokenizer=self.cvect)

    def test_one_empty_bow(self):
        ivecs = self.extractor.zero_bows(1)
        expectation = Variable(torch.zeros(1, 6))
        self.assertEqual(ivecs, expectation)

    def test_two_empty_bows(self):
        ivecs = self.extractor.zero_bows(2)
        expectation = Variable(torch.zeros(2, 6))
        self.assertEqual(ivecs, expectation)
