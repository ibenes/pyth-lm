import unittest
from hidden_state_reorganization import HiddenStateReorganizer

import torch

class Dummy_lstm():
    def __init__(self, nb_hidden):
        self._nb_hidden = nb_hidden

    def init_hidden(self, batch_size):
        return ( 
            torch.FloatTensor([[[0.0] * self._nb_hidden] * batch_size]),
            torch.FloatTensor([[[0.0] * self._nb_hidden] * batch_size])
        )

class HiddenStateReorganizerTests(unittest.TestCase):
    def setUp(self):
        lm = Dummy_lstm(nb_hidden=2)
        self.reorganizer = HiddenStateReorganizer(lm)

    def hidden_equal(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected):
            self.assertTrue(torch.equal(a, e))

    def test_passing(self):
        last_h = (
            torch.FloatTensor([[[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]]),
            torch.FloatTensor([[[1, 1], [2, 2], [3, 3]]]),
        )

        mask = torch.LongTensor([0, 1, 2])
        bsz = 3

        new_h = self.reorganizer(last_h, mask, bsz)
        self.hidden_equal(new_h, last_h)

    def test_shrinks(self):
        last_h = (
            torch.FloatTensor([[[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]]),
            torch.FloatTensor([[[1, 1], [2, 2], [3, 3]]]),
        )

        mask = torch.LongTensor([0, 2])
        bsz = 2

        new_h = self.reorganizer(last_h, mask, bsz)
        expected = (
            torch.FloatTensor([[[0.1, 0.1], [0.3, 0.3]]]),
            torch.FloatTensor([[[1, 1], [3, 3]]]),
        )
        self.hidden_equal(new_h, expected)

    def test_requires_bsz_greater_than_mask(self):
        last_h = (
            torch.FloatTensor([[[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]]),
            torch.FloatTensor([[[1, 1], [2, 2], [3, 3]]]),
        )

        mask = torch.LongTensor([0, 1, 2])
        bsz = 2

        self.assertRaises(ValueError, self.reorganizer, last_h, mask, bsz)

    def test_on_empty_mask_zeros(self):
        last_h = (
            torch.FloatTensor([[[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]]),
            torch.FloatTensor([[[1, 1], [2, 2], [3, 3]]]),
        )

        mask = torch.LongTensor([])
        bsz = 2

        new_h = self.reorganizer(last_h, mask, bsz)
        expected = (
            torch.FloatTensor([[[0.0, 0.0], [0.0, 0.0]]]),
            torch.FloatTensor([[[0.0, 0.0], [0.0, 0.0]]]),
        )
        self.hidden_equal(new_h, expected)

    def test_completion_by_zeros(self):
        last_h = (
            torch.FloatTensor([[[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]]),
            torch.FloatTensor([[[1, 1], [2, 2], [3, 3]]]),
        )

        mask = torch.LongTensor([1])
        bsz = 2

        new_h = self.reorganizer(last_h, mask, bsz)
        expected = (
            torch.FloatTensor([[[0.2, 0.2], [0.0, 0.0]]]),
            torch.FloatTensor([[[2.0, 2.0], [0.0, 0.0]]]),
        )
        self.hidden_equal(new_h, expected)

    def test_bug_regression_single_addition(self):
        last_h = (
            torch.FloatTensor([[[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]]),
            torch.FloatTensor([[[1, 1], [2, 2], [3, 3]]]),
        )

        mask = torch.LongTensor([1,2])
        bsz = 3

        new_h = self.reorganizer(last_h, mask, bsz)
        expected = (
            torch.FloatTensor([[[0.2, 0.2], [0.3, 0.3], [0.0, 0.0]]]),
            torch.FloatTensor([[[2.0, 2.0], [3.0, 3.0], [0.0, 0.0]]]),
        )
        self.hidden_equal(new_h, expected)
