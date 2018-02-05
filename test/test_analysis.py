import unittest
import torch 

from analysis import categorical_entropy


class CategoricalEntropyTests(unittest.TestCase):
    def setUp(self):
        pass

    def test_uniform_2d(self):
        p_x = torch.FloatTensor([0.5, 0.5])
        H_x = torch.FloatTensor([1.0])

        self.assertTrue(torch.equal(categorical_entropy(p_x), H_x))

    def test_uniform_4d(self):
        p_x = torch.FloatTensor([0.25, 0.25, 0.25, 0.25])
        H_x = torch.FloatTensor([2.0])

        self.assertTrue(torch.equal(categorical_entropy(p_x), H_x))

    def test_nonuniform_3d(self):
        p_x = torch.FloatTensor([0.5, 0.25, 0.25])
        H_x = torch.FloatTensor([1.5])

        self.assertTrue(torch.equal(categorical_entropy(p_x), H_x))

    def test_sparse_2d(self):
        p_x = torch.FloatTensor([0.0, 1.0])
        H_x = torch.FloatTensor([0.0])

        H_x_hat = categorical_entropy(p_x)

        self.assertTrue(torch.equal(H_x_hat, H_x))
