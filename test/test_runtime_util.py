import torch
from test.common import TestCase

from runtime.runtime_utils import repackage_hidden


class TensorReorganizerTests(TestCase):
    def setUp(self):
        self.computed_tensor = torch.FloatTensor([[[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]])

    def test_data_kept(self):
        repackaged = repackage_hidden(self.computed_tensor)
        self.assertEqual(self.computed_tensor, repackaged)

    def test_data_requires_grad(self):
        repackaged = repackage_hidden(self.computed_tensor)
        self.assertTrue(repackaged.requires_grad_)
