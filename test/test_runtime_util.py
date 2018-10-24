import torch
from test.common import TestCase

from runtime.runtime_utils import repackage_hidden


class TensorReorganizerTests(TestCase):
    def test_data_kept(self):
        tensor = torch.FloatTensor([[[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]])

        repackaged = repackage_hidden(tensor)
        self.assertEqual(tensor, repackaged)
