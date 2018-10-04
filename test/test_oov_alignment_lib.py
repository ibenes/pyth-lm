from unittest import TestCase

from oov_alignment_lib import align


class AlignTest(TestCase):
    def test_trivial(self):
        a = "a b".split()
        b = "a b".split()

        expected = [
            (['a'], ['a']),
            (['b'], ['b']),
        ]

        self.assertEqual(align(a, b), expected)
