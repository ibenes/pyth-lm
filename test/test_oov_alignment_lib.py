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

    def test_single_element(self):
        a = "a".split()
        b = "a".split()

        expected = [
            (['a'], ['a']),
        ]

        self.assertEqual(align(a, b), expected)

    def test_substitution_only(self):
        a = "a".split()
        b = "b".split()

        expected = [
            (['a'], ['b']),
        ]

        self.assertEqual(align(a, b), expected)
