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

    def test_single_insertion(self):
        a = "a".split()
        b = "a b".split()

        expected = [
            (['a'], ['a', 'b']),
        ]

        self.assertEqual(align(a, b), expected)

    def test_double_insertion(self):
        a = "a".split()
        b = "a b c".split()

        expected = [
            (['a'], ['a', 'b', 'c']),
        ]

        self.assertEqual(align(a, b), expected)

    def test_single_insertion_reversed(self):
        a = "b".split()
        b = "a b".split()

        expected = [
            (['b'], ['a', 'b']),
        ]

        self.assertEqual(align(a, b), expected)

    def test_double_insertion_reversed(self):
        a = "b".split()
        b = "c a b".split()

        expected = [
            (['b'], ['c', 'a', 'b']),
        ]

        self.assertEqual(align(a, b), expected)

    def test_single_deletion(self):
        a = "a b".split()
        b = "a".split()

        expected = [
            (['a', 'b'], ['a']),
        ]

        self.assertEqual(align(a, b), expected)

    def test_double_deletion(self):
        a = "a b c".split()
        b = "a".split()

        expected = [
            (['a', 'b', 'c'], ['a']),
        ]

        self.assertEqual(align(a, b), expected)

    def test_single_deletion_reversed(self):
        a = "a b".split()
        b = "b".split()

        expected = [
            (['a', 'b'], ['b']),
        ]

        self.assertEqual(align(a, b), expected)

    def test_double_deletion_reversed(self):
        a = "c a b".split()
        b = "b".split()

        expected = [
            (['c', 'a', 'b'], ['b']),
        ]

        self.assertEqual(align(a, b), expected)
