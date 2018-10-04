from unittest import TestCase

from oov_alignment_lib import align, extract_mismatch


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

    def test_double_substituion(self):
        a = "a b c d".split()
        b = "a x y d".split()

        expected = [
            (['a'], ['a']),
            (['b'], ['x']),
            (['c'], ['y']),
            (['d'], ['d']),
        ]

        self.assertEqual(align(a, b), expected)

    def test_inner_insertion(self):
        a = "a d".split()
        b = "a b c d".split()

        expected_1 = [
            (['a'], ['a', 'b', 'c']),
            (['d'], ['d']),
        ]
        expected_2 = [
            (['a'], ['a']),
            (['d'], ['b', 'c', 'd']),
        ]

        self.assertIn(align(a, b), [expected_1, expected_2])


class MismatchExtractionTest(TestCase):
    def test_trivial(self):
        ali = [
            (['a'], ['a'])
        ]
        expectation = [
        ]

        self.assertEqual(extract_mismatch(ali), expectation)

    def test_single_substitution(self):
        ali = [
            (['a'], ['b'])
        ]
        expectation = [
            (['a'], ['b'])
        ]

        self.assertEqual(extract_mismatch(ali), expectation)

    def test_double_substitution(self):
        ali = [
            (['a'], ['a']),
            (['a'], ['b']),
            (['a'], ['b']),
            (['a'], ['a']),
            (['a'], ['b']),
        ]
        expectation = [
            (['a', 'a'], ['b', 'b']),
            (['a'], ['b'])
        ]

        self.assertEqual(extract_mismatch(ali), expectation)

    def test_substitution_with_insertion(self):
        ali = [
            (['c'], ['a', 'b']),
        ]
        expectation = [
            (['c'], ['a', 'b']),
        ]

        self.assertEqual(extract_mismatch(ali), expectation)

    def test_insertion_only(self):
        ali = [
            (['a'], ['a', 'b']),
        ]
        expectation = [
            ([], ['b']),
        ]

        self.assertEqual(extract_mismatch(ali), expectation)

    def test_substitution_with_deletion(self):
        ali = [
            (['a', 'b'], ['c']),
        ]
        expectation = [
            (['a', 'b'], ['c']),
        ]

        self.assertEqual(extract_mismatch(ali), expectation)

    def test_deletion_only(self):
        ali = [
            (['a', 'b'], ['a']),
        ]
        expectation = [
            (['b'], []),
        ]

        self.assertEqual(extract_mismatch(ali), expectation)
