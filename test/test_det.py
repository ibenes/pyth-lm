from unittest import TestCase

from det import det_points_from_score_tg
from det import subsample_list


class DetPointTests(TestCase):
    def test_trivial(self):
        score_tg = [
            (0.0, 0),
            (1.0, 1)
        ]

        det_points = [
            [0.0, 0.5],
            [0.0, 0.0],
            [0.5, 0.0],
        ]

        self.assertEqual(det_points_from_score_tg(score_tg), det_points)

    def test_permutation(self):
        score_tg = [
            (1.0, 1),
            (0.0, 0),
        ]

        det_points = [
            [0.0, 0.5],
            [0.0, 0.0],
            [0.5, 0.0],
        ]

        self.assertEqual(det_points_from_score_tg(score_tg), det_points)

    def test_simple_bad_system(self):
        score_tg = [
            (1.0, 0),
            (0.0, 1),
        ]

        det_points = [
            [0.0, 0.5],
            [0.5, 0.5],
            [0.5, 0.0],
        ]

        self.assertEqual(det_points_from_score_tg(score_tg), det_points)

    def test_multiple_points(self):
        score_tg = [
            (0.0, 0),
            (0.1, 1),
            (0.2, 0),
            (0.3, 1),
            (0.4, 1),
        ]

        det_points = [
            [0.0, 0.4],
            [0.0, 0.2],
            [0.2, 0.2],
            [0.2, 0.0],
            [0.4, 0.0],
            [0.6, 0.0],
        ]

        self.assertEqual(det_points_from_score_tg(score_tg), det_points)


class ListSubsamplingTests(TestCase):
    def test_trivial(self):
        self.assertEqual(
            subsample_list([1, 2], 2),
            [1, 2]
        )
