from unittest import TestCase

from det import det_points_from_score_tg


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
