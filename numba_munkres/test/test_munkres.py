import unittest
import numpy as np

from numba_munkres import munkres


class TestMunkres(unittest.TestCase):
    def test_munkres_finds_lowest_cost_assignment(self):
        square = np.array([
            [1, 2, 3],
            [2, 4, 6],
            [3, 6, 9]
        ])
        optimal_assignment = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0]
        ], dtype=bool)
        np.testing.assert_array_equal(munkres(square), optimal_assignment)

    def test_munkres_returns_same_shape(self):
        rectangle = np.array([
            [1, 2, 3],
            [4, 5, 6]
        ])
        self.assertEqual(munkres(rectangle).shape, rectangle.shape)
