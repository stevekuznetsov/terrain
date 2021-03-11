import unittest
import math
import numpy
from tiff import support


class TestSupports(unittest.TestCase):
    def test_euclidean_distance(self):
        self.assertEqual(support.euclidean_distance((0, 0, 0), (1, 1, 1)), math.sqrt(3))
        self.assertEqual(support.euclidean_distance((1, 1, 1), (0, 0, 0)), math.sqrt(3))
        self.assertEqual(support.euclidean_distance((1, 1, 1), (5, 12, 7)), math.sqrt(173))

    def test_supporting_nodes_for_node(self):
        self.assertEqual(support.supporting_nodes_for_node(
            (10, 10, 10),
            1, 45
        ), {
            (10, 10, 9),
        })

        self.assertEqual(support.supporting_nodes_for_node(
            (10, 10, 10),
            3, 45
        ), {
            (10, 9, 9),
            (9, 10, 9), (10, 10, 9), (11, 10, 9),
            (10, 11, 9),

            (10, 8, 8),
            (10, 9, 8),
            (8, 10, 8), (9, 10, 8), (10, 10, 8), (11, 10, 8), (12, 10, 8),
            (10, 11, 8),
            (10, 12, 8),

            (10, 10, 7),
        })

    def test_local_neighborhood_nodes_for_element(self):
        self.assertEqual(support.local_neighborhood_nodes_for_element(
            (0, 0, 0),
            1,
        ), {
            (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1), (1, 1, 0),
        })

        self.assertEqual(support.local_neighborhood_nodes_for_element(
            (10, 10, 10), 2,
        ), {
            (10, 10, 9), (10, 11, 9),
            (11, 10, 9), (11, 11, 9),

            (9, 10, 10), (9, 11, 10),
            (10, 9, 10), (10, 10, 10), (10, 11, 10), (10, 12, 10),
            (11, 9, 10), (11, 10, 10), (11, 11, 10), (11, 12, 10),
            (12, 10, 10), (12, 11, 10),

            (9, 10, 11), (9, 11, 11),
            (10, 9, 11), (10, 10, 11), (10, 11, 11), (10, 12, 11),
            (11, 9, 11), (11, 10, 11), (11, 11, 11), (11, 12, 11),
            (12, 11, 11), (12, 10, 11),

            (10, 10, 12), (10, 11, 12),
            (11, 10, 12), (11, 11, 12),
        })

    def test_nodes_within_bounds(self):
        self.assertEqual(support.nodes_within_bounds(
            {(-1, 0, 1), (0, 0, 0), (0, 10, 1), (1, 1, 1)},
            (2, 2, 2)
        ), {
            (0, 0, 0), (1, 1, 1),
        })

    def test_weighting_factor(self):
        self.assertEqual(support.weighting_factor((0, 0, 0), (2, 2, 2), 10), (1 - 2 * math.sqrt(3) / 10))

    def test_elemental_index_to_nodal_index(self):
        self.assertEqual(support.elemental_index_to_nodal_index((0, 0, 0)), (0.5, 0.5, 0.5))

    def test_weighted_filtered_local_neighborhood_nodes_for_element(self):
        dist = 1 - math.sqrt(3 / 4) / 2
        self.assertCountEqual(support.weighted_filtered_local_neighborhood_nodes_for_element(
            (0, 0, 0), 2, (2, 2, 2), 10 * numpy.ones((2, 2))
        ), [
            ((0, 0, 0), dist),
            ((1, 0, 0), dist),
            ((0, 1, 0), dist),
            ((0, 0, 1), dist),
            ((0, 1, 1), dist),
            ((1, 1, 1), dist),
            ((1, 0, 1), dist),
            ((1, 1, 0), dist),
        ])

    def test_node_below_adjacent_elements(self):
        surface = numpy.array([
            [0.0, 1.0],
            [2.0, 3.0],
        ])
        self.assertEqual(support.node_below_adjacent_elements((0, 0, 0), surface), True)
        self.assertEqual(support.node_below_adjacent_elements((0, 0, 0), surface), True)
        self.assertEqual(support.node_below_adjacent_elements((0, 1, 1), surface), True)
        self.assertEqual(support.node_below_adjacent_elements((1, 0, 1), surface), True)
        self.assertEqual(support.node_below_adjacent_elements((1, 1, 3), surface), True)


if __name__ == '__main__':
    unittest.main()
