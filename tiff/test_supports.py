import logging
import unittest
import math
from tiff import support


class TestSupports(unittest.TestCase):
    def test_euclidian_distance(self):
        self.assertEqual(support.euclidian_distance((0, 0, 0), (1, 1, 1)), math.sqrt(3))
        self.assertEqual(support.euclidian_distance((1, 1, 1), (0, 0, 0)), math.sqrt(3))
        self.assertEqual(support.euclidian_distance((1, 1, 1), (5, 12, 7)), math.sqrt(173))

    def test_neighboring_set_for(self):
        self.assertEqual(support.neighboring_set_for(
            (10, 10, 10),
            {"xy_resolution_microns": 1000},
            {
                "minimum_feature_radius_millimeters": 1,
                "self_supporting_angle_degrees": 45,
            }
        ), {
            (10, 10, 9),
        })

        self.assertEqual(support.neighboring_set_for(
            (10, 10, 10),
            {"xy_resolution_microns": 1000},
            {
                "minimum_feature_radius_millimeters": 3,
                "self_supporting_angle_degrees": 45,
            }
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


if __name__ == '__main__':
    unittest.main()
