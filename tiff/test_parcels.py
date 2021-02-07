import logging
import unittest
import numpy
from tiff import parcels


class TestParcels(unittest.TestCase):
    def test_build_flanges(self):
        dataset = numpy.ones((20, 22))
        config = {
            "model": {
                "surface_thickness_millimeters": 1.0,
                "flange_thickness_millimeters": 1000.0,
            },
            "printer": {
                "xy_resolution_microns": 1000.0,
            }
        }
        parcel_shape = (7, 6)
        with_flanges = parcels.build_flanges(config, dataset, parcel_shape, logging.getLogger("test"))
        expected = numpy.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0],
            [0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0],
            [0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0],
            [0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0],
            [0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0],
            [0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0],
            [0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0],
            [0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0],
            [0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0],
            [0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0],
            [0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0],
            [0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]).astype(numpy.float32)
        numpy.testing.assert_array_equal(with_flanges, expected)

    def test_parcels(self):
        # we want to be able to visually identify if there's an item not in the correct parcel, so we can create
        # a 2D array here where the X and Y coordinates are visually distinct by shifting one of them out by two
        # orders of magnitude, so the value mod 100 is one index and the value / 100 is the other - this array
        # will look like:
        # [    0,    1,    2,    3 ...   20,   21] - row 0
        # [  100,  101,  102,  103 ... 1020, 1021] - row 1
        # ...
        # [ 1900, 1901, 1912, 1913 ... 1920, 1921] - row 19
        dataset = numpy.arange(20).reshape(20, 1) * 100 + numpy.arange(22)
        actual = [parcel for index, parcel in parcels.parcels(dataset, (7, 6))]
        expected = [
            [[0, 1, 2, 3, 4, 5],
             [100, 101, 102, 103, 104, 105],
             [200, 201, 202, 203, 204, 205],
             [300, 301, 302, 303, 304, 305],
             [400, 401, 402, 403, 404, 405],
             [500, 501, 502, 503, 504, 505],
             [600, 601, 602, 603, 604, 605]],
            [[6, 7, 8, 9, 10, 11],
             [106, 107, 108, 109, 110, 111],
             [206, 207, 208, 209, 210, 211],
             [306, 307, 308, 309, 310, 311],
             [406, 407, 408, 409, 410, 411],
             [506, 507, 508, 509, 510, 511],
             [606, 607, 608, 609, 610, 611]],
            [[12, 13, 14, 15, 16, 17],
             [112, 113, 114, 115, 116, 117],
             [212, 213, 214, 215, 216, 217],
             [312, 313, 314, 315, 316, 317],
             [412, 413, 414, 415, 416, 417],
             [512, 513, 514, 515, 516, 517],
             [612, 613, 614, 615, 616, 617]],
            [[18, 19, 20, 21],
             [118, 119, 120, 121],
             [218, 219, 220, 221],
             [318, 319, 320, 321],
             [418, 419, 420, 421],
             [518, 519, 520, 521],
             [618, 619, 620, 621]],
            [[700, 701, 702, 703, 704, 705],
             [800, 801, 802, 803, 804, 805],
             [900, 901, 902, 903, 904, 905],
             [1000, 1001, 1002, 1003, 1004, 1005],
             [1100, 1101, 1102, 1103, 1104, 1105],
             [1200, 1201, 1202, 1203, 1204, 1205],
             [1300, 1301, 1302, 1303, 1304, 1305]],
            [[706, 707, 708, 709, 710, 711],
             [806, 807, 808, 809, 810, 811],
             [906, 907, 908, 909, 910, 911],
             [1006, 1007, 1008, 1009, 1010, 1011],
             [1106, 1107, 1108, 1109, 1110, 1111],
             [1206, 1207, 1208, 1209, 1210, 1211],
             [1306, 1307, 1308, 1309, 1310, 1311]],
            [[712, 713, 714, 715, 716, 717],
             [812, 813, 814, 815, 816, 817],
             [912, 913, 914, 915, 916, 917],
             [1012, 1013, 1014, 1015, 1016, 1017],
             [1112, 1113, 1114, 1115, 1116, 1117],
             [1212, 1213, 1214, 1215, 1216, 1217],
             [1312, 1313, 1314, 1315, 1316, 1317]],
            [[718, 719, 720, 721],
             [818, 819, 820, 821],
             [918, 919, 920, 921],
             [1018, 1019, 1020, 1021],
             [1118, 1119, 1120, 1121],
             [1218, 1219, 1220, 1221],
             [1318, 1319, 1320, 1321]],
            [[1400, 1401, 1402, 1403, 1404, 1405],
             [1500, 1501, 1502, 1503, 1504, 1505],
             [1600, 1601, 1602, 1603, 1604, 1605],
             [1700, 1701, 1702, 1703, 1704, 1705],
             [1800, 1801, 1802, 1803, 1804, 1805],
             [1900, 1901, 1902, 1903, 1904, 1905]],
            [[1406, 1407, 1408, 1409, 1410, 1411],
             [1506, 1507, 1508, 1509, 1510, 1511],
             [1606, 1607, 1608, 1609, 1610, 1611],
             [1706, 1707, 1708, 1709, 1710, 1711],
             [1806, 1807, 1808, 1809, 1810, 1811],
             [1906, 1907, 1908, 1909, 1910, 1911]],
            [[1412, 1413, 1414, 1415, 1416, 1417],
             [1512, 1513, 1514, 1515, 1516, 1517],
             [1612, 1613, 1614, 1615, 1616, 1617],
             [1712, 1713, 1714, 1715, 1716, 1717],
             [1812, 1813, 1814, 1815, 1816, 1817],
             [1912, 1913, 1914, 1915, 1916, 1917]],
            [[1418, 1419, 1420, 1421],
             [1518, 1519, 1520, 1521],
             [1618, 1619, 1620, 1621],
             [1718, 1719, 1720, 1721],
             [1818, 1819, 1820, 1821],
             [1918, 1919, 1920, 1921]]
        ]

        self.assertEqual(len(actual), len(expected))
        for i in range(len(actual)):
            numpy.testing.assert_array_equal(actual[i], expected[i])


if __name__ == '__main__':
    unittest.main()
