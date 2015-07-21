""" Tests for the sanitizer functions. """


import sys, unittest

sys.path.insert(0, '../bin/')
from sanitizer import *


class SanitizerTest(unittest.TestCase):

    def test_sanitize(self):
        data = [(3232, 3, 3, 3),
                (3244, 3, 3, 3),
                (3255, 3, 3, 3),
                (3276, 2, 2, 2),
                (3289, 1, 1, 1)]

        clean_data = sanitize(data, sloppy=False, time_unit='seconds')

        t_0, x_0, y_0, z_0 = clean_data[0]
        self.assertEqual(t_0, 0)

        t_1, x_1, y_1, z_1 = clean_data[1]
        self.assertEqual(t_1, 12)

        del clean_data
        clean_data = sanitize(data, sloppy=True, time_unit='seconds')

        t_1, x_1, y_1, z_1 = clean_data[1]
        self.assertEqual(t_1, 10)

        del clean_data
        clean_data = sanitize(data, sloppy=True, time_unit='milliseconds')

        t_1, x_1, y_1, z_1 = clean_data[1]
        self.assertEqual(t_1, 0.01)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(SanitizerTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
