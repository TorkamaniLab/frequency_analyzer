""" Tests for the Matrix module. """


import unittest, sys
from math import pi

sys.path.insert(0, '../bin/')
from matrix import *


err = 1e-10


class MatrixTest(unittest.TestCase):

    def test_transformation_matrix(self):
        angles = [('z', pi / 2.0)]  # 90 degrees
        r = transformation_matrix(angles)[0]
        self.assertTrue(r is not None)

        self.assertTrue(abs(r[0][0]) < err)
        self.assertTrue(abs(r[0][1]) - 1 < err)

        self.assertTrue(abs(r[1][0]) - 1 < err)
        self.assertTrue(abs(r[1][1]) < err)

    def test_transform(self):
        v = [1, 0, 0]
        R = transformation_matrix([('z', pi/2)])
        v_t = transform(v, R)

        self.assertTrue(v_t is not None)

        self.assertTrue(abs(v_t[0]) < err)
        self.assertTrue(v_t[1] - 1 < err)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(MatrixTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
