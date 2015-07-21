""" Tests for the integrator module. """


import sys, unittest

sys.path.insert(0, '../bin/')
from integrator import *


class IntegratorTest(unittest.TestCase):

    def test_trapezoid(self):
        x_0, t_0 = 0, 0
        x_1, t_1 = 1, 1
        dx = trapezoid(x_0, t_0, x_1, t_1)
        self.assertEqual(dx, 0.5)

    def test_integrate(self):
        s_t = [(0, 0, 0, 0),
                (1, 2, 2, 2),
                (2, 4, 4, 4),
                (3, 6, 6, 6),
                (4, 8, 8, 8),
                (5, 10, 10, 10)]
        v_t = integrate(s_t)

        self.assertTrue(v_t is not None)

        t, x, y, z = v_t[1]
        self.assertEqual(t, 2)
        self.assertEqual(x, 4)
        self.assertEqual(y, 4)
        self.assertEqual(z, 4)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(IntegratorTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
