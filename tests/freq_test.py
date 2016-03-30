""" This test quite generates a series of pure sine waves
and checks to see if the analyzer does it's job.

author: Brian Schrader
since: 2015-08-05
"""

import sys
from math import sin, cos, pi

from numpy import arange

sys.path.insert(0, '../')
import frequency_analyzer as fq


def get_sin_wave(start, end, rate):
    # x = 2m @ 2Hz
    # y = 1m @ 4Hz
    # z = 1m @ 2Hz
    w = 2 * pi * 2
    return [(t, 18*sin(t*w), 0*sin(2*t*w), 0*cos(w*t)) for t in arange(start, end, 1.0/rate)]


if __name__ == '__main__':
    seq = get_sin_wave(0.0, 200, 50.0)
    s = [','.join([str(t), str(x), str(y), str(z)]) for (t, x, y, z) in seq]
    s = '\n'.join(s)
    with open('sin_wave.csv', 'w') as f:
        f.write(s)
