""" Given a data set (t, x, y, z) correct for time drift
in the data collection using interpolation.

author: Brian Schrader
since: 2015-06-30
"""

from __future__ import print_function
from scipy.interpolate import griddata
import numpy as np

from timer import timeit


method = 'linear'


def interpolate(grid, points, values):
    """ Given a grid, raw data points, and values interpolate the
    data to fit on the grid.
    """
    return griddata(points, values, grid, method=method)


def main():
    import sys, time
    try:
        interval, input_filename, output_filename = sys.argv[1:4]
    except ValueError:
        print('Usage: python interpolater.py <interval> <input_file>, <output_file>')
    with open(input_filename) as f, open(output_filename, 'w') as o:
        data = []
        # Ignore first line
        f.readline()
        for line in f:
            t, x, y, z = line.split(',')
            data.append((int(time), int(x), int(y), int(z)))
        grid = [range(len(data)) for i in range(3)]

        o_data = interpolate(grid, points, values)
        o.writelines(str(o_data))


if __name__ == '__main__':
    timeit(main)
