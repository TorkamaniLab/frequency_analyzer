""" Given data with time, and x, y, z coordinates
divide the given frequencies in 5 groups.

author: Brian Schrader
since: 2015-06-30
"""


from __future__ import print_function
from scipy.fftpack import fftn as fourier_transform

from timer import timeit


""" A list of names and ranges for the divisions
of the frequencies (Hz).
"""
default_buckets = [
        ['00.87-1.75Hz', 0.87, 1.75],
        ['01.75-3.5Hz', 1.75, 3.5],
        ['03.50-7.0Hz', 3.5, 7],
        ['07.0-14.0Hz', 7, 14],
        ['14.0-28.0Hz', 14, 28]
        ]


def categorize(data, buckets=default_buckets):
    """ Sorts the given data array into the buckets. """
    if buckets is None: buckets = default_buckets
    filled_buckets = {}
    for name, floor, ceiling in buckets:
        bucket = []

        for x, f_x, f_y, f_z in data:
            if x <= ceiling and x >= floor:
                bucket.append((x, f_x, f_y, f_z))
        filled_buckets[name] = sorted(bucket)
    return filled_buckets


def main():
    import sys, time
    try:
        input_filename, output_filename = sys.argv[1:3]
    except ValueError:
        print('Usage: python categorizer.py <input_file>, <output_file>')
    with open(input_filename) as f, open(output_filename, 'w') as o:
        data = []
        # Ignore first line
        f.readline()
        for line in f:
            t, x, y, z = line.split(',')
            data.append((int(x), int(y), int(z)))
        o_data = categorize(fourier_transform(data))
        o.writelines(str(o_data))


if __name__ == '__main__':
    timeit(main)
