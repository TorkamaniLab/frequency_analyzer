""" Resets the time interval to be in milliseconds.
Removes the epoch.

author: Brian Schrader
since: 2015-07-10
"""


from timer import timeit


""" Should the sanitization be done so that
the resulting intervals are rounded (sloppy)
or exact?
"""
sloppy = True

""" What time unit is the data given in.
This unit effects the conversion during
sanitization. Time will be converted to
seconds.
'seconds', 'milliseconds'
"""
time_unit = 'milliseconds'


def sanitize(data):
    """ Converts time in data from epoch to
    delta. Assumes the data is sorted.
    """
    start = data[0][0]
    res = []
    for row in data:
        t, x, y, z = row
        t = t - start if not sloppy else sloppify(t - start)
        t = t / 1000.0 if time_unit == 'milliseconds' else t
        res.append((t - start, x, y, z))
    return res


def sloppify(n):
    """ Removes the ones place from the integer provided. """
    return int(n / 10) * 10


def main():
    import sys, time
    try:
        input_filename, output_filename = sys.argv[1:3]
    except ValueError:
        print('Usage: python sanitizer.py <input_file>, <output_file>')
    with open(input_filename) as f, open(output_filename, 'w') as o:
        data = []
        # Ignore first line
        first_line = f.readline()
        for line in f:
            t, x, y, z = line.split(',')
            data.append((int(t), int(x), int(y), int(z)))
        o_data = sanitize(data)
        o_data = [','.join([str(t), str(x), str(y), str(z)]) for t, x, y, z in o_data]
        o.write(first_line)
        o.writelines('\n'.join(o_data))


if __name__ == '__main__':
    timeit(main)
