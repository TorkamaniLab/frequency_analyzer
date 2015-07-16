#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import print_function
import sys, getopt, os
from math import cos, sin, radians

import matplotlib.pyplot as plt

from bin.sanitizer import sanitize, sloppy
from bin.categorizer import categorize, fourier_transform
from bin.timer import timeit
from bin.complex_utilities import conjugate_filter


usage = """
Description
-----------
A utility for isolating, grouping, and visualizing
frequencies found in raw accelerometer data.

author: Brian Schrader
since: 2015-07-13

Usage
-----
python {} [options] <inputfile>

Options
-------
-h --help       : Display this message.
   --format     : Another help message detailing the format of
                  inputfiles.
-o --outputfile : Write output to a file instead of to the
                  console. (Data is formatted for csv)
-i --inputfile  : Specifies that the next value is the desired
                  input filename. Use this option in place of the
                  input filename argument.
-n --no-header  : Indicates that the inputfile contains no header.
-s --sloppy     : Input times should be treated as roundable to
                  ~10ms ignoring single digit ms. If turned off, data will
                  be interpolated to fit. Defaults to True
-v --verbose    : Verbose output.
-p --print      : Print results to stdout. If no output file is provided,
                  this option is set by default.
-g --graph      : Shows the results visually in another window (using matplotlib).
-e --save       : Save intermediate data (This option will create a number
                  of files in the cwd).
-a --angle      : The angle from the axis(es) to which gravity was acting.
                  Defaults to z. Order of angles is preserved and counted
                  as the order for translation. (e.x. z:15 or z:15,x:30)
                  0º being on the top of the device, perpendicular to the
                  center of the screen, extending out of the top(z),
                  front(y), and right side(x).
                  http://developer.getpebble.com/guides/pebble-apps/sensors/accelerometer/
   --no-gravity : The effects of the Earth's gravity will not be factored out of the
                   acceleration data, nor will the device's frame of reference be
                   transformed to the Earth's inertial frame.
                   Use this if no transformation angles are known.
""".format(os.path.basename(__file__))

frmt="""
inputfile
---------

Time,X,Y,Z\\n
12345,54,343,232\\n
12365,32,232,556\\n
...

Optionally, you may omit the header if the --no-header
option is provided.
"""

all_args = 'ho:i:s:va:pet:ng'
long_args = ['help', 'verbose', 'inputfile=', 'outputfile=', 'sloppy=',
    'angle=', 'print', 'save', 'time=', 'format', 'no-header', 'graph',
    'no-gravity']

g = 9.81 # m/s**2
R = {
        'x': [['1', '0', '0'],
              ['0', 'cos(ang)', '-sin(ang)'],
              ['0', 'sin(ang)', 'cos(ang)']
            ],
        'y': [['cos(ang)', '0', 'sin(ang)'],
              ['0', '1', '0'],
              ['-sin(ang)', '0', 'cos(ang)']
            ],
        'z': [['cos(ang)', '-sin(ang)', '0'],
              ['sin(ang)', 'cos(ang)', '0'],
              ['0', '0', '1']
            ]
    }


def quit(msg, err=True):
    """ Abruptly quit. """
    print(msg)
    code = 2 if err else 0
    sys.exit(code)


def save(data, filename, header=''):
    """ Saves the given data to a file. """
    with open(filename, 'w') as f:
        f.write('{}\n{}'.format(header,str(data)))


def trapezoid(x_0, t_0, x_1, t_1):
    dt = abs(t_1 - t_0)
    return 0.5 * dt * (x_1 + x_0)


def integrate(dI_x):
    """ Given a matrix of time dependent values,
    integrate them based on the trapezoid rule.
    """
    I_x = []
    for i in xrange(len(dI_x) - 1):
        t_0, ax_0, ay_0, az_0 = dI_x[i]
        t_1, ax_1, ay_1, az_1 = dI_x[i+1]

        dv_x = trapezoid(ax_0, t_0, ax_1, t_1)
        dv_y = trapezoid(ay_0, t_0, ay_1, t_1)
        dv_z = trapezoid(az_0, t_0, az_1, t_1)

        I_x.append((t_0, dv_x, dv_y, dv_z))
    return I_x


def transformation_matrix(angles):
    """ Given an set of transformation angles,
    generate a series of transformation matrices
    """
    R_t = []
    for axis, ang in angles:
        r_t = []
        for row in R[axis]:
            r_t_i = []
            for el in row:
                r_t_i.append(eval(el))
            r_t.append(r_t_i)
        R_t.append(r_t)
    return R_t


def transpose(v, R):
    """ Given a 3d vector and a list of transformation
    matrices, transpose the given vector using those matrices.
    """
    V = [0, 0, 0]
    x, y, z = v
    for r_x in R:
        V_i = []
        for a, b, c in r_x:
            p_i = a*x + b*y + c*z
            V_i.append(p_i)
        V[0] += V_i[0]
        V[1] += V_i[1]
        V[2] += V_i[2]
    return V


def main(args, kwargs):
    verbose = False
    print_results = False
    save_data = False
    angles = [('z', 180)]
    no_header = False
    graph = False
    gravity = True

    try:
        opts, args = getopt.getopt(sys.argv[1:], all_args, long_args)
    except getopt.GetoptError as err:
        quit('{}\n{}'.format(str(err), usage))

    try:
        input_filename = args[0]
    except IndexError:
        input_filename = None
    output_filename = None

    for o, a in opts:
        if o in ('-h', '--help'):
            print(usage)
            sys.exit(0)
        if o in ('--format'):
            print(frmt)
            sys.exit(0)
        elif o in ('-o', '--outputfile'):
            output_filename = a
        elif o in ('-i', '--inputfile'):
            input_filename = a
        elif o in ('-n', '--no-header'):
            no_header = True
        elif o in ('-s', '--sloppy'):
            sloppy = True if a.lower() in ['true', 'yes', '1'] else False
        elif o in ('-v', '--verbose'):
            verbose = True
        elif o in ('--no-gravity'):
            gravity = False
            print('Ignoring gravity')
        elif o in ('-g', '--graph'):
            graph = True
        elif o in ('-p', '--print'):
            print_results = True
        elif o in ('-e', '--save'):
            save_data = True
        elif o in ('-a', '--angle'):
            axis_angles = [k for k in a.split(',')]
            for val in axis_angles:
                angles.append(val.split(':')[:2])
            try:
                angles = [(ax.lower(), float(deg)) for ax, deg in angles]
            except ValueError:
                quit('Angle must be a valid float. See --help for more information.')
            invalids = [True for ang, deg in angles if ang not in ['x', 'y', 'z'] or ang == '']
            if len(invalids) > 0: quit('Axis must be either "x, y, z". See --help.')
        else:
            quit("Undefined option. See --help for more information.")

    if input_filename is None:
        quit('No input file supplied. See --help for more information.')
    with open(input_filename) as f:
        data = []
        if not no_header:
            first_line = f.readline()
        for line in f:
            t, x, y, z = line.split(',')
            data.append((int(t), int(x), int(y), int(z)))
    sanitized_data = sanitize(data)

    # Remove the Gravity vector from the data using the provided axis
    # and angle.
    def dim(n):
        """ Dimensionalize the given mG value to metric units. """
        return float(n) / 1000.0 * g

    if verbose: print('Dimensionalizing...')
    A_t_d = []
    for t, x, y, z in sanitized_data:
        a_x, a_y, a_z = dim(x), dim(y), dim(z)
        A_t_d.append((t, a_x, a_y, a_z))

    A_t = []
    if gravity:
        r = [0, 0, 9.81]
        # Given a frame in which gravity acts along the Z axis, and given
        # transformation angles, transform the device frame to the inertial frame.
        if verbose: print('Generating transformation matrix...')
        R_t = transformation_matrix(angles)
        if save_data:
            contents = ''
            for r_t in R_t:
                contents += '{}\n\n'.format('\n'.join([','.join([str(el) for el \
                        in row]) for row in r_t]))
            save(contents, 'Transformation_Matrices.csv')

        if verbose: print('Transposing...')
        for a_t in A_t_d:
            t, x, y, z = a_t
            x_i, y_i, z_i = transpose((x, y, z), R_t)
            A_t.append((t, x_i, y_i, z_i))
        if save_data:
            save('\n'.join([','.join([str(t), str(x), str(y), str(z)])\
                    for t, x, y, z in A_t]),
                    'Transposed_Sensor_Data.csv',
                    header='Time(s),X(m/s^2),Y(m/s^2),Z(m/s^2)')
    else:
        A_t = A_t_d

    if verbose: print('Integrating...')
    V_t = integrate(A_t)
    if save_data:
        save('\n'.join([','.join([str(t), str(x), str(y), str(z)])\
                for t, x, y, z in V_t]),
                'Integrated_Velocity_Data.csv',
                header='Time(s),X(m/s),Y(m/s),Z(m/s)')

    S_t = integrate(V_t)
    if save_data:
        save('\n'.join([','.join([str(t), str(x), str(y), str(z)])\
                for t, x, y, z in V_t]),
                'Integrated_Position_Data.csv',
                header='Time(s),X(m),Y(m),Z(m)')

    if verbose: print('Transforming...')
    data_sans_time = [(x, y, z) for t, x, y, z in S_t]
    transformed_data = fourier_transform(data_sans_time)

    # The data coming out of the fourier transform is
    # for cycles/millisecond. We need this in cycles/sec.
    for data in transformed_data:
        data *= 1000.0

    if verbose: print('Filling buckets...')
    sorted_data = categorize(transformed_data)
    buckets = {}
    for name, data in sorted_data.iteritems():
        buckets[name] = {'Frequencies': data}

    # Calculations
    raw_freq = []
    for bucket, data in buckets.iteritems():
        freq = conjugate_filter(data['Frequencies'])
        raw_freq = [(f.real, f.imag) for f in freq]
        data['Max Freq. (Hz)'] = '' if len(freq) == 0 else max(freq).real
        data['Avg. Freq. (Hz)'] = '' if len(freq) == 0 else sum(freq).real / float(len(freq))
        data['Total Movement (m)'] = sum([1.0 / f for f in freq]).real
        data['Total Energy / k (Jm/N))'] = sum([0.5 * f.imag for f in freq])
        # TODO: Add more

        data['Frequencies'] = ['{}λ:{}A'.format(f.real, f.imag) for f in freq]

    # Output
    if output_filename is None or print_results:
        fields = []
        print('\nResults\n============')
        for bucket, data in sorted(buckets.iteritems()):
            fields = sorted(data.keys()) if len(fields) == 0 else fields
            print('\n{}\n------------'.format(bucket))
            for field in fields:
                out_str = ', '.join([str(el) for el in data[field]]) \
                        if type(data[field]) is list else str(data[field])
                print('{}: {}'.format(field, out_str))

    if output_filename is not None:
        contents = 'Bucket Name, Field, Value(s)\n'
        fields = []
        for bucket, data in sorted(buckets.iteritems()):
            fields = sorted(data.keys()) if len(fields) == 0 else fields
            data_str = ''
            for field in fields:
                data_str += ',{},{}\n'.format(field, ', '.join([str(el) \
                        for el in data[field]]) if type(data[field]) \
                        is list else str(data[field]))
            contents += '{}\n{}'.format(bucket, data_str)
        save(contents, output_filename)

    if verbose: print('Making graphs...')
    # Convert to graphable data
    freq, amp = [], []
    for f, a in raw_freq:
        freq.append(f)
        amp.append(a)
    dist, tim = [], []
    for t, x, y, z in S_t:
        dist.append((x, y, z))
        tim.append(t)

    # Subplots
    fig = plt.figure()
    # Frequency vs. Amplitude
    plt.subplot(2, 1, 1)
    plt.plot(freq, amp)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Frequency vs. Amplitude')
    plt.grid(True)
    # Distance vs. Time
    plt.subplot(2, 1, 2)
    plt.plot(tim, dist)
    plt.xlabel('Time (s)')
    plt.ylabel('Distance (m)')
    plt.title('Distance vs. Time')
    plt.grid(True)
    plt.legend(['x', 'y', 'z'], loc='lower right', fontsize='x-small')

    # Layout
    plt.subplots_adjust(hspace=0.4)

    if save_data: plt.savefig('Freq_vs_Amp_and_x_vs_t.png')
    if graph: plt.show()
    print('Done!')


if __name__ == '__main__':
    timeit(main, sys.argv[1:])
