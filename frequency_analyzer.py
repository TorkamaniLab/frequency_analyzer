#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import print_function
import sys, getopt, os

import matplotlib.pyplot as plt
from math import sqrt
from numpy.fft import fftn, fftfreq
from numpy import absolute, mean

from bin.sanitizer import sanitize
from bin.categorizer import categorize, default_buckets
from bin.timer import timeit
from bin.integrator import integrate
from bin.matrix import transformation_matrix, transform


usage = """
Description
-----------
A utility for isolating, grouping, and visualizing
frequencies found in raw accelerometer data.

author: Brian Schrader
since: 2015-07-22

Usage
-----
python {} [options] <inputfile>

Options
-------
-h --help       : Display this message.
-f --format     : Another help message detailing the format of
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
-r --no-gravity : The effects of the Earth's gravity will not be factored out of the
                  acceleration data, nor will the device's frame of reference be
                  transformed to the Earth's inertial frame.
                  Use this if no transformation angles are known.
-b --bins       : Explicitly specify the frequency bins for the resulting
                  analysis to be divided. All units are considered Hz.
                  (e.x. 1.2-4.0,4.2-34.4,0.1-3)
-k --sig-factor : When calculating a significance factor to isolate relevant
                  frequencies, what should be the cutoff? This value is relative
                  to the max amplitude of the discovered frequencies.
                  Max values are calculated per bin.
                  Values are from 0<x<1. The default is 0.9. (e.x. 0.85)
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

all_args = 'hvi:o:s:a:pet:fngrb:k:'
long_args = ['help', 'verbose', 'inputfile=', 'outputfile=', 'sloppy=',
    'angle=', 'print', 'save', 'time=', 'format', 'no-header', 'graph',
    'no-gravity', 'bins=', 'sig-factor=']

g = 9.81 # m/s**2


def quit(msg, err=True):
    """ Abruptly quit. """
    print(msg)
    code = 2 if err else 0
    sys.exit(code)


def save(data, filename, header=''):
    """ Saves the given data to a file. """
    with open(filename, 'w') as f:
        f.write('{}\n{}'.format(header,str(data)))


def main(args, kwargs):
    verbose = False
    print_results = False
    save_data = False
    angles = [('z', 180)]
    no_header = False
    graph = False
    gravity = True
    time_unit = 'milliseconds'
    sloppy = False
    bins = default_buckets
    sig_factor = 0.9

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
        if o in ('-f', '--format'):
            print(frmt)
            sys.exit(0)
        elif o in ('-o', '--outputfile'):
            output_filename = a
        elif o in ('-i', '--inputfile'):
            input_filename = a
        elif o in ('-n', '--no-header'):
            no_header = True
        elif o in ('-k', '--sig-factor'):
            sig_factor = float(a)
            if sig_factor < 0 or sig_factor >= 1:
                quit('Significance factor must be 0<x<1. See --help for more information.')
        elif o in ('-s', '--sloppy'):
            sloppy = True if a.lower() in ['true', 'yes', '1'] else False
        elif o in ('-v', '--verbose'):
            verbose = True
        elif o in ('-r', '--no-gravity'):
            gravity = False
            print('Ignoring gravity')
        elif o in ('-g', '--graph'):
            graph = True
        elif o in ('-p', '--print'):
            print_results = True
        elif o in ('-e', '--save'):
            save_data = True
        elif o in ('-t', '--time'):
            time_unit = a
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
        elif o in ('-b', '--bins'):
            try:
                raw_bins = [b.split('-') for b in a.split(',')]
                bins = [('{}-{}Hz'.format(a,b), float(a), float(b)) for a, b in raw_bins]
            except Exception as e:
                quit("Incorrect bin syntax. See --help for more information.")
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
    sanitized_data = sanitize(data, sloppy=sloppy, time_unit=time_unit)
    sample_rate = 1 / sanitized_data[1][0] - sanitized_data[0][0]

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
            x_i, y_i, z_i = transform((x, y, z), R_t)
            A_t.append((t, x_i, y_i, z_i))
        if save_data:
            save('\n'.join([','.join([str(t), str(x), str(y), str(z)])\
                    for t, x, y, z in A_t]),
                    'Transformd_Sensor_Data.csv',
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
    amplitudes = absolute(fftn(data_sans_time)) / len(data_sans_time)

    n = len(S_t)
    freq_axis = fftfreq(n, d=1/sample_rate)
    frequencies = []
    for i, amp  in enumerate(amplitudes[-n:]):
        x, y, z = amp
        frequencies.append((freq_axis[i], x, y, z))

    if save_data:
        save('\n'.join([','.join([str(f), str(x), str(y), str(z)]) \
                for f, x, y, z, in frequencies]),
                'Frequencies.csv',
                header='Frequency,X,Y,Z')

    if verbose: print('Using bins: {}'.format(', '.join(\
            [str(a) for a, _, __ in bins])))
    if verbose: print('Filling buckets...')
    sorted_data = categorize(frequencies, buckets=bins)
    buckets = {}
    for name, data in sorted_data.iteritems():
        buckets[name] = {'Frequencies': data}

    # Calculations

    # Isolate the significant frequencies from the signal.
    # Proposed solution: If an amplitude is above the mean amplitude.
    # Gather a set of sig_freqs for later (set is faster to 'in' than list).
    sig_freq = set()
    for name, data in buckets.iteritems():
        freqs = data['Frequencies']
        sig_amp = {
                'x': max([x for f, x, y, z in freqs]) * sig_factor,
                'y': max([y for f, x, y, z in freqs]) * sig_factor,
                'z': max([z for f, x, y, z in freqs]) * sig_factor
                }
        if verbose: print('Cutoff: {} X:{} Y:{} Z:{}'.format(
            name, sig_amp['x'], sig_amp['y'], sig_amp['z']))
        per_buck_sig_freq = []
        for f, x, y, z in freqs:
            if x > sig_amp['x']:
                per_buck_sig_freq.append((f, 'x', x))
            if y > sig_amp['y']:
                per_buck_sig_freq.append((f, 'y', y))
            if z > sig_amp['z']:
                per_buck_sig_freq.append((f, 'z', z))
        data['Significant Frequencies'] = per_buck_sig_freq
        sig_freq.update(set(per_buck_sig_freq))

    # Serialize
    # TODO: This code is getting out of hand.
    filtered_freq = []
    for bucket, data in sorted(buckets.iteritems()):
        filtered_freq.extend(data['Frequencies'])
        freq = data['Frequencies']
        just_freq = [f for f, x, y, z in freq]
        data['Max Freq. (Hz)'] = '' if len(just_freq) == 0 else max(just_freq)
        data['Avg. Freq. (Hz)'] = '' if len(just_freq) == 0 else sum(just_freq) / float(len(just_freq))
        data['Total Movement (m)'] = sum([1.0 / f for f in just_freq])
        data['Total Energy / k (Jm/N))'] = sum([0.5 * sum([x, y, z]) for f in freq])
        # TODO: Add more
        data['Significant Frequencies'] = ['F:{} {}:{}'.format(f, n.upper(), v) \
                for f, n, v in data['Significant Frequencies']]
        data['Frequencies'] = ['F:{} X:{} Y:{} Z:{}'.format(f, x, y, z) for f, x, y, z in freq]

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
    dist, tim = [], []
    for t, x, y, z in S_t:
        dist.append((x, y, z))
        tim.append(t)
    freqs, amps  = [], []
    sig_freqs, sig_amps = [], []
    for f, x, y, z in filtered_freq:
        freqs.append(f)
        amps.append((x, y, z))
        # Add sig_freqs
        if (f, 'x', x) in sig_freq:
            sig_amps.append((x, 0, 0))
            sig_freqs.append(f)
        if (f, 'y', y) in sig_freq:
            sig_amps.append((0, y, 0))
            sig_freqs.append(f)
        if (f, 'z', z) in sig_freq:
            sig_amps.append((0, 0, z))
            sig_freqs.append(f)
        else:
            sig_amps.append((0, 0, 0))
            sig_freqs.append(f)
    try:
        fig = plt.figure()

        # Frequency vs. Amplitude
        a1 = plt.subplot(3, 1, 1)
        plt.plot(freqs, amps)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (m)')
        plt.title('Frequency vs. Magnitude')
        plt.grid(True)
        plt.legend(['x', 'y', 'z'], loc='lower right', fontsize='x-small')
        # Significant Frequency vs. Amplitude
        a2 = plt.subplot(3, 1, 2, sharex=a1, sharey=a1)
        plt.plot(sig_freqs, sig_amps)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (m)')
        plt.title('Significant Frequency vs. Magnitude (Cutoff={})'.format(sig_factor))
        plt.grid(True)
        plt.legend(['x', 'y', 'z'], loc='lower right', fontsize='x-small')
        # Distance vs. Time
        plt.subplot(3, 1, 3)
        plt.plot(tim, dist)
        plt.xlabel('Time (s)')
        plt.ylabel('Distance (m)')
        plt.title('Distance vs. Time')
        plt.grid(True)
        plt.legend(['x', 'y', 'z'], loc='lower right', fontsize='x-small')

        # Layout
        plt.subplots_adjust(hspace=0.7)

        if save_data: plt.savefig('Freq_vs_Amp_and_x_vs_t.png')
        if graph: plt.show()
    except Exception as e:
        print(e)
        print('Displaying and generating graphs isn\'t supported.')
    if verbose: print('Done!')


if __name__ == '__main__':
    timeit(main, sys.argv[1:])
