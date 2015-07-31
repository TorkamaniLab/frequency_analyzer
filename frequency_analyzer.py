#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import print_function
import sys, getopt, os
from itertools import izip

import matplotlib.pyplot as plt
from math import sqrt
from scipy.interpolate import splrep, splev
from numpy import absolute, append, mean, ndarray, arange
from pywt import wavedec

from bin.sanitizer import sanitize
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
-d --downsample : The frequency to downsample the data to. Defaults to 28Hz
-l --levels     : The number of levels desired in the DWT process.
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

all_args = 'hvi:o:s:a:pet:fngrb:k:d:l:'
long_args = ['help', 'verbose', 'inputfile=', 'outputfile=', 'sloppy=',
    'angle=', 'print', 'save', 'time=', 'format', 'no-header', 'graph',
    'no-gravity', 'bins=', 'sig-factor=', 'downsample=', 'levels=']

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


def serialize(data_to_serialize):
    """ Serializes the given dictionary. """
    fields = []
    contents = ''
    for name, data in data_to_serialize:
        fields = sorted(data.keys()) if len(fields) == 0 else fields
        data_str = ''
        for field in fields:
            data_str += ',{},{}\n'.format(field, ', '.join([str(el) \
                    for el in data[field]]) \
                    if type(data[field]) is list \
                        or type(data[field]) is ndarray \
                        or type(data[field]) is tuple \
                        else str(data[field]))
        contents += '{}\n{}'.format(name, data_str)
    return contents


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
    downsample = 28.0
    sig_factor = 0.9
    levels = 5

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
        elif o in ('-l', '--levels'):
            levels = int(a)
        elif o in ('-d', '--downsample'):
            downsample = float(a)
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
    sample_rate = 1.0 / sanitized_data[1][0] - sanitized_data[0][0]

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


    if verbose: print('Interpolating...')
    t_new = arange(S_t[0][0], S_t[-1][0], 1.0/downsample)

    S_t_i = [[t] for t in t_new]
    for axis in [1, 2, 3]:
        t, f_t = [], []
        # Populate the sequence to be interpolated.
        for val in S_t:
            t.append(val[0])
            f_t.append(val[axis])
        # Interpolate
        tck = splrep(t, f_t, s=0)
        f_t_new = splev(t_new, tck, der=0)
        [s_t_i.append(f_t_i) for f_t_i, s_t_i in izip(f_t, S_t_i)]

    if verbose: print('Filling buckets...')
    buckets = []
    for axis in [0, 1, 2]:
        f_t  = []
        for s_t_i in S_t_i:
            f_t.append(s_t_i[axis+1])
        bucket = wavedec(f_t, 'db8', level=levels)
        buckets.append(bucket)

    if verbose: print('Calculating frequency bins...')
    bins = [(downsample, downsample/2.0),]
    for i in range(levels-1):
        _, prev_min = bins[i]
        bins.append((prev_min, prev_min/2.0))
    _, last_min = bins[-1]
    bins.append((last_min, 0.0))
    bins = list(reversed(bins))
    bin_names = ['{}-{}Hz'.format(min_, max_) for max_, min_ in bins]

    if verbose: print('Using bins: {}'.format(', '.join(bin_names)))
    final_buckets = {}
    for index, axis in enumerate(['x', 'y', 'z']):
        for name, max_min in izip(bin_names, bins):
            if name not in final_buckets.keys(): final_buckets[name] = {}
            final_buckets[name]['max_min'] =  max_min
        for name, bucket in izip(bin_names, enumerate(buckets[index])):
            i, values_per_axis = bucket
            if axis not in final_buckets[name].keys(): final_buckets[name][axis] = []
            final_buckets[name][axis] = values_per_axis
            # Add the scales.
            if 'scale' not in final_buckets[name].keys():
                final_buckets[name]['scale'] = arange(0, len(S_t_i),
                        float(len(S_t_i))/float(len(values_per_axis)))


    # Calculations
    if verbose: print('Picking out the cool stuff...')
    for name, data in final_buckets.iteritems():
        for axis in ['x', 'y', 'z']:
            vals = data[axis]
            data['max_amp'] = max(vals)
            data['avg_amp'] = float(sum(vals)) / float(len(vals))
            field = 'sig_thresh_{}'.format(axis)
            sig_threshold = sig_factor * data['max_amp']
            data[field] = sig_threshold
            field = 'sig_freq_{}'.format(axis)
            data[field] = [freq if freq > sig_threshold else 0 for freq in vals]


    # Output
    if output_filename is None or print_results:
        header = '\nResults\n============\nBucket Name, Field, Value(s)\n'
        contents = serialize(sorted(list(final_buckets.iteritems()),
                cmp=lambda a,b: int(a[1]['max_min'][0] - b[1]['max_min'][0])))
        print(header, contents)
    if output_filename is not None:
        header = 'Bucket Name, Field, Value(s)\n'
        contents = serialize(sorted(list(final_buckets.iteritems()),
                cmp=lambda a,b: int(a[1]['max_min'][0] - b[1]['max_min'][0])))
        save(contents, output_filename, header=header)


    if verbose: print('Making graphs...')
    freqs = []
    sig_freqs = []
    for name, data in sorted(list(final_buckets.iteritems()),
            cmp=lambda a,b: int(a[1]['max_min'][0] - b[1]['max_min'][0])):
        new_data = []
        for x, y, z in izip(data['x'], data['y'], data['z']):
            new_data.append((x, y, z))
        freqs.append((name, data['scale'], new_data))

        new_sig_data = []
        for x, y, z in izip(data['sig_freq_x'], data['sig_freq_y'], data['sig_freq_z']):
            new_sig_data.append((x, y, z))
        sig_freqs.append((name, data['scale'], new_sig_data))

    tim, dist = [], []
    for t, x, y, z in S_t_i:
        tim.append(t)
        dist.append((x, y, z))

    try:
        # Main Frequency Plot

        num_plots = len(freqs) + 1
        fig = plt.figure(figsize=(12, 10))

        # Distance vs. Time
        a1 = fig.add_subplot(num_plots, 1, 1)
        a1.plot(tim, dist)
        a1.set_xlabel('Time (s)')
        a1.set_ylabel('Distance (m)')
        a1.set_title('Distance vs. Time')
        a1.grid(True)
        a1.legend(['x', 'y', 'z'], loc='lower right', fontsize='x-small')
        #a1.set_xticklabels(a1.get_xticklabels(), fontsize=10)

        named = False
        ax = []
        for i, freq in enumerate(freqs):
            name, scale, data = freq
            if len(ax) == 0:
                a = fig.add_subplot(num_plots, 1, i+2)
            else:
                a = fig.add_subplot(num_plots, 1, i+2, sharex=ax[0])
            a.plot(scale, data)
            if not named:
                a.set_title('Amplitude vs. Time by Frequency Bin')
                named = True
            a.set_xticklabels(a.get_xticklabels(), visible=False)
            a.grid(True)
            a.set_ylabel(name)
            a.legend(['x', 'y', 'z'], loc='lower right', fontsize='x-small')
            ax.append(a)
        #ax[-1].set_xticklabels(ax[-1].get_xticklabels(), fontsize=10, visible=True)
        if save_data: fig.savefig('Freq_vs_Amp_and_x_vs_t.png')


        # Significant Frequency Plot
        fig2 = plt.figure(figsize=(12, 10))

        # Distance v. Time (again)
        a2 = fig2.add_subplot(num_plots, 1, 1)
        a2.plot(tim, dist)
        a2.set_xlabel('Time (s)')
        a2.set_ylabel('Distance (m)')
        a2.set_title('Distance vs. Time')
        a2.grid(True)
        a2.legend(['x', 'y', 'z'], loc='lower right', fontsize='x-small')
        #a1.set_xticklabels(a1.get_xticklabels(), fontsize=10)

        named = False
        ax2 = []
        for i, sig_freq in enumerate(sig_freqs):
            name, scale, data = sig_freq
            if len(ax2) == 0:
                a2 = fig2.add_subplot(num_plots, 1, i+2)
            else:
                a2 = fig2.add_subplot(num_plots, 1, i+2, sharex=ax2[0])
            a2.plot(scale, data)
            if not named:
                a2.set_title('Amplitude vs. Time by Frequency Bin')
                named = True
            a2.set_xticklabels(a2.get_xticklabels(), visible=False)
            a2.grid(True)
            a2.set_ylabel(name)
            a2.legend(['x', 'y', 'z'], loc='lower right', fontsize='x-small')
            ax2.append(a2)
        #ax[-1].set_xticklabels(ax[-1].get_xticklabels(), fontsize=10, visible=True)
        if save_data: fig2.savefig('Sig_Freq_vs_Amp_and_x_vs_t.png')
        if graph: plt.show()
    except Exception as e:
        print(e)
        print('Displaying and generating graphs isn\'t supported.')
    if verbose: print('Done!')
    return True if verbose else False


if __name__ == '__main__':
    timeit(main, sys.argv[1:])
