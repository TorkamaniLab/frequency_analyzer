#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import print_function
from itertools import izip
import sys, getopt, json, time, os

from math import sqrt, ceil
from numpy import absolute, append, mean, ndarray, arange, std
from pywt import wavedec
from scipy.interpolate import splrep, splev
from scipy.linalg import svdvals
from scipy.signal import medfilt
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot

from bin.filters import Filter
from bin.matrix import transformation_matrix, transform
from bin.sanitizer import sanitize
from bin.timer import timeit


usage = """
Description
-----------
A utility for isolating, grouping, and visualizing
frequencies found in raw accelerometer data.

author: Brian Schrader
since: 2015-07-31

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
-d --downsample : The frequency to downsample the data to. Defaults to 25Hz
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
-m --svd        : Generate the single value decompositions of the given signal
                  matrix normalized by their std. This saves those values to a file.
   --filter []  : Filter the interpolated data with one of the following
                  filters:
                    - median
                    - low_pass
                    - high_pass
                  Note: For low/high pass filters, a cutoff must be supplied.
-c --cutoff []  : A cutoff frequency to use in filtering low and high pass filters (Hz).
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

all_args = 'hvi:o:s:a:pet:fngrb:k:d:l:c:'
long_args = ['help', 'verbose', 'inputfile=', 'outputfile=', 'sloppy=',
    'angle=', 'print', 'save', 'time=', 'format', 'no-header', 'graph',
    'no-gravity', 'bins=', 'sig-factor=', 'downsample=', 'levels=',
    'filter=', 'cutoff=']

g = 9.81 # m/s**2


def quit(msg, err=True):
    """ Abruptly quit. """
    print(msg)
    code = 2 if err else 0
    sys.exit(code)


def save(data, filename, header=''):
    """ Saves the given data to a file. """
    print(filename)
    with open(filename, 'w') as f:
        f.write('{}\n{}\n'.format(header,str(data)))


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


def get_data(filename, no_header=False):
    """ Given a filename, get the contents as a list of 4D tuples.
    :params no_header: True if the csv has no header formatting.
    :returns data: [(t, x, y, z), ...]
    """
    with open(filename) as f:
        data = []
        if not no_header:
            first_line = f.readline()
        for line in f:
            t, x, y, z = line.split(',')
            data.append((float(t), float(x), float(y), float(z)))
    return data


def do_interpolate(F_t, downsample=25.0):
    """ Given a sequence of (t, x, y, z) values at a given sample
    rate, interpolate values at a downsampled rate.

    :params F_t: A sequence of (t, x, y, z) values at a given rate.
    :params downsample: The Hz to be converted to (e.g. 25).

    :returns F_t_i: A new sequence of (t, x, y, z) values interpolated
    at the downsampled rate.
    """
    downsample = float(downsample)
    t_new = arange(F_t[0][0], F_t[-1][0], 1.0/downsample)

    # If the desired rate is an even division of
    # the current rate, then just strip out the rest.
    curr_sample_rate = 1.0 / F_t[1][0] - F_t[0][0]
    if curr_sample_rate % downsample == 0:
        k = curr_sample_rate / downsample
        F_t_i = [f_t for i, f_t in enumerate(F_t) if i % k == 0]
    else:
        F_t_i = [[t] for t in t_new]
        for axis in [1, 2, 3]:
            t, f_t = [], []
            # Populate the sequence to be interpolated.
            for val in F_t:
                t.append(val[0])
                f_t.append(val[axis])
            # Interpolate
            tck = splrep(t, f_t, s=0)
            f_t_new = splev(t_new, tck, der=0)
            [f_t_i_i.append(f_t_i) for f_t_i, f_t_i_i in izip(f_t, F_t_i)]
    return F_t_i


def get_frequencies(A_t, wavelet='db8', levels=5, sample_rate=25.0):
    """ Given a sequence of (t, s) points in a uniform
    distribution, extract the frequencies using a Discrete Wavelet
    Transform and the provided wavelet.

    :params A_t: A sequence of (t, s) points at a fixed
    sample rate.
    :params wavelet: The wavelet to use in the DWT.
    :params sample_rate: The sample rate of the data in Hz.

    :returns filled_buckets: A dict of groups whose name is the
    grouping of frequencies (e.g. '2.4-4.8Hz') and whose values
    are a dict containing a 'scale' of time, and x, y, and z
    amplutides at that given time.

    Example
    -------
    sample = [(0, 2), (0.1, 3), ...]
    buckets = get_frequency_buckets(sample, 10)
    buckets = {
        '2.5-0.0Hz': {
            max_min: (2.5, 0.0)
            scale: [1, 2, 3, 4, ...],
            s: [2, 0, -3, 4, ...],
            },
        ...
        }
    """
    data = [s for t, s in A_t]
    empty_bucket = wavedec(data, wavelet, level=levels)

    bins = [(sample_rate, sample_rate/2.0),]
    for i in range(levels-1):
        _, prev_min = bins[i]
        bins.append((prev_min, prev_min/2.0))
    _, last_min = bins[-1]
    bins.append((last_min, 0.0))
    bins = list(reversed(bins))
    bin_names = ['{}-{}Hz'.format(min_, max_) for max_, min_ in bins]

    filled_buckets = {}
    for name, max_min in izip(bin_names, bins):
        if name not in filled_buckets.keys(): filled_buckets[name] = {}
        filled_buckets[name]['max_min'] =  max_min
    for name, bucket in izip(bin_names, enumerate(empty_bucket)):
        i, values_per_axis = bucket
        if 's' not in filled_buckets[name].keys(): filled_buckets[name]['s'] = []
        filled_buckets[name]['s'] = values_per_axis
        # Add the scales.
        if 'scale' not in filled_buckets[name].keys():
            filled_buckets[name]['scale'] = arange(0, len(A_t),
                   float(len(A_t))/float(len(values_per_axis)))
    return filled_buckets


def do_calculations(buckets, sig_factor=0.9):
    """ Given a dictionary of values {group_name: {x: [], y:[], z[]},}
    do the required calculations.

    Calculations Performed
    ----------------------
    max_amp: Maximum amplitude per group.
    avg_amp: Average amplitude per group.
    sig_thresh_{axis}: A subset of values where:
    x_t > max_amp * sig_factor

    :returns buckets: A modified version of the original dict.
    All original values remain untouched.
    """

    for name, data in buckets.iteritems():
        for axis in ['s']:
            vals = data[axis]
            data['max_amp'] = max(vals)
            data['avg_amp'] = float(sum(abs(vals))) / float(len(vals))
            data['energy'] = sum(abs(vals))
            field = 'sig_thresh_{}'.format(axis)
            sig_threshold = sig_factor * data['max_amp']
            data[field] = sig_threshold
            field = 'sig_freq_{}'.format(axis)
            data[field] = [freq if freq > sig_threshold else 0 for freq in vals]
    return buckets


def root_mean_square(data):
    """ Combine the 3-D signal into a 1-D signal using
    x = sqrt(x**2 + y**2 + z**2)
    """
    rms = []
    for t, x, y, z in data:
        rms.append((t, sqrt(x**2 + y**2 + z**2)))
    return rms


def filter_with_filter(filter, data, sample_rate, cutoff=None):
    """ Apply the given filter to the data. Filters must be one of the
    following:
        - median
        - low_pass
        - high_pass
    """
    filter_fn = None
    if filter == 'median':
        filter_fn = lambda x: medfilt(x, 5)
    elif filter == 'low_pass':
        filter_fn = Filter(len(data), sample_rate, cutoff, 'low')
    elif filter == 'high_pass':
        filter_fn = Filter(len(data), sample_rate, cutoff, 'high')
    else:
        return data

    data_minus_time = []
    for t, s in data:
        data_minus_time.append(s)
    data_minus_time = filter_fn(data_minus_time)

    # Put it back
    new_data = []
    for a_t_i, temp in izip(data, data_minus_time):
        t = a_t_i[0]
        s = temp

        new_data.append((t, s))
    return new_data


def do_svd(buckets):
    """ Given a set of frequency bins, computes the singluar values
    of the feature vector matrix.

    :returns: A vector of singular values.
    """
    l = 0
    for _, data in buckets.iteritems():
        if len(data['scale']) > l: l = len(data['scale'])

    M = []
    l_t_new = xrange(l)
    for name, data in buckets.iteritems():
        _, min_ = data['max_min']
        rms = data['scale']
        l_t = xrange(len(rms))
        tck = splrep(l_t, rms, s=0)
        rms_new = splev(l_t_new, tck, der=0)
        M.append((min_, rms_new))

    M = [rms for _, rms in reversed(sorted(M))]
    svd = svdvals(M)
    m, s = mean(svd), std(svd)
    return [((x - m) / s) for x in svd]


def get_metadata(data):
    """ Given the original data set, extract useful data.

    Example
    -------
    - date/time
    - length of sample
    - sample id
    """
    start, end = time.localtime(data[0][0]/1000), time.localtime(data[-1][0]/1000)
    meta = {
            'total_time': data[-1][0] - data[0][0],
            'sample_rate': 1 / (data[1][0] - data[0][0]),
            'start_date': '%s-%s %s:%s' % start[1:5],
            'end_date': ' %s-%s %s:%s' % end[1:5]
        }

    return meta


def make_output(meta, data):
    """Given a data set and associated metadata, make a csv of
    the output.
    """
    contents, col_names = [], []
    cols = ['energy', 'avg_amp', 'max_amp']

    for bucket, water in sorted(data.iteritems()):
        contents.extend(water[val] for val in cols)
        col_names.extend('{}_{}'.format(bucket, col) for col in cols)

    col_names = ['date_time'] + col_names
    date = meta['start_date']
    contents = [date] + contents

    return col_names, contents


def main():
    """ Given an intial data set, and various other parameters, calculate
    the frequency spread of the data. Data provided should be in the form
    of acceleration in mG with a regular sample rate.
    For an explanation of the parameters, see usage.
    """
    svd = False
    input_filename = None
    verbose = False
    print_results = False
    save_data = False
    angles = [('z', 180)]
    no_header = False
    graph = False
    gravity = True
    time_unit = 'milliseconds'
    sloppy = False
    downsample = 25.0
    sig_factor = 0.9
    levels = 5
    filter = None
    cutoff = None

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
            quit(usage, err=False)
        if o in ('-f', '--format'):
            quit(fmt, err=False)
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
        elif o in ('--filter'):
            filter = a
        elif o in ('-t', '--time'):
            time_unit = a
        elif o in ('-m', '--svd'):
            svd = True
        elif o in ('-c', '--cutoff'):
            cutoff = float(a)
        elif o in ('-a', '--angle'):
            axis_angles = [k for k in a.split(',')]
            for val in axis_angles:
                angles.append(val.split(':')[:2])
            try:
                angles = [(ax.lower(), float(deg)) for ax, deg in angles]
            except ValueError:
                quit('Angle must be a valid float. See --help for more information.')
            invalid = any(True for a, _ in angles if a not in ['x', 'y',
                'z'] or a == '')
            if invalid: quit('Axis must be either "x, y, z". See --help.')
        else:
            quit("Undefined option. See --help for more information.")
    if input_filename is None:
        quit('No input file supplied. See --help for more information.')
    data = get_data(input_filename, no_header=no_header)

    if save_data:
        save_path = os.path.abspath('%s.work' % output_filename)
        try:
            os.mkdir(save_path)
        except OSError:
            pass

    sanitized_data = sanitize(data, sloppy=sloppy, time_unit=time_unit)
    sample_rate = 1.0 / sanitized_data[1][0] - sanitized_data[0][0]
    # Remove the Gravity vector from the data using the provided axis
    # and angle.
    def dim(n):
        """ Dimensionalize the given mG value to metric units. """
        return float(n) / 1000.0 * g

    if verbose: print('Cleaning the data...')
    A_t_d = []
    for t, x, y, z in sanitized_data:
        a_x, a_y, a_z = dim(x), dim(y), dim(z)
        A_t_d.append((t, a_x, a_y, a_z))

    A_t = []
    if gravity:
        r = [0, 0, 9.81]
        # Given a frame in which gravity acts along the Z axis, and given
        # transformation angles, transform the device frame to
        # the inertial frame.
        R_t = transformation_matrix(angles)
        if save_data:
            contents = ''
            for r_t in R_t:
                contents += ('{}\n\n'.format('\n'.join([','.join([str(el) for
                    el in row]) for row in r_t])))
            save(contents, '%s/Transformation_Matrices.csv' % save_path)

        for a_t in A_t_d:
            t, x, y, z = a_t
            x_i, y_i, z_i = transform((x, y, z), R_t)
            A_t.append((t, x_i, y_i, z_i))
        if save_data:
            save('\n'.join([','.join([str(t), str(x), str(y), str(z)])\
                    for t, x, y, z in A_t]),
                    '%s/Transformed_Sensor_Data.csv' % save_path,
                    header='Time(s),X(m/s^2),Y(m/s^2),Z(m/s^2)')
    else:
        A_t = A_t_d

    A_t_i = do_interpolate(A_t, downsample)

    combined = root_mean_square(A_t_i)

    filtered = filter_with_filter(filter, combined, downsample, cutoff)

    if verbose: print('Extracting Frequencies...')
    filled_buckets = get_frequencies(filtered, wavelet='db8',
            levels=levels, sample_rate=downsample)

    if verbose: print('Picking out the cool stuff...')
    do_calculations(filled_buckets, sig_factor=sig_factor)

    if verbose: print('Calculating SVD...')
    sigmas = do_svd(filled_buckets)
    if save_data: save('\n'.join(str(x) for x in sigmas),
            '%s/singular_values.csv' % save_path)

    # Output
    meta = get_metadata(data)
    header, contents = make_output(meta, filled_buckets)
    header = ','.join(header)
    contents = ','.join(str(x) for x in contents)

    if output_filename is None or print_results:
        print(header, contents)
    if output_filename is not None:
        save(contents, output_filename, header=header)

    # Serialize the data for graphing.
    if verbose: print('Making graphs...')
    freqs = []
    sig_freqs = []
    sorted_freqs = []
    for name, data in sorted(list(filled_buckets.iteritems()),
            cmp=lambda a,b: int(a[1]['max_min'][0] - b[1]['max_min'][0])):
        new_data = []
        for s in data['s']:
            new_data.append(s)
        freqs.append((name, data['scale'], new_data))

        new_sig_data = []
        for s in data['sig_freq_s']:
            new_sig_data.append(s)
        sig_freqs.append((name, data['scale'], new_sig_data))

        sorted_freqs.append((data['max_min'][0], name, data['s']))

    tim, dist = [], []
    for t, x, y, z in A_t_i:
        tim.append(t)
        dist.append((x, y, z))

    # Make the plots
    try:
        # Main Frequency Plot

        num_plots = len(freqs) + 1
        fig = pyplot.figure(figsize=(12, 10))

        # Distance vs. Time
        a1 = fig.add_subplot(num_plots, 1, 1)
        a1.plot(tim, dist)
        a1.set_xlabel('Time (s)')
        a1.set_ylabel('Accel (m/s^2)')
        a1.set_title('Acceleration vs. Time')
        a1.grid(True)
        a1.legend(['x', 'y', 'z'], loc='lower right', fontsize='x-small')

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
            a.legend(['s'], loc='lower right', fontsize='x-small')
            ax.append(a)
        if save_data: fig.savefig('%s/Freq_vs_Amp_and_x_vs_t.png' % save_path)

        # Significant Frequency Plot
        fig2 = pyplot.figure(figsize=(12, 10))

        # Distance v. Time (again)
        a2 = fig2.add_subplot(num_plots, 1, 1)
        a2.plot(tim, dist)
        a2.set_xlabel('Time (s)')
        a2.set_ylabel('Accel (m/s^2)')
        a2.set_title('Acceleration vs. Time')
        a2.grid(True)
        a2.legend(['x', 'y', 'z'], loc='lower right', fontsize='x-small')

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
                a2.set_title('Significant Amplitudes vs. Time by Frequency Bin @{}'\
                        .format(sig_factor))
                named = True
            a2.set_xticklabels(a2.get_xticklabels(), visible=False)
            a2.grid(True)
            a2.set_ylabel(name)
            a2.legend(['s'], loc='lower right', fontsize='x-small')
            ax2.append(a2)
        if save_data: fig2.savefig('%s/Sig_Freq_vs_Amp_and_x_vs_t.png' %
                save_path)

        # Histogram

        fig3 = pyplot.figure(figsize=(12, 10))
        l = len(sorted_freqs)
        for i, val in enumerate(sorted_freqs):
            _, name, data =  val
            s = data
            a3 = fig3.add_subplot(l, 1, i+1)
            a3.hist([s], bins=30)
            a3.set_ylabel(name)
        if save_data: fig3.savefig('%s/Histogram.png' % save_path)

        if graph: pyplot.show()
    except Exception as e:
        print(e)
        print('Displaying and generating graphs isn\'t supported.')
    if verbose: print('Done!')
    return True if verbose else False


if __name__ == '__main__':
    timeit(main)
