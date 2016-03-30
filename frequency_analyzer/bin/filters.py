""" Some wrappers for numpy filters.
Filter taken from:
http://stackoverflow.com/questions/23341566/python-high-pass-fir-filter-by-windowing#23345244
"""
from scipy.signal import firwin, lfilter


class Filter(object):
    """ A stateful object for applying a low or high pass filter to a
    given data set.
    """

    def __init__(self, length, sample_rate, cutoff, filter_type='low'):
        self.length = length
        self.sample_rate = sample_rate
        self.filter_type = filter_type

        if cutoff:
            self.fir = firwin(length, cutoff=cutoff/sample_rate, window="hann")
        else:
            raise ValueError('You must supply a cutoff freqency.')

    def __call__(self, data):
        if self.filter_type == 'low':
            return low_pass(data, self.fir)
        else:
            return high_pass(data, self.fir, self.length)


def low_pass(data, fir):
    vals = lfilter(fir, 1, data)
    return vals


def high_pass(data, fir, length):
    fir = -fir
    fir[length/2] += 1
    vals = lfilter(fir, 1, data)
    return vals
