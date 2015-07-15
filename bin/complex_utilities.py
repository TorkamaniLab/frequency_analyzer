""" A library of simple utilities for handling numpy complex numbers.

author: Brian Schrader
since: 2015-07-14
"""

import numpy as np


def conjugate_filter(numbers, favor='positive'):
    """ Given a list of complex numbers, return a
    filtered list containing no conjugates.

    :param favor: Given 2 complex conjugates, favor
    the 'positive' or 'negative' conjugate.
    """
    results = []
    for number in numbers:
        # Remove complex conj.
        n_type = 'positive' if number.imag > 0 else 'negative'
        conj = number.conjugate()
        present = complex_search(number, results) \
                or complex_search(conj, results)
        if not present:
            if n_type == favor:
                results.append(number)
            else:
                results.append(conj)
    return results


def complex_search(needle, haystack):
    """ Returns true if the needle is found in the haystack. """
    result = [True for hay in haystack if needle.real == hay.real \
            and needle.imag == hay.imag]
    return True if len(result) > 0 else False

