""" Convinience functions for timing method execution.

author: Brian Schrader
since: 2015-07-13
"""

from __future__ import print_function
import time


def timeit(f, *args, **kwargs):
    """ Executes the given function with the given arguments.
    :returns: the time it took to execute the given function.

    Also prints the time.
    """
    start = time.time()
    flag = f(*args, **kwargs)
    end = time.time()
    if flag:
        print('That took {}s'.format(end - start))
    return end - start
