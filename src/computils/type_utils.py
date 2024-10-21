""" This module implements helping functions for the introduced types. """

import numpy as np
from .globals import ScalarOrArray


def size(x: ScalarOrArray) -> int:
    """ Returns the number of elements in x.

    :param x:
    :return:
    """

    if isinstance(x, np.ndarray):
        return x.size
    else:
        return 1
