import numpy as np

from .weights import x_weighted


def ssr(a, x, y, yl):

    xw, _ = x_weighted(x, a[2], a[3])

    error = y - a[0] - a[1] * xw
    if yl is not None:
        error -= np.dot(yl, a[4:])

    return np.dot(error, error)
