import numpy as np

from .weights import x_weighted


def ssr(a, x, y, yl):

    xw, _ = x_weighted(x, a[2], a[3])

    error = y - a[0] - a[1] * xw
    if yl is not None:
        error -= np.dot(yl, a[4:])

    return error


def jacobian(a, x, y, yl):
    jwx = jacobian_wx(x, a[2], a[3])

    xw, _ = x_weighted(x, a[2], a[3])

    if yl is None:
      jac_e = np.concatenate([np.ones((len(xw), 1)),  xw.reshape((len(xw), 1)), (a[1] * jwx)], axis=1)
    else:
      jac_e = np.concatenate([np.ones((len(xw), 1)),  xw.reshape((len(xw), 1)), (a[1] * jwx), yl], axis=1)

    return -1.0 * jac_e


def jacobian_wx(x, theta1, theta2):
    eps = 1e-6

    xt1p, w = x_weighted(x, theta1 + eps / 2, theta2)
    xt1m, w = x_weighted(x, theta1 - eps / 2, theta2)
    jt1 = (xt1p - xt1m) / eps

    xt2p, w = x_weighted(x, theta1, theta2 + eps / 2)
    xt2m, w = x_weighted(x, theta1, theta2 - eps / 2)
    jt2 = (xt2p - xt2m) / eps

    return np.column_stack([jt1, jt2])
