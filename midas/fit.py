import numpy as np

from .weights import x_weighted


def resid(a, x, y, yl):

    xw, _ = x_weighted(x, a[2], a[3])

    error = y - a[0] - a[1] * xw
    if yl is not None:
        error -= np.dot(yl, a[4:])

    return error


def ssr(a, x, y, yl):

    error = resid(a, x, y, yl)

    return np.dot(error, error)


def jacobian(a, x, y, yl):
    jwx = jacobian_wx(x, a[2], a[3])

    xw, _ = x_weighted(x, a[2], a[3])

    error = resid(a, x, y, yl)
    if yl is None:
      jac_e = np.concatenate([np.ones((len(xw), 1)),  xw.reshape((len(xw), 1)), (a[0] * jwx)], axis=1)
    else:
      jac_e = np.concatenate([np.ones((len(xw), 1)),  xw.reshape((len(xw), 1)), (a[0] * jwx), yl], axis=1)

    jac = np.zeros(jac_e.shape)
    for i in range(len(a)):
        jac[:, i] = -2 * jac_e[:, i] * error

    return np.sum(jac, axis=0).T


def jacobian_wx(x, theta1, theta2):
    eps = 1e-6

    xt1p, w = x_weighted(x, theta1 + eps / 2, theta2)
    xt1m, w = x_weighted(x, theta1 - eps / 2, theta2)
    jt1 = (xt1p - xt1m) / eps

    xt2p, w = x_weighted(x, theta1, theta2 + eps / 2)
    xt2m, w = x_weighted(x, theta1, theta2 - eps / 2)
    jt2 = (xt2p - xt2m) / eps

    return np.column_stack([jt1, jt2])
