import numpy as np


def ssr(a, x, y, yl, weight_method):
    """
    Calculate the sum of the squared residuals of the MiDAS equations.  Parameters are arranged
    a_h = a[0]
    b_h = a[1]
    theta is a[2:n]
    y lag params are a[n:]

    Args:
        a:
        x:
        y:
        yl:

    Returns:

    """
    xw, _ = weight_method.x_weighted(x, a[2:4])

    error = y - a[0] - a[1] * xw
    if yl is not None:
        error -= np.dot(yl, a[4:])

    return error


def jacobian(a, x, y, yl, weight_method):

    jwx = jacobian_wx(x, a[2:4], weight_method)

    xw, _ = weight_method.x_weighted(x, a[2:4])

    if yl is None:
        jac_e = np.concatenate([np.ones((len(xw), 1)), xw.reshape((len(xw), 1)), (a[1] * jwx)], axis=1)
    else:
        jac_e = np.concatenate([np.ones((len(xw), 1)), xw.reshape((len(xw), 1)), (a[1] * jwx), yl], axis=1)

    return -1.0 * jac_e


def jacobian_wx(x, params, weight_method):
    eps = 1e-6

    jt = []
    for i, p in enumerate(params):
        dp = np.concatenate([params[0:i], [p + eps / 2], params[i + 1:]])
        dm = np.concatenate([params[0:i], [p - eps / 2], params[i + 1:]])
        jtp, _ = weight_method.x_weighted(x, dp)
        jtm, _ = weight_method.x_weighted(x, dm)
        jt.append((jtp - jtm) / eps)

    return np.column_stack(jt)
