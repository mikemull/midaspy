import numpy as np
import pandas as pd

from scipy.optimize import least_squares

from midas.weights import x_weighted
from midas.fit import ssr, jacobian


def estimate(y, yl, x):
    xw, w = x_weighted(x, 1., 5.)

    # First we do OLS to get initial parameters
    c = np.linalg.lstsq(np.concatenate([np.ones((len(xw), 1)), xw.reshape((len(xw), 1)), yl], axis=1), y)[0]

    print(c)

    f = lambda v: ssr(v, x.values, y.values, yl.values)
    jac = lambda v: jacobian(v, x.values, y.values, yl.values)

    opt_res = least_squares(f, np.concatenate([c[0:2], [1., 5.], c[2:]]), jac, xtol=1e-10, verbose=2)

    print(opt_res.x)
    print(opt_res.grad)

    return opt_res


def forecast(xfc, yfcl, res):

    a, b, theta1, theta2, l = res.x

    xw, w = x_weighted(xfc.values, theta1, theta2)

    yf = a + b * xw + l * yfcl.values[:, 0]

    return pd.DataFrame(yf, index=xfc.index)
