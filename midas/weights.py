import numpy as np


def beta_weights_es(n, theta1, theta2):
    """ Evenly-spaced beta weights
    """
    eps = np.spacing(1)
    u = np.linspace(eps, 1.0 - eps, n)

    beta_vals = u ** (theta1 - 1) * (1 - u) ** (theta2 - 1)

    return beta_vals / sum(beta_vals)


def exp_almon_weights(lag, k1, k2):
    """
    Exponential Almon weights

    Returns:
        array: Array of weights

    """
    ilag = np.arange(1, lag + 1)
    z = np.exp((k1 * ilag) + (k2 * ilag) ** 2)
    return z / sum(z)


def x_weighted(x, theta1, theta2):
    """
    Weight the matrix of regressors according to the specified weighting method
    Args:
        x:
        theta1:
        theta2:

    Returns:

    """
    w = beta_weights_es(x.shape[1], theta1, theta2)

    return np.dot(x, w), np.tile(w.T, (x.shape[1], 1))
