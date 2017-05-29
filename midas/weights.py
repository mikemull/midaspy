import numpy as np


def beta_weights_es(n, theta1, theta2):
    """ Evenly-spaced beta weights
    """
    eps = np.spacing(1)
    u = np.linspace(eps, 1.0 - eps, n)

    beta_vals = u ** (theta1 - 1) * (1 - u) ** (theta2 - 1)

    return beta_vals / sum(beta_vals)


def x_weighted(x, theta1, theta2):
    w = beta_weights_es(x.shape[1], theta1, theta2)

    return np.dot(x, w), np.tile(w.T, (x.shape[1], 1))
