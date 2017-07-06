import numpy as np


def polynomial_weights(poly):
    poly_class = {
        'beta': BetaWeights(1., 5.),
        'beta_nz': BetaWeights(1., 5.),
        'expalmon': ExpAlmonWeights(-1., 0.)
    }

    return poly_class[poly]


class WeightMethod(object):
    def __init__(self):
        pass

    def weights(self):
        pass


class BetaWeights(WeightMethod):
    def __init__(self, theta1, theta2, theta3=None):
        self.theta1 = theta1
        self.theta2 = theta2
        self.theta3 = theta3

    def weights(self, nlags):
        """ Evenly-spaced beta weights
        """
        eps = np.spacing(1)
        u = np.linspace(eps, 1.0 - eps, nlags)

        beta_vals = u ** (self.theta1 - 1) * (1 - u) ** (self.theta2 - 1)

        return beta_vals / sum(beta_vals)

    def x_weighted(self, x, params):
        self.theta1, self.theta2 = params

        w = self.weights(x.shape[1])

        return np.dot(x, w), np.tile(w.T, (x.shape[1], 1))

    @staticmethod
    def init_params():
        return np.array([1., 5.])


class ExpAlmonWeights(WeightMethod):
    def __init__(self, theta1, theta2):
        self.theta1 = theta1
        self.theta2 = theta2

    def weights(self, nlags):
        """
        Exponential Almon weights

        Returns:
            array: Array of weights

        """
        ilag = np.arange(1, nlags + 1)
        z = np.exp((self.theta1 * ilag) + (self.theta2 * ilag) ** 2)
        return z / sum(z)

    def x_weighted(self, x, params):
        self.theta1, self.theta2 = params

        w = self.weights(x.shape[1])

        return np.dot(x, w), np.tile(w.T, (x.shape[1], 1))

    @staticmethod
    def init_params():
        return np.array([-1., 0.])


def beta_weights_es(n, theta1, theta2, theta3=None):
    """ Evenly-spaced beta weights
    """
    eps = np.spacing(1)
    u = np.linspace(eps, 1.0 - eps, n)

    beta_vals = u ** (theta1 - 1) * (1 - u) ** (theta2 - 1)

    beta_vals = beta_vals / sum(beta_vals)

    if theta3 is not None:
        w = beta_vals + theta3
        return w / sum(w)
    else:
        return beta_vals


def exp_almon_weights(lag, k1, k2):
    """
    Exponential Almon weights

    Returns:
        array: Array of weights

    """
    ilag = np.arange(1, lag + 1)
    z = np.exp((k1 * ilag) + (k2 * ilag) ** 2)
    return z / sum(z)


def x_weighted(x, params, poly='beta'):
    """
    Weight the matrix of regressors according to the specified weighting method
    Args:
        x:
        theta1:
        theta2:

    Returns:

    """
    if poly == 'beta':
        theta1, theta2 = params
        w = beta_weights_es(x.shape[1], theta1, theta2)
    elif poly == 'beta_nz':
        try:
            theta1, theta2, theta3 = params
        except ValueError:
            theta2, theta3 = params
            theta1 = 1.
        w = beta_weights_es(x.shape[1], theta1, theta2, theta3)
    elif poly == 'exp':
        k1, k2 = params
        w = exp_almon_weights(x.shape[1], k1, k2)
    else:
        pass

    return np.dot(x, w), np.tile(w.T, (x.shape[1], 1))
