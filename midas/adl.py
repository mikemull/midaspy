import datetime
import numpy as np
import pandas as pd

from scipy.optimize import least_squares

from midas.weights import polynomial_weights

from .mix import mix_freq
from .fit import ssr, jacobian


def estimate(y, yl, x, poly='beta'):
    """
    Fit MIDAS model

    Args:
       y (Series): Low-frequency data
       yl (DataFrame): Lags of low-frequency data
       x (DataFrame): High-frequency regressors

    Returns:
        scipy.optimize.OptimizeResult
    """

    weight_method = polynomial_weights(poly)

    xw, w = weight_method.x_weighted(x, weight_method.init_params())

    # First we do OLS to get initial parameters
    c = np.linalg.lstsq(np.concatenate([np.ones((len(xw), 1)), xw.reshape((len(xw), 1)), yl], axis=1), y)[0]

    f = lambda v: ssr(v, x.values, y.values, yl.values, weight_method)
    jac = lambda v: jacobian(v, x.values, y.values, yl.values, weight_method)

    opt_res = least_squares(f,
                            np.concatenate([c[0:2], weight_method.init_params(), c[2:]]),
                            jac,
                            xtol=1e-9,
                            ftol=1e-9,
                            max_nfev=5000,
                            verbose=0)

    return opt_res


def forecast(xfc, yfcl, res, poly='beta'):
    """
    Use the results of MIDAS regression to forecast new periods
    """
    weight_method = polynomial_weights(poly)

    a, b, theta1, theta2, l = res.x

    xw, w = weight_method.x_weighted(xfc.values, [theta1, theta2])

    yf = a + b * xw + l * yfcl.values[:, 0]

    return pd.DataFrame(yf, index=xfc.index, columns=['yfh'])


def midas_adl(y_in, x_in, start_date, end_date, xlag, ylag, horizon, forecast_horizon=1, poly='beta', method='fixed'):
    methods = {'fixed': fixed_window,
               'rolling': rolling,
               'recursive': recursive}

    return methods[method](y_in, x_in, start_date, end_date, xlag, ylag, horizon, forecast_horizon, poly)


def fixed_window(y_in, x_in, start_date, end_date, xlag, ylag, horizon, forecast_horizon=1, poly='beta'):

    y, yl, x, yf, ylf, xf = mix_freq(y_in, x_in, xlag, ylag, horizon,
                                     start_date=start_date,
                                     end_date=end_date)

    res = estimate(y, yl, x, poly=poly)

    fc = forecast(xf, ylf, res, poly=poly)

    return (rmse(fc.yfh, yf),
            pd.DataFrame({'preds': fc.yfh, 'targets': yf}, index=yf.index))


def rolling(y_in, x_in, start_date, end_date, xlag, ylag, horizon, forecast_horizon=1, poly='beta'):
    """
    Make a series of forecasts using a fixed-size "rolling window" to fit the
    model

    Args:
        y_in (Series): Dependent variable
        x_in (Series): Independent variables
        start_date: Initial start date for window
        window_size: Number of periods in window
        max_horizon: Maximum horizon to forecast

    Returns:
        rmse (float64), predicted and target values (DataFrame)

    """
    preds = []
    targets = []
    dt_index = []
    start_loc = y_in.index.get_loc(start_date)
    window_size = 60
    if end_date is not None:
        end_loc = y_in.index.get_loc(end_date)
        window_size = end_loc - start_loc

    while start_loc + window_size < (len(y_in.index) - forecast_horizon):
        y, yl, x, yf, ylf, xf = mix_freq(y_in, x_in, xlag, ylag, horizon,
                                         start_date=y_in.index[start_loc],
                                         end_date=y_in.index[start_loc + window_size])
        if len(xf) - forecast_horizon <= 0:
            break

        res = estimate(y, yl, x)

        fc = forecast(xf, ylf, res)

        preds.append(fc.iloc[forecast_horizon - 1].values[0])
        targets.append(yf.iloc[forecast_horizon - 1])
        dt_index.append(yf.index[forecast_horizon - 1])

        start_loc += 1

    preds = np.array(preds)
    targets = np.array(targets)

    return (rmse(preds, targets),
            pd.DataFrame({'preds': preds, 'targets': targets}, index=pd.DatetimeIndex(dt_index)))


def recursive(y_in, x_in, start_date, end_date, xlag, ylag, horizon, forecast_horizon=1, poly='beta'):
    preds = []
    targets = []
    dt_index = []

    forecast_start_loc = y_in.index.get_loc(end_date)

    model_end_dates = y_in.index[forecast_start_loc:-forecast_horizon]

    for estimate_end in model_end_dates:
        y, yl, x, yf, ylf, xf = mix_freq(y_in, x_in, xlag, ylag, horizon,
                                         start_date=start_date,
                                         end_date=estimate_end)
        if len(xf) - forecast_horizon <= 0:
            break

        res = estimate(y, yl, x)

        fc = forecast(xf, ylf, res)

        preds.append(fc.iloc[forecast_horizon - 1].values[0])
        targets.append(yf.iloc[forecast_horizon - 1])
        dt_index.append(yf.index[forecast_horizon - 1])

    preds = np.array(preds)
    targets = np.array(targets)

    return (rmse(preds, targets),
            pd.DataFrame({'preds': preds, 'targets': targets}, index=pd.DatetimeIndex(dt_index)))


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
