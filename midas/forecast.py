import numpy as np
import pandas as pd

from .mix import mix_freq
from .regression import estimate, forecast


def rolling(y_in, x_in, start_date, end_date, xlag, ylag, horizon, window_size=60, forecast_horizon=1):
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


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
