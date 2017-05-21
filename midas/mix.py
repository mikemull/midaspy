import datetime
import pandas as pd
import numpy as np


def mix_freq(lf_data, hf_data, xlag, ylag, horizon, start_date=None, end_date=None):
    """
    Set up data for mixed-frequency regression

    Args:
        lf_data (Series): Low-frequency time series
        hf_data (Series): High-frequency time series
        xlag (int): Number of high frequency lags
        ylag (int): Number of low-frequency lags
        horizon (int):
        start_date (date): Date on which to start estimation
        end_date (date); Date on which to end estimation

    Returns:

    """
    if start_date is None:
        start_date = lf_data.index[ylag]
    if end_date is None:
        end_date = lf_data.index[xlag + horizon]

    forecast_start_date = lf_data.index[lf_data.index.get_loc(end_date) + 1]

    ylags = pd.concat([lf_data.shift(l) for l in range(1, ylag + 1)], axis=1)

    x_rows = []

    for lfdate in lf_data.loc[start_date:].index:
        start_hf = hf_data.index.get_loc(lfdate, method='bfill')  # @todo Find a more efficient way
        x_rows.append(hf_data.iloc[start_hf - horizon: start_hf - xlag - horizon: -1].values)

    x = pd.DataFrame(data=x_rows, index=lf_data.loc[start_date:].index)

    return (lf_data.loc[start_date:end_date],
            ylags.loc[start_date:end_date],
            x.loc[start_date:end_date],
            lf_data[forecast_start_date:],
            ylags[forecast_start_date:],
            x.loc[forecast_start_date:])


if __name__ == '__main__':

    lf_data = pd.read_csv('./tests/data/gdp.csv', parse_dates=['DATE'])
    lf_data.set_index('DATE', inplace=True)

    hf_data = pd.read_csv('./tests/data/farmpay.csv', parse_dates=['DATE'])
    hf_data.set_index('DATE', inplace=True)

    lf_g = np.log(1 + lf_data.pct_change()).dropna() * 100.
    hf_g = np.log(1 + hf_data.pct_change()).dropna() * 100.

    y, x = mix_freq(lf_g.VALUE, hf_g, 9, 1, 1,
                    start_date=datetime.datetime(1985, 1, 1),
                    end_date=datetime.datetime(2009,1,1))
