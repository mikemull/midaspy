import datetime
import pandas as pd
import numpy as np


def mix_freq(lf_data, hf_data, xlag, ylag, horizon, start_date=None, end_date=None):
    if start_date is None:
        start_date = lf_data.index[ylag]
    if end_date is None:
        end_date = lf_data.index[xlag + horizon]
        
    ylags = pd.concat([lf_data.shift(l) for l in range(1,ylag + 1)], axis=1)
    y = lf_data.loc[start_date:end_date]

    x_rows = []

    for i, lfdate in enumerate(y.index):
        start_hf = hf_data.index.get_loc(lfdate, method='bfill')  # @todo Find a more efficient way
        x_rows.append(hf_data.iloc[start_hf - horizon: start_hf - xlag - horizon: -1, 0].values)

    x = np.array(x_rows)
    x = np.concatenate([ylags.loc[start_date:end_date], x], axis=1)

    return y, x


if __name__ == '__main__':

    lf_data = pd.read_csv('./tests/data/gdp.csv', parse_dates=['DATE'])
    lf_data.set_index('DATE', inplace=True)

    hf_data = pd.read_csv('./tests/data/farmpay.csv', parse_dates=['DATE'])
    hf_data.set_index('DATE', inplace=True)

    lf_g = np.log(1 + lf_data.pct_change()).dropna() * 100.
    hf_g = np.log(1 + hf_data.pct_change()).dropna() * 100.

    y, x = mix_freq(lf_g.VALUE, hf_g, 9, 1, 1, start_date=datetime.datetime(1985,1,1), end_date=datetime.datetime(2009,1,1))
