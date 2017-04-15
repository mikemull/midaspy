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
    
    #First HF index >= first y date
    start_hf = [i for i in range(len(hf_data)) if hf_data.index[i] >= lf_data.index[0]][0]

    x = np.array([hf_data.iloc[start_hf + i * xlag: start_hf + (i+1)*xlag, 0].values 
                  for i in range(len(y))])
    
    x = np.concatenate([ylags.loc[start_date:end_date], x], axis=1)
    # Need consecutive xlag-sized chunks of HF data
    
    return lf_data.loc[start_date:end_date], x


if __name__ == '__main__':

    lf_data = pd.read_csv('./tests/data/gdp.csv', parse_dates=['DATE'])
    lf_data.set_index('DATE', inplace=True)

    hf_data = pd.read_csv('./tests/data/farmpay.csv', parse_dates=['DATE'])
    hf_data.set_index('DATE',inplace=True)
    y, x = mix_freq(lf_data, hf_data, 4, 2, 1, start_date=datetime.date(1985,1,1), end_date=datetime.date(2009,1,1))

