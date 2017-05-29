import os
import pytest

import pandas as pd
import numpy as np


@pytest.fixture()
def gdp_data(request):
    df = pd.read_csv(os.path.join(os.path.dirname(request.module.__file__), 'data', 'gdp.csv'),
                     parse_dates=['DATE'])

    df['gdp'] = (np.log(df.VALUE) - np.log(df.VALUE.shift(1))) * 100.

    return df.set_index('DATE')


@pytest.fixture()
def farmpay_data(request):
    df = pd.read_csv(os.path.join(os.path.dirname(request.module.__file__), 'data', 'farmpay.csv'),
                     parse_dates=['DATE'])

    df['farmpay'] = (np.log(df.VALUE) - np.log(df.VALUE.shift(1))) * 100.

    return df.set_index('DATE')
