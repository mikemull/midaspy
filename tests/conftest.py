import os
import pytest

import pandas as pd
import numpy as np


@pytest.fixture()
def gdp_data(request):
    df = pd.read_csv(os.path.join(os.path.dirname(request.module.__file__), 'data', 'gdp.csv'),
                     parse_dates=['DATE'])

    df['gdp'] = (np.log(df.GDP) - np.log(df.GDP.shift(1))) * 100.

    return df.set_index('DATE')


@pytest.fixture()
def pay_data(request):
    df = pd.read_csv(os.path.join(os.path.dirname(request.module.__file__), 'data', 'pay.csv'),
                     parse_dates=['DATE'])

    df['pay'] = (np.log(df.PAY) - np.log(df.PAY.shift(1))) * 100.

    return df.set_index('DATE')
