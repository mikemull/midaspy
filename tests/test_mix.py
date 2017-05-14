import pytest
import datetime
import pandas as pd

from midas import mix


@pytest.fixture()
def lf_data():
    df = pd.DataFrame({'date': ['2009-04-01', '2009-07-01', '2009-10-01', '2010-01-01', '2010-04-01'],
                       'val': [1.0, 2.0, 3.0, 4.0, 5.0]})
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    return df


@pytest.fixture()
def hf_data():
    df = pd.DataFrame({'date': ['2009-01-01', '2009-02-01', '2009-03-01', '2009-04-01', '2009-05-01',
                                '2009-06-01', '2009-07-01', '2009-08-01', '2009-09-01', '2009-10-01',
                                '2009-11-01', '2009-12-01', '2010-01-01', '2010-02-01', '2010-03-01',
                                '2010-04-01'],
                       'val': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]})
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    return df


def test_mix(lf_data, hf_data):
    y, yl, x, yf, ylf, xf = mix.mix_freq(lf_data, hf_data, 3, 1, 1,
                                         start_date=datetime.datetime(2009, 7, 1),
                                         end_date=datetime.datetime(2010, 1, 1))

    assert all(x.loc['2009-07-01'].values == [0.6, 0.5, 0.4])
    assert all(x.loc['2010-01-01'].values == [1.2, 1.1, 1.0])
    assert yl.loc['2009-07-01'].values[0] == 1.0
