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
    y, yl, x, yf, ylf, xf = mix.mix_freq(lf_data.val, hf_data.val, 3, 1, 1,
                                         start_date=datetime.datetime(2009, 7, 1),
                                         end_date=datetime.datetime(2010, 1, 1))

    assert all(x.loc['2009-07-01'].values == [0.6, 0.5, 0.4])
    assert all(x.loc['2010-01-01'].values == [1.2, 1.1, 1.0])
    assert yl.loc['2009-07-01'].values[0] == 1.0


def test_mix_no_ylag(lf_data, hf_data):
    y, yl, x, yf, ylf, xf = mix.mix_freq(lf_data.val, hf_data.val, 3, 0, 1,
                                         start_date=datetime.datetime(2009, 7, 1),
                                         end_date=datetime.datetime(2010, 1, 1))

    assert all(x.loc['2009-07-01'].values == [0.6, 0.5, 0.4])
    assert all(x.loc['2010-01-01'].values == [1.2, 1.1, 1.0])
    assert yl is None


def test_mix_lag_string(lf_data, hf_data):
    y, yl, x, yf, ylf, xf = mix.mix_freq(lf_data.val, hf_data.val, "3M", 1, 1,
                                         start_date=datetime.datetime(2009, 7, 1),
                                         end_date=datetime.datetime(2010, 1, 1))

    assert all(x.loc['2009-07-01'].values == [0.6, 0.5, 0.4])
    assert all(x.loc['2010-01-01'].values == [1.2, 1.1, 1.0])
    assert yl.loc['2009-07-01'].values[0] == 1.0


def test_mix_no_start_end(lf_data, hf_data):
    y, yl, x, yf, ylf, xf = mix.mix_freq(lf_data.val, hf_data.val, "3M", 1, 1)

    assert all(x.loc['2009-07-01'].values == [0.6, 0.5, 0.4])
    assert all(x.loc['2010-01-01'].values == [1.2, 1.1, 1.0])
    assert yl.loc['2009-07-01'].values[0] == 1.0


def test_mix_lf_after_hf(lf_data, hf_data):
    """ Case where high-freq data ends before last low-freq date
    """
    y, yl, x, yf, ylf, xf = mix.mix_freq(lf_data.val, hf_data.iloc[0:-2].val, 3, 0, 1,
                                         start_date=datetime.datetime(2009, 7, 1),
                                         end_date=datetime.datetime(2010, 1, 1))

    assert all(x.loc['2009-07-01'].values == [0.6, 0.5, 0.4])
    assert all(x.loc['2010-01-01'].values == [1.2, 1.1, 1.0])
    assert yl is None


def test_mix_hf_starts_after_lf(lf_data, hf_data):
    """ Case where high-freq data ends before last low-freq date
    """
    y, yl, x, yf, ylf, xf = mix.mix_freq(lf_data.val, hf_data.iloc[4:].val, 3, 0, 1,
                                         start_date=datetime.datetime(2009, 4, 1),
                                         end_date=datetime.datetime(2010, 1, 1))

    assert y.index[0] == datetime.datetime(2009, 10, 1)
    assert all(x.loc['2009-10-01'].values == [0.9, 0.8, 0.7])


def test_mix_negative_h(lf_data, hf_data):
    y, yl, x, yf, ylf, xf = mix.mix_freq(lf_data.val, hf_data.val, 3, 1, -1,
                                         start_date=datetime.datetime(2009, 7, 1),
                                         end_date=datetime.datetime(2010, 1, 1))

    assert all(x.loc['2009-07-01'].values == [0.8, 0.7, 0.6])
    assert all(x.loc['2010-01-01'].values == [1.4, 1.3, 1.2])
    assert yl.loc['2009-07-01'].values[0] == 1.0


def test_mix_gdp(gdp_data, pay_data):

    y, yl, x, yf, ylf, xf = mix.mix_freq(gdp_data.gdp, pay_data.pay, 3, 1, 1,
                                         start_date=datetime.datetime(1985, 1, 1),
                                         end_date=datetime.datetime(2009, 1, 1))

    assert len(y) == 97

    assert all(x.loc['1985-01-01'].values == [pay_data.loc['1984-12-01'].pay,
                                              pay_data.loc['1984-11-01'].pay,
                                              pay_data.loc['1984-10-01'].pay])


def test_data_freq(lf_data, hf_data):

    assert mix.data_freq(lf_data)[0] == 'Q'
    assert mix.data_freq(hf_data)[0] == 'M'

    idx = pd.date_range(start='2012-03-31', periods=5, freq='Q-DEC')
    assert mix.data_freq(pd.Series(lf_data.val.values, index=idx)) == 'Q-DEC'


def test_data_freq_infer():
    dfm = pd.DataFrame(data={'a': [1, 2, 3, 4]},
                       index=[datetime.datetime(2016, 1, 31),
                              datetime.datetime(2016, 2, 29),
                              datetime.datetime(2016, 3, 31),
                              datetime.datetime(2016, 4, 30)])

    assert mix.data_freq(dfm.a) == 'M'

    dfb = pd.DataFrame(data={'a': [1, 2, 3, 4, 5, 6]},
                       index=[datetime.datetime(2017, 7, 10),
                              datetime.datetime(2017, 7, 11),
                              datetime.datetime(2017, 7, 12),
                              datetime.datetime(2017, 7, 13),
                              datetime.datetime(2017, 7, 14),
                              datetime.datetime(2017, 7, 17),
                              ])

    assert mix.data_freq(dfb.a)[0] == 'B'

    dfb2 = pd.DataFrame(data={'a': [1, 2, 3, 4, 5, 6]},
                        index=pd.date_range('2017-7-10', '2017-7-17', freq='B'))

    assert mix.data_freq(dfb2.a)[0] == 'B'


@pytest.mark.parametrize("lag_string, freq, expected", [
    ("3M", "D", 90),
    ("3M", "B", 66),
    ("3M", "M", 3),
    ("12Q", "M", 36),
    ("3m", "d", 90),
    ("3m", "b", 66),
    ("2y", "q", 8),
    ("2y", "a", 2),
])
def test_parse_lag_string(lag_string, freq, expected):
    assert mix.parse_lag_string(lag_string, freq) == expected
