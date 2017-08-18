import datetime

from midas.forecast import rolling


def test_rolling(gdp_data, pay_data):

    rmse, yh_df = rolling(gdp_data.gdp, pay_data.pay, datetime.datetime(1985, 1, 1), None,
                          "3m", 1, 1)

    assert 0.6 < rmse < 0.7
