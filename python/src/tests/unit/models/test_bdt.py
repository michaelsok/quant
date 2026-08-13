import numpy as np

from quant.models.bdt import BlackDermanToy
from quant.swaption import Swaption


def test_bdt():
    expected_a = np.array([
        7.29999643, 7.92110414, 9.02117693, 9.43570858,
        12.13024853, 11.71923728, 12.85018206,
        12.56599101, 12.91852594, 15.19503948,
        14.53647872, 15.63621893, 15.15403114,
        13.44778152
    ]) / 100

    periods = 14
    rates = np.array([
        7.3, 7.62, 8.1, 8.45, 9.2, 9.64, 10.12, 10.45,
        10.75, 11.22, 11.55, 11.92, 12.2, 12.32
    ]) / 100
    a = np.ones_like(rates) * .05
    b = .005
    q = .5

    bdt = BlackDermanToy(q=q, b=b, periods=periods)
    bdt.fit(rates, initialization=a, seed=42, tol=1e-10)

    np.testing.assert_allclose(bdt.a, expected_a, atol=1e-5)


def test_swaption_on_bdt_low_volatility():
    periods = 14
    rates = np.array([
        7.3, 7.62, 8.1, 8.45, 9.2, 9.64, 10.12, 10.45,
        10.75, 11.22, 11.55, 11.92, 12.2, 12.32
    ]) / 100
    a = np.ones_like(rates) * .05
    b = .005
    q = .5

    bdt = BlackDermanToy(q=q, b=b, periods=periods)
    bdt.fit(rates, initialization=a, seed=42, tol=1e-10)
    calibrated_rates = bdt.get_rates()

    swaption = Swaption(q=q, swap_expiration=9, expiration=2, r=0.1165, strike=0, option_name='european', call=True)
    prices = swaption.get_prices_from(calibrated_rates)
    np.testing.assert_allclose(prices[0, 0], 0.00133901, atol=1e-4)


def test_swaption_on_bdt_high_volatility():
    periods = 14
    rates = np.array([
        7.3, 7.62, 8.1, 8.45, 9.2, 9.64, 10.12, 10.45,
        10.75, 11.22, 11.55, 11.92, 12.2, 12.32
    ]) / 100
    a = np.ones_like(rates) * .05
    b = .01
    q = .5

    bdt = BlackDermanToy(q=q, b=b, periods=periods)
    bdt.fit(rates, initialization=a, seed=42, tol=1e-10)
    calibrated_rates = bdt.get_rates()

    swaption = Swaption(q=q, swap_expiration=9, expiration=2, r=0.1165, strike=0, option_name='european', call=True)
    prices = swaption.get_prices_from(calibrated_rates)
    np.testing.assert_allclose(prices[0, 0], 0.00196257, atol=1e-4)
