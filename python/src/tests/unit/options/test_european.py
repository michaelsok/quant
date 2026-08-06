import numpy as np

from quant.black_scholes import BlackScholes
from quant.futures import Futures
from quant.bonds.zero import ZeroCouponBond
from quant.options.european import EuropeanCall, EuropeanPut
from quant.lattices.binomial import BinomialLattice, Lattice
from quant.lattices.up_n_down import UpNDownLattice


def test_european_call():
    expected_prices = np.array([
        [6.57596511, 2.12898278, 0., 0.],
        [0., 10.230831, 3.86042478, 0.],
        [0., 0., 15.4810793, 7.],
        [0., 0., 0., 22.5043]
    ])

    price = 100
    strike = 100
    periods = 3
    R = 1.01001

    u = 1.07
    d = 1 / u
    q = (R - d) / (u - d)

    lattice = BinomialLattice(base=price, periods=periods, up=u)
    call = EuropeanCall(strike=strike, q=q, R=R)
    prices = call.get_prices_from(lattice)

    np.testing.assert_allclose(prices, expected_prices, atol=1e-8)


def test_european_put_on_futures():
    expected_prices = np.array([
        [5.2124, 7.2381, 9.7711, 12.7841, 16.1680, 19.7380, 23.2879, 26.6887, 29.9468, 33.0683, 36.0593],
        [0., 3.1515, 4.6625, 6.7098, 9.3495, 12.5500, 16.1477, 19.8574, 23.4109, 26.8152, 30.0767],
        [0., 0., 1.6127, 2.5784, 4.0240, 6.0957, 8.8963, 12.3869, 16.2636, 19.9771, 23.5343],
        [0., 0., 0., 0.6285, 1.1054, 1.9136, 3.2441, 5.3448, 8.4475, 12.4991, 16.3798],
        [0., 0., 0., 0., 0.1421, 0.2812, 0.5568, 1.1024, 2.1827, 4.3215, 8.5559],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    ])

    
    price = 100
    strike = 100
    T = .5
    volatility = .2
    periods = 10
    r = .02
    div = .01

    black_scholes = BlackScholes(T=T, sigma=volatility, periods=periods, r=r, div=div)

    u, q = black_scholes.u, black_scholes.q

    lattice = BinomialLattice(base=price, periods=periods, up=u)

    futures = Lattice(base=None, periods=lattice.periods)
    futures.values = Futures(q=q).get_prices_from(lattice=lattice)
    
    put = EuropeanPut(strike=strike, q=q, R=(1 + r))
    prices = put.get_prices_from(futures, discount=black_scholes.discount)

    np.testing.assert_allclose(prices, expected_prices, atol=1e-4)


def test_european_call_on_zcb():
    expected_prices = np.array([
        [2.96947445, 4.73721378, 6.63619172],
        [0., 1.55807206, 3.34985493],
        [0., 0., 0.]
    ])

    r_0_0 = 0.06
    u = 1.25
    d = 0.9
    q = 0.5
    periods = 5
    lattice = UpNDownLattice(base=r_0_0, periods=periods, up=u, down=d)

    zcb = Lattice(base=None, periods=lattice.periods)
    zcb.values = ZeroCouponBond(coupon=100, maturation=4, q=.5).get_prices_from(lattice)
    
    call = EuropeanCall(strike=84, q=q, R=1 + r_0_0)
    prices = call.get_prices_from(zcb, expiration=2, discount=1 / (1 + lattice.values))

    np.testing.assert_allclose(prices, expected_prices, atol=1e-8)
