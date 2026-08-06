import numpy as np

from quant.futures import Futures
from quant.black_scholes import BlackScholes
from quant.lattices.binomial import BinomialLattice


def test_futures():
    expected_prices = np.array([
        [100.50125209, 92.46718837, 85.07536719, 78.27444773],
        [0., 108.86985493, 100.16680563, 92.15947754],
        [0., 0., 117.93529998, 108.50755958],
        [0., 0., 0., 127.75561233]
    ])

    price = 100
    T = .5
    volatility = .2
    periods = 3
    r = .02
    div = .01

    black_scholes = BlackScholes(T=T, sigma=volatility, periods=periods, r=r, div=div)

    u, d, q = black_scholes.u, black_scholes.d, black_scholes.q

    lattice = BinomialLattice(base=price, periods=periods, up=u)
    futures = Futures(q=q)
    prices = futures.get_prices_from(lattice)

    np.testing.assert_allclose(prices, expected_prices, atol=1e-8)
