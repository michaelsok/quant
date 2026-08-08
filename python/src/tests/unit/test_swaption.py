import numpy as np

from quant.swaption import Swaption
from quant.lattices.up_n_down import UpNDownLattice


def test_swaption():
    expected_prices = np.array([
        [0.0620, 0.0406, 0.0191, 0.0000],
        [0., 0.0908, 0.0665, 0.0400],
        [0., 0., 0.1286, 0.1021],
        [0., 0., 0., 0.1793]
    ])

    r_0_0 = 0.06
    periods = 5
    swap_expiration = 5
    expiration = 3
    u = 1.25
    d = .9
    q = .5
    r = .05

    rates = UpNDownLattice(base=r_0_0, periods=periods, up=u, down=d)

    swap = Swaption(q=q, swap_expiration=swap_expiration, expiration=expiration, r=r, strike=0)

    prices = swap.get_prices_from(rates=rates)

    np.testing.assert_allclose(prices, expected_prices, atol=1e-4)
