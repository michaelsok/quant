import numpy as np

from quant.swap import Swap
from quant.lattices.up_n_down import UpNDownLattice


def test_swap():
    expected_prices = np.array([
        [0.0990, 0.0496, 0.0137, -0.0085, -0.0174, -0.0141],
        [0., 0.1403, 0.0829, 0.0400, 0.0122, -0.0008],
        [0., 0., 0.1686, 0.1021, 0.0512, 0.0172],
        [0., 0., 0., 0.1793, 0.1014, 0.0410],
        [0., 0., 0., 0., 0.1648, 0.0723],
        [0., 0., 0., 0., 0., 0.1125]
    ])

    r_0_0 = 0.06
    periods = 5
    expiration = 5
    u = 1.25
    d = .9
    q = .5
    r = .05

    rates = UpNDownLattice(base=r_0_0, periods=periods, up=u, down=d)

    swap = Swap(q=q, expiration=expiration, r=r)

    prices = swap.get_prices_from(rates=rates)

    np.testing.assert_allclose(prices, expected_prices, atol=1e-4)
