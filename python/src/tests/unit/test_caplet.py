import numpy as np

from quant.caplet import Caplet
from quant.lattices.up_n_down import UpNDownLattice


def test_caplet():
    expected_prices = np.array([
        [0.0420, 0.0376, 0.0323, 0.0264, 0.0206, 0.0149],
        [0., 0.0515, 0.0471, 0.0412, 0.0346, 0.0278],
        [0., 0., 0.0637, 0.0592, 0.0528, 0.0453],
        [0., 0., 0., 0.0800, 0.0756, 0.0684],
        [0., 0., 0., 0., 0.1032, 0.0988],
        [0., 0., 0., 0., 0., 0.1379]
    ])

    r_0_0 = 0.06
    periods = 5
    expiration = 5
    u = 1.25
    d = .9
    q = .5
    r = .02

    rates = UpNDownLattice(base=r_0_0, periods=periods, up=u, down=d)

    caplet = Caplet(q=q, expiration=expiration, r=r)

    prices = caplet.get_prices_from(lattice=rates)

    np.testing.assert_allclose(prices, expected_prices, atol=1e-4)
