import numpy as np

from quant.elementary import Elementary
from quant.lattices.up_n_down import UpNDownLattice


def test_elementary():
    expected_prices = np.array([
        [1.0000, 0.4717, 0.2238, 0.1067, 0.0511, 0.0246, 0.0119],
        [0., 0.4717, 0.4432, 0.3143, 0.1992, 0.1190, 0.0686],
        [0., 0., 0.2194, 0.3079, 0.2901, 0.2293, 0.1640],
        [0., 0., 0., 0.1003, 0.1868, 0.2193, 0.2075],
        [0., 0., 0., 0., 0.0449, 0.1041, 0.1461],
        [0., 0., 0., 0., 0., 0.0196, 0.0543],
        [0., 0., 0., 0., 0., 0., 0.0083],
    ])

    r_0_0 = 0.06
    periods = 5
    u = 1.25
    d = .9
    q = .5

    rates = UpNDownLattice(base=r_0_0, periods=periods, up=u, down=d)

    elementary = Elementary(q=q)

    prices = elementary.get_prices_from(rates=rates)

    np.testing.assert_allclose(prices, expected_prices, atol=1e-4)
