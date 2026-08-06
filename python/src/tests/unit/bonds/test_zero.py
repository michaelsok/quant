import numpy as np

from quant.lattices.up_n_down import UpNDownLattice
from quant.bonds.zero import ZeroCouponBond


def test_binomial_lattice():
    expected_zcb_prices = np.array([
        [77.21774033, 84.43360847, 90.63619172, 95.80930117, 100.],
        [0., 79.26800103, 87.34985493, 94.27292010, 100.],
        [0., 0., 83.07634728, 92.21902017, 100.],
        [0., 0., 0., 89.51048951, 100.],
        [0., 0., 0., 0., 100.]
    ])

    r_0_0 = 0.06
    u = 1.25
    d = 0.9
    q = 0.5
    periods = 5
    lattice = UpNDownLattice(base=r_0_0, periods=periods, up=u, down=d)

    zcb = ZeroCouponBond(coupon=100, maturation=4, q=.5)
    prices = zcb.get_prices_from(lattice)

    np.testing.assert_allclose(prices, expected_zcb_prices, atol=1e-8)
