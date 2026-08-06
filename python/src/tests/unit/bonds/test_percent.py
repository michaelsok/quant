import numpy as np

from quant.lattices.up_n_down import UpNDownLattice
from quant.bonds.percent import PercentCouponBond


def test_binomial_lattice():
    expected_pcb_prices = np.array([
        [124.1371, 126.1409, 126.2709, 124.5687, 121.1626, 116.2361, 110.],
        [0., 115.8298, 118.5542, 119.2747, 117.9973, 114.8410, 110.],
        [0., 0., 108.9798, 112.4884, 113.8289, 112.9631, 110.],
        [0., 0., 0., 104.03, 108.4429, 110.4638, 110.],
        [0., 0., 0., 0., 101.6554, 107.1872, 110.],
        [0., 0., 0., 0., 0., 102.9757, 110.],
        [0., 0., 0., 0., 0., 0., 110.],
    ])

    r_0_0 = 0.06
    u = 1.25
    d = 0.9
    q = 0.5
    periods = 5
    lattice = UpNDownLattice(base=r_0_0, periods=periods, up=u, down=d)

    pcb = PercentCouponBond(coupon=100, maturation=6, q=.5, percent=0.1)
    prices = pcb.get_prices_from(lattice)

    np.testing.assert_allclose(prices, expected_pcb_prices, atol=1e-4)
