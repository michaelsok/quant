import numpy as np

from quant.bonds.zero import ZeroCouponBond
from quant.options.american import AmericanPut
from quant.lattices.binomial import BinomialLattice, Lattice
from quant.lattices.up_n_down import UpNDownLattice


def test_american_put():
    expected_prices = np.array([
        [3.82275086, 7.13338998, 12.65612717, 18.37021231],
        [0., 1.25849479, 2.86934548, 6.54205607],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]
    ])

    price = 100
    strike = 100
    periods = 3
    R = 1.01001

    u = 1.07
    d = 1 / u
    q = (R - d) / (u - d)

    lattice = BinomialLattice(base=price, periods=periods, up=u)
    put = AmericanPut(strike=strike, q=q, R=R)
    prices = put.get_prices_from(lattice)

    np.testing.assert_allclose(prices, expected_prices, atol=1e-8)



def test_american_put_on_zcb():
    expected_prices = np.array([
        [10.78225967, 3.56639153, 0., 0.],
        [0., 8.73199897, 0.65014507, 0.],
        [0., 0., 4.92365272, 0.],
        [0., 0., 0., 0.]
    ])

    r_0_0 = 0.06
    u = 1.25
    d = 0.9
    q = 0.5
    periods = 5
    lattice = UpNDownLattice(base=r_0_0, periods=periods, up=u, down=d)

    zcb = Lattice(base=None, periods=lattice.periods)
    zcb.values = ZeroCouponBond(coupon=100, maturation=4, q=.5).get_prices_from(lattice)
    
    call = AmericanPut(strike=88, q=q, R=1 + r_0_0)
    prices = call.get_prices_from(zcb, expiration=3, discount=1 / (1 + lattice.values))

    np.testing.assert_allclose(prices, expected_prices, atol=1e-8)
