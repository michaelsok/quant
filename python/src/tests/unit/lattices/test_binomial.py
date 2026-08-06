import numpy as np

from quant.lattices.binomial import BinomialLattice


def test_binomial_lattice():
    expected_stocks = np.array([
        [100., 93.45794393, 87.34387283, 81.62978769],
        [0., 107., 100., 93.45794393],
        [0., 0., 114.49, 107.],
        [0., 0., 0., 122.5043]
    ])

    price = 100
    periods = 3
    u = 1.07
    lattice = BinomialLattice(base=price, periods=periods, up=u)

    np.testing.assert_allclose(lattice.values, expected_stocks, atol=1e-8)
