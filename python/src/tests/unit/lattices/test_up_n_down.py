import numpy as np

from quant.lattices.up_n_down import UpNDownLattice


def test_binomial_lattice():
    expected_short_rate = np.array([
        [.06, 0.054, 0.0486, 0.04374, 0.039366, 0.03542940],
        [0., 0.075, 0.0675, 0.06075, 0.054675, 0.0492075],
        [0., 0., 0.09375, 0.084375, 0.0759375, 0.06834375],
        [0., 0., 0., 0.1171875, 0.10546875, 0.09492188],
        [0., 0., 0., 0., 0.14648438, 0.13183594],
        [0., 0., 0., 0., 0., 0.18310547]
    ])

    r_0_0 = 0.06
    u = 1.25
    d = 0.9
    q = 0.5
    periods = 5
    lattice = UpNDownLattice(base=r_0_0, periods=periods, up=u, down=d)

    np.testing.assert_allclose(lattice.values, expected_short_rate, atol=1e-8)
