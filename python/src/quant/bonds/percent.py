from dataclasses import dataclass

import numpy as np

from .zero import ZeroCouponBond


@dataclass
class PercentCouponBond(ZeroCouponBond):
    percent: float

    def get_prices_from(self, lattice):
        bearing = self.percent * self.coupon
        prices = np.zeros(self.shape)
        prices[:, -1] = self.coupon + bearing
        rates = 1 + lattice.values
        for step in range(self.maturation - 1, -1, -1):
            discount = 1 / rates[:step+1, step]
            prices[:step+1, step] = bearing + self.backward(prices[:step+2, step+1], discount=discount)
        return prices
