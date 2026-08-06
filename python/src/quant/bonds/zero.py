from dataclasses import dataclass

import numpy as np


@dataclass
class ZeroCouponBond:
    coupon: float
    maturation: int
    q: float

    @property
    def shape(self):
        return (self.maturation + 1, self.maturation + 1)

    def backward(self, prices, discount):
        return (prices[1:] * self.q + prices[:-1] * (1 - self.q)) * discount

    def get_prices_from(self, lattice):
        prices = np.zeros(self.shape)
        prices[:, -1] = self.coupon
        rates = 1 + lattice.values
        for step in range(self.maturation - 1, -1, -1):
            discount = 1 / rates[:step+1, step]
            prices[:step+1, step] = self.backward(prices[:step+2, step+1], discount=discount)
        return prices
