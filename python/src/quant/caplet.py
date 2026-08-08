from dataclasses import dataclass

import numpy as np


@dataclass
class Caplet:
    q: float
    expiration: int
    r: float

    @property
    def size(self):
        return self.expiration + 1

    def backward(self, prices, discount):
        return (prices[1:] * self.q + prices[:-1] * (1 - self.q)) * discount

    def get_prices_from(self, lattice):
        prices = lattice.values[:self.size, :self.size].copy()
        discount = 1 / (1 + prices)
        prices[:, -1] = (prices[:, -1] - self.r) * discount[:, -1]

        for step in range(self.expiration - 1, -1, -1):
            discount_ = discount[:step+1, step]
            prices[:step+1, step] = self.backward(prices[:step+2, step+1], discount_)
        return prices
