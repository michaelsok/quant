from dataclasses import dataclass

import numpy as np


@dataclass
class Forward:
    q: float
    maturity: int

    def backward(self, prices, discount):
        return (prices[1:] * self.q + prices[:-1] * (1 - self.q)) * discount

    def get_prices_from(self, value, discount):
        prices = np.zeros((self.maturity + 1, self.maturity + 1))
        prices[:, -1] = value

        for step in range(self.maturity - 1, -1, -1):
            discount_ = discount
            if isinstance(discount, np.ndarray):
                discount_ = discount[:step+1, step]
            prices[:step+1, step] = self.backward(prices[:step+2, step+1], discount_)
        return prices


@dataclass
class BondForward(Forward):    
    def get_prices_from(self, lattice, discount):
        prices = lattice.values[:self.maturity+1, :self.maturity+1].copy()
        for step in range(lattice.periods - 1, -1, -1):
            discount_ = discount
            if isinstance(discount, np.ndarray):
                discount_ = discount[:step+1, step]
            prices[:step+1, step] = self.backward(prices[:step+2, step+1], discount_)
        return prices

    def get_prices_from(self, value, discount, zcb):
        
        return super().get_prices_from(value, discount)
