from dataclasses import dataclass

import numpy as np

from quant.options.base import Option

class European(Option):
    def get_prices_from(self, lattice, expiration=None, discount=None):
        if discount is None:
            discount = self.discount

        if expiration is None:
            expiration = lattice.periods

        prices = lattice.values[:expiration+1, :expiration+1].copy()
        prices[:, -1] = np.maximum(self.call_put * (prices[:, -1] - self.strike), 0)
        for step in range(expiration - 1, -1, -1):
            discount_ = discount
            if isinstance(discount, np.ndarray):
                discount_ = discount[:step+1, step]
            prices[:step+1, step] = self.backward(prices[:step+2, step+1], discount_)
        return prices


@dataclass
class EuropeanCall(European):
    call: bool = True


@dataclass
class EuropeanPut(European):
    call: bool = False
