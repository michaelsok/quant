from dataclasses import dataclass

import numpy as np


@dataclass
class Elementary:
    q: float
    expiration: int | None = None

    def forward(self, prices, discount):
        results = np.zeros(prices.shape[0] + 1)
        results[:-1] = (prices * (1 - self.q)) * discount
        results[1:] += (prices * self.q) * discount
        return results

    def get_prices_from(self, rates):
        size = rates.periods + 1
        if self.expiration:
            size = self.expiration

        prices = np.zeros(shape=(size + 1, size + 1))
        prices[:size, :size] = rates.values[:size, :size].copy()
        discount = 1 / (1 + prices)
        prices[0, 0] = 1

        for step in range(1, size + 1):
            discount_ = discount[:step, step - 1]
            prices[:step+1, step] = self.forward(prices[:step, step - 1], discount_)
        return prices
