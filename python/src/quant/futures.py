from dataclasses import dataclass


@dataclass
class Futures:
    q: float

    def backward(self, prices):
        return prices[1:] * self.q + prices[:-1] * (1 - self.q)

    def get_prices_from(self, lattice):
        prices = lattice.values.copy()
        for step in range(lattice.periods - 1, -1, -1):
            prices[:step+1, step] = self.backward(prices[:step+2, step+1])
        return prices
