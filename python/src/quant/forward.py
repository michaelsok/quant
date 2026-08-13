from dataclasses import dataclass

import numpy as np

from .bonds.zero import ZeroCouponBond


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
    coupon: float
    maturation: int
    _zcb: ZeroCouponBond | None = None

    @property
    def zcb(self):
        if self._zcb is None:
            self._zcb = ZeroCouponBond(coupon=self.coupon, maturation=self.maturation, q=self.q)
        return self._zcb

    def get_price_from(self, rates):
        prices = self.zcb.get_prices_from(rates)
        discount = 1 / (1 + rates.values)
        divider = super().get_prices_from(value=1, discount=discount)

        return prices[0, 0] / divider[0, 0]
