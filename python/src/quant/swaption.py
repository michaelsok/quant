from dataclasses import dataclass

import numpy as np

from .swap import Swap
from .options.european import European
from .options.american import American
from quant.lattices.base import Lattice

OPTIONS = {
    'european': European, 'american': American
}



@dataclass
class Swaption:
    q: float
    swap_expiration: int
    expiration: int
    r: float
    strike: float
    option_name: str = 'european'
    call: bool = True

    @property
    def size(self):
        return self.expiration + 1

    @property
    def option(self):
        return OPTIONS[self.option_name](strike=self.strike, q=self.q, R=self.r, call=self.call)

    def get_prices_from(self, rates):
        swap = Lattice(base=None, periods=self.swap_expiration)
        swap.values = Swap(q=self.q, expiration=self.swap_expiration, r=self.r).get_prices_from(rates)
        discount = 1 / (1 + rates.values)

        return self.option.get_prices_from(swap, expiration=self.expiration, discount=discount)
