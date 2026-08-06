from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Option(ABC):
    strike: float
    q: float
    R: float
    call: bool

    def __post_init__(self):
        self.discount = 1 / self.R

    @property
    def call_put(self):
        return 2 * self.call - 1

    def backward(self, prices, discount):
        return (prices[1:] * self.q + prices[:-1] * (1 - self.q)) * discount

    @abstractmethod
    def get_prices_from(self, lattice):
        raise NotImplementedError()
