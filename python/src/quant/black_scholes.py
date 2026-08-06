from dataclasses import dataclass

import numpy as np


@dataclass
class BlackScholes:
    T: float
    sigma: float
    periods: int
    r: float
    div: float

    def __post_init__(self):
        self.u: float = np.exp(self.sigma * np.sqrt(self.T / self.periods))
        self.d = 1 / self.u
        self.q = (np.exp((self.r - self.div) * self.T / self.periods) - self.d) / (self.u - self.d)
        self.discount = np.exp(-self.r * self.T /self.periods)
