from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base import Lattice


@dataclass
class UpNDownLattice(Lattice):
    up: float
    down: float
    _values: Optional[np.ndarray] = None

    @property
    def shape(self):
        return (self.periods + 1, self.periods + 1)

    @property
    def values(self):
        if self._values is None:
            self._values = self.propagate()
        return self._values

    def propagate(self):
        periods = np.arange(self.periods + 1)[:, None]
        lattice: np.ndarray = np.triu(np.pow(self.up, periods) * np.pow(self.down, periods.T - periods))
        return lattice * self.base
