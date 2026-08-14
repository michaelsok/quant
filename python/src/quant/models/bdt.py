from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from quant.elementary import Elementary
from quant.lattices.base import Lattice


@dataclass
class BlackDermanToy:
    q: float
    periods: int
    a: float | None = None
    b: float | None = None
    _target: str | None = None

    @property
    def target(self):
        if self._target is None:
            if self.a is not None:
                self._target = 'b'
            elif self.b is not None:
                self._target = 'a'
            else:
                raise ValueError("Cannot define a target")
        return self._target

    def get_objective_function_from(self, a, b, rates, elementary, lattice):
        if (a is None) and (b is None):
            raise ValueError("Either a or b should be passed")

        periods = lattice.periods + 1
        if a is None:
            def objective(a):
                lattice.values = np.triu(a * np.exp(self.b * np.arange(periods))[:, None])
                prices = elementary.get_prices_from(lattice).sum(axis=0)[1:]
                predictions = ((1 / prices) ** (1 / (np.arange(periods) + 1)) - 1)
                return np.sum(np.square(predictions - rates))
        else:
            def objective(b):
                lattice.values = np.triu(self.a * np.exp(b * np.arange(periods))[:, None])
                prices = elementary.get_prices_from(lattice).sum(axis=0)[1:]
                predictions = ((1 / prices) ** (1 / (np.arange(periods) + 1)) - 1)
                return np.sum(np.square(predictions - rates))
        return objective

    def get_objective_function_from_prices(self, a, b, prices, elementary, lattice):
        if (a is None) and (b is None):
            raise ValueError("Either a or b should be passed")

        periods = lattice.periods + 1
        if a is None:
            def objective(a):
                lattice.values = np.triu(a * np.exp(self.b * np.arange(periods))[:, None])
                predictions = elementary.get_prices_from(lattice).sum(axis=0)[1:]
                return np.sum(np.square(predictions - prices))
        else:
            def objective(b):
                lattice.values = np.triu(self.a * np.exp(b * np.arange(periods))[:, None])
                predictions = elementary.get_prices_from(lattice).sum(axis=0)[1:]
                return np.sum(np.square(predictions - prices))
        return objective

    def fit(self, rates=None, initialization: np.ndarray | None = None, seed=None, prices=None, **kwargs):
        x0 = initialization

        if rates is None:
            target = prices
        else:
            target = rates

        if initialization is None:
            x0 = np.ones_like(target) * target[0]

        if self.target == 'a':
            a, b = None, self.b
        elif self.target == 'b':
            a, b = self.a, None
        else:
            raise ValueError("Cannot determine target")

        elementary = Elementary(q=self.q)
        lattice = Lattice(base=None, periods=self.periods - 1)

        if prices is None:
            objective = self.get_objective_function_from(
                a=a, b=b, rates=rates,
                elementary=elementary, lattice=lattice
            )
        else:
            objective = self.get_objective_function_from_prices(
                a=a, b=b, prices=prices,
                elementary=elementary, lattice=lattice
            )

        np.random.seed(seed)
        self.report = minimize(objective, x0, **kwargs)
        setattr(self, self.target, self.report.x)
        return self

    def get_rates(self):
        rates = np.triu(self.a * np.exp(self.b * np.arange(self.periods))[:, None])
        lattice = Lattice(base=rates[0, 0], periods=self.periods - 1)
        lattice.values = rates
        return lattice
