from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from quant.elementary import Elementary
from quant.lattices.base import Lattice


@dataclass
class BlackDermanToy:
    q: float
    b: float
    periods: int

    def get_objective_function_from(self, b, rates, elementary, lattice):
        periods = lattice.periods + 1
        def objective(a):
            lattice.values = np.triu(a * np.exp(self.b * np.arange(periods))[:, None])
            prices = elementary.get_prices_from(lattice).sum(axis=0)[1:]
            predictions = ((1 / prices) ** (1 / (np.arange(periods) + 1)) - 1)
            return np.sum(np.square(predictions - rates))
        return objective

    def fit(self, rates, initialization: np.ndarray | None = None, seed=None, **kwargs):
        a = initialization
        if initialization is None:
            a = np.ones_like(rates) * rates[0, 0]

        elementary = Elementary(q=self.q)
        lattice = Lattice(base=None, periods=self.periods - 1)
        objective = self.get_objective_function_from(
            b=self.b, rates=rates,
            elementary=elementary, lattice=lattice
        )
        
        np.random.seed(seed)
        self.report = minimize(objective, a, **kwargs)
        self.a = self.report.x
        return self

    def get_rates(self):
        rates = np.triu(self.a * np.exp(self.b * np.arange(self.periods))[:, None])
        lattice = Lattice(base=rates[0, 0], periods=self.periods - 1)
        lattice.values = rates
        return lattice



if __name__ == '__main__':
    periods = 14
    rates = np.array([7.3, 7.62, 8.1, 8.45, 9.2, 9.64, 10.12, 10.45, 10.75, 11.22, 11.55, 11.92, 12.2, 12.32]) / 100
    a = np.ones_like(rates) * .05
    b = .005
    q = .5

    bdt = BlackDermanToy(q=q, b=b, periods=periods)
    bdt.fit(rates, initialization=a, options={'disp': True})
    import ipdb; ipdb.set_trace()

