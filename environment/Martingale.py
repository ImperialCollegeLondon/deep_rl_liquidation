from abc import ABC, abstractmethod
import numpy as np
import copy


class Martingale(ABC):

    def __init__(
        self,
        volatility: float,
        initialization: float,
        step_size: int = 1,
        seed: int = None,
    ):
        self.initialization = initialization
        self.price = initialization
        self.volatility = volatility
        self.rng = np.random.default_rng(seed)
        self.step_size = step_size

    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self):
        pass

    def reset(self):
        self.price = copy.deepcopy(self.initialization)


class BrownianMotion(Martingale):

    def __next__(self):
        self.price += (
            self.volatility * np.sqrt(self.step_size) * self.rng.normal(0, 1, size=1)
        )
        return float(self.price)


class GeometricBrownianMotion(Martingale):

    def __next__(self):
        self.price *= np.exp(
            -0.5 * (self.volatility**2) * self.step_size
            + self.volatility * np.sqrt(self.step_size) * self.rng.normal(0, 1, size=1)
        )
        return float(self.price)
