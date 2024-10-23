from abc import ABC, abstractmethod
import numpy as np
import itertools
import copy


class Kernel(ABC):

    def __init__(self, init_decay: float, init_scale: float, name=None):
        self.name = name

        self.init_decay = (
            init_decay
            if hasattr(init_decay, "__iter__")
            else itertools.repeat(init_decay)
        )
        self.init_scale = (
            init_scale
            if hasattr(init_scale, "__iter__")
            else itertools.repeat(init_scale)
        )

        # set kernel parameters
        self.reset_parameter()

        # keep track of history
        self.reset_history()

    @abstractmethod
    def __call__(self, x):
        pass

    def reset_parameter(self):
        # copy the parameters over which to iterate
        self.iter_decay = copy.deepcopy(self.init_decay)
        self.iter_scale = copy.deepcopy(self.init_scale)

        # iterate on first value of parameters
        self.current_scale = next(self.iter_scale)
        self.current_decay = next(self.iter_decay)

    def reset_history(self):
        self.scale_history = []
        self.decay_history = []
        self.scale_history.append(self.current_scale)
        self.decay_history.append(self.current_decay)

    def update(self):
        # update parameters
        self.current_scale = next(self.iter_scale)
        self.current_decay = next(self.iter_decay)

        # keep track to retrace history
        self.scale_history.append(self.current_scale)
        self.decay_history.append(self.current_decay)


class ExponentialKernel(Kernel):

    def __init__(self, init_decay: float, init_scale: float):
        super().__init__(
            init_decay=init_decay, init_scale=init_scale, name="Exponential kernel"
        )

    def __call__(self, x):
        return float(self.current_scale * np.exp(-x * self.current_decay))


class PowerLawKernel(Kernel):

    def __init__(self, init_decay: float, init_scale: float):
        super().__init__(
            init_decay=init_decay, init_scale=init_scale, name="Power Law kernel"
        )

    def __call__(self, x):
        return float(self.current_scale * np.power(1 + x, -self.current_decay))


class LinearResilienceKernel(Kernel):

    def __init__(self, init_decay: float, init_scale: float):
        super().__init__(
            init_decay=init_decay,
            init_scale=init_scale,
            name="Linear Resilience kernel",
        )

    def __call__(self, x):
        resilience = 1 - self.current_decay * x
        if resilience > 0:
            return float(self.current_scale * resilience)
        else:
            return float(0.0)
