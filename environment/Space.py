import copy
import gymnasium as gym
from gymnasium.spaces import Dict


class ActionSpace(gym.spaces.Box):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_low = copy.deepcopy(self.low)
        self.init_high = copy.deepcopy(self.high)

    def update(self, low=None, high=None):
        if low:
            self.low = low
        if high:
            self.high = high

    def reset(self):
        self.low = copy.deepcopy(self.init_low)
        self.high = copy.deepcopy(self.init_high)


class ObservationSpace:

    def __init__(
        self, initial_inventory: float, initial_price: float, time_horizon: int
    ):
        self.initial_inventory = initial_inventory
        self.initial_price = initial_price
        self.time_horizon = time_horizon

    def __call__(self):
        return Dict(
            {
                "inventory": gym.spaces.Box(low=0, high=self.initial_inventory),
                "price": gym.spaces.Box(low=0, high=self.initial_price * 100),
                "time": gym.spaces.Box(low=0, high=self.time_horizon),
                "impact": gym.spaces.Box(low=-self.initial_price * 100, high=0),
            }
        )
