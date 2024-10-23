class State(dict):

    def __init__(
        self,
        init: dict,
        inputs: list,
        terminal_time: int,
        terminal_inventory: float,
        dtype,
    ):
        for k, v in init.items():
            super().__setitem__(k, v)
        self._init = init
        self._inputs = inputs
        self._terminal_time = terminal_time
        self._terminal_inventory = terminal_inventory
        self.dtype = dtype

    def reset(self):
        self.clear()
        self.update(self._init)

    def get_observation(self):
        observation = {obs: self[obs] for obs in self._inputs}
        return observation

    def get_current_price(self):
        return self["price"]

    def is_done(self):
        return (self["time"] == self._terminal_time + 1) or (
            self["inventory"] == self._terminal_inventory
        )

    def is_terminal(self):
        return self["time"] == self._terminal_time
