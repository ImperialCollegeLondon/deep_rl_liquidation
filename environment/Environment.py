import gymnasium as gym
from environment.Martingale import Martingale
from environment.Kernel import Kernel
from environment.State import State
from environment.Space import ActionSpace, ObservationSpace


class Environment(gym.Env):
    def __init__(
        self,
        martingale: Martingale,
        kernel: Kernel,
        state: State,
        observation_space: ObservationSpace,
        action_space: ActionSpace,
        tolerance: float,
    ):
        self.martingale = martingale
        self.kernel = kernel
        self.state = state
        self.observation_space = observation_space
        self.action_space = action_space
        self.tolerance = tolerance
        self.termination_time = int(self.observation_space["time"].high[0] + 1)
        self.actions_history = []

    def reset(self):
        self.martingale.reset()
        self.state.reset()
        self.action_space.reset()
        self.actions_history = []
        return (self.state, {})

    def save_action(self, action):
        # We cache time at which the action was performed
        # and what value for the action was chosen
        time = self.state["time"]
        self.actions_history.append((time, action))

    def transition_function(self, action):
        self.save_action(action)
        # Implement logic for transitioning from current state to next state based on action
        next_state = {}
        next_state["inventory"] = self.state["inventory"] + action
        next_state["inventory"] = (
            0
            if (next_state["inventory"] - 0) < self.tolerance
            else next_state["inventory"]
        )
        next_state["time"] = self.state["time"] + 1
        next_state["impact"] = sum(
            [
                self.kernel(next_state["time"] - pair[0]) * pair[1]
                for pair in self.actions_history
            ]
        )
        unaffected_price = next(self.martingale)
        next_state["price"] = unaffected_price + next_state["impact"]

        next_state["past_actions"] = [0] * int(self.termination_time)
        next_state["past_actions"][0 : len(self.actions_history)] = [
            action[1] for action in self.actions_history
        ]
        return next_state

    def get_reward(self, action):
        # We compute the jump price S_{i+} = S_{i} + \xi_{t_{n}}*G(0)  used in the penalty computation
        jump_price = self.state["price"] + action * self.kernel(0)
        penalty = pow(jump_price, 2) - pow(self.state["price"], 2)
        penalty /= 2 * self.kernel(0)
        return -penalty

    def step(self, action):
        # Perform action and move to next state
        next_state = self.transition_function(action)
        # Update the action space
        self.action_space.update(low=-next_state["inventory"])
        # Calculate reward
        reward = self.get_reward(action)
        # Update state
        self.state.update(next_state)
        # Check for terminal condition
        done = self.state.is_done()
        # Observe state
        observation = self.state.get_observation()
        return (
            observation,
            float(reward),
            bool(done),
        )
