import numpy as np
import gymnasium as gym

from agents.Agent import Agent


class OptimalAgent(Agent):
    """An agent that computes and executes the optimal strategy for a given environment."""

    def __init__(self, env: gym.Env):
        """
        Args:
            env (gym.Env): The environment to interact with.
        """

        # Call the parent class constructor
        super().__init__("Optimal agent")

        # Initialize agent with environment object
        self.env = env
        self.termination_time = int(self.env.observation_space["time"].high[0])

    def act(self, state: np.array):
        """Returns the optimal action given the current state.

        Args:
            state (np.array): Current state.

        Returns:
            float: The optimal action.
        """

        action = self.get_optimal_strategy()[state["time"]]

        return action

    def get_optimal_strategy(self):
        """
        Computes the optimal strategy for the given environment.

        Returns:
            np.ndarray: The optimal strategy.
        """
        M = np.matrix(
            [
                [
                    self.env.kernel(abs(i - j))
                    for i in range(0, self.termination_time + 1)
                ]
                for j in range(0, self.termination_time + 1)
            ]
        )
        inverse_M = np.linalg.inv(M)
        strategy = self.env.state._init["inventory"]
        all_ones = np.ones(self.termination_time + 1)
        strategy /= np.matmul(np.matmul(all_ones, inverse_M), all_ones)
        strategy = strategy * np.matmul(inverse_M, all_ones)
        return -np.ravel(strategy)
