import torch
import pickle
import gymnasium as gym
import copy

from experiment.utils_experiment import *
from agents.Agent import Agent


class Experiment:

    def __init__(
        self,
        env: gym.Env,
        agent: Agent,
        max_iterations=None,
        max_episodes=None,
        monitor_freq: int = 100,
        plot_actions: bool = True,
        plot_rewards: bool = True,
        plot_loss: bool = True,
        plot_environment: bool = True,
        file_path: str = None,
    ):
        """Initalize a class Experiment.

        Args:
            env (gym.Env):
                The environment instance created using gym.make with a valid id.
            agent (Agent):
                The agent that interacts with the environment.
            max_iterations (int, optional):
                The maximum number of training iterations.
            max_episodes (int, optional):
                The maximum number of training episodes.
            monitor_freq (int, optional):
                The frequency (in episodes) to log information during training.
                Defaults to 100.
            plot_actions (bool, optional):
                Whether to plot best actions during training.
                Defaults to True.
            plot_rewards (bool, optional):
                Whether to plot cumulative rewards during training.
                Defaults to True.
            plot_loss (bool, optional):
                Whether to plot the loss during training.
                Defaults to True.
            plot_environment (bool, optional):
                Whether to plot environment.
                Defaults to True.
            file_path (str, optional):
                The path to save training results (e.g., plots, best actions).
                Defaults to None.
        """

        self.env = env  # env to generate episode
        self.env_simulation = copy.deepcopy(env)  # env to track best actions
        self.agent = agent
        self.max_iterations = max_iterations
        self.max_episodes = max_episodes
        self.monitor_freq = monitor_freq
        self.plot_actions = plot_actions
        self.plot_rewards = plot_rewards
        self.plot_loss = plot_loss
        self.plot_environment = plot_environment
        self.file_path = file_path
        self.tolerance = self.agent.tolerance
        self.dtype = self.agent.dtype
        self.device = self.agent.device

        # initialize lists to store best actions and rewards
        self.best_strategy = []
        self.executed_strategy = []
        self.current_states = []

        # store optimal actions and associated rewards
        (
            self.optimal_actions,
            self.optimal_best_rewards,
            self.optimal_total_reward,
        ) = get_optimal_strategy(self.env_simulation)
        self.optimal_cumulative_rewards = np.cumsum(self.optimal_best_rewards)

    def fit(self):
        """Trains the agent for the specified number of episodes."""

        self.current_iteration = 0
        self.current_episode = 0

        while (
            self.current_iteration < self.max_iterations
            and self.current_episode < self.max_episodes
        ):

            self.current_episode += 1
            self._reset()  # Reset at the beginning of each episode
            self._run_episode()  # Run episode
            self._monitor_fit()  # Monitor Variables during training

    def save(self):
        """Saves training results, plots, and agent state."""

        # Extract best strategy from list
        episodes = [entry["episode"] for entry in self.best_strategy]
        iterations = [entry["iteration"] for entry in self.best_strategy]
        best_actions = [entry["best_actions"] for entry in self.best_strategy]
        best_cumulative_rewards = [
            entry["best_cumulative_rewards"] for entry in self.best_strategy
        ]
        best_total_reward = [entry["best_total_reward"] for entry in self.best_strategy]

        # Extract actual strategy from list
        executed_actions = [
            entry["executed_actions"] for entry in self.executed_strategy
        ]
        executed_cumulative_rewards = [
            entry["executed_cumulative_rewards"] for entry in self.executed_strategy
        ]
        executed_total_reward = [
            entry["executed_total_reward"] for entry in self.executed_strategy
        ]

        # Move all tensors to cpu device
        cpu_device = torch.device(type="cpu")
        if self.device != cpu_device:
            best_actions = list_tensors_to_device(best_actions, cpu_device)
            best_cumulative_rewards = list_tensors_to_device(
                best_cumulative_rewards, cpu_device
            )
            best_total_reward = list_tensors_to_device(best_total_reward, cpu_device)
            executed_actions = list_tensors_to_device(executed_actions, cpu_device)
            executed_cumulative_rewards = list_tensors_to_device(
                executed_cumulative_rewards, cpu_device
            )
            executed_total_reward = list_tensors_to_device(
                executed_total_reward, cpu_device
            )
            self.agent.critic_loss = list_tensors_to_device(
                self.agent.critic_loss, cpu_device
            )
            self.agent.actor_loss = list_tensors_to_device(
                self.agent.actor_loss, cpu_device
            )
            self.agent.memory = list_tuple_tensors_to_device(
                self.agent.memory, cpu_device
            )

        # reset simulation environment
        self.env_simulation.reset()
        self.env_simulation.kernel.reset_parameter()

        # save loss, actions and rewards objects
        save_object(self.file_path, "device", self.device)
        save_object(self.file_path, "memory", self.agent.memory)
        save_object(self.file_path, "environment", self.env_simulation)
        save_object(self.file_path, "current_episode", self.current_episode)
        save_object(self.file_path, "current_iteration", self.current_iteration)
        save_object(self.file_path, "episodes", episodes)
        save_object(self.file_path, "iterations", iterations)
        save_object(self.file_path, "cache_size", self.agent.cache_size)
        save_object(self.file_path, "kernel_scale", self.env.kernel.scale_history)
        save_object(self.file_path, "kernel_decay", self.env.kernel.decay_history)
        save_object(self.file_path, "best_actions", best_actions)
        save_object(self.file_path, "best_cumulative_rewards", best_cumulative_rewards)
        save_object(self.file_path, "best_total_reward", best_total_reward)
        save_object(self.file_path, "executed_actions", executed_actions)
        save_object(
            self.file_path, "executed_cumulative_rewards", executed_cumulative_rewards
        )
        save_object(self.file_path, "executed_total_reward", executed_total_reward)
        save_object(self.file_path, "optimal_actions", self.optimal_actions)
        save_object(
            self.file_path,
            "optimal_cumulative_rewards",
            self.optimal_cumulative_rewards,
        )
        save_object(self.file_path, "optimal_total_reward", self.optimal_total_reward)
        save_object(self.file_path, "critic_loss", self.agent.critic_loss)
        if hasattr(self.agent, "actor_loss"):
            save_object(self.file_path, "actor_loss", self.agent.actor_loss)

        # Save nn weights
        if hasattr(self.agent, "critic_policy_network"):
            critic_dict_cpu = {
                key: value.cpu()
                for key, value in self.agent.critic_policy_network.state_dict().items()
            }
            torch.save(
                critic_dict_cpu,
                self.file_path + "/weights_critic.pth",
            )
        if hasattr(self.agent, "actor_policy_network"):
            actor_dict_cpu = {
                key: value.cpu()
                for key, value in self.agent.actor_policy_network.state_dict().items()
            }
            torch.save(
                actor_dict_cpu,
                self.file_path + "/weights_actor.pth",
            )

        # save plot of the loss, actions and rewards
        self._plot_experiment()

    def _reset(self):
        """
        Resets the environment and state variables for a new episode.
        """

        # Update the kernel (parameters are changing for online experiement)
        if self.current_episode > 1:
            self.env.kernel.update()  # used for data (episode) generation
            self.env_simulation.kernel.update()  # used to track best action

        # Reset the environment
        _ = self.env.reset()

        # Reset the noise
        if hasattr(self.agent, "noise"):
            self.agent.noise.reset()

        # Initialize episode variables
        self.reward = torch.tensor([0], dtype=self.dtype, device=self.device)
        self.current_state = torch.tensor(
            [
                item
                for sublist in self.env.state.get_observation().values()
                for item in (sublist if isinstance(sublist, list) else [sublist])
                if not isinstance(item, list)
            ],
            dtype=self.dtype,
            device=self.device,
        )

    def _run_episode(self):
        """
        Runs a single episode of interaction between the agent and environment.
        """

        while True:

            # 1. Agent selects action
            action = self.agent.act(self.current_state)

            # Validate action (prevent selling more than inventory)
            if (self.current_state[0] + action) < -self.tolerance:
                raise ValueError("action larger than inventory")

            # 2. Environment Steps
            next_state, reward, done = self.env.step(action)

            # Move to tensors
            next_state = torch.tensor(
                [
                    item
                    for sublist in next_state.values()
                    for item in (sublist if isinstance(sublist, list) else [sublist])
                    if not isinstance(item, list)
                ],
                dtype=self.dtype,
                device=self.device,
            )
            reward = torch.tensor([reward], dtype=self.dtype, device=self.device)
            done = torch.tensor([done], device=self.device)

            # 3. Check whether episode is done based on inventory level
            if next_state[0] < self.tolerance:
                done = torch.tensor([True], device=self.device).bool()

            # 4. Store Experience in Memory
            self.current_states.append(
                self.env.state.get_observation()
            )  # current state
            self.agent.memory.append(
                (self.current_state, action, reward, next_state, done)
            )

            # 5. Learn from Experience
            self.current_iteration += 1
            self.agent.replay()  # Update agent's networks

            # 6. Accumulate Reward
            self.reward += reward

            # 7. Track best actions
            self._track_best_actions()

            # 8. Iterate or End Episode
            if done:

                # Handle premature liquidations (remove invalid transitions from memory)
                if self.current_state[1] < self.env.observation_space["time"].high[0]:
                    diff_T = (1 + self.current_state[1]).int()
                    self.agent.memory = self.agent.memory[0:-diff_T]
                    if len(self.agent.memory) < self.agent.cache_size:
                        self.current_episode -= 1
                        self.current_iteration -= int(diff_T)
                else:
                    # Track actions used during episode
                    self._track_executed_actions()
                break
            elif self.current_iteration == self.max_iterations:
                break
            else:
                self.current_state = next_state  # Move to next state

    def _track_best_actions(self):
        """
        Tracks performance of the best possible strategy elicitated by agent.
        """

        if self.current_iteration % 100 == 0:
            # use environmnent to generate best action (to avoid interfering with undergoing episode)
            self.agent.env = self.env_simulation

            # find best actions and associated rewards
            best_actions, best_rewards, best_total_reward = get_best_strategy(
                self.env_simulation,
                self.agent,
                self.dtype,
                self.device,
            )
            best_cumulative_rewards = torch.cumsum(best_rewards, dim=0)

            # Keep track
            self.best_strategy.append(
                {
                    "iteration": self.current_iteration,
                    "episode": self.current_episode,
                    "best_actions": best_actions,
                    "best_cumulative_rewards": best_cumulative_rewards,
                    "best_total_reward": best_total_reward,
                }
            )

            # set-back env used in episode
            self.agent.env = self.env

    def _track_executed_actions(self):
        """
        Tracks performance of the actual strategy used by agent.
        """
        termination_time = int(self.env.observation_space["time"].high[0] + 1)

        if (
            len(self.best_strategy) > 0
            and self.current_episode == self.best_strategy[-1]["episode"]
        ):
            # select memory of last episode
            last_episode = self.agent.memory[-termination_time:]

            # find action actions and associated rewards
            executed_actions = torch.stack(
                [tensor[1] for tensor in last_episode]
            ).squeeze(1)
            executed_rewards = torch.stack(
                [tensor[2] for tensor in last_episode]
            ).squeeze(1)
            executed_total_reward = torch.sum(executed_rewards).unsqueeze(0)
            executed_cumulative_rewards = torch.cumsum(executed_rewards, dim=0)

            # Keep track
            self.executed_strategy.append(
                {
                    "iteration": self.current_iteration,
                    "episode": self.current_episode,
                    "executed_actions": executed_actions,
                    "executed_cumulative_rewards": executed_cumulative_rewards,
                    "executed_total_reward": executed_total_reward,
                }
            )

    def _monitor_fit(self):
        """
        Monitors training progress by logging information and plotting results.
        """

        if len(self.agent.memory) < self.agent.cache_size:
            return

        # Print loss every 1000th iteration
        if self.current_iteration % 1000 == 0:
            print(f"Episode {self.current_episode + 1}")
            print(f"Critic Loss: {self.agent.critic_loss[-1]}")
            if hasattr(self.agent, "actor_loss"):
                print(f"Actor Loss: {self.agent.actor_loss[-1]}")

        # Monitor experiment every `monitor_freq` episodes
        if (
            self.current_iteration % self.monitor_freq == 0
            and self.current_episode != 0
        ):
            print(f"Episode {self.current_episode + 1}")
            print(f"Total Reward: {self.reward.item()}")
            print("Memory size:", len(self.agent.memory))
            print(f"Epsilon={self.agent.epsilon}", flush=True)
            self._plot_experiment()

    def _plot_experiment(self):
        """
        Generates plots to track training progress.
        """

        # Extract best strategy across iterations
        iterations = [entry["iteration"] for entry in self.best_strategy]
        best_actions = [entry["best_actions"].cpu() for entry in self.best_strategy]
        best_cumulative_rewards = [
            entry["best_cumulative_rewards"].cpu() for entry in self.best_strategy
        ]
        best_total_reward = [
            entry["best_total_reward"].cpu() for entry in self.best_strategy
        ]

        # Extract current best strategy
        current_best_actions = best_actions[-1]
        current_best_cumulative_rewards = best_cumulative_rewards[-1]
        current_best_total_reward = best_total_reward[-1]

        # Plot best actions if enabled
        if self.plot_environment:
            plot_states(states=self.current_states, file_path=self.file_path)
            plot_kernel(
                param=self.env.kernel.scale_history,
                y_label="scale",
                file_path=self.file_path,
            )
            plot_kernel(
                param=self.env.kernel.decay_history,
                y_label="decay",
                file_path=self.file_path,
            )

        if self.plot_actions:
            print("Optimal actions:", self.optimal_actions)
            print("Best actions:", current_best_actions)
            plot_strategy_multiple_iterations(
                iterations,
                best_actions,
                self.optimal_actions,
                y_label="Action",
                file_path=self.file_path,
            )
            plot_random_actions(self.agent.memory, file_path=self.file_path)

        # Plot best rewards if enabled
        if self.plot_rewards:
            print("Optimal cumulative rewards:", self.optimal_total_reward)
            print("Best cumulative rewards:", current_best_total_reward)
            plot_strategy(
                current_best_cumulative_rewards,
                self.optimal_cumulative_rewards,
                y_label="cumulative_rewards",
                file_path=self.file_path,
            )
            plot_total_reward(
                iterations,
                best_total_reward,
                self.optimal_total_reward,
                file_path=self.file_path,
            )

        # Plot losses if enabled
        if self.plot_loss:
            plot_loss(self.agent.critic_loss, "Critic", file_path=self.file_path)
            if hasattr(self.agent, "actor_loss"):
                plot_loss(
                    self.agent.actor_loss, "Actor", file_path=self.file_path, log=False
                )
