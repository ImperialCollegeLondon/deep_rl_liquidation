import torch
import numpy as np
import copy
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

from agents.Agent import Agent
from noise.OrnsteinUhlenbeckNoise import OrnsteinUhlenbeckNoise


class DeepDPGAgent(Agent):
    """Deep Deterministic Policy Gradient (DDPG) agent for Reinforcement Learning."""

    def __init__(
        self,
        env: gym.Env,
        critic_architecture: nn.Module,
        actor_architecture: nn.Module,
        noise: OrnsteinUhlenbeckNoise,
        tau_critic: float,
        tau_actor: float,
        epsilon: float,
        epsilon_min: float,
        epsilon_decay: float,
        weight_decay_critic: float,
        batch_size: int,
        cache_size: int,
        fifo_replay: bool,
        lr_critic_optimizer: float,
        lr_actor_optimizer: float,
        critic_optimizer: optim.Optimizer,
        actor_optimizer: optim.Optimizer,
        criterion_critic: nn.Module = nn.MSELoss(),
        criterion_actor: nn.Module = nn.MSELoss(),
        price_in_input: bool = False,
        gamma: float = 1,
        auxiliary_bellman_equation: bool = True,
        current_price_in_auxiliary: bool = False,
        tolerance: float = 1e-6,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Args:
            env (gym.Env):
                The environment to interact with.
            critic_architecture (nn.Module):
                The architecture of the critic network.
                This network should take (state, action) as input and output a
                predicted Q-value.
            actor_architecture (nn.Module):
                The architecture of the actor network.
                This network should take state as input and output an action.
            noise (OrnsteinUhlenbeckNoise):
                A Ornstein-Uhlenbeck noise process used for exploration during training.
            tau_critic (float):
                Target network update rate for critic.
            tau_actor (float):
                Target network update rate for actor
            epsilon (float):
                Exploration rate at the start of the experiment.
            epsilon_min (float):
                Minimum exploration rate.
            epsilon_decay (float):
                Exploration rate decay.
            weight_decay_critic (float):
                Weight decay parameter for critic optimizer.
            batch_size (int):
                Batch size for experience replay.
            cache_size (int):
                Experience replay buffer size.
            fifo_replay (bool):
                Memory replay buffer is a FIFO Queue
            device (torch.device):
                Device on which to perform torch computation
            lr_critic_optimizer (float):
                Learning rate for the critic optimizer.
            lr_actor_optimizer (float):
                Learning rate for the actor optimizer.
            critic_optimizer (optim.Optimizer):
                The optimizer class used to train the critic network.
            actor_optimizer (optim.Optimizer):
                The optimizer class used to train the actor network.
            criterion_critic (nn.Module, optional):
                Loss function for critic network.
                Defaults to nn.MSELoss().
            criterion_actor (nn.Module, optional):
                Loss function for actor network.
                Defaults to nn.MSELoss().
            gamma (float, optional):
                Discount factor.
                Defaults to 1.0.
            auxiliary_bellman_equation (bool, optional):
                Whether to use the auxiliary Bellman equation.
                Defaults to True.
            tolerance (float, optional):
                Tolerance level to declare liquidation.
                Defaults to 1e-6.
            dtype (torch.dtype, optional):
                Data type for tensors.
                Defaults to torch.float32.
        """

        # Call the parent class constructor
        super().__init__("Deep DPG Agent")

        # Save environment parameters
        self.env = env
        self.termination_time = self.env.observation_space["time"].high[0]
        self.tolerance = tolerance
        self.dtype = dtype
        self.device = device
        self.noise = noise

        # Define Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.tau_critic = tau_critic
        self.tau_actor = tau_actor
        self.batch_size = batch_size
        self.cache_size = cache_size
        self.fifo_replay = fifo_replay
        self.auxiliary_bellman_equation = auxiliary_bellman_equation
        self.current_price_in_auxiliary = current_price_in_auxiliary
        self.price_in_input = price_in_input

        # Initialize critic network
        self.critic_policy_network = copy.deepcopy(critic_architecture)

        # Initialize critic target network
        self.critic_target_network = copy.deepcopy(critic_architecture)
        self.critic_target_network.load_state_dict(
            copy.deepcopy(self.critic_policy_network.state_dict())
        )

        # Set-up critic optimizer
        self.critic_optimizer = critic_optimizer(
            self.critic_policy_network.parameters(),
            lr=lr_critic_optimizer,
            weight_decay=weight_decay_critic,
        )
        self.criterion_critic = criterion_critic

        # Initialize actor policy network
        self.actor_policy_network = copy.deepcopy(actor_architecture)

        # Initialize actor policy and target networks
        self.actor_target_network = copy.deepcopy(actor_architecture)
        self.actor_target_network.load_state_dict(
            copy.deepcopy(self.actor_policy_network.state_dict())
        )

        # Set-up actor optimizer
        self.actor_optimizer = actor_optimizer(
            self.actor_policy_network.parameters(), lr=lr_actor_optimizer
        )
        self.criterion_actor = criterion_actor

        # initialize experience replay
        self.memory = []
        self.critic_loss = []
        self.actor_loss = []

        # set agent to training mode
        self.train()

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def act(self, state):
        """Choose action using policy networks.

        Args:
            state (torch.tensor): Current state.

        Raises:
            ValueError: if action is NaN.
            ValueError: if action is positive (i.e., buy).

        Returns:
            torch.tensor: Choosen action at current state.
        """

        # Set policy networks to evaluation mode
        self.actor_policy_network.eval()

        # Liquidate everything if termination time is reached
        if self.env.state.is_terminal():
            best_action = -torch.tensor(
                [self.env.state["inventory"]], dtype=self.dtype, device=self.device
            )
        else:

            # Get action from actor network
            best_action = self.actor_policy_network(
                self.get_input_actor(
                    state.unsqueeze(0),
                )
            ).squeeze(0)

            # Add noise if required
            if self.training and np.random.random() < self.epsilon:
                noise = self.noise.sample()  # randomly sample noise
                best_action = self.actor_policy_network.transform(
                    (self.actor_policy_network.inv_transform(best_action) + noise)
                )

            # Multiply best action with current inventory
            best_action *= -self.env.state["inventory"]

            if torch.isnan(best_action):
                raise ValueError("best action is nan")

            if best_action > 0:
                raise ValueError("best action should be negative")

        # Set actor policy networks back to training mode
        self.actor_policy_network.train()

        return best_action.detach()

    def optimize(
        self,
        states: torch.tensor,
    ):
        """Value function under optimal policy given target networks.

        Args:
            states (torch.tensor): torch.tensor of states.

        Returns:
            torch.tensor: Q-value under optimal policy at each state.
        """

        # Set traget networks to evaluation mode
        self.actor_target_network.eval()
        self.critic_target_network.eval()

        # Create mask for non-terminal states
        mask = states[:, 1] < self.termination_time

        # Calculate optimal actions with actor network
        best_actions = -states[:, 0].view(-1, 1)  # inventory
        best_actions[mask] *= self.actor_target_network(self.get_input_actor(states))[
            mask
        ]  # multiply inventory by output of NN (in [0,1]) for non-terminal states

        # Calculate Q-values with critic network
        max_q_values = self.critic_target_network(
            self.get_input_critic(states, best_actions)
        )

        # Set target networks back to training mode
        self.actor_target_network.train()
        self.critic_target_network.train()

        return max_q_values.squeeze().detach()

    def replay(self):
        if len(self.memory) < self.cache_size:
            return

        if self.fifo_replay and len(self.memory) > self.cache_size:
            memory = self.memory[-self.cache_size :]
        else:
            memory = self.memory

        # Sample a batch of experience tuples from memory
        batch_index = np.random.choice(
            range(0, len(memory)),
            size=min(len(memory), self.batch_size),
            replace=False,
        ).astype(int)
        batch = [memory[i] for i in batch_index]
        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = zip(*batch)

        # Convert to tensors
        dones = torch.cat(dones).float()
        rewards = torch.cat(rewards)
        states = torch.stack(states, dim=0)
        actions = torch.stack(actions, dim=0)
        next_states = torch.stack(next_states, dim=0)

        # Calculate target Q-values
        with torch.no_grad():
            q_values_next_states = self.optimize(
                states=next_states,
            )
            target_q_values = rewards + self.gamma * q_values_next_states * (1 - dones)
            if self.current_price_in_auxiliary:
                target_q_values += (
                    self.env.state.get_current_price() * actions.squeeze(1)
                ) * self.auxiliary_bellman_equation
            else:
                target_q_values += (
                    self.env.state._init["price"] * actions.squeeze(1)
                ) * self.auxiliary_bellman_equation

        # Critic Loss
        q_values = self.critic_policy_network(self.get_input_critic(states, actions))
        critic_loss = self.criterion_critic(q_values.squeeze(), target_q_values).to(
            self.device
        )
        self.critic_loss.append(critic_loss.detach())

        # Critic backward pass
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor Loss
        mask = states[:, 1] < self.termination_time
        actions_values = -states[:, 0].view(-1, 1)
        actions_values[mask] *= self.actor_policy_network(self.get_input_actor(states))[
            mask
        ]

        self.critic_policy_network.eval()
        actor_loss = (
            -self.critic_policy_network(self.get_input_critic(states, actions_values))
            .mean()
            .to(self.device)
        )
        self.critic_policy_network.train()

        self.actor_loss.append(actor_loss.detach())

        # Actor backward pass
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Update target critic network weights
        for t, p in zip(
            self.critic_target_network.parameters(),
            self.critic_policy_network.parameters(),
        ):
            t.data = (1 - self.tau_critic) * t.data + self.tau_critic * p.data

        # Update target actor network weights
        for t, p in zip(
            self.actor_target_network.parameters(),
            self.actor_policy_network.parameters(),
        ):
            t.data = (1 - self.tau_actor) * t.data + self.tau_actor * p.data

    def get_input_critic(self, states, actions):

        return torch.cat(
            (
                self.normalize_states(states),
                self.normalize_actions(actions, states[:, 0].unsqueeze(1)),
            ),
            dim=1,
        )

    def get_input_actor(self, states):

        return self.normalize_states(states)

    def normalize_states(self, states):

        normalized_states = copy.deepcopy(states)

        # Normalize first column: inventory_t / inventory_0
        normalized_states[:, 0] = (
            normalized_states[:, 0] / self.env.state._init["inventory"]
        )

        # Normalize second column:  t / T
        normalized_states[:, 1] = normalized_states[:, 1] / (
            self.env.state._terminal_time + 1
        )

        # Normalize third column: past actions
        normalized_states[:, 2:] = self.normalize_actions(
            -normalized_states[:, 2:],
            torch.ones_like(normalized_states[:, 2:]),
        )

        # If price in input, normalize fourth column: (price_t - price_0) / inventory_0
        if self.price_in_input:
            normalized_states[:, 3] -= self.env.state._init["price"]
            normalized_states[:, 3] /= self.env.state._init["inventory"]

        return normalized_states

    def normalize_actions(self, actions, inventories):

        return actions / self.env.state._init["inventory"]
