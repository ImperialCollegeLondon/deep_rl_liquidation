import torch
import json
import argparse
import numpy as np
import random
import torch.optim as optim
import pickle

from experiment.experiment import Experiment
from agents.DeepDPGAgent import DeepDPGAgent
from architectures.deep_dpg_architectures import FCN, ActorFCN
from noise.OrnsteinUhlenbeckNoise import OrnsteinUhlenbeckNoise
from environment.Environment import Environment
from environment.Space import ActionSpace, ObservationSpace
from environment.Martingale import BrownianMotion, GeometricBrownianMotion
from environment.Kernel import ExponentialKernel, PowerLawKernel, LinearResilienceKernel
from environment.State import State


torch.manual_seed(42)


def run_experiment(
    file_path: str,
    martingale_type: str = "BrownianMotion",
    initial_inventory: int = 10,
    initial_price: float = 50.0,
    time_horizon: int = 9,
    martingale_scale: float = 0.0001,
    kernel_type: str = "ExponentialKernel",
    kernel_decay: float = 1.0,
    kernel_scale: float = 1.0,
    kernel_decay_change: float = None,  # None if no change, else value of the decay after change
    kernel_change_type: str = "linear",  # Only if kernel_decay_change not None
    sigma_noise: float = 1.0,  # OU process
    theta_noise: float = 0.15,  # OU process
    critic_input_dim: int = 13,
    critic_num_layers: int = 14,
    critic_feature_dim: int = 64,
    actor_input_dim: int = 12,
    actor_num_layers: int = 10,
    actor_feature_dim: int = 54,
    critic_activation: str = "ReLU",
    actor_activation: str = "sigmoid",
    batch_size: int = 100,
    cache_size: int = 10000,
    fifo_replay: bool = True,  # if true, only cache_size most recent episodes are used from memory
    tau_critic: float = 0.005,
    tau_actor: float = 0.005,
    lr_critic_optimizer: float = 0.0005,
    lr_actor_optimizer: float = 0.00005,
    critic_optimizer_name: str = "Adam",
    actor_optimizer_name: str = "Adam",
    epsilon: float = 1.0,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 1.0,
    weight_decay_critic: float = 0.0,
    agent_name: str = "DDPG",
    max_iterations: int = None,
    max_episodes: int = None,
    price_in_input: bool = False,  # projected input space does not contain price
    auxiliary_bellman_equation: bool = True,
    current_price_in_auxiliary: bool = False,
    plot_actions: bool = True,
    plot_rewards: bool = True,
    plot_loss: bool = True,
    plot_environment: bool = True,
    file_pre_trained_critic_weights: str = None,
    file_pre_trained_actor_weights: str = None,
    file_pre_trained_memory: str = None,
    tolerance: float = 1e-15,
    dtype: torch.dtype = torch.float32,
    seed=12,
):

    # set a seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Check if GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU is not available, using CPU")

    # Either max_iterations or max_episodes need to be specified
    if max_episodes is None and max_episodes is None:
        ValueError("Either max_episodes or max_episodes need to be specified")
    if max_iterations is None:
        max_iterations = max_episodes * (time_horizon + 1)
    elif max_episodes is None:
        max_episodes = max_iterations / (time_horizon + 1)

    # Define martingale for environment
    if martingale_type == "BrownianMotion":
        martingale = BrownianMotion(
            initialization=initial_price, volatility=martingale_scale, seed=seed
        )
    elif martingale_type == "GeometricBrownianMotion":
        martingale = GeometricBrownianMotion(
            initialization=initial_price, volatility=martingale_scale, seed=seed
        )
    else:
        raise ValueError("Martingale not implemented.")

    # Define kernel for environment
    if kernel_decay_change is not None:
        if kernel_change_type == "linear":
            kernel_decay = iter(
                np.linspace(
                    kernel_decay, kernel_decay_change, num=max_episodes
                ).tolist()
            )
        elif kernel_change_type == "exponential":
            kernel_decay = iter(
                (
                    kernel_decay * kernel_decay_change ** np.arange(0, max_episodes)
                ).tolist()
            )

    if kernel_type == "ExponentialKernel":
        kernel = ExponentialKernel(init_decay=kernel_decay, init_scale=kernel_scale)
    elif kernel_type == "PowerLawKernel":
        kernel = PowerLawKernel(init_decay=kernel_decay, init_scale=kernel_scale)
    elif kernel_type == "LinearResilienceKernel":
        kernel = LinearResilienceKernel(
            init_decay=kernel_decay, init_scale=kernel_scale
        )
    else:
        raise ValueError("Kernel not implemented.")

    # Define obervational state for environment
    if price_in_input:
        inputs = ["inventory", "time", "past_actions", "price"]
    else:
        inputs = ["inventory", "time", "past_actions"]

    state = State(
        init={
            "inventory": initial_inventory,
            "price": initial_price,
            "impact": 0.0,
            "time": 0,
            "past_actions": [0] * int(time_horizon + 1),
        },
        inputs=inputs,
        terminal_time=time_horizon,
        terminal_inventory=0.0,
        dtype=dtype,
    )

    # Define observation and action space
    observation_space = ObservationSpace(
        initial_inventory, initial_price, time_horizon
    )()
    action_space = ActionSpace(low=-initial_inventory, high=0)

    # Define environment
    env = Environment(
        observation_space=observation_space,
        action_space=action_space,
        martingale=martingale,
        kernel=kernel,
        state=state,
        tolerance=tolerance,
    )

    # Define critic activation function
    if critic_activation == "sigmoid":
        critic_activation = torch.sigmoid
    elif critic_activation == "ReLU":
        critic_activation = torch.nn.ReLU()
    elif critic_activation == "Tanh":
        critic_activation = torch.nn.Tanh()
    elif critic_activation == "LeakyReLU":
        critic_activation = torch.nn.LeakyReLU()

    # Define actor activation function
    if actor_activation == "sigmoid":
        actor_activation = torch.sigmoid
    elif actor_activation == "ReLU":
        actor_activation = torch.nn.ReLU()
    elif actor_activation == "Tanh":
        actor_activation = torch.nn.Tanh()
    elif actor_activation == "LeakyReLU":
        actor_activation = torch.nn.LeakyReLU()

    # Define actor and critic architectures
    critic = FCN(
        input_dim=critic_input_dim,
        num_layers=critic_num_layers,
        feature_dim=critic_feature_dim,
        activation=critic_activation,
    )
    actor = ActorFCN(
        input_dim=actor_input_dim,
        num_layers=actor_num_layers,
        feature_dim=actor_feature_dim,
        activation=actor_activation,
    )

    # Assign pre-trained weight to netowrk if any
    if file_pre_trained_critic_weights is not None:
        critic_pretrained_weights = torch.load(file_pre_trained_critic_weights)
        critic.load_state_dict(critic_pretrained_weights)

    if file_pre_trained_actor_weights is not None:
        actor_pretrained_weights = torch.load(file_pre_trained_actor_weights)
        actor.load_state_dict(actor_pretrained_weights)

    # Move the models to the selected device
    critic.to(device)
    actor.to(device)

    # Define critic optimizer
    if critic_optimizer_name == "Adam":
        critic_optimizer = optim.Adam
    elif critic_optimizer_name == "RMSprop":
        critic_optimizer = optim.RMSprop

    # Define actor optimizer
    if actor_optimizer_name == "Adam":
        actor_optimizer = optim.Adam
    elif actor_optimizer_name == "RMSprop":
        actor_optimizer = optim.RMSprop

    # Define noise for exploration
    noise = OrnsteinUhlenbeckNoise(
        1, sigma=sigma_noise, theta=theta_noise, device=device
    )

    # Define agents
    if agent_name == "DDPG":
        agent = DeepDPGAgent(
            env=env,
            noise=noise,
            critic_architecture=critic,
            actor_architecture=actor,
            batch_size=batch_size,
            cache_size=cache_size,
            tau_critic=tau_critic,
            tau_actor=tau_actor,
            lr_critic_optimizer=lr_critic_optimizer,
            lr_actor_optimizer=lr_actor_optimizer,
            critic_optimizer=critic_optimizer,
            actor_optimizer=actor_optimizer,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            weight_decay_critic=weight_decay_critic,
            price_in_input=price_in_input,
            dtype=dtype,
            tolerance=tolerance,
            fifo_replay=fifo_replay,
            auxiliary_bellman_equation=auxiliary_bellman_equation,
            current_price_in_auxiliary=current_price_in_auxiliary,
            device=device,
        )
    else:
        raise ValueError("Agent not implemented yet.")

    # Append existing memory
    if file_pre_trained_memory is not None:
        with open(file_pre_trained_memory, "rb") as f:
            agent.memory = pickle.load(f)["memory"]

    # Setup Experiment
    discrete_experiment = Experiment(
        env=env,
        agent=agent,
        max_iterations=max_iterations,
        max_episodes=max_episodes,
        monitor_freq=max_iterations + 1,
        plot_actions=plot_actions,
        plot_rewards=plot_rewards,
        plot_loss=plot_loss,
        plot_environment=plot_environment,
        file_path=file_path,
    )

    # Run Experiment
    discrete_experiment.fit()

    # Save results
    discrete_experiment.save()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    parser.add_argument("--fpath", type=str, help="Path to save the results")
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config_data = json.load(file)

    if not isinstance(config_data, list):
        config_data = [config_data]

    for config_set in config_data:
        config_set.pop("job_name", None)  # Remove the "job_name" key from the config
        run_experiment(file_path=args.fpath, **config_set)

    # example call
    # /Users/melodiemonod/miniconda3/envs/deep_rl_liquidation/bin/python /Users/melodiemonod/git/deep_rl_liquidation/run_experiment.py --config=/Users/melodiemonod/git/deep_rl_liquidation/config.json --fpath="/Users/melodiemonod/projects/2024/deep_rl_liquidation/debug"
