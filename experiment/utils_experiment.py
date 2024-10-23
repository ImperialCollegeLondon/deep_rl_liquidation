# global modules
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle

# local modules
from agents.OptimalAgent import OptimalAgent


def get_best_strategy(env, agent, dtype, device):

    # Initialize lists
    best_rewards_list = []
    best_total_reward_list = []
    best_actions_list = []

    # Number of simulations
    n = 50

    # Simulate best strategy for n market simulations
    for _ in range(n):
        best_actions, best_rewards, best_total_reward = (
            get_best_strategy_one_simulation(env, agent, dtype, device)
        )
        best_rewards_list.append(best_rewards)
        best_total_reward_list.append(best_total_reward)
        best_actions_list.append(best_actions)

    # Convert list of tensors to a single tensor
    best_total_reward_stack = torch.stack(best_total_reward_list)
    best_rewards_stack = torch.stack(best_rewards_list)
    best_actions_stack = torch.stack(best_actions_list)

    # Compute average
    best_total_reward = best_total_reward_stack.mean(dim=0)
    best_rewards = best_rewards_stack.mean(dim=0)
    best_actions = best_actions_stack.mean(dim=0)

    return best_actions, best_rewards, best_total_reward


def get_best_strategy_one_simulation(env, agent, dtype, device):

    # set agent to evaluation mode
    agent.eval()

    # initialize episode
    _ = env.reset()
    agent.env = env
    best_total_reward = torch.tensor([0.0], dtype=dtype, device=device)
    done = False

    best_actions = torch.tensor([], dtype=dtype, device=device)
    best_rewards = torch.tensor([], dtype=dtype, device=device)
    best_state = torch.tensor(
        [
            item
            for sublist in env.state.get_observation().values()
            for item in (sublist if isinstance(sublist, list) else [sublist])
            if not isinstance(item, list)
        ],
        dtype=dtype,
        device=device,
    )

    while True:
        best_action = agent.act(best_state)
        best_actions = torch.cat((best_actions, best_action), dim=0)

        next_state, reward, done = env.step(best_action)
        next_state = torch.tensor(
            [
                item
                for sublist in next_state.values()
                for item in (sublist if isinstance(sublist, list) else [sublist])
                if not isinstance(item, list)
            ],
            dtype=dtype,
            device=device,
        )

        best_rewards = torch.cat(
            (best_rewards, torch.tensor([reward], dtype=dtype, device=device)), dim=0
        )
        best_total_reward += reward

        if done:
            # Handle for premature liquidations
            if best_state[1] < env.observation_space["time"].high[0]:
                reduced_time = len(best_actions)
                empty_tensor = torch.zeros(
                    size=(
                        int(env.observation_space["time"].high[0] - reduced_time + 1),
                    ),
                    dtype=dtype,
                    device=device,
                )
                best_actions = torch.cat((best_actions, empty_tensor))
                best_rewards = torch.cat((best_rewards, empty_tensor))
            break
        else:
            best_state = next_state

    # set agent back to training mode
    agent.train()

    return best_actions, best_rewards, best_total_reward


def get_optimal_strategy(env, env_agent=None):

    # Initialize lists
    optimal_rewards_list = []
    optimal_total_reward_list = []

    # Number of simulations
    n = 50

    # Simulate optimal strategy for n market simulations
    for _ in range(n):
        optimal_actions, optimal_rewards, optimal_total_reward = (
            get_optimal_strategy_one_simulation(env, env_agent)
        )
        optimal_rewards_list.append(optimal_rewards)
        optimal_total_reward_list.append(optimal_total_reward)

    # Convert list of tensors to a single NumPy array
    optimal_total_reward_stack = np.stack(optimal_total_reward_list)
    optimal_rewards_stack = np.stack(optimal_rewards_list)

    # Compute average
    optimal_total_reward = np.mean(optimal_total_reward_stack, axis=0)
    optimal_rewards = np.mean(optimal_rewards_stack, axis=0)

    return optimal_actions, optimal_rewards, optimal_total_reward


def get_optimal_strategy_one_simulation(env, env_agent):

    # env_agent is the environment assumed by optimal agent
    # if not specified, assumed that it is the same as the true environment
    if env_agent is None:
        env_agent = env

    # Define optimal agent
    optimal_agent = OptimalAgent(env_agent)

    optimal_actions = []
    optimal_rewards = []

    # Reset the environment
    state, _ = env.reset()
    done = False

    while not done:
        # Take an action based on the current state
        action = optimal_agent.act(state)

        # Step through the environment and receive feedback
        next_state, reward, done = env.step(action)

        optimal_actions.append(action)
        optimal_rewards.append(reward)

        # Update the state
        state = next_state

    return optimal_actions, np.array(optimal_rewards), np.sum(optimal_rewards)


def plot_strategy_multiple_iterations(
    iterations,
    best_actions,
    optimal_actions,
    y_label,
    file_path=None,
):
    # number of iterations to plot
    n = 5

    # select last 5 iterations and last 5 actions
    for i in range(n):
        i += 1
        plot_strategy(
            best_actions[-i],
            optimal_actions,
            y_label=y_label + "_iteration_" + str(iterations[-i]),
            file_path=file_path,
        )


def plot_strategy(best_actions, optimal_actions, y_label, file_path=None):
    SE = round(
        float(np.sum((best_actions.numpy() - np.array(optimal_actions)) ** 2)), 8
    )
    AE = round(
        float(np.sum(np.abs(best_actions.numpy() - np.array(optimal_actions)))), 8
    )

    fig, ax = plt.subplots()
    plt.gcf().set_size_inches(10, 6)

    # Plot best actions
    ax.plot(
        range(len(best_actions)),
        best_actions,
        label="Best Actions",
        marker="o",
        color="blue",
    )

    # Plot optimal actions in red
    ax.plot(
        range(len(optimal_actions)),
        optimal_actions,
        label="Optimal Actions",
        marker="x",
        color="red",
    )

    ax.set_xlabel("Time")
    ax.set_ylabel(y_label)
    ax.legend(title="Strategy")
    ax.grid()

    # Adding annotation
    plt.text(
        len(optimal_actions) - 1,  # x-coordinate of the annotation
        min(min(optimal_actions), best_actions.min()),  # y-coordinate of the annotation
        f"Squared error: {SE}\nAbsolute error: {AE}",  # Text of the annotation
        horizontalalignment="right",  # Align the text to the right
        verticalalignment="bottom",  # Align the text to the bottom
    )
    plt.show()

    if file_path is not None:
        file_name = "/plot_strategy_" + y_label + ".png"
        fig.savefig(file_path + file_name, dpi=300, bbox_inches="tight", format="png")
    else:
        plt.show()
    plt.clf()


def plot_states(states, file_path=None):

    fig, ax = plt.subplots()

    if "impact" in states[0]:
        impact = [s["impact"] for s in states]
        fig, ax = plt.subplots()
        ax.plot(range(0, len(states)), impact)
        ax.set_title("Impact")
        ax.set_xlabel("Date")
        plt.grid(True)

        if file_path is not None:
            file_name = "/plot_impact.png"
            fig.savefig(
                file_path + file_name, dpi=300, bbox_inches="tight", format="png"
            )
        else:
            plt.show()
        plt.clf()

    if "inventory" in states[0]:
        inventory = [float(s["inventory"]) for s in states]
        fig, ax = plt.subplots()
        ax.plot(range(0, len(states)), inventory)
        ax.set_title("Inventory")
        ax.set_xlabel("Time")
        plt.grid(True)
        plt.show()

        if file_path is not None:
            file_name = "/plot_inventory.png"
            fig.savefig(
                file_path + file_name, dpi=300, bbox_inches="tight", format="png"
            )
        else:
            plt.show()
        plt.clf()

    if "price" in states[0]:
        price = [float(s["price"]) for s in states]
        fig, ax = plt.subplots()
        ax.plot(range(0, len(states)), price)
        ax.set_title("Price with Impact")
        ax.set_xlabel("Time")
        plt.grid(True)
        plt.show()

        if file_path is not None:
            file_name = "/plot_price_w_impact.png"
            fig.savefig(
                file_path + file_name, dpi=300, bbox_inches="tight", format="png"
            )
        else:
            plt.show()
        plt.clf()

    if "time" in states[0]:
        time = [s["time"] for s in states]
        fig, ax = plt.subplots()
        ax.plot(range(0, len(states)), time)
        ax.set_title("Time")
        ax.set_xlabel("Time")
        plt.grid(True)
        plt.show()

        if file_path is not None:
            file_name = "/plot_time.png"
            fig.savefig(
                file_path + file_name, dpi=300, bbox_inches="tight", format="png"
            )
        else:
            plt.show()
        plt.clf()


def plot_loss(loss_values, y_label, file_path=None, log=True):
    """
    Plot the loss of a neural network across iterations.

    Parameters:
    - loss_values (list): List of loss values across iterations.
    """
    iterations = range(1, len(loss_values) + 1)

    plt.gcf().set_size_inches(10, 6)
    plt.plot(iterations, loss_values, marker="o")
    plt.title("Neural Network Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss " + y_label)
    plt.grid(True)

    if log:
        plt.yscale("log")

    if file_path is not None:
        file_name = "/plot_loss_" + y_label + ".png"
        plt.savefig(file_path + file_name, dpi=300, bbox_inches="tight", format="png")
    else:
        plt.show()

    plt.clf()


def plot_kernel(param, y_label, file_path=None):
    """
    Plot the kernel scale across episodes.

    Parameters:
    - param (list): List of kernel param across episodes.
    """
    episodes = range(1, len(param) + 1)

    plt.gcf().set_size_inches(10, 6)
    plt.plot(episodes, param, marker="o")
    plt.title("Kernel " + y_label)
    plt.xlabel("Episode")
    plt.ylabel("Kernel " + y_label)
    plt.grid(True)

    if file_path is not None:
        file_name = "/plot_kernel_" + y_label + ".png"
        plt.savefig(file_path + file_name, dpi=300, bbox_inches="tight", format="png")
    else:
        plt.show()

    plt.clf()


def plot_total_reward(
    iterations, total_best_reward, total_optimal_reward, file_path=None, log=False
):
    SE = round(float((total_optimal_reward - total_best_reward[-1]) ** 2), 8)
    AE = round(float(abs(total_optimal_reward - total_best_reward[-1])), 8)

    plt.gcf().set_size_inches(10, 6)

    plt.plot(iterations, total_best_reward, marker="o", label="best reward")
    plt.axhline(
        y=total_optimal_reward, color="r", linestyle="--", label="Optimal reward"
    )
    plt.title("Neural Network Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Cumulative rewards")
    plt.grid(True)

    # Adding annotation
    plt.text(
        max(iterations),  # x-coordinate of the annotation
        min(total_best_reward),  # y-coordinate of the annotation
        f"Squared error: {SE}\nAbsolute error: {AE}",  # Text of the annotation
        horizontalalignment="right",  # Align the text to the right
        verticalalignment="bottom",  # Align the text to the bottom
    )

    if log:
        plt.yscale("log")

    if file_path is not None:
        file_name = "/plot_total_best_reward.png"
        plt.savefig(file_path + file_name, dpi=300, bbox_inches="tight", format="png")
    else:
        plt.show()

    plt.clf()


def plot_random_actions(memory, file_path=None):
    states, actions, *_ = zip(*memory)

    # Convert to tensors
    states = torch.stack(states, dim=0)
    actions = torch.stack(actions, dim=0)

    # create input
    input = torch.cat((states, actions), dim=1)

    # Find at first time
    selected_indices = torch.nonzero(input[:, 1] == 0.0, as_tuple=False).squeeze()

    # extract action
    selected_values = actions[selected_indices]

    # Plot histogram
    if len(selected_values.numpy()) == 0:
        print("states", states)
        print("actions", actions)
        print("input", input)
        print("selected_indices", selected_indices)
        print("selected_values", selected_values)
    else:
        plt.gcf().set_size_inches(10, 6)
        plt.hist(selected_values.numpy(), bins=20, color="blue", edgecolor="black")
        plt.xlabel("Values in 4th Column")
        plt.ylabel("Frequency")
        plt.title("Histogram of 4th Column Values where 2nd Column is 0.0")

        if file_path is not None:
            file_name = "/plot_random_actions.png"
            plt.savefig(
                file_path + file_name, dpi=300, bbox_inches="tight", format="png"
            )
        else:
            plt.show()


def list_tensors_to_device(list_tensors, device):
    return [tensor.to(device) for tensor in list_tensors]


def list_tuple_tensors_to_device(list_tuple_tensors, device):
    # Iterate over each tuple in the list
    for i, memory_tuple in enumerate(list_tuple_tensors):
        # Convert each tensor in the tuple to the specified device
        converted_tensors = tuple(tensor.to(device) for tensor in memory_tuple)
        # Update the memory tuple with the converted tensors
        list_tuple_tensors[i] = converted_tensors

    return list_tuple_tensors


def save_object(output_path, name, object):

    output_path = output_path + "/outputs_" + name + ".pkl"
    with open(output_path, "wb") as f:
        pickle.dump(
            {name: object},
            f,
        )


def load_object(output_path, name):

    output_path = output_path + "/outputs_" + name + ".pkl"
    with open(output_path, "rb") as f:
        saved_data = pickle.load(f)

    return saved_data[name]
