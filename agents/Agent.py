from abc import ABC, abstractmethod


class Agent(ABC):
    """Abstract class representing an agent."""

    def __init__(self, name=None):
        """Initializes the agent with a name (optional for subclasses).

        Args:
        name: (Optional) The name of the agent.
        """
        self.name = name

    @abstractmethod
    def act(self, state):
        """Abstract method representing the agent's action.
        Subclasses must implement this method.

        Args:
            state (torch.tensor): Current state.
        """
        pass
