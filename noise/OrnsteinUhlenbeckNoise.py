import torch


# Ornstein-Uhlenbeck noise process
class OrnsteinUhlenbeckNoise:
    def __init__(self, size, mu=0, theta=0.15, sigma=1, device=torch.device("cpu")):
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.device = device
        self.reset()

    def reset(self):
        # Initialize the noise state using PyTorch
        self.state = torch.ones(self.size) * self.mu

        # Move the tensor to the selected device
        self.state = self.state.to(self.device)

    def sample(self):

        # Get random gaussian noise
        noise = torch.randn(self.size)
        noise = noise.to(self.device)

        # Update the noise state using the Ornstein-Uhlenbeck process
        dx = self.theta * (self.mu - self.state) + self.sigma * noise
        self.state += dx

        return self.state
