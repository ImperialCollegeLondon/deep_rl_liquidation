import torch
from torch import nn


class FCN(nn.Module):
    def __init__(
        self,
        input_dim=4,
        feature_dim=20,
        num_layers=4,
        output_dim=1,
        activation=torch.nn.ReLU(),
    ):
        super().__init__()

        # map u to y -- REVIEWED
        fcls = list()
        fcls.append(nn.Linear(input_dim, feature_dim, bias=True))
        for k in range(num_layers - 2):
            fcls.append(nn.Linear(feature_dim, feature_dim, bias=True))
        fcls.append(nn.Linear(feature_dim, output_dim, bias=True))
        self.fc = nn.ModuleList(fcls)

        self.activation = activation

    def forward(self, x):
        x = self.activation(self.fc[0](x))
        for linear in self.fc[1:-1]:
            x = self.activation(linear(x))
        x = self.fc[-1](x)
        return x


class ActorFCN(nn.Module):
    def __init__(
        self,
        input_dim=4,
        feature_dim=20,
        num_layers=4,
        output_dim=1,
        activation=torch.sigmoid,
        transform=torch.sigmoid,
        inv_transform=torch.special.logit,
    ):
        super().__init__()

        # map u to y -- REVIEWED
        fcls = list()
        fcls.append(nn.Linear(input_dim, feature_dim, bias=True))
        for k in range(num_layers - 2):
            fcls.append(nn.Linear(feature_dim, feature_dim, bias=True))
        fcls.append(nn.Linear(feature_dim, output_dim, bias=True))
        self.fc = nn.ModuleList(fcls)

        self.activation = activation
        self.transform = transform
        self.inv_transform = inv_transform

    def forward(self, x):
        x = self.activation(self.fc[0](x))
        for linear in self.fc[1:-1]:
            x = self.activation(linear(x))
        x = self.transform(self.fc[-1](x))
        return x
