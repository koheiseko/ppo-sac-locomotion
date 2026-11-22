import torch
from jaxtyping import Float
from src.policies.utils import layer_init
from torch import nn


class ValueNetwork(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        hidden_dim: int,
        dtype: torch.dtype,
        device: str,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.observation_dim = observation_dim
        self.dtype = dtype
        self.device = device

        self.network = nn.Sequential(
            layer_init(
                nn.Linear(
                    self.observation_dim,
                    self.hidden_dim,
                    dtype=self.dtype,
                    device=self.device,
                )
            ),
            nn.Tanh(),
            layer_init(
                nn.Linear(
                    self.hidden_dim,
                    self.hidden_dim,
                    dtype=self.dtype,
                    device=self.device,
                )
            ),
            nn.Tanh(),
            layer_init(
                nn.Linear(
                    self.hidden_dim, 1, dtype=self.dtype, device=self.device
                ),
                std=1.0,
            ),
        )

    def forward(
        self, observation: Float[torch.Tensor, "... observation_dim"]
    ) -> Float[torch.Tensor, "... 1"]:
        return self.network(observation).squeeze(-1)


class QNetwork(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        observation_dim: int,
        action_dim: int,
        dtype: torch.dtype,
        device: str,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.dtype = dtype
        self.device = device

        self.network = nn.Sequential(
            layer_init(
                nn.Linear(
                    self.observation_dim + self.action_dim,
                    self.hidden_dim,
                    dtype=self.dtype,
                    device=self.device,
                )
            ),
            nn.Tanh(),
            layer_init(
                nn.Linear(
                    self.hidden_dim,
                    self.hidden_dim,
                    dtype=self.dtype,
                    device=self.device,
                )
            ),
            nn.Tanh(),
            layer_init(
                nn.Linear(
                    self.hidden_dim, 1, dtype=self.dtype, device=self.device
                ),
                std=1.0,
            ),
        )

    def forward(
        self,
        observation: Float[torch.Tensor, "... observation_dim"],
        action: Float[torch.Tensor, "... action_dim"],
    ) -> Float[torch.Tensor, "... 1"]:
        x = torch.cat([observation, action], dim=1)

        return self.network(x).squeeze(-1)
