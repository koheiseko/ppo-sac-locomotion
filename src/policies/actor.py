import math

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import nn
from torch.distributions import Normal


class GaussianActorPolicy(nn.Module):
    def __init__(
        self,
        action_dim: Int,
        observation_dim: Int,
        min_log_std,
        max_log_std,
        hidden_dim: Int,
        dtype: torch.dtype,
        device: str,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.dtype = dtype
        self.device = device

        self.network = nn.Sequential(
            nn.Linear(
                observation_dim,
                hidden_dim,
                dtype=self.dtype,
                device=self.device,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim, hidden_dim, dtype=self.dtype, device=self.device
            ),
            nn.ReLU(),
        )

        self.mean = nn.Linear(
            hidden_dim,
            action_dim,
            dtype=self.dtype,
            device=self.device,
        )

        self.log_std = nn.Linear(
            self.hidden_dim,
            self.action_dim,
            dtype=self.dtype,
            device=self.device,
        )

    def forward(
        self,
        observation: Float[torch.Tensor, "... observations_dim"],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.network(observation)

        mean = self.mean(features)
        log_std = self.log_std(features).clamp(
            self.min_log_std, self.max_log_std
        )

        return mean, log_std

    def get_distribution(
        self,
        observation: Float[torch.Tensor, "... obs_dim"],
    ) -> Normal:
        mean, log_std = self.forward(observation)

        return Normal(mean, log_std.exp())

    @staticmethod
    def _squashed_log_prob(
        distribution: Normal,
        pre_tanh_action: torch.Tensor,
    ) -> torch.Tensor:
        u = pre_tanh_action
        log_det = 2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))
        return (distribution.log_prob(u) - log_det).sum(dim=-1)

    def sample(
        self,
        observation: torch.Tensor,
        *,
        reparameterize: bool,
    ):
        distribution = self.get_distribution(observation)
        pre_tanh_action = (
            distribution.rsample() if reparameterize else distribution.sample()
        )

        action = torch.tanh(pre_tanh_action)
        log_prob = self._squashed_log_prob(distribution, pre_tanh_action)

        return action, pre_tanh_action, log_prob

    def log_prob_from_pre_tanh(
        self,
        observation: torch.Tensor,
        pre_tanh_action: torch.Tensor,
    ) -> torch.Tensor:
        distribution = self.get_distribution(observation)
        return self._squashed_log_prob(distribution, pre_tanh_action)

    def predict(
        self,
        observation: Float[torch.Tensor, "... obs_dim"],
        deterministic: bool = True,
    ) -> Float[torch.Tensor, "... action_dim"]:
        distribution = self.get_distribution(observation)

        u = distribution.mean if deterministic else distribution.sample()

        return torch.tanh(u)
