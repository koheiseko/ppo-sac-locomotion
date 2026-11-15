import torch
from jaxtyping import Float, Int
from src.policies.utils import layer_init
from torch import nn
from torch.distributions import Normal


class GaussianActorPolicy(nn.Module):
    def __init__(
        self,
        action_dim: Int,
        observation_dim: Int,
        hidden_dim: Int,
        dtype: torch.dtype,
        device: str,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
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
                    self.hidden_dim,
                    self.action_dim,
                    dtype=self.dtype,
                    device=self.device,
                ),
                std=0.01,
            ),
        )
        self.logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_distribution(
        self, observation: Float[torch.Tensor, "... obs_dim"]
    ) -> torch.distributions.Distribution:
        action_mean = self.network(observation)
        action_logstd = self.logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)

        return Normal(action_mean, action_std)

    def get_action(
        self,
        observation: Float[torch.Tensor, "... obs_dim"],
        deterministic: bool = False,
    ) -> Float[torch.Tensor, "... action_dim"]:
        if deterministic:
            return self.network(observation)

        distribution = self.get_distribution(observation)

        return distribution.sample()

    def evaluate_actions(
        self,
        observation: Float[torch.Tensor, "... observation_dim"],
        action: Float[torch.Tensor, "... action_dim"] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.get_distribution(observation)
        if action is None:
            action = distribution.sample()

        log_prob = distribution.log_prob(action).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        return action, log_prob, entropy
