from copy import deepcopy

import torch
from jaxtyping import Float
from src.policies.actor import GaussianActorPolicy
from src.policies.critic import QNetwork
from torch import nn


class SACCritic(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int,
        dtype: torch.dtype,
        device: str,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.observation_dim = observation_dim

        self.q_network_1 = QNetwork(
            hidden_dim=hidden_dim,
            observation_dim=observation_dim,
            action_dim=action_dim,
            dtype=dtype,
            device=device,
        )

        self.q_network_2 = QNetwork(
            hidden_dim=hidden_dim,
            observation_dim=observation_dim,
            action_dim=action_dim,
            dtype=dtype,
            device=device,
        )

    def forward(
        self,
        observation: Float[torch.Tensor, "... observation_dim"],
        action: Float[torch.Tensor, "... action_dim"],
    ) -> Float[torch.Tensor, "... 1"]:
        return (
            self.q_network_1(observation, action),
            self.q_network_2(observation, action),
        )

    def min_q(
        self,
        observation: Float[torch.Tensor, "... observation_dim"],
        action: Float[torch.Tensor, "... action_dim"],
    ) -> Float[torch.Tensor, "... 1"]:
        q_value_1, q_value_2 = self.forward(observation, action)

        q_min_value = torch.minimum(q_value_1, q_value_2)

        return q_min_value


class SACAgent(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int,
        min_log_std: int,
        max_log_std: int,
        dtype: torch.dtype,
        device: str,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

        self.actor = GaussianActorPolicy(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            min_log_std=self.min_log_std,
            max_log_std=self.max_log_std,
            dtype=dtype,
            device=device,
        )

        self.critic = SACCritic(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            dtype=dtype,
            device=device,
        )

        self.target_critic = deepcopy(self.critic)
        self.target_critic.requires_grad_(False)

    @torch.no_grad()
    def predict(
        self,
        observations: Float[torch.Tensor, "... obs_dim"],
        *,
        deterministic: bool = False,
    ) -> Float[torch.Tensor, "... action_dim"]:
        return self.actor.predict(
            observations,
            deterministic=deterministic,
        )

    @torch.no_grad()
    def update_target_network(self, tau: float) -> None:
        for param, target_param in zip(
            self.critic.parameters(),
            self.target_critic.parameters(),
            strict=True,
        ):
            target_param.lerp_(param, tau)
