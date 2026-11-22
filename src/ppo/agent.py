import torch
from jaxtyping import Float
from src.policies.actor import GaussianActorPolicy
from src.policies.critic import ValueNetwork
from torch import nn


class PPOAgent(nn.Module):
    def __init__(
        self,
        action_dim: int,
        observation_dim: int,
        hidden_dim: int,
        dtype: torch.dtype,
        device: str,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.dtype = dtype
        self.device = device

        self.critic = ValueNetwork(
            observation_dim=self.observation_dim,
            hidden_dim=self.hidden_dim,
            dtype=self.dtype,
            device=self.device,
        )
        self.actor = GaussianActorPolicy(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            min_log_std=-5.0,
            max_log_std=2.0,
            dtype=self.dtype,
            device=self.device,
        )

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

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        pre_tanh_actions: torch.Tensor,
    ) -> torch.Tensor:
        return self.actor.log_prob_from_pre_tanh(observations, pre_tanh_actions)
