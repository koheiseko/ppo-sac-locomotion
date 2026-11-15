import torch
from jaxtyping import Float
from src.policies.actor import GaussianActorPolicy
from src.policies.critc import ValueNetwork
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
            dtype=self.dtype,
            device=self.device,
        )

    def get_value(
        self, observation: Float[torch.Tensor, "... observation_dim"]
    ) -> Float[torch.Tensor, "... 1"]:
        return self.critic(observation)

    def get_action(
        self,
        observation: Float[torch.Tensor, "... observation_dim"],
        deterministic: bool = False,
    ) -> Float[torch.Tensor, "... action_dim"]:
        return self.actor.get_action(observation, deterministic)

    def get_action_and_value(
        self,
        observation: Float[torch.Tensor, "... observation_dim"],
        action: Float[torch.Tensor, "... action_dim"] | None = None,
    ) -> tuple[
        Float[torch.Tensor, "... action_dim"],
        Float[torch.Tensor, "... action_dim"],
        Float[torch.Tensor, "... action_dim"],
        Float[torch.Tensor, "... 1"],
    ]:
        action, log_prob, entropy = self.actor.evaluate_actions(
            observation, action
        )
        value = self.critic(observation)

        return (
            action,
            log_prob,
            entropy,
            value,
        )
