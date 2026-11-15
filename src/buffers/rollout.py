import torch
from jaxtyping import Bool, Float


class RolloutBuffer:
    def __init__(
        self,
        buffer_size: int,
        n_envs: int,
        observation_dim: int,
        action_dim: int,
        gamma: float,
        gae_lambda: float,
        dtype: torch.dtype,
        device: str,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.dtype = dtype
        self.device = device

        self.buffer_size = buffer_size
        self.n_envs = n_envs

        self.observations = torch.empty(
            (self.buffer_size, self.n_envs, observation_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.actions = torch.empty(
            (self.buffer_size, self.n_envs, action_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.action_log_probs = torch.empty(
            (self.buffer_size, self.n_envs),
            dtype=self.dtype,
            device=self.device,
        )
        self.rewards = torch.empty(
            (self.buffer_size, self.n_envs),
            dtype=self.dtype,
            device=self.device,
        )
        self.values = torch.empty(
            (self.buffer_size, self.n_envs),
            dtype=self.dtype,
            device=self.device,
        )
        self.dones = torch.empty(
            (self.buffer_size, self.n_envs),
            dtype=torch.bool,
            device=self.device,
        )

        self.pos = 0

    def reset(self) -> None:
        self.pos = 0

    def add(
        self,
        observation: Float[torch.Tensor, "... observation_dim"],
        action: Float[torch.Tensor, "... action_dim"],
        action_log_prob: Float[torch.Tensor, "..."],
        reward: Float[torch.Tensor, "..."],
        value: Float[torch.Tensor, "..."],
        done: Bool[torch.Tensor, "..."],
    ) -> None:
        self.observations[self.pos] = observation
        self.actions[self.pos] = action
        self.action_log_probs[self.pos] = action_log_prob
        self.rewards[self.pos] = reward
        self.values[self.pos] = value.reshape(-1)
        self.dones[self.pos] = done

        self.pos += 1

    def compute_gae(
        self, last_value: Float[torch.Tensor, "..."]
    ) -> tuple[Float[torch.Tensor, "..."], Float[torch.Tensor, "..."]]:
        last_value = last_value.reshape(-1)

        next_values = torch.cat(
            [self.values[1:], last_value.unsqueeze(0)], dim=0
        )

        next_non_terminal = (~self.dones).float()

        delta = (
            self.rewards
            + self.gamma * next_values * next_non_terminal
            - self.values
        )

        advantages = torch.zeros_like(self.rewards)
        last_gae = torch.zeros_like(last_value)

        for t in reversed(range(self.buffer_size)):
            last_gae = (
                delta[t]
                + self.gamma * self.gae_lambda * next_non_terminal[t] * last_gae
            )

            advantages[t] = last_gae

        returns = advantages + self.values

        return advantages, returns
