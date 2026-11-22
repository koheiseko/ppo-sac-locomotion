from collections.abc import Iterator

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
        self.pre_tanh_actions = torch.empty_like(self.actions)
        self.action_log_probs = torch.empty(
            (self.buffer_size, self.n_envs), dtype=dtype, device=device
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

        self.reset()

    def reset(self) -> None:
        self.pos = 0
        self.advantages = None
        self.returns = None

    @torch.no_grad()
    def add(
        self,
        observation: Float[torch.Tensor, "... observation_dim"],
        action: Float[torch.Tensor, "... action_dim"],
        action_log_prob: Float[torch.Tensor, "..."],
        reward: Float[torch.Tensor, "..."],
        value: Float[torch.Tensor, "..."],
        done: Bool[torch.Tensor, "..."],
        *,
        pre_tanh_action: torch.Tensor,
    ) -> None:
        self.observations[self.pos] = observation
        self.actions[self.pos] = action
        self.pre_tanh_actions[self.pos] = pre_tanh_action
        self.action_log_probs[self.pos] = action_log_prob
        self.rewards[self.pos] = reward
        self.values[self.pos] = value.reshape(-1)
        self.dones[self.pos] = done
        self.pos += 1

    def iter_minibatches(
        self, mini_batch_size: int
    ) -> Iterator[dict[str, torch.Tensor]]:
        advantages = self.advantages
        returns = self.returns

        if advantages is None or returns is None:
            raise RuntimeError("Calcule o GAE antes de iterar pelo rollout.")

        permutation = torch.randperm(self.pos * self.n_envs, device=self.device)

        for flat_ids in permutation.split(mini_batch_size):
            step_ids = flat_ids // self.n_envs
            env_ids = flat_ids % self.n_envs
            indices = (step_ids, env_ids)

            yield {
                "observations": self.observations[indices],
                "actions": self.actions[indices],
                "pre_tanh_actions": self.pre_tanh_actions[indices],
                "old_log_probs": self.action_log_probs[indices],
                "advantages": advantages[indices],
                "returns": returns[indices],
            }

    @torch.no_grad()
    def compute_gae(
        self, last_value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pos == 0:
            raise RuntimeError(
                "Colete pelo menos um passo antes de calcular o GAE."
            )

        values = self.values[: self.pos]
        rewards = self.rewards[: self.pos]
        continues = (~self.dones[: self.pos]).to(values.dtype)

        last_value = last_value.reshape(self.n_envs)
        next_values = torch.cat(
            [values[1:], last_value.unsqueeze(0)],
            dim=0,
        )
        deltas = rewards + self.gamma * next_values * continues - values

        advantages = torch.empty_like(rewards)
        last_gae = torch.zeros_like(last_value)

        for step in reversed(range(self.pos)):
            last_gae = (
                deltas[step]
                + self.gamma * self.gae_lambda * continues[step] * last_gae
            )
            advantages[step] = last_gae

        self.advantages = advantages
        self.returns = advantages + values
