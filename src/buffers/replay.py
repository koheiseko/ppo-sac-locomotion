import torch


class ReplayBuffer:
    def __init__(
        self,
        buffer_size,
        n_envs,
        observation_dim,
        action_dim,
        dtype,
        device,
    ):
        self.dtype = dtype
        self.device = device

        self.buffer_size = buffer_size
        self.n_envs = n_envs

        self.observations = torch.empty(
            (self.buffer_size, self.n_envs) + (observation_dim,),
            dtype=self.dtype,
            device=self.device,
        )
        self.next_observations = torch.empty(
            (self.buffer_size, self.n_envs) + (observation_dim,),
            dtype=self.dtype,
            device=self.device,
        )
        self.actions = torch.empty(
            (self.buffer_size, self.n_envs) + (action_dim,),
            dtype=self.dtype,
            device=self.device,
        )
        self.rewards = torch.empty(
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
        self.full = False

    def reset(self) -> None:
        self.pos = 0
        self.full = False

    @torch.no_grad()
    def add(
        self,
        observation,
        next_observation,
        action,
        reward,
        done,
    ) -> None:
        self.observations[self.pos] = observation
        self.next_observations[self.pos] = next_observation
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done

        self.pos += 1

        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size):
        if self.full:
            batch_ids = torch.randint(0, self.buffer_size, (batch_size,))
        else:
            batch_ids = torch.randint(0, self.pos, (batch_size,))

        return self._get_samples(batch_ids)

    def _get_samples(self, batch_ids):
        env_ids = torch.randint(0, self.n_envs, size=(len(batch_ids),))

        return {
            "observations": self.observations[batch_ids, env_ids, :],
            "next_observations": self.next_observations[batch_ids, env_ids, :],
            "actions": self.actions[batch_ids, env_ids, :],
            "rewards": self.rewards[batch_ids, env_ids],
            "dones": self.dones[batch_ids, env_ids],
        }
