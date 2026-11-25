from collections.abc import Callable
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from src.env import make_env


@torch.no_grad()
def evaluate(
    agent: torch.nn.Module,
    env: gym.Env,
    n_episodes: int = 5,
    seed: int = 42,
) -> tuple[float, float, float]:
    device = next(agent.parameters()).device

    was_training = agent.training
    agent.eval()

    returns = []
    lengths = []

    try:
        for i in range(n_episodes):
            obs, _ = env.reset(seed=seed + i)

            episode_return = 0.0
            episode_length = 0

            terminated = False
            truncated = False

            with torch.inference_mode():
                while not (terminated or truncated):
                    obs = torch.as_tensor(
                        obs,
                        dtype=torch.float32,
                        device=device,
                    ).unsqueeze(0)

                    action = agent.predict(obs)

                    action = action.squeeze(0).cpu().numpy()

                    obs, reward, terminated, truncated, _ = env.step(action)

                    episode_return += float(reward)
                    episode_length += 1

            returns.append(episode_return)
            lengths.append(episode_length)

    finally:
        agent.train(was_training)

    return (
        float(np.mean(returns)),
        float(np.std(returns)),
        float(np.mean(lengths)),
    )


def record_eval_video(
    config: any,
    policy_fn: Callable[[np.ndarray], np.ndarray],
    video_dir: str | Path = "videos/eval",
    name_prefix: str = "eval",
    num_episodes: int = 5,
    seed: int = 42,
) -> list[float]:
    env = make_env(
        config=config,
        record_video=True,
        video_dir=video_dir,
        name_prefix=name_prefix,
        episode_trigger=lambda ep: True,
        seed=seed,
    )()

    episode_returns: list[float] = []

    try:
        with torch.inference_mode():
            for ep in range(num_episodes):
                obs, _ = env.reset(seed=seed + ep)
                done = False
                ep_return = 0.0

                while not done:
                    action = policy_fn(obs)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    ep_return += float(reward)
                    done = terminated or truncated

                episode_returns.append(ep_return)
    finally:
        env.close()

    return episode_returns
