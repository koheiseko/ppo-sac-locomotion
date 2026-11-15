import gymnasium as gym
import numpy as np
import torch


def evaluate(
    agent: torch.nn.Module,
    env: gym.Env,
    n_episodes: int = 5,
    seed: int = 42,
    deterministic: bool = False,
) -> tuple[float, float, float]:
    device = next(agent.parameters()).device

    was_training = agent.training
    agent.eval()

    returns = []
    lengths = []

    try:
        for _ in range(n_episodes):
            obs, _ = env.reset(seed=seed)

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

                    action = agent.get_action(
                        obs,
                        deterministic=deterministic,
                    )

                    action = action.squeeze(0).cpu().numpy()

                    obs, reward, terminated, truncated, _ = env.step(action)

                    episode_return += float(reward)
                    episode_length += 1

            returns.append(episode_return)
            lengths.append(episode_length)

    finally:
        env.close()
        agent.train(was_training)

    return (
        float(np.mean(returns)),
        float(np.std(returns)),
        float(np.mean(lengths)),
    )
