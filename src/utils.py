import gymnasium as gym
import numpy as np
import torch


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
