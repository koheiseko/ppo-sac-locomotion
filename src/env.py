from collections.abc import Callable
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym


@dataclass
class HopperConfig:
    id: str = "Hopper-v5"

    forward_reward_weight: float = 1.0
    ctrl_cost_weight: float = 1e-3
    healthy_reward: float = 1.0

    terminate_when_unhealthy: bool = True

    healthy_state_range: tuple[float, float] = (-100.0, 100.0)
    healthy_z_range: tuple[float, float] = (0.7, float("inf"))
    healthy_angle_range: tuple[float, float] = (-0.2, 0.2)

    reset_noise_scale: float = 5e-3

    exclude_current_positions_from_observation: bool = True


@dataclass
class HalfCheetahConfig:
    id: str = "HalfCheetah-v5"

    forward_reward_weight: float = 1.0
    ctrl_cost_weight: float = 0.1

    reset_noise_scale: float = 0.1

    exclude_current_positions_from_observation: bool = True


@dataclass
class Walker2DConfig:
    id: str = "Walker2d-v5"

    forward_reward_weight: float = 1.0
    ctrl_cost_weight: float = 1e-3
    healthy_reward: float = 1.0

    terminate_when_unhealthy: bool = True

    healthy_z_range: tuple[float, float] = (0.8, 2.0)
    healthy_angle_range: tuple[float, float] = (-1.0, 1.0)

    reset_noise_scale: float = 5e-3

    exclude_current_positions_from_observation: bool = True


@dataclass
class HumanoidConfig:
    id: str = "Humanoid-v5"

    forward_reward_weight: float = 1.25
    ctrl_cost_weight: float = 0.1
    contact_cost_weight: float = 5e-7
    contact_cost_range: tuple[float, float] = (float("-inf"), 10.0)
    healthy_reward: float = 5.0

    terminate_when_unhealthy: bool = True

    healthy_z_range: tuple[float, float] = (1.0, 2.0)

    reset_noise_scale: float = 1e-2

    exclude_current_positions_from_observation: bool = True

    include_cinert_in_observation: bool = True
    include_cvel_in_observation: bool = True
    include_qfrc_actuator_in_observation: bool = True
    include_cfrc_ext_in_observation: bool = True


def make_env(
    config: Any,
    render_mode: str | None = None,
    record_video: bool = False,
    video_dir: str | Path | None = None,
    name_prefix: str = "rl-video",
    episode_trigger: Callable[[int], bool] | None = None,
    seed: int = 42,
):
    def thunk():
        env_kwargs = asdict(config) if is_dataclass(config) else dict(config)
        effective_render_mode = "rgb_array" if record_video else render_mode

        env = gym.make(render_mode=effective_render_mode, **env_kwargs)

        if record_video:
            if video_dir is None:
                raise ValueError(
                    "video_dir deve ser informado quando record_video=True."
                )
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=str(video_dir),
                name_prefix=name_prefix,
                episode_trigger=episode_trigger,
                disable_logger=True,
            )

        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)

        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        return env

    return thunk
