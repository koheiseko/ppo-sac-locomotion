from dataclasses import dataclass

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
    config,
    render_mode: str | None = None,
    record_video: bool | None = None,
    video_dir: str | None = None,
    seed: int | None = 42,
):
    def thunk():
        env = gym.make(render_mode=render_mode, **config)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)

        if record_video:
            env = gym.wrappers.RecordVideo(env, video_folder=str(video_dir))

        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        return env

    return thunk
