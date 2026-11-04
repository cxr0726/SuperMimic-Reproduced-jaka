"""All training configuration in one dataclass tree (tyro-friendly)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from supermimic.imagination.policy_trainer import PolicyTrainerCfg
from supermimic.observation_layout import (
    goal_dimension,
    resolve_observation_layout,
    resolve_observation_sizes,
)
from supermimic.policy.nets import CriticCfg, PolicyCfg
from supermimic.robots.registry import RobotName
from supermimic.world_model.wm_trainer import WMTrainerCfg
from supermimic.world_model.world_model import WorldModelCfg


@dataclass
class DataCfg:
    npz_dir: Path | None = None  # None selects the robot's default dataset directory
    clips: list[str] | None = None  # None = all npz in npz_dir
    limit: int | None = None
    manifest: Path | None = None  # csv with name,keep (filter_bones_seed.py output)
    library: Path | None = None  # packed compact library (.pt from build_library.py); overrides npz_dir


@dataclass
class EnvCfg:
    num_envs: int = 4096
    episode_length_s: float = 10.0
    action_noise_std: float = 0.1
    obs_noise: bool = True
    reset_noise: bool = True
    domain_randomization: bool = True
    actuator_delay: bool | None = None  # Khan on, G1 off; nominal evaluation disables it
    actuator_delay_max_ms: float = 20.0
    reset_root_z_offset: float | None = None  # Actual robot only: Khan .05 m, G1 zero
    sampling_mode: str = "return"
    replay_capacity: int = 1000
    future_frames: int | None = None  # robot/layout default: G1=10, Khan Isaac=5
    observation_layout: Literal["auto", "supermimic", "isaac_khan"] = "auto"
    compact_library: bool = False
    anchor_obs: bool = False


@dataclass
class ScheduleCfg:
    total_iterations: int = 320_000  # env steps (plan D12 reading A)
    updates_per_env_step: float = 0.4  # UTD
    policy_updates_per_wm_update: int = 1
    warmup_iterations: int = 40  # env steps before the first update (fills k-step windows)
    lr_final_ratio: float = 0.03
    lr_decay_fraction: float = 250_000 / 320_000
    log_interval: int = 50  # iterations
    checkpoint_interval: int = 4000  # iterations
    max_hours: float | None = None
    stop_after_iterations: int | None = None  # execution limit; does not shorten the LR schedule


@dataclass
class TrainConfig:
    run_name: str = "debug"
    log_root: Path = Path("logs")
    seed: int = 0
    device: str = "cuda:0"
    robot: RobotName = "g1"
    data: DataCfg = field(default_factory=DataCfg)
    env: EnvCfg = field(default_factory=EnvCfg)
    world_model: WorldModelCfg = field(default_factory=WorldModelCfg)
    wm_trainer: WMTrainerCfg = field(default_factory=WMTrainerCfg)
    policy: PolicyCfg = field(default_factory=lambda: PolicyCfg(history=None))
    critic: CriticCfg = field(default_factory=CriticCfg)
    policy_trainer: PolicyTrainerCfg = field(default_factory=PolicyTrainerCfg)
    schedule: ScheduleCfg = field(default_factory=ScheduleCfg)
    resume: Path | None = None


def resolve_training_observations(cfg: TrainConfig) -> None:
    """Resolve robot defaults before constructing networks or saving configuration.

    None denotes an unspecified window, so an explicit incompatible Isaac window
    is rejected rather than silently replaced. Network hidden layers and all
    optimization settings remain independent of the observation layout.
    """
    from supermimic.robots.registry import get_robot_spec, resolve_robot_environment

    robot = get_robot_spec(cfg.robot)
    from supermimic.sim_context import context_kind_for_robot

    cfg.world_model.state_spec.context_layout = context_kind_for_robot(cfg.robot)
    settings = resolve_robot_environment(
        cfg.robot, actuator_delay=cfg.env.actuator_delay,
        actuator_delay_max_ms=cfg.env.actuator_delay_max_ms,
        reset_root_z_offset=cfg.env.reset_root_z_offset,
    )
    for name, value in settings.items():
        setattr(cfg.env, name, value)
    layout = resolve_observation_layout(cfg.robot, cfg.env.observation_layout)
    future, history = resolve_observation_sizes(
        layout, cfg.env.future_frames, cfg.policy.history, cfg.env.anchor_obs,
    )
    cfg.env.observation_layout = layout
    cfg.env.future_frames = future
    cfg.policy.observation_layout = layout
    cfg.policy.future_frames = future
    cfg.policy.history = history
    cfg.policy.num_joints = robot.num_joints
    cfg.policy.num_tracked_bodies = len(robot.tracked_bodies)
    cfg.policy.goal_dim = goal_dimension(layout, robot.num_joints, future, cfg.env.anchor_obs)
    if layout == "isaac_khan":
        # The environment and analytic observations already apply Isaac's scales.
        cfg.policy.obs_term_scale = (1.0,) * 6
        cfg.policy.goal_term_scale = (1.0,) * 3
    cfg.policy_trainer.observation_layout = layout
    cfg.policy_trainer.anchor_obs = cfg.env.anchor_obs
    cfg.policy_trainer.future_frames = future
    cfg.policy_trainer.history = history
