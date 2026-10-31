"""Load the locally converted Khan_mini asset as an mjlab entity."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ASSET_DIR = ROOT / "assets" / "robots" / "khan_mini"


def load_profile(asset_dir: str | Path | None = None) -> dict:
    directory = Path(asset_dir) if asset_dir is not None else DEFAULT_ASSET_DIR
    path = directory / "profile.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}; run uv run python scripts/setup/build_khan_mini.py first"
        )
    profile = json.loads(path.read_text())
    if profile.get("robot") != "khan_mini" or profile.get("num_joints") != 27:
        raise ValueError(f"Not a 27-joint Khan_mini profile: {path}")
    return profile


def get_khan_mini_spec(asset_dir: str | Path | None = None) -> mujoco.MjSpec:
    directory = Path(asset_dir) if asset_dir is not None else DEFAULT_ASSET_DIR
    return mujoco.MjSpec.from_file(str(directory / "robot.xml"))


def get_khan_mini_cfg(asset_dir: str | Path | None = None) -> EntityCfg:
    """Fresh configuration with the PD gains and pose from Khan_mini_CFG.

    Velocity limits remain in profile.json: MuJoCo's built-in position servos do
    not implement Isaac Lab's velocity_limit_sim constraint.
    """
    directory = Path(asset_dir) if asset_dir is not None else DEFAULT_ASSET_DIR
    profile = load_profile(directory)
    actuators = tuple(
        BuiltinPositionActuatorCfg(
            target_names_expr=(name,),
            stiffness=profile["stiffness"][name],
            damping=profile["damping"][name],
            effort_limit=profile["effort_limit"][name],
            armature=profile["armature"][name],
        )
        for name in profile["joint_names"]
    )
    # Collision masks and groups are already stored in robot.xml, including
    # self-collisions. Do not apply the retargeted repository's no-self mask.
    return EntityCfg(
        spec_fn=partial(get_khan_mini_spec, directory),
        init_state=EntityCfg.InitialStateCfg(
            pos=tuple(profile["initial_pos"]),
            joint_pos=profile["default_joint_pos"],
            joint_vel={".*": 0.0},
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=actuators,
            soft_joint_pos_limit_factor=profile["soft_joint_pos_limit_factor"],
        ),
    )


def get_khan_mini_action_cfg(asset_dir: str | Path | None = None, noise_std: float = 0.1):
    """Use SuperMimic's joint-target action interface with Khan's stored scale.

    q_target = q_home + action_scale * (action + execution_noise).
    The training environment selects the same interface via the robot registry.
    """
    from supermimic.envs.actions import NoisyJointPositionActionCfg

    return NoisyJointPositionActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        scale=load_profile(asset_dir)["action_scale"],
        use_default_offset=True,
        noise_std=noise_std,
    )
