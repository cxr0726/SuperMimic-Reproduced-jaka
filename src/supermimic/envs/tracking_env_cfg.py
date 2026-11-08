"""SuperMimic tracking environment on mjlab (plan Phase 2.3).

Differences from mjlab's BeyondMimic re-implementation (``mjlab.tasks.tracking``):
  * observations: G1 uses paper Table A.1 with 10-frame proprioceptive history;
    Khan uses the Isaac 155-d goal and 465-d proprioception, with five future
    orientations and five sensor-history frames in term-major order;
  * action: ``NoisyJointPositionAction`` (clean action observed, noise only executed);
  * multi-clip command with return-based reset sampling and clip-end time-outs;
  * rewards: self-collision term removed, joint-limit weight -2; Khan additionally
    tracks the two reference neck joint positions and velocities;
  * terminations: pelvis height / tilt vs reference, no end-effector term, clip-end time-out;
  * events: no push; friction and encoder bias re-sampled per episode; COM at startup;
    Khan also randomizes PD gains and ankle armature per episode, and waist mass at startup;
  * a ``ReplayRecorder`` streams transitions into a GPU replay buffer.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs import mdp as base_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.tracking import mdp as track_mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from supermimic.data.recorder import ReplayRecorderCfg
from supermimic.envs.actions import NoisyJointPositionActionCfg
from supermimic.envs.mdp import observations as sm_obs
from supermimic.envs.mdp import rewards as sm_rewards
from supermimic.envs.mdp import terminations as sm_term
from supermimic.envs.mdp.commands import MultiClipMotionCommandCfg
from supermimic.envs.mdp.rewards import TrackingRewardCfg
from supermimic.envs.partial_constants import khan_joint_armature
from supermimic.observation_layout import (
    ObservationLayoutOption,
    resolve_observation_layout,
    resolve_observation_sizes,
)
from supermimic.robots.registry import (
    G1_TRACKED_BODIES,
    RobotName,
    get_robot_spec,
    resolve_robot_environment,
)
from supermimic.sim_context import KHAN_ANKLE_JOINTS


class QuietSimulationCfg(SimulationCfg):
    """SimulationCfg that disables MuJoCo Warp's per-step overflow printf (a kernel build-time constant,
    so it must be set before the CUDA graphs are captured)."""

    def apply_wp_opt(self, wp_opt) -> None:
        super().apply_wp_opt(wp_opt)
        wp_opt.warn_overflow = 0


# Compatibility exports for existing G1 diagnostics; runtime selection is below.
TRACKED_BODIES = G1_TRACKED_BODIES
ANCHOR_BODY = "torso_link"
ROOT_BODY = "pelvis"
FOOT_GEOM_REGEX = r"^(left|right)_foot[1-7]_collision$"


@dataclass
class SuperMimicEnvCfgArgs:
    npz_files: list[str]
    num_envs: int = 4096
    future_frames: int | None = None
    history_length: int | None = None
    episode_length_s: float = 10.0
    action_noise_std: float = 0.1
    obs_noise: bool = True
    reset_noise: bool = True
    domain_randomization: bool = True
    actuator_delay: bool | None = None
    actuator_delay_max_ms: float = 20.0
    reset_root_z_offset: float | None = None
    sampling_mode: str = "return"  # return | uniform | start
    replay_capacity: int = 1000
    record_transitions: bool = True
    seed: int | None = 0
    root_height_threshold: float = 0.25
    root_tilt_threshold: float = 0.8
    joint_limit_weight: float = -2.0
    bin_frames: int = 50
    merge_short_tail_bins: bool = False
    compact_library: bool = False
    reference_fk_compile: bool = False
    anchor_obs: bool = False  # add BeyondMimic anchor-relative pose (9) to the goal group (plan D5 alt.)
    debug_vis: bool = False  # draw the reference ghost in the viewer (scripts/play.py)
    robot: RobotName = "g1"
    observation_layout: ObservationLayoutOption = "auto"
    extra_command_kwargs: dict = field(default_factory=dict)


def make_env_cfg(a: SuperMimicEnvCfgArgs) -> ManagerBasedRlEnvCfg:
    robot_spec = get_robot_spec(a.robot)
    settings = resolve_robot_environment(
        a.robot, actuator_delay=a.actuator_delay,
        actuator_delay_max_ms=a.actuator_delay_max_ms,
        reset_root_z_offset=a.reset_root_z_offset,
    )
    for name, value in settings.items():
        setattr(a, name, value)
    entity_cfg = robot_spec.entity_cfg()
    # Match MorphMimic: one held lag per episode and fused actuator group.
    # Nominal evaluation uses domain_randomization=False and has no delay.
    lag = round(a.actuator_delay_max_ms / 5) if a.actuator_delay and a.domain_randomization else 0
    if entity_cfg.articulation is not None and (a.actuator_delay or a.robot == "khan_mini"):
        entity_cfg.articulation.actuators = tuple(
            replace(actuator, delay_min_lag=0, delay_max_lag=lag, delay_hold_prob=0.0,
                    delay_update_period=2**40, delay_per_env_phase=False)
            for actuator in entity_cfg.articulation.actuators
        )
    a.observation_layout = resolve_observation_layout(a.robot, a.observation_layout)
    a.future_frames, a.history_length = resolve_observation_sizes(
        a.observation_layout, a.future_frames, a.history_length, a.anchor_obs,
    )
    tracking_rewards = TrackingRewardCfg.for_joint_names(robot_spec.joint_names)
    # ---------------------------------------------------------------- observations
    noise = (lambda lo, hi: Unoise(n_min=lo, n_max=hi)) if a.obs_noise else (lambda lo, hi: None)
    policy_terms = {
        "joint_pos": ObservationTermCfg(func=base_mdp.joint_pos_rel, params={"biased": True}, noise=noise(-0.01, 0.01)),
        "joint_vel": ObservationTermCfg(func=base_mdp.joint_vel_rel, noise=noise(-0.5, 0.5)),
        "base_ang_vel": ObservationTermCfg(func=base_mdp.base_ang_vel, noise=noise(-0.2, 0.2)),
        "projected_gravity": ObservationTermCfg(func=base_mdp.projected_gravity, noise=noise(-0.05, 0.05)),
        "actions": ObservationTermCfg(func=base_mdp.last_action),
    }
    observations = {
        "policy": ObservationGroupCfg(
            terms=policy_terms,
            concatenate_terms=True,
            enable_corruption=a.obs_noise,
            history_length=a.history_length,
            flatten_history_dim=True,
        ),
        "goal": ObservationGroupCfg(
            terms={
                "future_frames": ObservationTermCfg(func=sm_obs.goal_future_frames, params={"command_name": "motion"}),
                **(
                    {
                        "anchor_pos": ObservationTermCfg(func=track_mdp.motion_anchor_pos_b, params={"command_name": "motion"}, noise=noise(-0.25, 0.25)),
                        "anchor_ori": ObservationTermCfg(func=track_mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=noise(-0.05, 0.05)),
                    }
                    if a.anchor_obs
                    else {}
                ),
            },
            concatenate_terms=True,
            enable_corruption=a.obs_noise,
        ),
    }

    if a.observation_layout == "isaac_khan":
        if robot_spec.anchor_body != robot_spec.root_body:
            raise ValueError("isaac_khan requires the anchor to be the free-root base_link")
        # Only sensor terms accumulate history. Future orientation is recomputed
        # from the current robot pose and the current five-frame reference window.
        sensor_terms = {
            "projected_gravity": ObservationTermCfg(
                func=base_mdp.projected_gravity, noise=noise(-0.07, 0.07)),
            "base_ang_vel": ObservationTermCfg(
                func=base_mdp.base_ang_vel, noise=noise(-0.2, 0.2), scale=0.5),
            "joint_pos": ObservationTermCfg(
                func=base_mdp.joint_pos_rel, params={"biased": True}, noise=noise(-0.01, 0.01)),
            "joint_vel": ObservationTermCfg(
                func=base_mdp.joint_vel_rel, noise=noise(-0.5, 0.5), scale=0.2),
            "actions": ObservationTermCfg(func=base_mdp.last_action),
        }
        for term in sensor_terms.values():
            term.history_length = a.history_length
            term.flatten_history_dim = True
        observations = {
            "policy": ObservationGroupCfg(
                terms={
                    "motion_anchor_ori_b": ObservationTermCfg(
                        func=sm_obs.isaac_motion_anchor_orientation,
                        params={"command_name": "motion"}, noise=noise(-0.05, 0.05)),
                    **sensor_terms,
                }, concatenate_terms=True, enable_corruption=a.obs_noise,
            ),
            "goal": ObservationGroupCfg(
                terms={"command": ObservationTermCfg(
                    func=sm_obs.isaac_goal_command, params={"command_name": "motion"},
                )}, concatenate_terms=True, enable_corruption=False,
            ),
        }

    # ---------------------------------------------------------------- actions
    actions = {
        "joint_pos": NoisyJointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=robot_spec.action_scale,
            use_default_offset=True,
            noise_std=a.action_noise_std,
        )
    }

    # ---------------------------------------------------------------- commands
    pose_range = {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01), "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.2, 0.2)}
    vel_range = {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.2, 0.2), "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78)}
    command_kwargs = {
        # Preserve the original sampler's reward-relative temperature after adding neck rewards.
        "g_max": 10.0 * (tracking_rewards.max_step_reward / 0.1),
        **a.extra_command_kwargs,
    }
    commands = {
        "motion": MultiClipMotionCommandCfg(
            npz_files=[str(f) for f in a.npz_files],
            anchor_body_name=robot_spec.anchor_body,
            root_body_name=robot_spec.root_body,
            body_names=robot_spec.tracked_bodies,
            robot_name=robot_spec.name,
            entity_name="robot",
            future_frames=a.future_frames,
            bin_frames=a.bin_frames,
            merge_short_tail_bins=a.merge_short_tail_bins,
            pose_range=pose_range if a.reset_noise else {},
            velocity_range=vel_range if a.reset_noise else {},
            joint_position_range=(-0.1, 0.1) if a.reset_noise else (0.0, 0.0),
            sampling_mode=a.sampling_mode,  # type: ignore[arg-type]
            reset_root_z_offset=a.reset_root_z_offset,
            compact_library=a.compact_library,
            reference_fk_compile=a.reference_fk_compile,
            debug_vis=a.debug_vis,
            **command_kwargs,
        )
    }

    # ---------------------------------------------------------------- events (plan D8)
    events: dict[str, EventTermCfg] = {}
    if a.domain_randomization:
        events["base_com"] = EventTermCfg(
            mode="startup",
            func=dr.body_com_offset,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=(robot_spec.com_body,)),
                "operation": "add",
                "ranges": {0: (-0.025, 0.025), 1: (-0.05, 0.05), 2: (-0.05, 0.05)},
            },
        )
        events["encoder_bias"] = EventTermCfg(
            mode="reset",
            func=dr.encoder_bias,
            params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.01, 0.01)},
        )
        events["foot_friction"] = EventTermCfg(
            mode="reset",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=robot_spec.foot_geom_regex),
                "operation": "abs",
                "ranges": (0.3, 1.2),
                "shared_random": True,
            },
        )
        if a.robot == "khan_mini":
            # Sample gains independently for each joint and for Kp / Kd. The
            # built-in position servo's actual gainprm / biasprm are randomized.
            events["actuator_gains"] = EventTermCfg(
                mode="reset",
                func=dr.pd_gains,
                params={
                    "asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",)),
                    "operation": "scale", "kp_range": (0.8, 1.2), "kd_range": (0.8, 1.2),
                },
            )
            events["ankle_armature"] = EventTermCfg(
                mode="reset",
                func=khan_joint_armature,
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot", joint_names=KHAN_ANKLE_JOINTS, preserve_order=True,
                    ),
                    "operation": "scale", "ranges": (0.8, 1.2), "shared_random": False,
                },
            )
            # Add/remove a point payload at the randomized waist COM, retaining
            # the body's authored rotational inertia. This is not density DR.
            # Nominal waist mass 7.1277 kg remains positive throughout the range.
            events["waist_mass"] = EventTermCfg(
                mode="startup",
                func=dr.body_mass,
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=(robot_spec.com_body,)),
                    "operation": "add", "ranges": (-2.0, 3.0),
                },
            )
    else:
        # Evaluation protocol: foot friction fixed at 1.0, no other randomization.
        events["foot_friction"] = EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=robot_spec.foot_geom_regex),
                "operation": "abs",
                "ranges": (1.0, 1.0),
                "shared_random": True,
            },
        )

    # ---------------------------------------------------------------- rewards (Table A.2)
    rewards = {
        "motion_global_root_pos": RewardTermCfg(func=track_mdp.motion_global_anchor_position_error_exp, weight=0.5, params={"command_name": "motion", "std": 0.3}),
        "motion_global_root_ori": RewardTermCfg(func=track_mdp.motion_global_anchor_orientation_error_exp, weight=0.5, params={"command_name": "motion", "std": 0.4}),
        "motion_body_pos": RewardTermCfg(func=track_mdp.motion_relative_body_position_error_exp, weight=1.0, params={"command_name": "motion", "std": 0.3}),
        "motion_body_ori": RewardTermCfg(func=track_mdp.motion_relative_body_orientation_error_exp, weight=1.0, params={"command_name": "motion", "std": 0.4}),
        "motion_body_lin_vel": RewardTermCfg(func=track_mdp.motion_global_body_linear_velocity_error_exp, weight=1.0, params={"command_name": "motion", "std": 1.0}),
        "motion_body_ang_vel": RewardTermCfg(func=track_mdp.motion_global_body_angular_velocity_error_exp, weight=1.0, params={"command_name": "motion", "std": 3.14}),
        "action_rate_l2": RewardTermCfg(func=base_mdp.action_rate_l2, weight=-0.1),
        "joint_limit": RewardTermCfg(func=base_mdp.joint_pos_limits, weight=a.joint_limit_weight, params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))}),
    }
    if robot_spec.neck_joint_names:
        neck_asset = SceneEntityCfg("robot", joint_names=robot_spec.neck_joint_names, preserve_order=True)
        rewards["motion_neck_pos"] = RewardTermCfg(
            func=sm_rewards.motion_neck_position_error_exp,
            weight=tracking_rewards.w_neck_pos,
            params={"command_name": "motion", "std": tracking_rewards.std_neck_pos,
                    "asset_cfg": neck_asset},
        )
        rewards["motion_neck_vel"] = RewardTermCfg(
            func=sm_rewards.motion_neck_velocity_error_exp,
            weight=tracking_rewards.w_neck_vel,
            params={"command_name": "motion", "std": tracking_rewards.std_neck_vel,
                    "asset_cfg": neck_asset},
        )

    # ---------------------------------------------------------------- terminations
    terminations = {
        "time_out": TerminationTermCfg(func=base_mdp.time_out, time_out=True),
        "clip_end": TerminationTermCfg(func=sm_term.clip_end, params={"command_name": "motion"}, time_out=True),
        "root_height": TerminationTermCfg(func=sm_term.bad_root_height, params={"command_name": "motion", "threshold": a.root_height_threshold}),
        "root_tilt": TerminationTermCfg(func=sm_term.bad_root_tilt, params={"command_name": "motion", "threshold": a.root_tilt_threshold}),
    }

    recorders = {}
    if a.record_transitions:
        recorders["replay"] = ReplayRecorderCfg(params={
            "capacity": a.replay_capacity, "foot_geom_regex": robot_spec.foot_geom_regex,
            "com_body_name": robot_spec.com_body,
            "context_layout": "khan_dr_v1" if a.robot == "khan_mini" else "base",
        })

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=a.num_envs, entities={"robot": entity_cfg}),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        recorders=recorders,
        viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot", body_name=robot_spec.anchor_body, distance=2.8, fovy=55.0, elevation=-5.0, azimuth=120.0),
        sim=QuietSimulationCfg(nconmax=robot_spec.nconmax, njmax=robot_spec.njmax, mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20)),
        decimation=4,
        episode_length_s=a.episode_length_s,
        seed=a.seed,
    )
    return cfg


def make_env(a: SuperMimicEnvCfgArgs, device: str = "cuda:0") -> ManagerBasedRlEnv:
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    return ManagerBasedRlEnv(make_env_cfg(a), device=device)
