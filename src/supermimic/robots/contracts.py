"""Checkpoint compatibility for observation, reward and robot-physics semantics."""

from __future__ import annotations

import hashlib
from dataclasses import asdict

from supermimic.envs.mdp.rewards import TrackingRewardCfg
from supermimic.observation_layout import resolve_observation_layout
from supermimic.robots.registry import RobotName, get_robot_spec, resolve_robot_environment
from supermimic.sim_context import KHAN_ANKLE_JOINTS, SimContextLayout


def robot_contract(robot: RobotName, observation_layout: str = "auto", *, env_config: dict | None = None) -> dict:
    spec = get_robot_spec(robot)
    contract = {
        "version": 1,
        "robot": robot,
        "root_body": spec.root_body,
        "anchor_body": spec.anchor_body,
        "com_body": spec.com_body,
        "tracked_bodies": list(spec.tracked_bodies),
        "joint_names": list(spec.joint_names),
        "reward_config": asdict(TrackingRewardCfg.for_joint_names(spec.joint_names)),
        "action_scale": spec.action_scale,
    }
    if robot == "khan_mini":
        from supermimic.robots.khan_mini import DEFAULT_ASSET_DIR

        # Neither equal tensor widths nor matching body names prove equal physics.
        contract["model_sha256"] = hashlib.sha256((DEFAULT_ASSET_DIR / "robot.xml").read_bytes()).hexdigest()
        contract["profile_sha256"] = hashlib.sha256((DEFAULT_ASSET_DIR / "profile.json").read_bytes()).hexdigest()
        layout = resolve_observation_layout(robot, observation_layout)
        contract["version"] = 6
        contract["observation_layout"] = layout
        env_config = env_config or {}
        settings = resolve_robot_environment(
            robot, **{key: env_config[key] for key in
                      ("actuator_delay", "actuator_delay_max_ms", "reset_root_z_offset")
                      if key in env_config},
        )
        context = SimContextLayout(spec.num_joints, "khan_dr_v1")
        contract["simulation_schema"] = {
            **settings,
            "actuator_delay_enabled_in_training": settings["actuator_delay"] and env_config.get("domain_randomization", True),
            "actuator_delay_sampling": "per_environment_per_fused_group_held_for_episode",
            "actuator_delay_physics_dt_s": 0.005,
            "nominal_evaluation_delay": False,
            "delay_initial_pending_target": "zero_physical_target_after_robot_reset",
            "world_model_delay_queue": False, "delay_in_xi": True,
            "context_layout": context.kind,
            "xi_dim": context.dim, "world_model_input_dim": spec.state_dim + spec.num_joints + context.dim,
            "xi_fields": context.sizes,
            "xi_offset": list(context.offset), "xi_scale": list(context.scale),
            "xi_readback": "applied_model_fields_in_robot_joint_order",
            "ankle_armature_joint_order": list(KHAN_ANKLE_JOINTS),
            "dynamics_randomization": {
                "enabled": env_config.get("domain_randomization", True),
                "kp_factor": {"range": [0.8, 1.2], "mode": "reset", "independent_per_joint": True},
                "kd_factor": {"range": [0.8, 1.2], "mode": "reset", "independent_per_joint": True},
                "ankle_armature_factor": {"range": [0.8, 1.2], "mode": "reset", "independent_per_joint": True},
                "torso_mass_delta_kg": {"range": [-2.0, 3.0], "mode": "startup",
                                        "body": spec.com_body, "inertia": "unchanged_COM_point_mass"},
            },
            "reset_offset_scope": "actual_robot_only_all_resets_including_nominal_evaluation",
            "reference_height_offset": 0.0,
            "foot_collision": "one_full_CAD_convex_mesh_per_foot",
        }
        if layout == "isaac_khan":
            contract["observation_schema"] = {
                "future_frames": 5, "history": 5, "goal_dim": 155, "policy_dim": 465,
                "critic_goal_dim": 365, "critic_input_dim": 67 + 365 + context.dim + 4 + 78,
                "critic_tracking_errors": {
                    "order": ["reference_minus_actual_joint_pos", "reanchored_body_position_error_heading",
                              "full_root_relative_rotation_6d"],
                    "dimensions": [27, 45, 6], "scales": [1.0, 1.0, 1.0],
                    "tracked_body_count": 15,
                    "append_after_original_root_error4": True,
                },
                "actor_goal_noise": False,
                "critic_reference_noise": False, "critic_future_orientation": True,
                "critic_reference_order": ["command", "future_anchor_ori_b", "extra_reference_frames"],
                "critic_extra_reference": {
                    "future_frames": 5, "per_frame_dim": 36, "flatten_order": "frame_major",
                    "order": ["joint_vel", "projected_gravity", "root_ang_vel_b", "root_lin_vel_h"],
                    "dimensions": [27, 3, 3, 3],
                    "coordinate_frames": {
                        "projected_gravity": "future_reference_root",
                        "root_ang_vel_b": "future_reference_root",
                        "root_lin_vel_h": "future_reference_heading",
                    },
                    "network_divisors": [5.0, 1.0, 2.0, 1.0],
                },
                "future_offsets": [0, 1, 2, 3, 4],
                "goal_order": ["reference_root_displacement_b", "reference_world_z", "reference_joint_pos"],
                "displacement_frame": "current_reference_base_full_rotation",
                "policy_order": ["future_anchor_ori_b", "projected_gravity_history",
                                 "base_ang_vel_history", "joint_pos_rel_history",
                                 "joint_vel_rel_history", "last_action_history"],
                "orientation": "inverse(robot_base_quat)*reference_base_quat; matrix[:,:2] row-major",
                "history_order": "oldest_to_newest_per_term",
                "sensor_scales": {"base_ang_vel": 0.5, "joint_vel": 0.2},
                "noise_half_width_before_scale": {
                    "goal_root_displacement": 0.0, "goal_root_z": 0.0, "goal_joint_pos": 0.0,
                    "future_anchor_ori_b": 0.05, "projected_gravity": 0.07,
                    "base_ang_vel": 0.2, "joint_pos_rel": 0.01, "joint_vel_rel": 0.5,
                    "last_action": 0.0,
                },
            }
    return contract


def validate_robot_contract(checkpoint: dict, robot: RobotName, observation_layout: str = "auto") -> None:
    saved = checkpoint.get("robot_contract")
    if saved is None:
        if robot == "g1":
            return  # Historical G1 checkpoints retain the original training semantics.
        raise ValueError(
            "Legacy Khan checkpoint has no robot_contract: its 14-body/waist-anchor "
            "training semantics cannot be assumed compatible with the current "
            "15-body/base-anchor/neck-reward model and its Isaac observation layout. Start a new run; evaluate old "
            "policies with scripts/analysis/eval_legacy_khan_benchmark.py, which exports "
            "the matching legacy source and assets without changing this checkout."
        )
    expected = robot_contract(robot, observation_layout,
                              env_config=checkpoint.get("config", {}).get("env", {}))
    differences = sorted(key for key in set(saved) | set(expected) if saved.get(key) != expected.get(key))
    if differences:
        raise ValueError("Checkpoint robot_contract differs from current model: " + ", ".join(differences))
