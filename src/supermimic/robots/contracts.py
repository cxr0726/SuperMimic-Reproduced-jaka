"""Checkpoint compatibility for observation, reward and robot-physics semantics."""

from __future__ import annotations

import hashlib
from dataclasses import asdict

from supermimic.envs.mdp.rewards import TrackingRewardCfg
from supermimic.observation_layout import resolve_observation_layout
from supermimic.robots.registry import RobotName, get_robot_spec, resolve_robot_environment
from supermimic.sim_context import (
    PHYSICS_STEP_MS,
    KHAN_ANKLE_JOINTS,
    KHAN_FOOT_BODIES,
    KHAN_LOWER_BODY_JOINTS,
    ContextKind,
    SimContextLayout,
    resolve_context_layout,
    validate_ankle_encoder_bias_half_range,
)


def checkpoint_context_layout(config: dict) -> ContextKind:
    """Resolve saved physics, without applying fresh-training DR defaults.

    The explicit environment field was introduced with v2. Older checkpoints
    record the v1 layout in their WM config, or predate that config field.
    """
    robot = config.get("robot", "g1")
    requested = config.get("env", {}).get("context_layout")
    if requested is None:
        requested = config.get("world_model", {}).get("state_spec", {}).get("context_layout")
    if requested is None:
        requested = "khan_dr_v1" if robot == "khan_mini" else "base"
    return resolve_context_layout(robot, requested)


def checkpoint_foot_tracking(config: dict) -> bool:
    """Historical v3 checkpoints coupled foot rewards to the DR layout."""
    saved = config.get("env", {}).get("foot_tracking")
    return checkpoint_context_layout(config) == "khan_dr_v3" if saved is None else saved


def robot_contract(robot: RobotName, observation_layout: str = "auto", *, env_config: dict | None = None) -> dict:
    spec = get_robot_spec(robot)
    reward_config = asdict(TrackingRewardCfg.for_joint_names(spec.joint_names))
    # These fields did not exist in historical G1 or Khan v1/v2 contracts.
    # Keep their serialized dictionaries identical for old play/eval/export.
    for key in ("std_foot_pos", "std_foot_ori", "w_foot_pos", "w_foot_ori"):
        reward_config.pop(key, None)
    contract = {
        "version": 1,
        "robot": robot,
        "root_body": spec.root_body,
        "anchor_body": spec.anchor_body,
        "com_body": spec.com_body,
        "tracked_bodies": list(spec.tracked_bodies),
        "joint_names": list(spec.joint_names),
        "reward_config": reward_config,
        "action_scale": spec.action_scale,
    }
    if robot == "khan_mini":
        from supermimic.robots.khan_mini import DEFAULT_ASSET_DIR, load_profile

        # Neither equal tensor widths nor matching body names prove equal physics.
        contract["model_sha256"] = hashlib.sha256((DEFAULT_ASSET_DIR / "robot.xml").read_bytes()).hexdigest()
        contract["profile_sha256"] = hashlib.sha256((DEFAULT_ASSET_DIR / "profile.json").read_bytes()).hexdigest()
        profile = load_profile()
        contract["nominal_joint_physics"] = {
            "training_semantics_version": profile["training_semantics_version"],
            "joint_frictionloss_nm": profile["joint_frictionloss"],
            "joint_frictionloss_randomized": False,
            "joint_position_limits_rad": {
                name: [profile["urdf_limits"][name][key] for key in ("lower", "upper")]
                for name in spec.joint_names
            },
            "soft_joint_pos_limit_factor": profile["soft_joint_pos_limit_factor"],
            "knee_soft_lower_limit_rad": profile["parameter_overrides"]["knee_lower_limit"]["soft_lower_limit"],
        }
        layout = resolve_observation_layout(robot, observation_layout)
        contract["version"] = 6
        contract["observation_layout"] = layout
        env_config = env_config or {}
        settings = resolve_robot_environment(
            robot, **{key: env_config[key] for key in
                      ("actuator_delay", "actuator_delay_max_ms", "reset_root_z_offset")
                      if key in env_config},
        )
        # An absent field denotes the historical contract, never a new-run
        # default. This keeps v6/v7 checkpoint validation unchanged.
        kind = resolve_context_layout(robot, env_config.get("context_layout") or "khan_dr_v1")
        context = SimContextLayout(spec.num_joints, kind)
        contract["simulation_schema"] = {
            **settings,
            "actuator_delay_enabled_in_training": settings["actuator_delay"] and env_config.get("domain_randomization", True),
            "actuator_delay_sampling": "per_environment_per_fused_group_held_for_episode",
            "actuator_delay_physics_dt_s": PHYSICS_STEP_MS / 1000,
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
            "foot_collision": profile["parameter_overrides"]["foot_collision"].get(
                "schema", "one_full_CAD_convex_mesh_per_foot"),
        }
        # Missing/False preserves v6 byte-for-byte for historical checkpoint
        # validation. New training records its distinct reset queue semantics.
        if env_config.get("delay_reset_first_step_no_delay", False):
            contract["version"] = 7
            contract["simulation_schema"].update({
                "delay_reset_first_step_no_delay": True,
                "delay_initial_pending_target": "first_processed_command_after_encoder_bias",
                "delay_reset_scope": "only_reset_worlds_first_control_step_all_substeps",
            })
        if kind in ("khan_dr_v2", "khan_dr_v3", "khan_dr_v4"):
            contract["version"] = 8
            schema = contract["simulation_schema"]
            del schema["ankle_armature_joint_order"]
            schema["lower_body_armature_joint_order"] = list(KHAN_LOWER_BODY_JOINTS)
            randomization = schema["dynamics_randomization"]
            del randomization["ankle_armature_factor"]
            randomization["lower_body_armature_factor"] = {
                "range": [0.8, 1.2], "mode": "startup", "independent_per_joint": True,
            }
            randomization["com_offset_m"] = {
                "range": [[-0.035, 0.035], [-0.07, 0.07], [-0.07, 0.07]],
                "mode": "startup", "body": spec.com_body, "frame": "body_local",
            }
        foot_tracking = checkpoint_foot_tracking({"robot": robot, "env": env_config})
        if foot_tracking and kind not in ("khan_dr_v3", "khan_dr_v4"):
            raise ValueError("Extra foot rewards require the Khan DR v3 or v4 layout")
        if kind in ("khan_dr_v3", "khan_dr_v4"):
            contract["version"] = 9 if foot_tracking else 10
            contract["simulation_schema"]["foot_mass_body_order"] = list(KHAN_FOOT_BODIES)
            contract["simulation_schema"]["dynamics_randomization"]["foot_mass_delta_kg"] = {
                "range": [0.0, 0.15 if kind == "khan_dr_v4" else 0.2],
                "mode": "startup", "independent_per_body": True,
                "bodies": list(KHAN_FOOT_BODIES), "operation": "add_to_nominal",
                "inertia": "unchanged_COM_point_mass", "com": "unchanged",
            }
        if foot_tracking:
            foot_reward = TrackingRewardCfg.for_joint_names(spec.joint_names, foot_tracking=True)
            contract["reward_config"] = asdict(foot_reward)
            contract["foot_tracking_schema"] = {
                "bodies": list(KHAN_FOOT_BODIES),
                "reference_frame": "robot_anchor_xy_yaw_reanchor_reference_z_retained",
                "aggregation": "mean_of_per_foot_exponentials",
                "physical_and_imagined": True,
                "max_step_reward": foot_reward.max_step_reward,
            }
        noise_std = env_config.get("action_noise_std", 0.1)
        neck_mask = env_config.get("disable_neck_action_noise", False)
        if noise_std == 0:
            contract["execution_noise_schema"] = {
                "std_normalized": 0.0,
                "replay_action": "clean_policy_action",
                "last_action_and_action_rate": "clean_policy_action",
            }
        elif neck_mask:
            contract["version"] = 12
            contract["execution_noise_schema"] = {
                "disabled_joint_names": list(profile["neck_joint_names"]),
                "other_joint_std_normalized": noise_std,
                "rng_sampling": "full_action_tensor_before_neck_mask",
                "replay_action": "executed_after_noise_mask",
                "last_action_and_action_rate": "clean_policy_action",
            }
        elif noise_std not in (None, 0.1):
            contract["execution_noise_schema"] = {
                "std_normalized": noise_std,
                "replay_action": "executed_after_noise",
                "last_action_and_action_rate": "clean_policy_action",
            }
        # Missing means the historical +/-0.01 rad draw. Keep those contracts
        # byte-for-byte identical, including explicit legacy resume overrides.
        ankle_bias = validate_ankle_encoder_bias_half_range(
            robot, env_config.get("ankle_encoder_bias_half_range", 0.01),
        )
        if ankle_bias != 0.01:
            contract["version"] = 13
            contract["simulation_schema"]["dynamics_randomization"]["encoder_bias_rad"] = {
                "ankle_joint_names": list(KHAN_ANKLE_JOINTS),
                "ankle_range": [-ankle_bias, ankle_bias], "other_range": [-0.01, 0.01],
                "mode": "reset", "independent_per_joint": True,
                "observation": "q_plus_bias", "physical_target": "command_minus_bias",
                "xi": "actual_bias_in_radians", "xi_divisor_rad": 0.01,
            }
        if kind == "khan_dr_v4":
            contract["version"] = 14
            contract["nominal_foot_inertials"] = profile["parameter_overrides"]["foot_nominal_inertia"]
            contract["nominal_joint_physics"]["joint_frictionloss_randomized"] = env_config.get("domain_randomization", True)
            contract["simulation_schema"]["ankle_frictionloss_joint_order"] = list(KHAN_ANKLE_JOINTS)
            contract["simulation_schema"]["dynamics_randomization"]["ankle_frictionloss_nm"] = {
                "delta_range": [0.0, 0.15], "nominal": 0.1, "total_range": [0.1, 0.25],
                "mode": "startup", "independent_per_joint": True,
                "operation": "add_to_nominal", "xi": "actual_total_frictionloss_nm",
            }
            if env_config.get("foot_mass_shared", False):
                contract["version"] = 15
                contract["simulation_schema"]["dynamics_randomization"]["foot_mass_delta_kg"][
                    "independent_per_body"
                ] = False
        if env_config.get("wm_delay_history", False):
            contract["version"] = 16
            contract["simulation_schema"].update({
                "world_model_delay_queue": True,
                "world_model_input_dim": spec.state_dim + 2 * spec.num_joints + context.dim,
                "world_model_delay_actions": "first_and_last_applied_commands",
                "world_model_delay_timing": "number_of_substeps_before_within_step_command_switch",
            })
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
                # Historical key: corruption in the environment-produced goal.
                # Policy encoder-only augmentation is separate, recorded in
                # config.policy and ONNX training_goal_noise. Keep this clean
                # reference contract stable for legacy checkpoint validation.
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


def robot_contract_differences(saved: dict, expected: dict) -> list[str]:
    """Compare contracts, allowing only the audited range-formatting migration."""
    differences = {key for key in set(saved) | set(expected) if saved.get(key) != expected.get(key)}
    # 81c5b42's .17g ranges and their shortest-round-trip spelling compile to
    # identical physics. No other model/profile/config change is exempted.
    equivalent_xml = {
        "ec825ef0e79e9b68782f2bbbe564aec63a686c129a6664c2109508202e77a1e8",
        "db6bc5d5192deeaceafb1f5c7ece4bbec7ae6a8ba3e196323077502903320294",
    }
    if (saved.get("robot") == expected.get("robot") == "khan_mini"
            and {saved.get("model_sha256"), expected.get("model_sha256")} == equivalent_xml):
        differences.discard("model_sha256")
    return sorted(differences)


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
    saved_env = checkpoint.get("config", {}).get("env", {})
    expected = robot_contract(robot, observation_layout, env_config=saved_env)
    differences = robot_contract_differences(saved, expected)
    # The older DR branch only serialized execution-noise metadata for masked
    # collection, even if an explicit standard deviation was saved in config.
    if (differences and robot == "khan_mini" and "wm_delay_history" not in saved_env
            and "foot_mass_shared" not in saved_env and "action_noise_std" in saved_env):
        legacy = robot_contract(robot, observation_layout, env_config={**saved_env, "action_noise_std": 0.1})
        if saved_env.get("disable_neck_action_noise", False):
            legacy["execution_noise_schema"]["other_joint_std_normalized"] = saved_env["action_noise_std"]
        if not robot_contract_differences(saved, legacy):
            return
    if differences:
        raise ValueError("Checkpoint robot_contract differs from current model: " + ", ".join(differences))
