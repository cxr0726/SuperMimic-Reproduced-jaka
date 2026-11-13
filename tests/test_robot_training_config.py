"""CPU contracts: robot selection changes bindings, not training semantics."""

import re
from dataclasses import asdict

import mujoco
import pytest
import torch
from mjlab.asset_zoo.robots import G1_ACTION_SCALE

from supermimic.config import TrainConfig, resolve_training_observations
from supermimic.envs.tracking_env_cfg import (
    ANCHOR_BODY,
    FOOT_GEOM_REGEX,
    ROOT_BODY,
    TRACKED_BODIES,
    SuperMimicEnvCfgArgs,
    make_env_cfg,
)
from supermimic.robots.registry import get_robot_spec
from supermimic.sim_context import SimContextLayout
from supermimic.training.loop import Trainer


@pytest.mark.parametrize("anchor_obs", [False, True])
@pytest.mark.parametrize("randomized", [False, True])
def test_robot_observation_reward_dr_and_solver_contract(anchor_obs, randomized):
    options = {"npz_files": ["unused.npz"], "anchor_obs": anchor_obs,
               "domain_randomization": randomized, "obs_noise": randomized,
               "reset_noise": randomized}
    g1 = make_env_cfg(SuperMimicEnvCfgArgs(**options))
    # Explicit paper mode retains the previous robot-neutral observation path.
    khan = make_env_cfg(SuperMimicEnvCfgArgs(**options, robot="khan_mini",
                                           observation_layout="supermimic"))
    assert tuple(khan.observations) == tuple(g1.observations) == ("policy", "goal")
    for name in khan.observations:
        assert asdict(khan.observations[name]) == asdict(g1.observations[name])
    assert tuple(khan.observations["policy"].terms) == (
        "joint_pos", "joint_vel", "base_ang_vel", "projected_gravity", "actions"
    )
    assert khan.observations["policy"].history_length == 10
    assert khan.observations["policy"].flatten_history_dim
    assert tuple(khan.observations["goal"].terms) == (
        ("future_frames", "anchor_pos", "anchor_ori") if anchor_obs else ("future_frames",)
    )
    assert {name: khan.rewards[name] for name in g1.rewards} == g1.rewards
    assert {name: term.weight for name, term in khan.rewards.items()} == {
        "motion_global_root_pos": 0.5, "motion_global_root_ori": 0.5,
        "motion_body_pos": 1.0, "motion_body_ori": 1.0,
        "motion_body_lin_vel": 1.0, "motion_body_ang_vel": 1.0,
        "action_rate_l2": -0.1, "joint_limit": -2.0,
        "motion_neck_pos": 0.5, "motion_neck_vel": 0.1,
    }
    assert sum(max(0.0, term.weight) for term in khan.rewards.values()) * .02 == pytest.approx(.112)
    assert khan.terminations == g1.terminations
    assert khan.terminations["root_height"].params["threshold"] == 0.25
    assert khan.terminations["root_tilt"].params["threshold"] == 0.8
    assert khan.terminations["time_out"].time_out and khan.terminations["clip_end"].time_out
    assert set(khan.events) - set(g1.events) == (
        {"actuator_gains", "lower_body_armature", "waist_mass", "foot_mass"} if randomized else set()
    )
    for name in g1.events:
        a, b = asdict(khan.events[name]), asdict(g1.events[name])
        # COM bounds are robot-specific; the event operation is identical.
        asset_a = a["params"].pop("asset_cfg")
        asset_b = b["params"].pop("asset_cfg")
        for key in ("body_names", "geom_names"):
            asset_a.pop(key)
            asset_b.pop(key)
        if name == "base_com":
            a["params"].pop("ranges")
            b["params"].pop("ranges")
        assert asset_a == asset_b
        assert a == b
    if randomized:
        assert tuple(g1.events) == ("base_com", "encoder_bias", "foot_friction")
        assert tuple(khan.events) == (
            "base_com", "encoder_bias", "foot_friction",
            "actuator_gains", "lower_body_armature", "waist_mass", "foot_mass",
        )
        assert khan.events["base_com"].mode == "startup"
        assert khan.events["base_com"].params["ranges"] == {
            0: (-0.035, 0.035), 1: (-0.07, 0.07), 2: (-0.07, 0.07)
        }
        armature = khan.events["lower_body_armature"]
        assert armature.mode == "startup"
        assert armature.params["ranges"] == (0.8, 1.2)
        assert len(armature.params["asset_cfg"].joint_names) == 12
        assert khan.events["actuator_gains"].mode == "reset"
        foot_mass = khan.events["foot_mass"]
        assert foot_mass.mode == "startup"
        assert foot_mass.params["operation"] == "add"
        assert foot_mass.params["ranges"] == (0.0, 0.2)
        assert foot_mass.params["shared_random"] is False
        assert foot_mass.params["asset_cfg"].body_names == (
            "Left_ankle_roll_Link", "Right_ankle_roll_Link",
        )
        assert khan.events["base_com"].params["asset_cfg"].body_names == ("waist_yaw_Link",)
        assert khan.events["encoder_bias"].mode == "reset"
        assert khan.events["encoder_bias"].params["bias_range"] == (-0.01, 0.01)
        assert khan.events["foot_friction"].mode == "reset"
        assert khan.events["foot_friction"].params["ranges"] == (0.3, 1.2)
    else:
        assert tuple(khan.events) == ("foot_friction",)
        assert khan.events["foot_friction"].mode == "startup"
        assert khan.events["foot_friction"].params["ranges"] == (1.0, 1.0)
    assert khan.events["foot_friction"].params["shared_random"]
    simulation_a, simulation_b = asdict(khan.sim), asdict(g1.sim)
    for key in ("nconmax", "njmax"):
        simulation_a.pop(key)
        simulation_b.pop(key)
    assert simulation_a == simulation_b
    assert khan.sim.mujoco.timestep == 0.005
    assert khan.sim.mujoco.iterations == 10 and khan.sim.mujoco.ls_iterations == 20
    assert khan.decimation == g1.decimation == 4
    assert khan.episode_length_s == g1.episode_length_s == 10.0
    action_a, action_b = asdict(khan.actions["joint_pos"]), asdict(g1.actions["joint_pos"])
    assert set(action_a.pop("scale").values()) == {0.5}
    assert action_b.pop("scale") == G1_ACTION_SCALE
    assert action_a == action_b
    assert action_a["noise_std"] == 0.1
    command_a, command_b = asdict(khan.commands["motion"]), asdict(g1.commands["motion"])
    assert command_a.pop("g_max") == pytest.approx(11.2)
    assert command_b.pop("g_max") == pytest.approx(10.0)
    assert command_a.pop("reset_root_z_offset") == 0.05
    assert command_b.pop("reset_root_z_offset") == 0.0
    for key in ("robot_name", "anchor_body_name", "root_body_name", "body_names"):
        command_a.pop(key)
        command_b.pop(key)
    assert command_a == command_b
    assert command_a["sampling_mode"] == "return"
    assert command_a["future_frames"] == 10 and command_a["bin_frames"] == 50
    assert command_a["joint_position_range"] == ((-0.1, 0.1) if randomized else (0.0, 0.0))
    for actuator in khan.scene.entities["robot"].articulation.actuators:
        assert actuator.delay_min_lag == 0
        assert actuator.delay_max_lag == (4 if randomized else 0)
        assert actuator.delay_update_period == 2**40
        assert actuator.delay_hold_prob == 0.0 and not actuator.delay_per_env_phase
    for actuator in g1.scene.entities["robot"].articulation.actuators:
        assert actuator.delay_min_lag == actuator.delay_max_lag == 0


def test_robot_bindings_and_g1_compatibility_defaults():
    assert TrainConfig().robot == SuperMimicEnvCfgArgs(npz_files=[]).robot == "g1"
    g1 = get_robot_spec()
    assert g1.root_body == ROOT_BODY == "pelvis"
    assert g1.anchor_body == ANCHOR_BODY == "torso_link"
    assert g1.com_body == "torso_link" and g1.neck_joint_names == ()
    assert g1.tracked_bodies == TRACKED_BODIES and len(TRACKED_BODIES) == 14
    assert g1.foot_geom_regex == FOOT_GEOM_REGEX
    assert (g1.num_joints, g1.nconmax, g1.njmax) == (29, 35, 250)
    assert not g1.default_actuator_delay and g1.default_reset_root_z_offset == 0.0
    khan = get_robot_spec("khan_mini")
    cfg = make_env_cfg(SuperMimicEnvCfgArgs(npz_files=["unused.npz"], robot=khan.name))
    assert khan.root_body == khan.anchor_body == "base_link"
    assert khan.com_body == "waist_yaw_Link"
    assert khan.neck_joint_names == ("Neck_yaw_joint", "Neck_pitch_joint")
    assert khan.default_actuator_delay and khan.default_reset_root_z_offset == 0.05
    assert len(khan.joint_names) == khan.num_joints == 27
    assert len(khan.tracked_bodies) == 15
    assert khan.tracked_bodies[0] == khan.root_body == khan.anchor_body
    assert khan.tracked_bodies[7] == khan.com_body
    assert khan.tracked_bodies[-1] == "Neck_pitch_Link"
    assert cfg.commands["motion"].robot_name == khan.name
    assert cfg.commands["motion"].body_names == khan.tracked_bodies
    assert cfg.commands["motion"].root_body_name == khan.root_body
    assert cfg.commands["motion"].anchor_body_name == khan.anchor_body
    assert cfg.viewer.body_name == khan.anchor_body
    assert cfg.events["base_com"].params["asset_cfg"].body_names == (khan.com_body,)
    assert cfg.events["foot_friction"].params["asset_cfg"].geom_names == khan.foot_geom_regex
    assert cfg.recorders["replay"].params == {
        "capacity": 1000, "foot_geom_regex": khan.foot_geom_regex,
        "com_body_name": khan.com_body, "context_layout": "khan_dr_v3",
    }
    model = khan.load_model()
    assert all(name in [model.body(i).name for i in range(model.nbody)] for name in khan.tracked_bodies)
    feet = [i for i in range(model.ngeom) if re.fullmatch(khan.foot_geom_regex, model.geom(i).name)]
    assert len(feet) == 2
    assert (model.geom_type[feet] == mujoco.mjtGeom.mjGEOM_MESH).all()
    assert (model.geom_priority[feet] == 1).all() and (model.geom_condim[feet] == 3).all()
    assert cfg.scene.entities["robot"].articulation.soft_joint_pos_limit_factor == 0.9
    assert (cfg.sim.nconmax, cfg.sim.njmax) == (128, 512)


def test_production_compact_encoder_and_ema_preserve_optimizer_defaults():
    cfg = TrainConfig(robot="khan_mini")
    resolve_training_observations(cfg)
    assert cfg.env.context_layout == cfg.world_model.state_spec.context_layout == "khan_dr_v3"
    assert SimContextLayout(27, kind=cfg.env.context_layout).dim == 101
    assert cfg.world_model.hidden == (2048, 2048, 2048)
    assert cfg.policy.encoder_hidden == (1024, 512, 256)
    assert cfg.policy.decoder_hidden == (2048, 1024, 512, 256)
    assert cfg.policy_trainer.ema_decay == .995
    assert cfg.policy.latent_dim == 32 and cfg.policy.history == 5
    assert cfg.env.future_frames == cfg.policy.future_frames == cfg.policy_trainer.history == 5
    assert cfg.policy.goal_dim == 155 and cfg.policy.observation_layout == "isaac_khan"
    assert cfg.critic.hidden == (256, 128)
    assert (cfg.wm_trainer.window, cfg.wm_trainer.batch_size, cfg.wm_trainer.batches_per_update) == (8, 2048, 10)
    assert (cfg.policy_trainer.horizon, cfg.policy_trainer.batch_size) == (32, 4096)
    assert (cfg.policy_trainer.critic_epochs, cfg.policy_trainer.critic_minibatches) == (5, 4)
    assert cfg.wm_trainer.compile == cfg.policy_trainer.compile == "default"
    assert cfg.schedule.updates_per_env_step == 0.4
    assert cfg.schedule.policy_updates_per_wm_update == 1
    assert cfg.schedule.warmup_iterations == 40
    assert cfg.schedule.total_iterations == 320000
    assert cfg.schedule.stop_after_iterations is None
    assert cfg.env.obs_noise and cfg.env.reset_noise and cfg.env.domain_randomization
    assert not cfg.env.anchor_obs


@pytest.mark.parametrize("requested,saved", [("g1", "khan_mini"), ("khan_mini", "g1"), ("khan_mini", None)])
def test_checkpoint_rejects_robot_mismatch_before_loading_parameters(tmp_path, requested, saved):
    trainer = Trainer.__new__(Trainer)
    trainer.cfg = TrainConfig(robot=requested, device="cpu")
    trainer.device = "cpu"
    path = tmp_path / "wrong_robot.pt"
    torch.save({"config": {} if saved is None else {"robot": saved}}, path)
    # No models or optimizers exist on this object: the robot check must run first.
    with pytest.raises(ValueError, match="Checkpoint robot"):
        trainer.load(path)
