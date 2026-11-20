"""Episode-held ankle offsets preserve the other axes and historical checkpoints."""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from mjlab.envs.mdp import dr
from mjlab.managers.scene_entity_config import SceneEntityCfg

from supermimic.config import TrainConfig, resolve_training_observations
from supermimic.envs import SuperMimicEnvCfgArgs, make_env_cfg
from supermimic.envs.encoder_bias import khan_encoder_bias
from supermimic.robots.contracts import robot_contract, validate_robot_contract
from supermimic.sim_context import KHAN_ANKLE_JOINTS
from supermimic.world_model.state_spec import StateSpec, StateSpecCfg

NPZ = Path(__file__).resolve().parents[1] / "data/npz/khan_mini/lafan1/walk1_subject1.npz"


@pytest.mark.parametrize("partial", [False, True])
def test_ankle_draw_preserves_other_joints_rng_and_unreset_worlds(partial):
    n = 64
    initial = torch.full((n, 27), .123)
    data = SimpleNamespace(encoder_bias=initial.clone())
    env = SimpleNamespace(num_envs=n, device="cpu", scene={
        "robot": SimpleNamespace(data=data, num_joints=27),
    })
    # Deliberately reordered IDs: the event must use resolved joint names.
    ankles = [21, 3, 9, 15]
    other = [i for i in range(27) if i not in ankles]
    ids = torch.arange(0, n, 2) if partial else None
    rows = ids if partial else torch.arange(n)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(631)
        dr.encoder_bias(env, ids, (-.01, .01))
        legacy = data.encoder_bias.clone()
        legacy_next = torch.rand(8)
        data.encoder_bias.copy_(initial)
        torch.manual_seed(631)
        khan_encoder_bias(env, ids, .02, SceneEntityCfg("robot", joint_ids=ankles))
        assert torch.equal(torch.rand(8), legacy_next)
    assert torch.equal(data.encoder_bias[:, other], legacy[:, other])
    assert torch.equal(data.encoder_bias[rows[:, None], ankles], 2 * legacy[rows[:, None], ankles])
    assert data.encoder_bias[rows[:, None], ankles].abs().max() <= .02
    assert data.encoder_bias[rows[:, None], other].abs().max() <= .01
    if partial:
        assert torch.equal(data.encoder_bias[1::2], initial[1::2])


@pytest.mark.parametrize("robot,layout,expected", [
    ("khan_mini", None, .02), ("khan_mini", "khan_dr_v3", .02),
    ("khan_mini", "khan_dr_v4", .02),
    ("khan_mini", "khan_dr_v2", .01), ("khan_mini", "khan_dr_v1", .01),
    ("g1", None, .01),
])
def test_fresh_training_selects_bias_without_changing_legacy_layouts(robot, layout, expected):
    cfg = TrainConfig(robot=robot)
    cfg.env.context_layout = layout
    resolve_training_observations(cfg)
    assert cfg.env.ankle_encoder_bias_half_range == expected
    env = make_env_cfg(SuperMimicEnvCfgArgs(
        npz_files=["unused.npz"], robot=robot, context_layout=cfg.env.context_layout,
        ankle_encoder_bias_half_range=cfg.env.ankle_encoder_bias_half_range,
    ))
    event = env.events["encoder_bias"]
    if expected == .02:
        assert event.func is khan_encoder_bias
        assert event.params["asset_cfg"].joint_names == KHAN_ANKLE_JOINTS
    else:
        assert event.func is dr.encoder_bias


@pytest.mark.parametrize("value", [-.01, float("nan"), float("inf")])
def test_invalid_bias_rejected_before_training(value):
    cfg = TrainConfig(robot="khan_mini")
    cfg.env.ankle_encoder_bias_half_range = value
    with pytest.raises(ValueError, match="finite and nonnegative"):
        resolve_training_observations(cfg)


def test_new_distribution_has_explicit_contract_without_reinterpreting_old_xi():
    cfg = TrainConfig(robot="khan_mini")
    cfg.env.context_layout = "khan_dr_v3"  # Preserve the v13 encoder-only contract.
    cfg.env.wm_delay_history = False
    cfg.env.action_noise_std = 0.1  # Exercise the historical masked-noise contract.
    cfg.env.disable_neck_action_noise = True
    resolve_training_observations(cfg)
    saved = asdict(cfg)
    new = robot_contract("khan_mini", "isaac_khan", env_config=saved["env"])
    assert new["version"] == 13
    schema = new["simulation_schema"]
    assert schema["xi_dim"] == 101
    bias = schema["dynamics_randomization"]["encoder_bias_rad"]
    assert bias["ankle_range"] == [-.02, .02] and bias["other_range"] == [-.01, .01]
    validate_robot_contract({"config": saved, "robot_contract": new}, "khan_mini", "isaac_khan")
    old_env = dict(saved["env"])
    del old_env["ankle_encoder_bias_half_range"]
    old = robot_contract("khan_mini", "isaac_khan", env_config=old_env)
    assert old["version"] == 12
    assert old == robot_contract("khan_mini", "isaac_khan", env_config={
        **old_env, "ankle_encoder_bias_half_range": .01,
    })
    with pytest.raises(ValueError, match="robot_contract differs"):
        validate_robot_contract({"config": saved, "robot_contract": old}, "khan_mini", "isaac_khan")
    spec = StateSpec(StateSpecCfg(context_layout="khan_dr_v3"), device="cpu", num_joints=27)
    xi = spec.xi_offset[None].clone()
    xi[:, 1:28] = .02
    torch.testing.assert_close(spec.normalize_xi(xi)[:, 1:28], torch.full((1, 27), 2.), rtol=0, atol=0)


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available() or not NPZ.is_file(), reason="needs CUDA and Khan LAFAN1")
def test_live_bias_matches_replay_pd_and_imagined_policy_observations():
    from supermimic.envs import make_env
    from supermimic.envs.state_io import StateIO
    from supermimic.imagination.rollout import imagine
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.policy.nets import Policy, PolicyCfg
    from supermimic.robots.registry import get_robot_spec
    from supermimic.world_model.heads import AnalyticHeads
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg
    from supermimic.world_model.wm_trainer import WorldModelTrainer, WMTrainerCfg

    env = make_env(SuperMimicEnvCfgArgs(
        npz_files=[str(NPZ)], robot="khan_mini", context_layout="khan_dr_v4",
        observation_layout="isaac_khan", num_envs=8, replay_capacity=32,
        sampling_mode="start", seed=631, actuator_delay=False,
        reset_noise=False, obs_noise=False, domain_randomization=True,
        action_noise_std=.1, disable_neck_action_noise=True,
        ankle_encoder_bias_half_range=.02,
    ), device="cuda:0")
    try:
        env.reset()
        robot = env.scene["robot"]
        spec = get_robot_spec("khan_mini")
        recorder = env.recorder_manager.get_term("replay")
        action = env.action_manager.get_term("joint_pos")
        io = StateIO(env)
        rows = torch.arange(env.num_envs, device=env.device)
        ankles = [robot.joint_names.index(name) for name in KHAN_ANKLE_JOINTS]
        other = [i for i in range(27) if i not in ankles]
        clean = torch.linspace(-.04, .04, env.num_envs * 27, device=env.device).reshape(-1, 27)
        initial = robot.data.encoder_bias.clone()
        assert initial[:, ankles].abs().max() <= .02
        assert (initial[:, ankles].abs() > .01).any()
        assert initial[:, other].abs().max() <= .01
        for step in range(8):
            if step == 3:
                before_reset = robot.data.encoder_bias.clone()
                env.reset(env_ids=rows[:2])
                assert torch.equal(robot.data.encoder_bias[2:], before_reset[2:])
                assert not torch.equal(robot.data.encoder_bias[:2], before_reset[:2])
            bias = robot.data.encoder_bias.clone()
            ptr = recorder.buffer.ptr.clone()
            obs, _, _, _, _ = env.step(clean)
            torch.testing.assert_close(recorder.buffer.xi[rows, ptr, 1:28], bias, rtol=0, atol=0)
            held = ~env.reset_buf
            assert held.any()
            assert torch.equal(robot.data.encoder_bias[held], bias[held])
            expected_target = (robot.data.default_joint_pos + .5 * recorder.buffer.a_exec[rows, ptr] - bias)
            torch.testing.assert_close(robot.data.joint_pos_target[held], expected_target[held], rtol=0, atol=2e-7)
            # Read the real actor input after its observation manager history stack.
            q_obs = obs["policy"][:, 60:195].reshape(-1, 5, 27)[:, -1]
            expected_obs = io.read()[:, 7:34] + robot.data.encoder_bias - robot.data.default_joint_pos
            torch.testing.assert_close(q_obs, expected_obs, rtol=0, atol=2e-7)

        heads = AnalyticHeads(
            G1Kinematics(spec.load_model(), device=env.device), spec.tracked_bodies,
            spec.anchor_body, spec.root_body, robot.data.soft_joint_pos_limits[0].clone(),
            robot.data.default_joint_pos[0].clone(),
        )
        wm = WorldModel(WorldModelCfg(
            hidden=(16,), input_normalization=True, termination_mode="analytic",
            state_spec=StateSpecCfg(context_layout="khan_dr_v4"),
        ), heads, env.device,
            action_scale=torch.full((27,), .5, device=env.device),
            action_offset=robot.data.default_joint_pos[0].clone(),
        )
        library = env.command_manager.get_term("motion").library
        wm.update_normalization(recorder.last_step_states, recorder.last_step_actions)
        trainer = WorldModelTrainer(wm, library, WMTrainerCfg(
            batch_size=8, window=2, batches_per_update=1, compile=None,
        ))
        metrics = trainer.update(recorder.buffer)
        assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
        policy = Policy(PolicyCfg(
            encoder_hidden=(16,), decoder_hidden=(16,), latent_dim=8,
            history=5, future_frames=5, goal_dim=155, num_joints=27,
            observation_layout="isaac_khan",
        )).to(env.device)
        start = recorder.buffer.sample_starts(8, hist=5)
        actor_inputs = []
        dynamics_contexts = []
        hook = policy.register_forward_pre_hook(lambda _module, args: actor_inputs.append(args[0].detach().clone()))
        wm_hook = wm.dynamics.register_forward_pre_hook(lambda _module, args: dynamics_contexts.append(args[2].detach().clone()))
        try:
            with torch.no_grad():
                rollout = imagine(wm, policy, library,
                                  start, horizon=3, future_frames=5, obs_noise=False)
        finally:
            hook.remove()
            wm_hook.remove()
        assert len(actor_inputs) == 3
        assert len(dynamics_contexts) == 3
        assert start.xi.shape == (8, 105)
        for context in dynamics_contexts:
            torch.testing.assert_close(context, wm.spec.normalize_xi(start.xi), rtol=0, atol=0)
        for t, obs in enumerate(actor_inputs):
            expected = rollout.states[t, :, 7:34] + start.xi[:, 1:28] - robot.data.default_joint_pos[0]
            torch.testing.assert_close(obs[:, 60:195].reshape(-1, 5, 27)[:, -1], expected, rtol=0, atol=2e-7)
    finally:
        env.close()
