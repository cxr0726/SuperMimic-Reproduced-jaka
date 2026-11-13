"""Compare the pushed acceleration recipe with its pre-acceleration oracle.

CDF and the subsequent full-window/network fusion are deliberately disabled.
The historical replay/rollout source is read from git rather than reproduced
as a second copy of the implementation under test.
"""

import copy
import gc
import json
import subprocess
import sys
import types
from functools import cache

import pytest
import torch

from supermimic.data.replay_buffer import ReplayBuffer, WindowBatch
from supermimic.imagination.policy_trainer import PolicyTrainer, PolicyTrainerCfg
from supermimic.imagination.rollout import imagine
from supermimic.policy.nets import Critic, CriticCfg
from supermimic.world_model.wm_trainer import WMTrainerCfg, WorldModelTrainer


@cache
def _historical_module(relative_path):
    try:
        source = subprocess.check_output(
            ["git", "show", f"b024710:{relative_path}"], text=True, stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError:
        pytest.skip("pre-acceleration commit b024710 is unavailable in this checkout")
    name = "_pre_acceleration_" + relative_path.replace("/", "_").removesuffix(".py")
    module = types.ModuleType(name)
    sys.modules[name] = module
    # The fixed oracle is trusted source from this repository, not user input.
    exec(compile(source, f"git:b024710:{relative_path}", "exec"), module.__dict__)  # noqa: S102
    return module


def test_prepared_multinomial_matches_historical_replay_samples_and_rng():
    from test_gpu_replay_sampling import _ring

    old_type = _historical_module("src/supermimic/data/replay_buffer.py").ReplayBuffer
    current = _ring(sampling="multinomial")
    historical = old_type(current.E, current.T, 2, 1, 2, "cpu")
    for name, value in vars(current).items():
        if isinstance(value, torch.Tensor) and hasattr(historical, name):
            getattr(historical, name).copy_(value)
    ends = torch.tensor([100])
    for method, arguments in (("sample_windows", {}),
                              ("sample_starts", {"horizon": 4, "clip_last": ends})):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(924)
            expected = getattr(historical, method)(64, 3, **arguments)
            expected_rng = torch.get_rng_state().clone()
            torch.manual_seed(924)
            actual = getattr(current, method)(64, 3, **arguments)
            actual_rng = torch.get_rng_state().clone()
        assert vars(actual).keys() == vars(expected).keys()
        for name, value in vars(expected).items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(getattr(actual, name), value, rtol=0, atol=0)
        torch.testing.assert_close(actual_rng, expected_rng, rtol=0, atol=0)


@pytest.mark.parametrize("noise", [False, True])
def test_h32_batch_encoder_matches_historical_rollout_actions_objective_and_gradients(tmp_path, noise):
    from test_batched_reference_encoder import _components, _loss

    old_imagine = _historical_module("src/supermimic/imagination/rollout.py").imagine
    torch.manual_seed(615)
    wm, original, lib, start = _components(tmp_path)
    old_actor, accelerated_actor = copy.deepcopy(original), copy.deepcopy(original)
    outputs, gradients, rng_states = [], [], []
    for actor, accelerated in ((old_actor, False), (accelerated_actor, True)):
        torch.manual_seed(388)
        if accelerated:
            rollout = imagine(wm, actor, lib, start, 32, 5, obs_noise=noise,
                              batch_reference_encoder=True, tensor_metrics=True)
        else:
            rollout = old_imagine(wm, actor, lib, start, 32, 5, obs_noise=noise)
        loss = _loss(rollout)
        loss.backward()
        outputs.append((rollout, loss.detach()))
        gradients.append([parameter.grad.clone() for parameter in actor.parameters()])
        rng_states.append(torch.get_rng_state().clone())
    expected, actual = outputs[0][0], outputs[1][0]
    for field in ("actions", "states", "rewards", "survival", "goals", "critic_goals", "e_root"):
        torch.testing.assert_close(getattr(actual, field), getattr(expected, field),
                                   rtol=3e-5, atol=2e-6, msg=field)
    torch.testing.assert_close(outputs[1][1], outputs[0][1], rtol=3e-5, atol=2e-6)
    for actual_gradient, expected_gradient in zip(gradients[1], gradients[0], strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-4, atol=3e-6)
    torch.testing.assert_close(rng_states[1], rng_states[0], rtol=0, atol=0)


def _steps(optimizer):
    return {int(value["step"]) for value in optimizer.state.values() if "step" in value}


def _max_parameter_error(actual, expected):
    return max(float((value.detach() - dict(expected.named_parameters())[name].detach()).abs().max())
               for name, value in actual.named_parameters())


def _fixed_khan_components(tmp_path, device):
    from test_batched_reference_encoder import _components

    wm, actor, lib, start = _components(tmp_path)
    # Reconstruct plain state-spec/FK attributes on the destination device;
    # Module.to alone would not move those non-registered tensors.
    if str(device) != "cpu":
        from supermimic.envs.mdp.motion_library import MotionLibrary
        from supermimic.kinematics.g1_fk import G1Kinematics
        from supermimic.policy.nets import Policy
        from supermimic.robots.registry import get_robot_spec
        from supermimic.world_model.heads import AnalyticHeads
        from supermimic.world_model.world_model import WorldModel

        robot = get_robot_spec("khan_mini")
        new_lib = MotionLibrary(
            [tmp_path / "batch_encoder_motion.npz"], robot.tracked_bodies,
            lib.default_joint_pos.cpu().numpy(), device=device, store_dtype=torch.float32,
            robot_name=robot.name, root_body_name=robot.root_body,
        )
        heads = AnalyticHeads(
            G1Kinematics(robot.load_model(), device=device), robot.tracked_bodies,
            robot.anchor_body, robot.root_body, wm.heads.soft_limits.to(device),
            new_lib.default_joint_pos, reward_cfg=copy.deepcopy(wm.heads.reward_cfg),
        )
        new_wm = WorldModel(copy.deepcopy(wm.cfg), heads, device,
                            action_scale=wm.action_scale.to(device),
                            action_offset=new_lib.default_joint_pos)
        new_wm.load_state_dict(wm.state_dict())
        new_actor = Policy(copy.deepcopy(actor.cfg), default_joint_pos=new_lib.default_joint_pos,
                           action_scale=torch.full((27,), .5, device=device)).to(device)
        new_actor.load_state_dict(actor.state_dict())
        wm, actor, lib = new_wm, new_actor, new_lib
    wm.requires_grad_(True)
    buffer = ReplayBuffer(1, 16, 67, 27, 101, device, sampling="multinomial")
    ids = torch.tensor([0], device=device)
    for row in range(16):
        buffer.add(ids, lib.ref71(torch.tensor([4], device=device)),
                   torch.zeros(1, 27, device=device), torch.zeros(1, 27, device=device),
                   wm.spec.xi_offset[None], torch.zeros(1, device=device), torch.zeros(1, device=device),
                   torch.tensor([4], device=device), torch.zeros(1, dtype=torch.long, device=device),
                   torch.zeros(1, dtype=torch.bool, device=device),
                   torch.full((1,), row == 0, dtype=torch.bool, device=device))
    frames = torch.tensor([2, 3, 6, 15], device=device)
    history_frames = (frames[None] + torch.arange(-4, 1, device=device)[:, None]).clamp_min(0)
    from supermimic.data.replay_buffer import StartBatch

    start = StartBatch(
        state=lib.ref71(frames), xi=wm.spec.xi_offset[None].expand(4, -1).clone(),
        frame=frames, prev_action=torch.zeros(4, 27, device=device),
        hist_states=lib.ref71(history_frames), hist_actions=torch.zeros(5, 4, 27, device=device),
        history_valid=frames[None] + torch.arange(-4, 1, device=device)[:, None] >= 0,
    )
    critic = Critic(CriticCfg(hidden=(8,), hidden_layer_norm=True, hidden_activation="gelu"),
                    pcfg=actor.cfg, xi_dim=wm.spec.xi_dim).to(device)
    return wm, actor, critic, lib, buffer, start


def _controlled_policy_trainer(components, *, accelerated, compile_mode):
    wm, actor, critic, lib, _buffer, _start = components
    return PolicyTrainer(
        copy.deepcopy(wm), copy.deepcopy(actor), copy.deepcopy(critic), lib,
        PolicyTrainerCfg(
            horizon=32, batch_size=4, history=5, future_frames=5,
            compile=compile_mode, compile_networks=False, observation_layout="isaac_khan",
            cuda_graph=accelerated, batch_reference_encoder=accelerated,
            obs_noise=False, critic_epochs=5, critic_minibatches=4, critic_value_bound=True,
        ),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA Graphs")
def test_h32_pushed_acceleration_preserves_functional_updates_and_budget(tmp_path, monkeypatch):
    components = _fixed_khan_components(tmp_path, "cuda:0")
    _wm, _actor, _critic, _lib, buffer, start = components
    assert buffer.sampling == "multinomial"
    # Fix samples and the critic ordering. This isolates execution changes
    # from graph/non-graph RNG reservation and the t0..t31 fit-data ordering.
    monkeypatch.setattr(buffer, "sample_starts", lambda *args, **kwargs: start)
    monkeypatch.setattr(torch, "randperm", lambda n, device=None: torch.arange(n, device=device))
    reference = _controlled_policy_trainer(components, accelerated=False, compile_mode="default")
    pushed = _controlled_policy_trainer(components, accelerated=True, compile_mode="default")
    initial_wm = copy.deepcopy(pushed.wm.state_dict())
    try:
        for update in range(6):
            lr = 0. if update == 4 else 2e-4
            reference.set_lr(lr)
            pushed.set_lr(lr)
            expected, actual = reference.update(buffer), pushed.update(buffer)
            for key in expected:
                assert actual[key] == pytest.approx(expected[key], rel=8e-4, abs=8e-6), key
            for name, value in pushed.policy.state_dict().items():
                torch.testing.assert_close(value, reference.policy.state_dict()[name], rtol=3e-4, atol=3e-6)
            for name, value in pushed.critic.state_dict().items():
                expected_value = reference.critic.state_dict()[name]
                if name == "mlp.net.0.weight":
                    # Heading removal makes canonical R[1,0] (feature index3)
                    # analytically zero. FP32 cancellation can give it a tiny
                    # gradient which Adam amplifies without functional effect.
                    value, expected_value = value.clone(), expected_value.clone()
                    value[:, 3] = expected_value[:, 3] = 0.
                torch.testing.assert_close(value, expected_value, rtol=3e-4, atol=3e-6)
            with torch.no_grad():
                probe = reference.wm.spec.canonical(start.state)
                assert probe[:, 3].abs().max() < 1e-6
                probe[:, 3] = 0.
                goal = torch.zeros(4, reference.critic.goal_scale.numel(), device="cuda:0")
                error = torch.zeros(4, reference.critic.error_dim, device="cuda:0")
                xi = reference.wm.spec.normalize_xi(start.xi)
                root = torch.zeros(4, 4, device="cuda:0")
                torch.testing.assert_close(pushed.critic(probe, goal, xi, root, error),
                                           reference.critic(probe, goal, xi, root, error),
                                           rtol=3e-4, atol=3e-6)
            assert _steps(pushed.opt) == {update + 1}
            assert _steps(pushed.critic_opt) == {20 * (update + 1)}
            assert pushed.ema.num_updates == update + 1
            torch.testing.assert_close(pushed.wm.state_dict(), initial_wm, rtol=0, atol=0)
            print(json.dumps({
                "path": "h32_policy", "update": update + 1,
                "max_abs_metric_error": max(abs(actual[key] - expected[key]) for key in expected),
                "max_abs_policy_parameter_error": _max_parameter_error(pushed.policy, reference.policy),
                "max_abs_critic_parameter_error_including_null_column": _max_parameter_error(pushed.critic, reference.critic),
                "actor_steps": sorted(_steps(pushed.opt)), "critic_steps": sorted(_steps(pushed.critic_opt)),
            }), flush=True)
        assert pushed._captured_update.captured
        assert pushed._captured_update.calls == 6
    finally:
        reference.reset_capture()
        pushed.reset_capture()
        torch._dynamo.reset()
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA Graphs")
def test_eight_transition_wm_pushed_capture_matches_pre_acceleration_budget(tmp_path, monkeypatch):
    wm, _actor, _critic, lib, buffer, start = _fixed_khan_components(tmp_path, "cuda:0")
    frames = start.frame[None] + torch.arange(8, device="cuda:0")[:, None]
    states = lib.ref71(start.frame[None] + torch.arange(9, device="cuda:0")[:, None])
    actions = (lib.ref(frames + 1).joint_pos - lib.default_joint_pos) / .5
    window = WindowBatch(states, actions, actions, start.xi,
                         torch.zeros(8, 4, device="cuda:0"), torch.zeros(8, 4, device="cuda:0"), frames)
    monkeypatch.setattr(buffer, "sample_windows", lambda *args, **kwargs: window)
    trainers = [WorldModelTrainer(copy.deepcopy(wm), lib, WMTrainerCfg(
        window=8, batch_size=4, batches_per_update=10, compile="default",
        compile_full_window=False, cuda_graph=accelerated,
    )) for accelerated in (False, True)]
    try:
        for update in range(6):
            for trainer in trainers:
                trainer.set_lr(0. if update == 4 else 2e-4)
            expected, actual = [trainer.update(buffer) for trainer in trainers]
            for key in expected:
                assert actual[key] == pytest.approx(expected[key], rel=8e-4, abs=8e-6), key
            for name, value in trainers[1].wm.state_dict().items():
                torch.testing.assert_close(value, trainers[0].wm.state_dict()[name], rtol=3e-4, atol=3e-6)
            assert _steps(trainers[1].opt) == {10 * (update + 1)}
            print(json.dumps({
                "path": "eight_step_wm", "update": update + 1,
                "max_abs_metric_error": max(abs(actual[key] - expected[key]) for key in expected),
                "max_abs_parameter_error": _max_parameter_error(trainers[1].wm, trainers[0].wm),
                "wm_steps": sorted(_steps(trainers[1].opt)),
            }), flush=True)
        assert trainers[1].update_graph.captured and trainers[1].update_graph.calls == 6
    finally:
        for trainer in trainers:
            trainer.reset_capture()
        torch._dynamo.reset()
        gc.collect()
        torch.cuda.empty_cache()
