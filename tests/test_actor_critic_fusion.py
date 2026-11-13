"""Functional actor/critic compilation keeps raw goals, gradients and budgets."""

import copy
import gc
import json
from types import SimpleNamespace

import pytest
import torch

from supermimic.imagination import compiled, policy_trainer
from supermimic.imagination.objective import gae_targets
from supermimic.imagination.policy_trainer import PolicyTrainer, PolicyTrainerCfg
from supermimic.imagination.rollout import ImagRollout
from supermimic.policy.nets import Critic, CriticCfg, Policy, PolicyCfg
from supermimic.world_model.state_spec import StateSpec, StateSpecCfg


def _networks(dtype=torch.float32):
    torch.manual_seed(13)
    cfg = PolicyCfg(
        encoder_hidden=(8,), decoder_hidden=(8,), latent_dim=3,
        num_joints=2, num_tracked_bodies=2, history=5, future_frames=5,
        observation_layout="isaac_khan", goal_dim=30,
        action_mode="reference_residual", running_normalization=True,
        hidden_layer_norm=True, hidden_activation="gelu",
    )
    actor = Policy(cfg, default_joint_pos=torch.tensor([-.2, .4]),
                   action_scale=torch.full((2,), .5)).to(dtype=dtype)
    actor.update_normalization(torch.randn(12, 90, dtype=dtype) * .3,
                               torch.randn(12, 30, dtype=dtype) * .2)
    critic = Critic(CriticCfg(hidden=(8,), hidden_layer_norm=True, hidden_activation="gelu"),
                    pcfg=cfg, xi_dim=6).to(dtype=dtype)
    return actor, critic


def _aot_compile_spy(monkeypatch):
    calls = []
    real_compile = torch.compile

    def build(fn, **kwargs):
        calls.append((fn, kwargs))
        return real_compile(fn, backend="aot_eager", dynamic=False, fullgraph=True)

    monkeypatch.setattr(torch, "compile", build)
    return calls


def _trainer(monkeypatch, *, enabled, dtype=torch.float32):
    actor, critic = _networks(dtype)
    wm = torch.nn.Linear(2, 2).to(dtype=dtype)
    wm.heads = SimpleNamespace(reward_cfg=SimpleNamespace(max_step_reward=.1))
    wm.spec = SimpleNamespace(normalize_xi=lambda xi: xi)
    # This test isolates network/optimizer fusion, not simulator or robot FK.
    monkeypatch.setattr(compiled, "ImagStep", lambda *args, **kwargs: object())
    cfg = PolicyTrainerCfg(
        horizon=2, batch_size=4, history=5, future_frames=5,
        compile="default" if enabled else None, compile_networks=enabled,
        observation_layout="isaac_khan", obs_noise=False, critic_epochs=2,
        critic_minibatches=2, critic_value_bound=True,
    )
    library = SimpleNamespace(default_joint_pos=torch.zeros(2), clip_last=torch.tensor([100]))
    trainer = PolicyTrainer(wm, actor, critic, library, cfg)
    return trainer


def test_training_compile_callbacks_preserve_raw_residual_and_model_schema(monkeypatch):
    calls = _aot_compile_spy(monkeypatch)
    trainer = _trainer(monkeypatch, enabled=True, dtype=torch.float64)
    actor = trainer.policy
    before = copy.deepcopy(actor.state_dict())
    original_methods = (actor.forward, actor.encode_goal, actor.decode_from_latent)
    obs = torch.randn(4, 90, dtype=torch.float64, requires_grad=True)
    clean = torch.randn(4, 30, dtype=torch.float64, requires_grad=True)
    corrupt = clean + .13
    expected = actor(obs, clean, encoder_goal=corrupt)
    fused = trainer._imagine_networks["policy_forward_fn"](obs, clean, encoder_goal=corrupt)
    latent = trainer._imagine_networks["encode_goal_fn"](clean, encoder_goal=corrupt)
    decoded = trainer._imagine_networks["decode_from_latent_fn"](obs, clean, latent)
    torch.testing.assert_close(fused, expected, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(decoded, expected, rtol=1e-11, atol=1e-12)
    targets = (obs, clean, *actor.parameters())
    expected_grad = torch.autograd.grad(expected.square().sum(), targets, retain_graph=True)
    actual_grad = torch.autograd.grad(decoded.square().sum(), targets)
    for actual, reference in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(actual, reference, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(actor.state_dict(), before, rtol=0, atol=0)
    assert (actor.forward, actor.encode_goal, actor.decode_from_latent) == original_methods
    assert all(not key.startswith("_orig_mod") for key in actor.state_dict())
    assert len(calls) == 6
    assert all(call[1]["dynamic"] is False for call in calls)
    assert all(call[1]["options"]["triton.cudagraphs"] is False for call in calls)


def test_inductor_actor_callbacks_read_in_place_updated_running_statistics(monkeypatch):
    # Actual Inductor, not backend=aot_eager: normalizer tensor contents must
    # remain live after compilation while addresses and checkpoint keys stay put.
    trainer = _trainer(monkeypatch, enabled=True)
    actor = trainer.policy
    obs, goal = torch.randn(4, 90), torch.randn(4, 30)
    noisy_goal = goal + .07
    encode = trainer._imagine_networks["encode_goal_fn"]
    decode = trainer._imagine_networks["decode_from_latent_fn"]
    forward = trainer._imagine_networks["policy_forward_fn"]
    before_latent = encode(goal, encoder_goal=noisy_goal)
    before_action = forward(obs, goal, encoder_goal=noisy_goal)
    parameters = copy.deepcopy(dict(actor.named_parameters()))
    keys = tuple(actor.state_dict())
    addresses = {name: value.data_ptr() for name, value in actor.named_buffers()}
    for _ in range(2):
        actor.update_normalization(torch.randn(16, 90) * .7 + .8,
                                   torch.randn(16, 30) * .5 - .4,
                                   encoder_goal=torch.randn(16, 30) * .4 + .6)
        current_buffers = {name: value.clone() for name, value in actor.named_buffers()}
        latent = encode(goal, encoder_goal=noisy_goal)
        expected = actor(obs, goal, encoder_goal=noisy_goal)
        torch.testing.assert_close(forward(obs, goal, encoder_goal=noisy_goal), expected,
                                   rtol=4e-5, atol=4e-6)
        torch.testing.assert_close(decode(obs, goal, latent), expected, rtol=4e-5, atol=4e-6)
        torch.testing.assert_close(dict(actor.named_buffers()), current_buffers, rtol=0, atol=0)
        assert {name: value.data_ptr() for name, value in actor.named_buffers()} == addresses
        assert tuple(actor.state_dict()) == keys
    assert (latent - before_latent).abs().max() > 1e-5
    assert (expected - before_action).abs().max() > 1e-6
    torch.testing.assert_close(dict(actor.named_parameters()), parameters, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_inductor_critic_gather_loss_gae_and_bootstrap_gradients_match_eager(dtype):
    # Use the actual Inductor backend on CPU to check its reductions/autograd,
    # separately from the inexpensive AOT full-update wiring tests.
    _, critic = _networks(dtype)
    trainer = SimpleNamespace(critic=critic)
    B, H = 5, 3
    feat = torch.randn((H + 1) * B, 17, dtype=dtype, requires_grad=True)
    goals = torch.randn((H + 1) * B, critic.goal_scale.numel(), dtype=dtype)
    xi = torch.randn((H + 1) * B, 6, dtype=dtype)
    root = torch.randn((H + 1) * B, 4, dtype=dtype)
    errors = torch.randn((H + 1) * B, critic.error_dim, dtype=dtype, requires_grad=True)
    indices = torch.tensor([4, 1, 1, 0])
    targets = torch.randn((H + 1) * B, dtype=dtype)

    def fit(feat, goals, xi, root, errors, indices, targets):
        return PolicyTrainer._critic_minibatch_loss(trainer, feat, goals, xi, root, errors,
                                                   indices, targets)

    fitted = torch.compile(fit, fullgraph=True, dynamic=False)
    expected = fit(feat, goals, xi, root, errors, indices, targets)
    actual = fitted(feat, goals, xi, root, errors, indices, targets)
    tolerance = {"rtol": 3e-5, "atol": 3e-6} if dtype == torch.float32 else {"rtol": 1e-10, "atol": 1e-12}
    torch.testing.assert_close(actual, expected, **tolerance)
    inputs = (feat, errors, *critic.parameters())
    expected_grad = torch.autograd.grad(expected, inputs, retain_graph=True)
    actual_grad = torch.autograd.grad(actual, inputs)
    for got, want in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(got, want, **tolerance)
    # The bootstrap must retain derivatives into state/error even when critic
    # parameters are frozen for the actor's objective.
    critic.requires_grad_(False)
    bootstrap = torch.compile(critic.forward, fullgraph=True, dynamic=False)
    want = critic(feat[-B:], goals[-B:], xi[-B:], root[-B:], errors[-B:]).sum()
    got = bootstrap(feat[-B:], goals[-B:], xi[-B:], root[-B:], errors[-B:]).sum()
    for actual, expected in zip(torch.autograd.grad(got, (feat, errors)),
                                torch.autograd.grad(want, (feat, errors)), strict=True):
        torch.testing.assert_close(actual, expected, **tolerance)
    rewards, done = torch.randn(H, B, dtype=dtype), torch.rand(H, B, dtype=dtype)
    values = torch.randn(H + 1, B, dtype=dtype)
    compiled_gae = torch.compile(gae_targets, fullgraph=True, dynamic=False)
    torch.testing.assert_close(compiled_gae(rewards, done, values, .99, .95),
                               gae_targets(rewards, done, values, .99, .95), **tolerance)


def test_fused_update_preserves_metrics_gradients_optimizer_ema_and_rng(monkeypatch):
    _aot_compile_spy(monkeypatch)
    eager, fused = [_trainer(monkeypatch, enabled=enabled) for enabled in (False, True)]
    fused.wm.load_state_dict(eager.wm.state_dict())
    fused.policy.load_state_dict(eager.policy.state_dict())
    fused.critic.load_state_dict(eager.critic.state_dict())
    fused.ema.module.load_state_dict(eager.ema.module.state_dict())

    def imagine(wm, policy, lib, start, horizon, future_frames, **kwargs):
        obs, goal = start.obs, start.goal
        encode = kwargs.get("encode_goal_fn", policy.encode_goal)
        decode = kwargs.get("decode_from_latent_fn", policy.decode_from_latent)
        latent = encode(goal)
        states = [torch.zeros(len(obs), 17)]
        actions = []
        for _ in range(horizon):
            action = decode(obs, goal, latent)
            states.append(torch.cat((action * .01, states[-1][:, 2:]), -1))
            actions.append(action)
        actions = torch.stack(actions)
        return ImagRollout(
            states=torch.stack(states), actions=actions, rewards=-(actions.square().mean(-1)) * .01,
            dhat=torch.zeros(horizon + 1, len(obs)), survival=torch.ones(horizon + 1, len(obs)),
            goals=goal[None].expand(horizon + 1, -1, -1),
            e_root=torch.zeros(horizon + 1, len(obs), 4),
            frames=torch.zeros(horizon + 1, len(obs), dtype=torch.long), reward_terms={},
        )

    monkeypatch.setattr(policy_trainer, "imagine", imagine)
    obs, goal = torch.randn(4, 90), torch.randn(4, 30)
    start = SimpleNamespace(obs=obs, goal=goal, xi=torch.randn(4, 6))
    buffer = SimpleNamespace(sample_starts=lambda *args, **kwargs: start)
    # Supply privileged reference fields at the real critic's expected width.
    for trainer in (eager, fused):
        trainer._critic_inputs = lambda ro, trainer=trainer: (
            ro.states, torch.nn.functional.pad(ro.goals, (0, trainer.critic.goal_scale.numel() - 30)),
            ro.e_root, torch.zeros(*ro.states.shape[:2], trainer.critic.error_dim),
        )
    for update in range(2):
        results, rng = [], []
        for trainer in (eager, fused):
            torch.manual_seed(92 + update)
            results.append(trainer.update(buffer))
            rng.append(torch.get_rng_state())
        for name in results[0]:
            assert results[1][name] == pytest.approx(results[0][name], rel=4e-5, abs=4e-6), name
        for name in ("policy", "critic"):
            torch.testing.assert_close(getattr(fused, name).state_dict(), getattr(eager, name).state_dict(),
                                       rtol=4e-5, atol=4e-6)
        torch.testing.assert_close(fused.ema.state_dict(), eager.ema.state_dict(), rtol=4e-5, atol=4e-6)
        assert fused.ema.num_updates == eager.ema.num_updates == update + 1
        assert {int(v["step"]) for v in fused.opt.state.values()} == {update + 1}
        assert {int(v["step"]) for v in fused.critic_opt.state.values()} == {4 * (update + 1)}
        torch.testing.assert_close(rng[0], rng[1], rtol=0, atol=0)


def _cuda_trainer(components, *, fused, activation_checkpoint, obs_noise=False,
                  horizon=2, critic_epochs=2, critic_minibatches=2, compile_steps=False):
    wm, actor, critic, library, _buffer = components
    cfg = PolicyTrainerCfg(
        horizon=horizon, batch_size=2, history=5, future_frames=5,
        observation_layout="isaac_khan", compile="default" if fused or compile_steps else None,
        compile_networks=fused, cuda_graph=fused or compile_steps, cuda_graph_warmup=3,
        critic_value_bound=True, critic_epochs=critic_epochs, critic_minibatches=critic_minibatches,
        activation_checkpoint=activation_checkpoint, obs_noise=obs_noise,
    )
    return PolicyTrainer(copy.deepcopy(wm), copy.deepcopy(actor), copy.deepcopy(critic), library, cfg)


def _cuda_models_match(actual, expected):
    for name in ("policy", "critic"):
        torch.testing.assert_close(getattr(actual, name).state_dict(), getattr(expected, name).state_dict(),
                                   rtol=5e-4, atol=5e-6)
    torch.testing.assert_close(actual.ema.state_dict(), expected.ema.state_dict(), rtol=5e-4, atol=5e-6)


def _cuda_steps(optimizer):
    return {int(value["step"]) for value in optimizer.state.values() if "step" in value}


def _parameter_error_report(actual, expected):
    """Report both absolute error and relative norm without zero-entry inflation."""
    differences, references = [], []
    for left, right in zip(actual.parameters(), expected.parameters(), strict=True):
        differences.append((left.detach() - right.detach()).double().flatten())
        references.append(right.detach().double().flatten())
    delta, reference = torch.cat(differences), torch.cat(references)
    return {"max_abs": float(delta.abs().max()),
            "relative_l2": float(delta.norm() / reference.norm().clamp_min(1e-12))}


def test_fp64_heading_coordinate_is_null_for_critic_values_and_gradients():
    # Rz(-yaw(R)) @ R has element [1,0] == 0 analytically. Its row-major
    # first-two-column representation is coordinate 3 after root height.
    _, critic = _networks(torch.float64)
    spec = StateSpec(StateSpecCfg(), "cpu", dtype=torch.float64, num_joints=2)
    state = torch.randn(64, spec.state_dim, dtype=torch.float64)
    state[:, 3:7] /= state[:, 3:7].norm(dim=-1, keepdim=True)
    features = spec.canonical(state)
    assert features[:, 3].abs().max() < 1e-13
    inputs = (features, torch.randn(64, critic.goal_scale.numel(), dtype=torch.float64),
              torch.randn(64, 6, dtype=torch.float64), torch.randn(64, 4, dtype=torch.float64),
              torch.randn(64, critic.error_dim, dtype=torch.float64))
    values = critic(*inputs)
    values.square().mean().backward()
    first = critic.mlp.net[0]
    assert first.weight.grad[:, 3].abs().max() < 1e-13
    assert first.weight.grad.abs().max() > 1e-4
    with torch.no_grad():
        # A large change in this unidentifiable weight leaves the exact
        # function unchanged; FP32 drift is assessed by its actual contribution.
        first.weight[:, 3].add_(2.)
        torch.testing.assert_close(critic(*inputs), values, rtol=1e-12, atol=1e-13)


def _assert_h32_functional_parity(actual, expected, observations, xi):
    """Keep strict parameter checks except the analytically null R[1,0] input.

    Heading removal makes local rotation R[1,0] exactly zero. FP32 computes a
    small residual there; Adam can amplify its ~1e-9 gradients near eps=1e-8.
    Its weight is unidentifiable, so verify the FP64 geometry and its bounded
    contribution instead of treating a null-space weight as a model error.
    All informative weights retain the original numerical tolerance.
    """
    torch.testing.assert_close(actual.policy.state_dict(), expected.policy.state_dict(),
                               rtol=5e-4, atol=5e-6)
    torch.testing.assert_close(actual.ema.state_dict(), expected.ema.state_dict(), rtol=5e-4, atol=5e-6)
    left, right = actual.critic.state_dict(), expected.critic.state_dict()
    for name in left:
        if name == "mlp.net.0.weight":
            informative = lambda value: torch.cat((value[:, :3], value[:, 4:]), dim=1)
            torch.testing.assert_close(informative(left[name]), informative(right[name]),
                                       rtol=5e-4, atol=5e-6)
        else:
            torch.testing.assert_close(left[name], right[name], rtol=5e-4, atol=5e-6)
    null_amplitude, oracle_amplitude = 0., 0.
    for trainer in (actual, expected):
        observation = observations[id(trainer)]
        null_amplitude = max(null_amplitude, float(observation["features"][..., 3].abs().max()))
        oracle = trainer.wm.spec.canonical(observation["states"].double())
        oracle_amplitude = max(oracle_amplitude, float(oracle[..., 3].abs().max()))
    assert null_amplitude < 1e-6
    assert oracle_amplitude < 1e-13
    weight_difference = (left["mlp.net.0.weight"][:, 3] - right["mlp.net.0.weight"][:, 3]).abs().max()
    contribution_bound = float(weight_difference) * null_amplitude
    assert contribution_bound < 1e-9
    reference = observations[id(expected)]
    H1, B, _ = reference["features"].shape
    inputs = (reference["features"].flatten(0, 1), reference["goals"].flatten(0, 1),
              xi.expand(H1 * B, -1), reference["root"].flatten(0, 1),
              reference["errors"].flatten(0, 1))
    with torch.no_grad():
        old_value, new_value = expected.critic(*inputs), actual.critic(*inputs)
    torch.testing.assert_close(new_value, old_value, rtol=5e-4, atol=5e-6)
    old_actions = reference["actions"]
    new_actions = observations[id(actual)]["actions"]
    torch.testing.assert_close(new_actions, old_actions, rtol=5e-4, atol=5e-6)
    return {"null_feature_fp32_max_abs": null_amplitude,
            "null_feature_fp64_max_abs": oracle_amplitude,
            "null_weight_contribution_bound": contribution_bound,
            "critic_value_max_abs_same_inputs": float((new_value - old_value).abs().max()),
            "actor_action_max_abs": float((new_actions - old_actions).abs().max())}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("sampling", ["multinomial", "cdf"])
def test_cuda_h32_fusion_preserves_twenty_critic_steps_and_reports_error(tmp_path, monkeypatch, sampling):
    # The full-size benchmark checks runtime but samples different replay rows.
    # Here actual Khan starts coincide and every Critic permutation is fixed.
    # The d98f4df baseline already compiles ImagStep and captures the complete
    # update; only the new actor/critic network compilation differs here.
    from test_policy_update_capture import _components

    components = _components(tmp_path, "cuda:0", goal_noise=True)
    buffer = components[-1]
    buffer.sampling = sampling
    eager, fused = [_cuda_trainer(
        components, fused=enabled, activation_checkpoint=False,
        horizon=32, critic_epochs=5, critic_minibatches=4, compile_steps=True,
    ) for enabled in (False, True)]
    monkeypatch.setattr(torch, "randperm", lambda n, *, device: torch.arange(n, device=device))
    initial_wm = copy.deepcopy(fused.wm.state_dict())
    initial_buffers = {key: value.clone() for key, value in fused.policy.named_buffers()}
    observations = {}
    for trainer in (eager, fused):
        original_inputs = trainer._critic_inputs
        identity = id(trainer)

        def observed_inputs(ro, original=original_inputs, identity=identity):
            result = original(ro)
            features, goals, root, errors = result
            observations[identity] = {
                "features": features.detach(), "goals": goals.detach(), "root": root.detach(),
                "errors": errors.detach(), "actions": ro.actions.detach(), "states": ro.states.detach(),
            }
            return result

        trainer._critic_inputs = observed_inputs
    xi = eager.wm.spec.normalize_xi(buffer.xi[0, 0])[None]
    report = {"sampling": sampling, "horizon": 32, "actor_updates": 0,
              "critic_steps_per_update": 20, "metric_max_abs": 0., "parameters": {}}
    try:
        for update in range(5):
            expected, actual = eager.update(buffer), fused.update(buffer)
            for key in expected:
                assert actual[key] == pytest.approx(expected[key], rel=8e-4, abs=8e-6), key
                report["metric_max_abs"] = max(report["metric_max_abs"], abs(actual[key] - expected[key]))
            functional = _assert_h32_functional_parity(fused, eager, observations, xi)
            report["functional"] = {key: max(value, report.get("functional", {}).get(key, 0.))
                                    for key, value in functional.items()}
            assert _cuda_steps(fused.opt) == {update + 1}
            assert _cuda_steps(fused.critic_opt) == {20 * (update + 1)}
            assert fused.ema.num_updates == update + 1
            torch.testing.assert_close(fused.wm.state_dict(), initial_wm, rtol=0, atol=0)
            torch.testing.assert_close(dict(fused.policy.named_buffers()), initial_buffers, rtol=0, atol=0)
            assert all(parameter.grad is None for parameter in fused.wm.parameters())
        assert fused._captured_update.captured and fused._captured_update.phase == "replay"
        report["actor_updates"] = fused.ema.num_updates
        report["parameters"] = {name: _parameter_error_report(getattr(fused, name), getattr(eager, name))
                                for name in ("policy", "critic")}
        print("FUSION_EQUIVALENCE " + json.dumps(report, sort_keys=True))
    finally:
        eager.reset_capture()
        fused.reset_capture()
        torch._dynamo.reset()
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("activation_checkpoint", [False, True])
def test_cuda_inductor_complete_actor_critic_graph_matches_eager_and_restores(tmp_path, activation_checkpoint, monkeypatch):
    # Reuse the actual Khan FK/DR-v3 fixture. All replay states are identical,
    # making different valid-row RNG draws irrelevant to mathematical parity.
    from test_policy_update_capture import _components

    components = _components(tmp_path, "cuda:0", goal_noise=True)
    buffer = components[-1]
    buffer.sampling = "cdf" if activation_checkpoint else "multinomial"
    eager = _cuda_trainer(components, fused=False, activation_checkpoint=activation_checkpoint)
    fused = _cuda_trainer(components, fused=True, activation_checkpoint=activation_checkpoint)
    initial_wm = copy.deepcopy(fused.wm.state_dict())
    initial_buffers = {key: value.clone() for key, value in fused.policy.named_buffers()}
    restored = noisy = None
    original_randperm = torch.randperm
    # Initial replay states coincide, but t=0 and t=1 critic targets differ.
    # Match minibatch composition explicitly; graph/compiled RNG offset usage
    # need not match eager draws, and a last-minibatch metric depends on order.
    monkeypatch.setattr(torch, "randperm", lambda n, *, device: torch.arange(n, device=device))
    try:
        phases = []
        for update in range(6):
            lr = 0. if update == 4 else fused.cfg.lr * (.5 if update == 5 else 1.)
            eager.set_lr(lr)
            fused.set_lr(lr)
            before = copy.deepcopy(fused.policy.state_dict())
            expected, actual = eager.update(buffer), fused.update(buffer)
            phases.append(fused._captured_update.phase)
            for key in expected:
                assert actual[key] == pytest.approx(expected[key], rel=8e-4, abs=8e-6), key
            _cuda_models_match(fused, eager)
            assert _cuda_steps(fused.opt) == {update + 1}
            assert _cuda_steps(fused.critic_opt) == {4 * (update + 1)}
            assert fused.ema.num_updates == update + 1
            torch.testing.assert_close(fused.wm.state_dict(), initial_wm, rtol=0, atol=0)
            torch.testing.assert_close(dict(fused.policy.named_buffers()), initial_buffers, rtol=0, atol=0)
            assert all(parameter.grad is None for parameter in fused.wm.parameters())
            if update == 4:
                torch.testing.assert_close(fused.policy.state_dict(), before, rtol=0, atol=0)
        assert phases == ["warmup", "warmup", "warmup", "capture", "replay", "replay"]

        if not activation_checkpoint:
            # Real CUDA optimizer state and compiled methods must survive the
            # same restore/invalidation boundary used by Trainer.load().
            snapshot = copy.deepcopy({
                "wm": fused.wm.state_dict(), "policy": fused.policy.state_dict(),
                "critic": fused.critic.state_dict(), "policy_opt": fused.opt.state_dict(),
                "critic_opt": fused.critic_opt.state_dict(), "policy_ema": fused.ema.state_dict(),
                "policy_ema_updates": fused.ema.num_updates, "policy_ema_decay": fused.ema.decay,
            })
            restored = _cuda_trainer(components, fused=True, activation_checkpoint=False)
            for name in ("wm", "policy", "critic"):
                getattr(restored, name).load_state_dict(snapshot[name])
            restored.opt.load_state_dict(snapshot["policy_opt"])
            restored.critic_opt.load_state_dict(snapshot["critic_opt"])
            restored.ema.restore(restored.policy, snapshot)
            restored.reset_capture()
            for _ in range(4):
                fused.set_lr(7e-5)
                restored.set_lr(7e-5)
                expected, actual = fused.update(buffer), restored.update(buffer)
                _cuda_models_match(restored, fused)
                for key in expected:
                    assert actual[key] == pytest.approx(expected[key], rel=8e-4, abs=8e-6), key
            assert restored._captured_update.captured
            assert restored.ema.num_updates == 10
            assert _cuda_steps(restored.opt) == {10} and _cuda_steps(restored.critic_opt) == {40}

        # A separate graph records noise=True from the outset. Replay RNG and
        # shuffles cannot alter actions on this constant-state fixture; with
        # LR=0, changed actions specifically demonstrate fresh policy noise.
        monkeypatch.setattr(torch, "randperm", original_randperm)
        noisy = _cuda_trainer(components, fused=True, activation_checkpoint=activation_checkpoint, obs_noise=True)
        noisy.set_lr(0.)
        initial_actor = copy.deepcopy(noisy.policy.state_dict())
        initial_noise_wm = copy.deepcopy(noisy.wm.state_dict())
        actions = []
        for index in range(6):
            metrics = noisy.update(buffer)
            if index >= 3:
                actions.append(metrics["action_abs_mean"])
        assert noisy._captured_update.captured
        assert len(set(actions)) == 3
        torch.testing.assert_close(noisy.policy.state_dict(), initial_actor, rtol=0, atol=0)
        torch.testing.assert_close(noisy.wm.state_dict(), initial_noise_wm, rtol=0, atol=0)
        assert noisy.ema.num_updates == 6
        assert _cuda_steps(noisy.opt) == {6} and _cuda_steps(noisy.critic_opt) == {24}
    finally:
        for trainer in (eager, fused, restored, noisy):
            if trainer is not None:
                trainer.reset_capture()
        torch._dynamo.reset()
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_captured_compiled_update_reads_changed_actor_and_wm_normalizers(tmp_path, monkeypatch):
    from test_policy_update_capture import _components

    components = _components(tmp_path, "cuda:0")
    buffer = components[-1]
    buffer.sampling = "cdf"
    # The ordinary reference needs no checkpointing. This also avoids fresh
    # TorchScript profiling changing saved-tensor order during its very first
    # eager recomputation, before any physical-simulation/JIT warm-up.
    eager = _cuda_trainer(components, fused=False, activation_checkpoint=False)
    fused = _cuda_trainer(components, fused=True, activation_checkpoint=True)
    monkeypatch.setattr(torch, "randperm", lambda n, *, device: torch.arange(n, device=device))
    for trainer in (eager, fused):
        trainer.set_lr(0.)
    parameters = {name: copy.deepcopy(dict(getattr(fused, name).named_parameters()))
                  for name in ("wm", "policy", "critic")}
    addresses = {name: {key: value.data_ptr() for key, value in getattr(fused, name).named_buffers()}
                 for name in ("wm", "policy", "critic")}
    keys = {name: tuple(getattr(fused, name).state_dict()) for name in ("wm", "policy", "critic")}
    try:
        for _ in range(4):
            expected, previous = eager.update(buffer), fused.update(buffer)
        assert fused._captured_update.captured
        states = components[3].ref71(torch.full((16,), 4, dtype=torch.long, device="cuda:0")).clone()
        states[:, 7:34] += torch.randn(16, 27, device="cuda:0") * .02
        states[:, 34:] += torch.randn(16, 33, device="cuda:0") * .03
        actions = torch.randn(16, 27, device="cuda:0") * .1
        obs = torch.randn(16, 465, device="cuda:0") * .3 + .2
        goal = torch.randn(16, 155, device="cuda:0") * .2 + .3
        for trainer in (eager, fused):
            trainer.policy.update_normalization(obs, goal)
            trainer.wm.update_normalization(states, actions)
        # Contents changed, storage did not. The already-recorded graph must
        # see these new statistics, without needing invalidation or recapture.
        expected_buffers = {name: {key: value.clone() for key, value in getattr(fused, name).named_buffers()}
                            for name in ("wm", "policy", "critic")}
        for _ in range(3):
            expected, actual = eager.update(buffer), fused.update(buffer)
            assert fused._captured_update.phase == "replay"
            for key in expected:
                assert actual[key] == pytest.approx(expected[key], rel=8e-4, abs=8e-6), key
            for name, reference_parameters in parameters.items():
                module = getattr(fused, name)
                torch.testing.assert_close(dict(module.named_parameters()), reference_parameters, rtol=0, atol=0)
                torch.testing.assert_close(dict(module.named_buffers()), expected_buffers[name], rtol=0, atol=0)
                assert {key: value.data_ptr() for key, value in module.named_buffers()} == addresses[name]
                assert tuple(module.state_dict()) == keys[name]
            torch.testing.assert_close(fused.ema.state_dict(), eager.ema.state_dict(), rtol=5e-4, atol=5e-6)
        assert abs(actual["action_abs_mean"] - previous["action_abs_mean"]) > 1e-6
        assert abs(actual["imag_reward_mean"] - previous["imag_reward_mean"]) > 1e-6
        assert fused._captured_update.calls == 7
        assert fused.ema.num_updates == 7
        assert _cuda_steps(fused.opt) == {7} and _cuda_steps(fused.critic_opt) == {28}
    finally:
        eager.reset_capture()
        fused.reset_capture()
        torch._dynamo.reset()
        gc.collect()
        torch.cuda.empty_cache()
