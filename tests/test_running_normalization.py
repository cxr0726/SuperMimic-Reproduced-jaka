"""Fresh-data empirical moments, actor history semantics, and deployment parity."""

import copy
from dataclasses import asdict

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from supermimic.normalization import RunningNormalizer
from supermimic.policy.ema import ActorEMA
from supermimic.policy.nets import Policy, PolicyCfg


def _cfg(**kwargs):
    return PolicyCfg(
        encoder_hidden=(12,), decoder_hidden=(12,), latent_dim=4,
        history=5, future_frames=5, observation_layout="isaac_khan",
        num_joints=27, num_tracked_bodies=15, goal_dim=155,
        obs_term_scale=(1.0,) * 6, goal_term_scale=(1.0,) * 3,
        running_normalization=True, **kwargs,
    )


def test_empirical_moments_match_concatenated_data_and_official_eps():
    normalizer = RunningNormalizer(4)
    rng = torch.Generator().manual_seed(93)
    first = torch.randn(17, 4, generator=rng) * 3 + 4
    second = torch.randn(5, 4, generator=rng) * 5 - 2
    addresses = [buffer.data_ptr() for buffer in normalizer.buffers()]
    normalizer.update(first)
    normalizer.update(second)
    joined = torch.cat([first, second])
    assert normalizer.count == 22
    torch.testing.assert_close(normalizer.mean, joined.mean(0))
    torch.testing.assert_close(normalizer.std, joined.std(0, unbiased=False))
    torch.testing.assert_close(normalizer(second), (second - joined.mean(0)) /
                               (joined.std(0, unbiased=False) + .01))
    assert addresses == [buffer.data_ptr() for buffer in normalizer.buffers()]


def test_forward_never_updates_and_eval_explicit_update_is_frozen():
    normalizer = RunningNormalizer(3)
    before = copy.deepcopy(normalizer.state_dict())
    inputs = torch.randn(8, 3, requires_grad=True)
    normalizer(inputs).sum().backward()
    assert torch.isfinite(inputs.grad).all()
    normalizer.eval().update(inputs)
    for name, value in before.items():
        torch.testing.assert_close(normalizer.state_dict()[name], value, atol=0, rtol=0)


def test_empty_batches_and_constant_features_are_finite():
    normalizer = RunningNormalizer(2)
    normalizer.update(torch.empty(0, 2))
    assert normalizer.count == 0
    normalizer.update(torch.full((8, 2), 1.25))
    assert torch.equal(normalizer.std, torch.zeros(2))
    torch.testing.assert_close(normalizer(torch.full((3, 2), 1.26)), torch.ones(3, 2),
                               atol=2e-6, rtol=0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_update_does_not_poison_statistics(bad):
    normalizer = RunningNormalizer(2)
    normalizer.update(torch.randn(3, 2))
    before = copy.deepcopy(normalizer.state_dict())
    with pytest.raises(RuntimeError, match="nonfinite"):
        normalizer.update(torch.tensor([[bad, 1.0]]))
    for name, value in before.items():
        torch.testing.assert_close(normalizer.state_dict()[name], value, atol=0, rtol=0)


def test_finite_but_unrepresentable_fp32_moments_do_not_poison_buffers():
    normalizer = RunningNormalizer(1)
    before = copy.deepcopy(normalizer.state_dict())
    with pytest.raises(RuntimeError, match="FP32 range"):
        normalizer.update(torch.tensor([[1e30], [-1e30]]))
    for name, value in before.items():
        torch.testing.assert_close(normalizer.state_dict()[name], value, atol=0, rtol=0)


@pytest.mark.parametrize("corruption", ["count", "variance", "std", "dtype"])
def test_checkpoint_rejects_invalid_running_statistics(corruption):
    normalizer = RunningNormalizer(2)
    normalizer.update(torch.randn(7, 2))
    before = copy.deepcopy(normalizer.state_dict())
    state = copy.deepcopy(before)
    if corruption == "count":
        state["count"].fill_(-1)
    elif corruption == "variance":
        state["_var"][0] = -1
    elif corruption == "std":
        state["_std"][0] += 2
    else:
        state["count"] = state["count"].float()
    with pytest.raises(RuntimeError, match="normalizer"):
        normalizer.load_state_dict(state)
    for name, value in before.items():
        torch.testing.assert_close(normalizer.state_dict()[name], value, atol=0, rtol=0)


def _distributed_moments_worker(rank, init_path, output_dir):
    dist.init_process_group("gloo", init_method=f"file://{init_path}", rank=rank, world_size=2)
    try:
        normalizer = RunningNormalizer(2)
        normalizer.update(torch.tensor([[1., 2.], [3., 6.]]) if rank == 0 else
                          torch.tensor([[10., -3.]]))
        torch.save(normalizer.state_dict(), f"{output_dir}/rank{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_distributed_moments_match_global_samples_with_unequal_rank_batches(tmp_path):
    mp.spawn(_distributed_moments_worker, args=(str(tmp_path / "init"), str(tmp_path)),
             nprocs=2, join=True)
    reference = RunningNormalizer(2)
    reference.update(torch.tensor([[1., 2.], [3., 6.], [10., -3.]]))
    for rank in range(2):
        actual = torch.load(tmp_path / f"rank{rank}.pt", weights_only=True)
        for name, expected in reference.state_dict().items():
            torch.testing.assert_close(actual[name], expected, rtol=0, atol=0)


def _pack_khan_history(raw_sensor, orientation):
    # raw_sensor is B,H,87 in gravity/omega/q/qd/action order.
    parts, offset = [orientation], 0
    for width, scale in ((3, 1.), (3, .5), (27, 1.), (27, .2), (27, 1.)):
        parts.append((raw_sensor[..., offset:offset + width] * scale).flatten(1))
        offset += width
    return torch.cat(parts, -1)


def test_actor_counts_only_latest_sensor_and_restores_physical_units():
    policy = Policy(_cfg())
    rng = torch.Generator().manual_seed(8)
    raw = torch.randn(11, 5, 87, generator=rng)
    raw[:, :-1] += 1000  # old history must not affect moment updates
    orientation = torch.randn(11, 30, generator=rng)
    goal = torch.randn(11, 155, generator=rng)
    actual_encoder_goal = goal + 2
    packed = _pack_khan_history(raw, orientation)
    policy.update_normalization(packed, goal, encoder_goal=actual_encoder_goal)
    observation_norm = policy.decoder.observation_normalizer
    assert observation_norm.sensor.count == observation_norm.orientation.count == 11
    assert policy.encoders["motion"].normalizer.count == 11
    torch.testing.assert_close(observation_norm.sensor.mean, raw[:, -1].mean(0))
    torch.testing.assert_close(observation_norm.sensor.std, raw[:, -1].std(0, unbiased=False))
    torch.testing.assert_close(policy.encoders["motion"].normalizer.mean, actual_encoder_goal.mean(0))
    torch.testing.assert_close(observation_norm.orientation.mean, orientation.mean(0))
    normalized = observation_norm(packed)
    reference_raw = (raw - raw[:, -1].mean(0)) / (raw[:, -1].std(0, unbiased=False) + .01)
    reference_ori = (orientation - orientation.mean(0)) / (orientation.std(0, unbiased=False) + .01)
    # Reference is already normalized, so pack without the environment scaling.
    parts, offset = [reference_ori], 0
    for width in (3, 3, 27, 27, 27):
        parts.append(reference_raw[..., offset:offset + width].flatten(1))
        offset += width
    torch.testing.assert_close(normalized, torch.cat(parts, -1), rtol=2e-5, atol=2e-4)


def test_running_goal_normalization_does_not_change_residual_skip():
    cfg = _cfg(action_mode="reference_residual", goal_noise_z=.01, goal_noise_joint_pos=.05)
    policy = Policy(cfg, default_joint_pos=torch.zeros(27), action_scale=torch.full((27,), .5))
    with torch.no_grad():
        for parameter in policy.decoder.parameters():
            parameter.zero_()
    obs, goal = torch.randn(9, 465), torch.randn(9, 155)
    encoder_goal = policy.make_encoder_goal(goal)
    policy.update_normalization(obs, goal, encoder_goal=encoder_goal)
    action = policy(obs, goal, encoder_goal=encoder_goal)
    torch.testing.assert_close(.5 * action, goal[:, 47:74], atol=0, rtol=0)


def test_forward_and_imagined_gradients_leave_stats_unchanged_and_ema_copies_buffers():
    policy = Policy(_cfg())
    ema = ActorEMA(policy)
    obs, goal = torch.randn(7, 465), torch.randn(7, 155)
    policy.update_normalization(obs, goal)
    before = {name: value.clone() for name, value in policy.named_buffers()}
    policy(obs.requires_grad_(), goal.requires_grad_()).square().mean().backward()
    assert torch.isfinite(obs.grad).all() and torch.isfinite(goal.grad).all()
    for name, value in policy.named_buffers():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
    ema.sync_buffers(policy)
    assert ema.num_updates == 0
    for (_, actual), (_, expected) in zip(ema.module.named_buffers(), policy.named_buffers(), strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("input_layout", ["packed", "split"])
def test_normalized_checkpoint_and_onnx_keep_frozen_statistics(tmp_path, input_layout):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from supermimic.deployment.onnx_export import export_policy
    from supermimic.eval.benchmark import load_policy_from_checkpoint
    from supermimic.robots.contracts import robot_contract

    cfg = _cfg()
    policy = Policy(cfg)
    obs, goal = torch.randn(19, 465) * 2 + 4, torch.randn(19, 155) * 3 - 2
    policy.update_normalization(obs, goal)
    policy.eval()
    checkpoint = {
        "iteration": 20, "policy": policy.state_dict(),
        "config": {"robot": "khan_mini", "policy": asdict(cfg),
                   "env": {"future_frames": 5, "observation_layout": "isaac_khan", "anchor_obs": False},
                   "world_model": {"state_spec": {"dt": .02}}},
        "robot_contract": robot_contract("khan_mini", "isaac_khan"),
    }
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)
    loaded, _ = load_policy_from_checkpoint(path, "cpu")
    torch.testing.assert_close(loaded(obs, goal), policy(obs, goal), atol=0, rtol=0)
    output = tmp_path / f"normalized_{input_layout}.onnx"
    report = export_policy(path, output, input_layout=input_layout, validation_samples=3)
    assert report["schema"]["running_normalization"]
    assert report["schema"]["normalization"]["embedded_in_onnx"]
    assert report["metadata"]["running_normalization"] == "true"
    onnx.checker.check_model(onnx.load(output), full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(output), options, providers=["CPUExecutionProvider"])
    for batch in (1, 3):
        obs_batch, goal_batch = obs[:batch], goal[:batch]
        feeds = ({"obs": torch.cat([goal_batch, obs_batch], -1).numpy()} if input_layout == "packed"
                 else {"policy_obs": obs_batch.numpy(), "goal": goal_batch.numpy()})
        with torch.no_grad():
            expected = policy(obs_batch, goal_batch).numpy()
        actual = session.run(["actions"], feeds)[0]
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
    assert loaded.encoders["motion"].normalizer.count == 19


def test_historical_fixed_scale_policy_has_no_normalizer_state():
    cfg = _cfg()
    cfg.running_normalization = False
    policy = Policy(cfg)
    assert not any("normalizer" in name for name in policy.state_dict())
