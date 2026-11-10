"""Checkpoint actor selection and deployment provenance with distinct EMA weights.

Tiny actors retain the real Khan residual-action and observation contracts. No
training run, simulation environment or private checkpoint is required.
"""

import copy
from dataclasses import asdict

import numpy as np
import pytest
import torch

from supermimic.eval import benchmark
from supermimic.policy.nets import Policy, PolicyCfg
from supermimic.robots.contracts import robot_contract
from supermimic.robots.khan_mini import load_profile


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_nonfinite_ema_rejected_without_hiding_valid_live(ema_checkpoint, bad_value):
    path, _, _, saved = ema_checkpoint
    name = next(key for key in saved["policy_ema"] if key.endswith("weight"))
    saved["policy_ema"][name] = torch.full_like(saved["policy_ema"][name], bad_value)
    torch.save(saved, path)
    with pytest.raises(ValueError, match="policy_ema.*nonfinite"):
        benchmark.load_policy_from_checkpoint(path, "cpu")
    loaded, _ = benchmark.load_policy_from_checkpoint(path, "cpu", policy_weights="live")
    assert loaded.checkpoint_policy_weights == "live"


@pytest.fixture
def ema_checkpoint(tmp_path):
    profile = load_profile()
    names = profile["joint_names"]
    q0 = torch.tensor([profile["default_joint_pos"][name] for name in names])
    scale = torch.tensor([profile["action_scale"][name] for name in names])
    cfg = PolicyCfg(
        encoder_hidden=(8,), decoder_hidden=(8,), latent_dim=4,
        num_joints=27, num_tracked_bodies=len(profile["tracked_bodies"]),
        goal_dim=155, history=5, future_frames=5, observation_layout="isaac_khan",
        obs_term_scale=(1.,) * 6, goal_term_scale=(1.,) * 3,
        action_mode="reference_residual",
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(137)
        live = Policy(cfg, default_joint_pos=q0, action_scale=scale).eval()
    ema = copy.deepcopy(live)
    with torch.no_grad():
        # Every input distinguishes the two actors, including all-zero inputs.
        ema.decoder.mlp.net[-1].bias.add_(.25)
    saved = {
        "iteration": 12,
        "policy": copy.deepcopy(live.state_dict()),
        "policy_ema": copy.deepcopy(ema.state_dict()),
        "policy_ema_updates": 11,
        "policy_ema_decay": .995,
        "config": {
            "robot": "khan_mini", "policy": asdict(cfg),
            "env": {"future_frames": 5, "observation_layout": "isaac_khan",
                    "anchor_obs": False},
            "world_model": {"state_spec": {"dt": .02}},
        },
        "robot_contract": robot_contract("khan_mini", "isaac_khan"),
    }
    path = tmp_path / "checkpoint.pt"
    torch.save(saved, path)
    return path, live, ema, saved


def _inputs(batch=3):
    rng = torch.Generator().manual_seed(19)
    return torch.randn(batch, 465, generator=rng), torch.randn(batch, 155, generator=rng)


def _without_ema(saved, availability):
    if availability == "missing":
        del saved["policy_ema"]
    else:
        saved["policy_ema"] = None


@pytest.mark.parametrize("selection,resolved", [("auto", "ema"), ("ema", "ema"), ("live", "live")])
def test_loader_selects_actor_and_preserves_bptt_gradients(ema_checkpoint, selection, resolved):
    path, live, ema, _ = ema_checkpoint
    loaded, _ = benchmark.load_policy_from_checkpoint(path, "cpu", policy_weights=selection)
    expected, other = (ema, live) if resolved == "ema" else (live, ema)
    obs, goal = _inputs()
    actual = loaded(obs, goal)
    torch.testing.assert_close(actual, expected(obs, goal), atol=0, rtol=0)
    assert (actual - other(obs, goal)).abs().max().item() > .2
    assert loaded.checkpoint_policy_weights == resolved
    assert loaded.checkpoint_policy_ema_updates == (11 if resolved == "ema" else None)
    assert loaded.checkpoint_policy_ema_decay == (.995 if resolved == "ema" else None)
    assert not loaded.training
    assert all(parameter.requires_grad for parameter in loaded.parameters())
    actual.sum().backward()
    # The inference loader also serves imagination benchmarks, whose gradients
    # must still propagate through the selected encoder and decoder.
    for module in (loaded.encoders["motion"], loaded.decoder):
        assert any(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
                   for parameter in module.parameters())


@pytest.mark.parametrize("availability", ["missing", "none"])
@pytest.mark.parametrize("selection", ["auto", "live"])
def test_legacy_and_disabled_ema_fall_back_to_live(ema_checkpoint, availability, selection):
    path, live, _, saved = ema_checkpoint
    _without_ema(saved, availability)
    torch.save(saved, path)
    loaded, _ = benchmark.load_policy_from_checkpoint(path, "cpu", policy_weights=selection)
    obs, goal = _inputs()
    torch.testing.assert_close(loaded(obs, goal), live(obs, goal), atol=0, rtol=0)
    assert loaded.checkpoint_policy_weights == "live"
    assert loaded.checkpoint_policy_ema_updates is None
    assert loaded.checkpoint_policy_ema_decay is None


@pytest.mark.parametrize("availability", ["missing", "none"])
def test_explicit_ema_without_weights_is_rejected(ema_checkpoint, availability):
    path, _, _, saved = ema_checkpoint
    _without_ema(saved, availability)
    torch.save(saved, path)
    with pytest.raises(ValueError, match="EMA.*no policy_ema"):
        benchmark.load_policy_from_checkpoint(path, "cpu", policy_weights="ema")


@pytest.mark.parametrize("selection", ["auto", "ema"])
@pytest.mark.parametrize("corruption", ["empty", "wrong_type", "missing_parameter", "wrong_shape"])
def test_invalid_ema_state_never_silently_uses_live(ema_checkpoint, selection, corruption):
    path, live, _, saved = ema_checkpoint
    if corruption == "empty":
        saved["policy_ema"] = {}
    elif corruption == "wrong_type":
        saved["policy_ema"] = ["not a state dict"]
    else:
        key = "encoders.motion.mlp.net.0.weight"
        if corruption == "missing_parameter":
            del saved["policy_ema"][key]
        else:
            saved["policy_ema"][key] = saved["policy_ema"][key][:1]
    torch.save(saved, path)
    error_type = ValueError if corruption in ("empty", "wrong_type") else RuntimeError
    message = ("policy_ema" if error_type is ValueError
               else "Missing key" if corruption == "missing_parameter" else "size mismatch")
    with pytest.raises(error_type, match=message):
        benchmark.load_policy_from_checkpoint(path, "cpu", policy_weights=selection)
    # The explicit live escape hatch ignores a damaged EMA copy.
    loaded, _ = benchmark.load_policy_from_checkpoint(path, "cpu", policy_weights="live")
    obs, goal = _inputs()
    torch.testing.assert_close(loaded(obs, goal), live(obs, goal), atol=0, rtol=0)


@pytest.mark.parametrize("buffer_name", ["reference_q_default", "reference_action_scale"])
@pytest.mark.parametrize("corruption", ["changed", "missing", "wrong_dtype"])
def test_ema_residual_physical_buffers_are_validated(ema_checkpoint, buffer_name, corruption):
    path, live, _, saved = ema_checkpoint
    if corruption == "missing":
        del saved["policy_ema"][buffer_name]
    elif corruption == "wrong_dtype":
        saved["policy_ema"][buffer_name] = saved["policy_ema"][buffer_name].double()
    else:
        saved["policy_ema"][buffer_name][0] += .01
    torch.save(saved, path)
    for selection in ("auto", "ema"):
        with pytest.raises(ValueError, match=buffer_name):
            benchmark.load_policy_from_checkpoint(path, "cpu", policy_weights=selection)
    loaded, _ = benchmark.load_policy_from_checkpoint(path, "cpu", policy_weights="live")
    obs, goal = _inputs()
    torch.testing.assert_close(loaded(obs, goal), live(obs, goal), atol=0, rtol=0)


@pytest.mark.parametrize("selection,resolved", [("auto", "ema"), ("ema", "ema"), ("live", "live")])
@pytest.mark.parametrize("layout", ["packed", "split"])
def test_onnx_metadata_and_runtime_use_selected_actor(
    ema_checkpoint, tmp_path, selection, resolved, layout,
):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from supermimic.deployment.onnx_export import export_policy

    path, live, ema, _ = ema_checkpoint
    output = tmp_path / f"actor_{layout}_{selection}.onnx"
    report = export_policy(path, output, input_layout=layout,
                           validation_samples=2, policy_weights=selection)
    exported = onnx.load(output)
    onnx.checker.check_model(exported, full_check=True)
    metadata = {item.key: item.value for item in exported.metadata_props}
    assert report["policy_weights"] == metadata["policy_weights"] == resolved
    assert report["policy_weights_requested"] == metadata["policy_weights_requested"] == selection
    assert report["policy_ema_updates"] == (11 if resolved == "ema" else None)
    assert report["policy_ema_decay"] == (.995 if resolved == "ema" else None)
    if resolved == "ema":
        assert metadata["policy_ema_updates"] == "11"
        assert metadata["policy_ema_decay"] == "0.995"
    else:
        assert "policy_ema_updates" not in metadata
        assert "policy_ema_decay" not in metadata
    assert report["validation"]["all_checks_passed"]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(output), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    expected, other = (ema, live) if resolved == "ema" else (live, ema)
    for batch in (1, 3):
        obs, goal = _inputs(batch)
        feeds = ({"obs": torch.cat([goal, obs], dim=-1).numpy()} if layout == "packed"
                 else {"policy_obs": obs.numpy(), "goal": goal.numpy()})
        actual = session.run(["actions"], feeds)[0]
        with torch.no_grad():
            wanted = expected(obs, goal).numpy()
            unwanted = other(obs, goal).numpy()
        np.testing.assert_allclose(actual, wanted, atol=1e-5, rtol=1e-5)
        assert float(np.max(np.abs(actual - unwanted))) > .2


@pytest.mark.parametrize("availability", ["missing", "none"])
def test_onnx_export_falls_back_to_live_for_legacy_or_disabled_ema(
    ema_checkpoint, tmp_path, availability,
):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from supermimic.deployment.onnx_export import export_policy

    path, _, _, saved = ema_checkpoint
    _without_ema(saved, availability)
    torch.save(saved, path)
    report = export_policy(path, tmp_path / "live_fallback.onnx", validation_samples=2)
    assert report["policy_weights"] == "live"
    assert report["metadata"]["policy_weights"] == "live"
    assert report["policy_ema_updates"] is None


def test_omitted_legacy_hidden_sizes_keep_original_four_layers(ema_checkpoint, monkeypatch):
    path, _, _, saved = ema_checkpoint
    del saved["config"]["policy"]["encoder_hidden"]
    del saved["config"]["policy"]["decoder_hidden"]
    torch.save(saved, path)
    captured = {}

    def inspect_construction(cfg, **reference_parameters):
        captured["cfg"] = cfg
        # Intercept construction to inspect historical dimensions without
        # allocating the large original four-layer actor in this CPU unit test.
        raise RuntimeError("legacy architecture inspected")

    monkeypatch.setattr(benchmark, "Policy", inspect_construction)
    with pytest.raises(RuntimeError, match="legacy architecture inspected"):
        benchmark.load_policy_from_checkpoint(path, "cpu")
    assert captured["cfg"].encoder_hidden == (2048, 1024, 512, 256)
    assert captured["cfg"].decoder_hidden == (2048, 1024, 512, 256)


def test_unknown_policy_weight_selection_is_rejected(ema_checkpoint):
    path, _, _, _ = ema_checkpoint
    with pytest.raises(ValueError, match="policy_weights must be"):
        benchmark.load_policy_from_checkpoint(path, "cpu", policy_weights="unknown")
