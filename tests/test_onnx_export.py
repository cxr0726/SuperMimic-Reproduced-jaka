"""CPU deployment-export contract and ONNX Runtime parity for the Khan actor.

Small networks exercise the production observation/robot interface without
depending on a private training run or creating a simulation environment.
"""

import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from supermimic.policy.nets import Policy, PolicyCfg
from supermimic.robots.contracts import robot_contract
from supermimic.robots.khan_mini import load_profile


@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(13)
    cfg = PolicyCfg(
        encoder_hidden=(32, 16), decoder_hidden=(32, 16), latent_dim=32,
        history=5, future_frames=5, observation_layout="isaac_khan",
        obs_term_scale=(1.,) * 6, goal_term_scale=(1.,) * 3,
        goal_dim=155, num_joints=27,
    )
    policy = Policy(cfg).eval()
    saved = {
        "iteration": 100000,
        "config": {
            "robot": "khan_mini", "policy": asdict(cfg),
            "env": {"future_frames": 5, "observation_layout": "isaac_khan",
                    "anchor_obs": False},
            "world_model": {"state_spec": {"dt": .02}},
        },
        "policy": policy.state_dict(),
        "robot_contract": robot_contract("khan_mini", "isaac_khan"),
    }
    path = tmp_path / "ckpt_0100000.pt"
    torch.save(saved, path)
    return path, policy, saved


def _metadata(path):
    model = onnx.load(path)
    onnx.checker.check_model(model)
    return model, {item.key: item.value for item in model.metadata_props}


@pytest.mark.parametrize("layout", ["packed", "split"])
def test_exported_actor_matches_pytorch_for_dynamic_batches(checkpoint, tmp_path, layout):
    from supermimic.deployment.onnx_export import export_policy

    ckpt, policy, _ = checkpoint
    output = tmp_path / f"actor_{layout}.onnx"
    report = export_policy(ckpt, output, input_layout=layout, validation_samples=8)
    assert output.is_file()
    assert isinstance(report, dict)
    model, _ = _metadata(output)
    assert len(model.graph.output) == 1
    assert model.graph.output[0].name == "actions"
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    assert [value.name for value in session.get_inputs()] == (
        ["obs"] if layout == "packed" else ["policy_obs", "goal"]
    )
    generator = torch.Generator().manual_seed(19)
    for batch in (1, 3):
        policy_obs = torch.randn(batch, 465, generator=generator)
        goal = torch.randn(batch, 155, generator=generator)
        feeds = ({"obs": torch.cat([goal, policy_obs], -1).numpy()} if layout == "packed"
                 else {"policy_obs": policy_obs.numpy(), "goal": goal.numpy()})
        actual = session.run(["actions"], feeds)[0]
        with torch.no_grad():
            expected = policy(policy_obs, goal).numpy()
        assert actual.shape == (batch, 27)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    # A deployment graph contains exactly the actor, independent of train-only
    # modules (the fixture deliberately has no WM/critic checkpoint entries).
    for value in model.graph.initializer:
        assert "critic" not in value.name.lower()
        assert "world_model" not in value.name.lower()
        assert value.data_location != onnx.TensorProto.EXTERNAL


def test_export_metadata_preserves_current_robot_and_observation_semantics(checkpoint, tmp_path):
    from supermimic.deployment.onnx_export import export_policy

    ckpt, _, saved = checkpoint
    output = tmp_path / "actor.onnx"
    export_policy(ckpt, output, validation_samples=4)
    model, metadata = _metadata(output)
    profile = load_profile()
    names = profile["joint_names"]
    assert metadata["joint_names"].split(",") == names
    for key, profile_key in (
        ("joint_stiffness", "stiffness"),
        ("joint_damping", "damping"),
        ("default_joint_pos", "default_joint_pos"),
        ("action_scale", "action_scale"),
    ):
        actual = np.fromstring(metadata[key], sep=",")
        expected = [profile[profile_key][name] for name in names]
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    assert metadata["anchor_body_name"] == "base_link"
    assert metadata["body_names"].split(",") == profile["tracked_bodies"]
    assert metadata["observation_names"].split(",") == [
        "command", "motion_anchor_ori_b", "projected_gravity", "base_ang_vel",
        "joint_pos", "joint_vel", "actions",
    ]
    assert metadata["command_names"] == "motion"
    assert json.loads(metadata["robot_contract"]) == saved["robot_contract"]
    schema = json.loads(metadata["onnx_deployment_schema"])
    assert schema["dimensions"] == {"obs": 620, "goal": 155, "policy_obs": 465, "actions": 27}
    assert schema["packed_order"] == ["goal", "policy_obs"]
    assert schema["history_layout"] == "oldest_to_newest_per_term"
    assert schema["future_offsets"] == [0, 1, 2, 3, 4]
    assert not schema["reference_motion_embedded"]
    assert not schema["running_normalization"]
    assert not schema["actor_stateful"]
    assert schema["action"]["clip"] is None
    assert not schema["action"]["tanh"]
    expected_policy_terms = [
        ("motion_anchor_ori_b", 0, 30, 1.),
        ("projected_gravity", 30, 45, 1.),
        ("base_ang_vel", 45, 60, .5),
        ("joint_pos", 60, 195, 1.),
        ("joint_vel", 195, 330, .2),
        ("actions", 330, 465, 1.),
    ]
    assert [(term["name"], term["start"], term["stop"], term["caller_scale"])
            for term in schema["policy_terms"]] == expected_policy_terms
    assert [(term["packed_start"], term["packed_stop"])
            for term in schema["policy_terms"]] == [
                (155 + start, 155 + stop) for _, start, stop, _ in expected_policy_terms
            ]
    # Current checkpoint scale buffers are identity because the actual sensor
    # values are already scaled: exporting may not apply omega/qd scaling twice.
    assert saved["policy"]["decoder.scale"].shape == (465,)
    assert torch.equal(saved["policy"]["decoder.scale"], torch.ones(465))
    assert model.graph.input[0].type.tensor_type.shape.dim[1].dim_value == 620
    assert model.graph.output[0].type.tensor_type.shape.dim[1].dim_value == 27


def test_packed_actor_splits_command_before_policy_history(checkpoint):
    from supermimic.deployment.onnx_export import PackedActor

    _, policy, _ = checkpoint
    obs = torch.randn(2, 620)
    with torch.no_grad():
        actual = PackedActor(policy)(obs)
        expected = policy(obs[:, 155:], obs[:, :155])
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_export_rejects_stale_contract_without_writing_model(checkpoint, tmp_path):
    from supermimic.deployment.onnx_export import export_policy

    ckpt, _, saved = checkpoint
    saved["robot_contract"]["version"] = 3
    torch.save(saved, ckpt)
    output = tmp_path / "must_not_exist.onnx"
    with pytest.raises(ValueError, match="robot_contract"):
        export_policy(ckpt, output, validation_samples=2)
    assert not output.exists()


def test_export_keeps_unclipped_normalized_action_semantics(checkpoint, tmp_path):
    from supermimic.deployment.onnx_export import export_policy

    ckpt, policy, saved = checkpoint
    # Saturation or incorporating q_target=q0+scale*a into the graph would
    # silently change the action stored in the next proprioceptive history.
    linear = [layer for layer in policy.decoder.mlp.net if isinstance(layer, torch.nn.Linear)]
    with torch.no_grad():
        linear[-1].weight.zero_()
        linear[-1].bias.copy_(torch.linspace(-3., 3., 27))
    saved["policy"] = policy.state_dict()
    torch.save(saved, ckpt)
    output = tmp_path / "raw_actions.onnx"
    export_policy(ckpt, output, validation_samples=2)
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    action = session.run(["actions"], {"obs": np.zeros((1, 620), dtype=np.float32)})[0]
    np.testing.assert_allclose(action[0], np.linspace(-3., 3., 27), atol=1e-6)
    assert action.min() < -1 and action.max() > 1


def test_export_rejects_nonfinite_actor_weights(checkpoint, tmp_path):
    from supermimic.deployment.onnx_export import export_policy

    ckpt, _, saved = checkpoint
    weight = next(value for name, value in saved["policy"].items() if "weight" in name)
    weight.flatten()[0] = float("nan")
    torch.save(saved, ckpt)
    output = tmp_path / "must_not_exist.onnx"
    with pytest.raises(ValueError, match="(?i)(finite|nan)"):
        export_policy(ckpt, output, validation_samples=2)
    assert not output.exists()


def test_export_rejects_checkpoint_with_extra_sensor_scaling(checkpoint, tmp_path):
    from supermimic.deployment.onnx_export import export_policy

    ckpt, _, saved = checkpoint
    saved["policy"]["decoder.scale"][45:60] = .5
    torch.save(saved, ckpt)
    output = tmp_path / "must_not_exist.onnx"
    with pytest.raises(ValueError, match="scale buffers must be identity"):
        export_policy(ckpt, output, validation_samples=2)
    assert not output.exists()


def test_failed_parity_validation_leaves_existing_export_and_report_intact(
    checkpoint, tmp_path, monkeypatch,
):
    from supermimic.deployment import onnx_export

    ckpt, _, _ = checkpoint
    output, report = tmp_path / "existing.onnx", tmp_path / "existing.json"
    output.write_bytes(b"previous valid model")
    report.write_bytes(b"previous parity report")

    def fail_validation(*args, **kwargs):
        raise ValueError("synthetic parity validation failure")

    monkeypatch.setattr(onnx_export, "_validate", fail_validation)
    with pytest.raises(ValueError, match="parity validation failure"):
        onnx_export.export_policy(ckpt, output, validation_samples=2, overwrite=True)
    assert output.read_bytes() == b"previous valid model"
    assert report.read_bytes() == b"previous parity report"
    assert not list(tmp_path.glob(".onnx_export_*"))


def test_export_preserves_existing_output_unless_overwrite_requested(checkpoint, tmp_path):
    from supermimic.deployment.onnx_export import export_policy

    ckpt, _, _ = checkpoint
    output = tmp_path / "existing.onnx"
    original = b"a pre-existing export must not be destroyed"
    output.write_bytes(original)
    with pytest.raises(FileExistsError):
        export_policy(ckpt, output, validation_samples=2)
    assert output.read_bytes() == original
    export_policy(ckpt, output, validation_samples=2, overwrite=True)
    assert output.read_bytes() != original
    _metadata(output)


def test_export_rejects_unknown_layout_and_zero_validation(checkpoint, tmp_path):
    from supermimic.deployment.onnx_export import export_policy

    ckpt, _, _ = checkpoint
    with pytest.raises(ValueError):
        export_policy(ckpt, tmp_path / "bad_layout.onnx", input_layout="legacy")
    with pytest.raises(ValueError):
        export_policy(ckpt, tmp_path / "no_validation.onnx", validation_samples=0)
