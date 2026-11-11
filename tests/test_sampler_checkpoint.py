"""Resume must preserve the new curriculum and discard incompatible old estimates."""

import copy
from types import SimpleNamespace

import pytest
import torch

from supermimic.config import TrainConfig, resolve_training_observations
from supermimic.robots.registry import get_robot_spec
from supermimic.training.loop import Trainer, load_sampler_state
from supermimic.world_model.world_model import WorldModelCfg


class FakeCommand:
    def __init__(self):
        self.sampler_contract = {
            "version": 2,
            "return_target": "start_bin_mean_reward_continuation",
            "ema_alpha": 0.2,
            "memory_alpha": 0.001,
            "motion_bin_hash": "test-motion-mapping",
        }
        self.bin_return = torch.tensor([1.0, 2.0, 3.0])
        self.bin_memory = torch.tensor([0.2, 0.3, 0.5])
        self.sampleable_bins = torch.ones(3, dtype=torch.bool)
        self.reset_calls = 0

    def reset_sampler_statistics(self):
        self.reset_calls += 1
        self.bin_return.zero_()
        self.bin_memory.copy_(self.sampleable_bins / self.sampleable_bins.sum())


def checkpoint(command):
    return {
        "sampler_contract": copy.deepcopy(command.sampler_contract),
        "bin_return": torch.tensor([-0.5, 6.0, 8.0]),
        "bin_memory": torch.tensor([0.1, 0.7, 0.2]),
    }


def test_matching_sampler_roundtrip_restores_statistics():
    command = FakeCommand()
    saved = checkpoint(command)
    assert load_sampler_state(command, saved)
    assert command.reset_calls == 0
    torch.testing.assert_close(command.bin_return, saved["bin_return"], rtol=0, atol=0)
    torch.testing.assert_close(command.bin_memory, saved["bin_memory"], rtol=0, atol=0)


def test_unmerged_v2_scores_cannot_seed_merged_v3_curriculum():
    command = FakeCommand()
    saved = checkpoint(command)
    command.sampler_contract.update(
        version=3, tail_bin_policy="merge_remainder_strictly_less_than_half_reference_bin",
    )
    with pytest.raises(ValueError, match="sampler_contract differs"):
        load_sampler_state(command, saved)
    assert command.reset_calls == 0
    torch.testing.assert_close(command.bin_return, torch.tensor([1.0, 2.0, 3.0]))


def test_legacy_sampler_is_reset_even_if_old_bins_have_different_shape():
    command = FakeCommand()
    saved = {
        "iteration": 32000,
        "num_updates": 12784,
        "bin_return": torch.full((4,), float("nan")),
        "bin_memory": torch.zeros(4),
    }
    with pytest.warns(UserWarning, match="Legacy checkpoint.*reinitialized"):
        assert not load_sampler_state(command, saved)
    assert command.reset_calls == 1
    torch.testing.assert_close(command.bin_return, torch.zeros(3), rtol=0, atol=0)
    torch.testing.assert_close(command.bin_memory, torch.full((3,), 1 / 3), rtol=0, atol=0)
    assert (saved["iteration"], saved["num_updates"]) == (32000, 12784)


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", 1),
        ("return_target", "visited_bin_raw_rtg"),
        ("ema_alpha", 0.05),
        ("memory_alpha", 0.01),
        ("motion_bin_hash", "same-shape-different-order"),
    ],
)
def test_semantics_or_same_shape_bin_mapping_mismatch_is_rejected(field, value):
    command = FakeCommand()
    saved = checkpoint(command)
    saved["sampler_contract"][field] = value
    old_return, old_memory = command.bin_return.clone(), command.bin_memory.clone()
    with pytest.raises(ValueError, match="sampler_contract differs"):
        load_sampler_state(command, saved)
    torch.testing.assert_close(command.bin_return, old_return, rtol=0, atol=0)
    torch.testing.assert_close(command.bin_memory, old_memory, rtol=0, atol=0)


@pytest.mark.parametrize("contract", [None, 2, "version2"])
def test_malformed_contract_is_not_treated_as_a_legacy_checkpoint(contract):
    command = FakeCommand()
    saved = checkpoint(command)
    saved["sampler_contract"] = contract
    with pytest.raises(ValueError, match="sampler_contract differs"):
        load_sampler_state(command, saved)
    assert command.reset_calls == 0


@pytest.mark.parametrize(
    "name,value,error",
    [
        ("bin_return", [1.0, 2.0, 3.0], "floating-point tensor"),
        ("bin_return", torch.tensor([1, 2, 3]), "floating-point tensor"),
        ("bin_return", torch.zeros(4), "shape"),
        ("bin_return", torch.tensor([0.0, float("nan"), 0.0]), "nonfinite"),
        ("bin_return", torch.tensor([0.0, float("inf"), 0.0]), "nonfinite"),
        ("bin_return", torch.tensor([0.0, 1e300, 0.0], dtype=torch.float64), "nonfinite"),
        ("bin_memory", torch.zeros(3), "probability distribution"),
        ("bin_memory", torch.tensor([-0.1, 0.6, 0.5]), "probability distribution"),
        ("bin_memory", torch.tensor([0.1, 0.1, 0.1]), "probability distribution"),
        ("bin_memory", torch.full((3,), float("nan")), "nonfinite"),
        ("bin_memory", torch.ones(1, 3) / 3, "shape"),
    ],
)
def test_bad_statistics_are_rejected_before_either_array_is_modified(name, value, error):
    command = FakeCommand()
    saved = checkpoint(command)
    saved[name] = value
    old_return, old_memory = command.bin_return.clone(), command.bin_memory.clone()
    with pytest.raises(ValueError, match=error):
        load_sampler_state(command, saved)
    torch.testing.assert_close(command.bin_return, old_return, rtol=0, atol=0)
    torch.testing.assert_close(command.bin_memory, old_memory, rtol=0, atol=0)


def test_missing_statistics_in_versioned_checkpoint_are_rejected():
    command = FakeCommand()
    saved = checkpoint(command)
    del saved["bin_return"]
    with pytest.raises(ValueError, match="bin_return.*floating-point tensor"):
        load_sampler_state(command, saved)


def test_history_cannot_reenable_a_final_state_only_bin():
    command = FakeCommand()
    command.sampleable_bins[-1] = False
    saved = checkpoint(command)
    with pytest.raises(ValueError, match="no valid reset frame"):
        load_sampler_state(command, saved)
    saved["bin_memory"] = torch.tensor([0.3, 0.7, 0.0])
    assert load_sampler_state(command, saved)
    torch.testing.assert_close(command.bin_memory, saved["bin_memory"], rtol=0, atol=0)


def small_trainer(tmp_path):
    trainer = object.__new__(Trainer)
    trainer.is_main = True
    trainer.log_dir = tmp_path
    trainer.device = "cpu"
    trainer.iteration, trainer.num_updates = 32000, 12784
    trainer.cfg = TrainConfig()
    # This fixture verifies historical sampler/optimizer recovery with toy MLPs.
    trainer.cfg.world_model = WorldModelCfg()
    trainer.cfg.policy.running_normalization = False
    trainer.cfg.policy_trainer.critic_value_bound = False
    trainer.cfg.env.reset_sampler = "legacy"
    resolve_training_observations(trainer.cfg)
    trainer.robot_spec = get_robot_spec(trainer.cfg.robot)
    trainer.command = FakeCommand()
    trainer.wm, trainer.policy, trainer.critic = [torch.nn.Linear(2, 2) for _ in range(3)]
    trainer.wm_trainer = SimpleNamespace(opt=torch.optim.Adam(trainer.wm.parameters()))
    trainer.policy_trainer = SimpleNamespace(
        opt=torch.optim.Adam(trainer.policy.parameters()),
        critic_opt=torch.optim.Adam(trainer.critic.parameters()),
    )
    return trainer


def test_trainer_save_keeps_audit_arrays_and_adds_sampler_contract(tmp_path):
    trainer = small_trainer(tmp_path)
    trainer.save()
    saved = torch.load(tmp_path / "ckpt_latest.pt", weights_only=False)
    assert saved["sampler_contract"] == trainer.command.sampler_contract
    assert (saved["iteration"], saved["num_updates"]) == (32000, 12784)
    torch.testing.assert_close(saved["bin_return"], trainer.command.bin_return, rtol=0, atol=0)
    torch.testing.assert_close(saved["bin_memory"], trainer.command.bin_memory, rtol=0, atol=0)


def test_legacy_resume_preserves_models_optimizer_moments_iteration_and_lr(tmp_path):
    original = small_trainer(tmp_path)
    pairs = [
        (original.wm, original.wm_trainer.opt),
        (original.policy, original.policy_trainer.opt),
        (original.critic, original.policy_trainer.critic_opt),
    ]
    # Populate real Adam moments so the resume check also covers optimizer state.
    for index, (model, optimizer) in enumerate(pairs):
        optimizer.param_groups[0]["lr"] = (index + 1) * 0.0001
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
    original.save()
    path = tmp_path / "ckpt_latest.pt"
    saved = torch.load(path, weights_only=False)
    del saved["sampler_contract"]
    saved["bin_return"] = torch.zeros(4)  # Legacy bin layouts need not match.
    torch.save(saved, path)

    resumed = small_trainer(tmp_path)
    resumed.iteration, resumed.num_updates = 0, 0
    with pytest.warns(UserWarning, match="Legacy checkpoint.*reinitialized"):
        resumed.load(path)
    assert (resumed.iteration, resumed.start_iteration, resumed.num_updates) == (
        32000, 32000, 12784,
    )
    restored_pairs = [
        (resumed.wm, resumed.wm_trainer.opt),
        (resumed.policy, resumed.policy_trainer.opt),
        (resumed.critic, resumed.policy_trainer.critic_opt),
    ]
    for (before, before_optimizer), (after, after_optimizer) in zip(pairs, restored_pairs):
        for before_parameter, after_parameter in zip(before.parameters(), after.parameters()):
            torch.testing.assert_close(before_parameter, after_parameter, rtol=0, atol=0)
        assert before_optimizer.param_groups[0]["lr"] == after_optimizer.param_groups[0]["lr"]
        for key, state in before_optimizer.state_dict()["state"].items():
            for name, value in state.items():
                torch.testing.assert_close(
                    value, after_optimizer.state_dict()["state"][key][name], rtol=0, atol=0,
                )
    torch.testing.assert_close(resumed.command.bin_return, torch.zeros(3), rtol=0, atol=0)
    torch.testing.assert_close(resumed.command.bin_memory, torch.full((3,), 1 / 3), rtol=0, atol=0)
