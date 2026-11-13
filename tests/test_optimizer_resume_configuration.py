"""Changing the optimization algorithm must not silently reinterpret a resume."""

from dataclasses import asdict

import pytest

from supermimic.config import TrainConfig
from supermimic.training.loop import validate_optimizer_configuration


def test_historical_missing_fields_mean_adam_and_require_explicit_adam_resume():
    cfg = TrainConfig()
    with pytest.raises(ValueError, match="WM optimizer differs"):
        validate_optimizer_configuration({}, cfg)
    cfg.wm_trainer.optimizer = "adam"
    cfg.policy_trainer.optimizer = cfg.policy_trainer.critic_optimizer = "adam"
    validate_optimizer_configuration({}, cfg)


@pytest.mark.parametrize("section, field", [
    ("wm_trainer", "muon_weight_decay"),
    ("policy_trainer", "muon_weight_decay"),
    ("policy_trainer", "critic_muon_weight_decay"),
])
def test_muon_decay_is_part_of_the_resume_algorithm(section, field):
    cfg = TrainConfig()
    saved = asdict(cfg)
    validate_optimizer_configuration(saved, cfg)
    saved[section][field] = .2
    with pytest.raises(ValueError, match="Muon weight decay differs"):
        validate_optimizer_configuration(saved, cfg)
