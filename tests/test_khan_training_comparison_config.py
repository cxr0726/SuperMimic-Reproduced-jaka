"""The new comparison changes delay/bin layout, not observations or rewards."""
from dataclasses import asdict

from supermimic.config import TrainConfig, resolve_training_observations
from supermimic.envs import SuperMimicEnvCfgArgs, make_env_cfg
from supermimic.robots.contracts import robot_contract, validate_robot_contract


def test_new_khan_training_disables_delay_and_merges_tails():
    cfg = TrainConfig(robot="khan_mini")
    resolve_training_observations(cfg)
    assert cfg.env.actuator_delay is False
    assert cfg.env.merge_short_tail_bins is True
    assert cfg.env.reset_root_z_offset == .05
    env = make_env_cfg(SuperMimicEnvCfgArgs(
        npz_files=["unused.npz"], robot=cfg.robot,
        actuator_delay=cfg.env.actuator_delay,
        merge_short_tail_bins=cfg.env.merge_short_tail_bins,
    ))
    assert env.commands["motion"].merge_short_tail_bins
    assert env.commands["motion"].reset_root_z_offset == .05
    for actuator in env.scene.entities["robot"].articulation.actuators:
        assert actuator.delay_min_lag == actuator.delay_max_lag == 0
    old = make_env_cfg(SuperMimicEnvCfgArgs(npz_files=["unused.npz"], robot=cfg.robot))
    assert asdict(env.rewards["action_rate_l2"]) == asdict(old.rewards["action_rate_l2"])
    assert {name: asdict(term) for name, term in env.rewards.items()} == {
        name: asdict(term) for name, term in old.rewards.items()
    }


def test_explicit_old_delay_and_unmerged_bins_remain_available():
    cfg = TrainConfig(robot="khan_mini")
    cfg.env.actuator_delay = True
    cfg.env.merge_short_tail_bins = False
    resolve_training_observations(cfg)
    assert cfg.env.actuator_delay is True
    assert cfg.env.merge_short_tail_bins is False
    historical_env = {"actuator_delay": True, "domain_randomization": True,
                      "actuator_delay_max_ms": 20., "reset_root_z_offset": .05}
    old = robot_contract("khan_mini", "isaac_khan", env_config=historical_env)
    validate_robot_contract({"robot_contract": old, "config": {"env": historical_env}},
                            "khan_mini", "isaac_khan")
    new = robot_contract("khan_mini", "isaac_khan", env_config=asdict(cfg.env))
    assert old == new


def test_g1_keeps_original_bin_layout_and_delay():
    cfg = TrainConfig()
    resolve_training_observations(cfg)
    assert cfg.env.actuator_delay is False
    assert cfg.env.merge_short_tail_bins is False
    assert cfg.env.reset_root_z_offset == 0.
