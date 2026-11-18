"""Khan's optional noise mask affects executed targets only."""

from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from supermimic.envs import SuperMimicEnvCfgArgs, make_env_cfg
from supermimic.envs.actions import NoisyJointPositionActionCfg
from supermimic.robots.registry import get_robot_spec


NECK = ("Neck_yaw_joint", "Neck_pitch_joint")


@pytest.mark.parametrize("noise_std", [0.0, 0.1])
def test_only_named_neck_actions_are_excluded_after_the_standard_noise_draw(noise_std):
    joint_names = get_robot_spec("khan_mini").joint_names
    neck = [joint_names.index(name) for name in NECK]
    other = [i for i, name in enumerate(joint_names) if name not in NECK]
    target_ids = [other[0], neck[1], *other[1:12], neck[0], *other[12:]]
    target_names = [joint_names[i] for i in target_ids]
    neck_columns = [target_names.index(name) for name in NECK]
    n_envs = 3
    data = SimpleNamespace(
        default_joint_pos=torch.linspace(-0.3, 0.15, len(joint_names)).expand(n_envs, -1),
        encoder_bias=torch.linspace(-0.02, 0.03, len(joint_names)).expand(n_envs, -1),
        joint_pos_target=torch.empty(n_envs, len(joint_names)),
    )

    def set_target(target, joint_ids):
        data.joint_pos_target[:, joint_ids] = target

    entity = SimpleNamespace(
        data=data,
        set_joint_position_target=set_target,
        find_joints_by_actuator_names=lambda _: (target_ids, target_names),
    )
    env = SimpleNamespace(num_envs=n_envs, device="cpu", scene={"robot": entity})
    actions = torch.linspace(-1.1, 0.8, n_envs * len(joint_names)).reshape(n_envs, -1)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(413)
        expected = actions + noise_std * torch.randn_like(actions) if noise_std else actions.clone()
        expected_rng = torch.random.get_rng_state()

        for exclusions in ((), NECK):
            action = NoisyJointPositionActionCfg(
                entity_name="robot",
                actuator_names=(".*",),
                scale=0.5,
                noise_std=noise_std,
                use_default_offset=True,
                clip={".*": (-0.3, 0.3)},
                noise_excluded_joint_names=exclusions,
            ).build(env)
            torch.manual_seed(413)
            action.process_actions(actions)
            assert torch.equal(action.raw_action, actions)
            assert torch.equal(torch.random.get_rng_state(), expected_rng)

            expected_exec = expected.clone()
            if exclusions:
                expected_exec[:, neck_columns] = actions[:, neck_columns]
            assert torch.equal(action.executed_raw, expected_exec)
            action.apply_actions()
            expected_target = (
                (expected_exec * 0.5 + data.default_joint_pos[:, target_ids]).clamp(-0.3, 0.3)
                - data.encoder_bias[:, target_ids]
            )
            assert torch.equal(data.joint_pos_target[:, target_ids], expected_target)


def test_environment_toggle_leaves_observations_rewards_and_randomization_unchanged():
    options = {"npz_files": ["unused.npz"], "robot": "khan_mini", "action_noise_std": 0.1}
    default = make_env_cfg(SuperMimicEnvCfgArgs(**options))
    masked = make_env_cfg(SuperMimicEnvCfgArgs(**options, disable_neck_action_noise=True))
    before = asdict(default.actions["joint_pos"])
    after = asdict(masked.actions["joint_pos"])
    assert before.pop("noise_excluded_joint_names") == ()
    assert after.pop("noise_excluded_joint_names") == NECK
    assert before == after and after["noise_std"] == 0.1
    for field in ("observations", "rewards", "events", "commands", "terminations"):
        assert getattr(masked, field) == getattr(default, field)


def test_neck_noise_toggle_rejects_robots_without_khan_neck_axes():
    with pytest.raises(ValueError, match="only supported for Khan Mini"):
        make_env_cfg(SuperMimicEnvCfgArgs(
            npz_files=["unused.npz"], robot="g1", disable_neck_action_noise=True,
        ))
