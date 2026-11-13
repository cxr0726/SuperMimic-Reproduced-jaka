"""The portable server launcher validates paths and parses the real CLI on CPU."""

import os
import shlex
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest
import tyro

from supermimic.config import TrainConfig, resolve_training_observations

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/training/train_khan_server.sh"


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "portable project"
    script = root / "scripts/training/train_khan_server.sh"
    script.parent.mkdir(parents=True)
    shutil.copyfile(SCRIPT, script)
    python = root / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    # Dry-run must never execute this fake interpreter.
    python.write_text("#!/bin/sh\nexit 93\n")
    python.chmod(0o755)
    assets = root / "assets/robots/khan_mini"
    assets.mkdir(parents=True)
    for name in ("robot.xml", "profile.json"):
        (assets / name).touch()
    data = root / "data/npz/khan_mini/lafan1"
    data.mkdir(parents=True)
    (data / "synthetic.npz").touch()
    return root, script


def run_launcher(project, *args):
    root, script = project
    env = os.environ.copy()
    env.update(PYTHONPATH="/unwanted/ros", PYTHONHOME="/unwanted/python")
    return subprocess.run(["bash", str(script), *args], cwd=root.parent,
                          env=env, text=True, capture_output=True, check=False)


def parse_config(output):
    line = next(line for line in output.splitlines() if line.startswith("[COMMAND]"))
    command = shlex.split(line.removeprefix("[COMMAND]"))
    assert command[1:4] == ["-u", "-m", "supermimic.scripts.train"]
    return tyro.cli(TrainConfig, args=command[4:], console_outputs=False)


@pytest.mark.parametrize("num_envs", [4096, 1024])
def test_dry_run_selects_isaac_observations_without_overriding_original_optimization(project, num_envs):
    root, _ = project
    result = run_launcher(project, "--dry-run", "--num-envs", str(num_envs))
    assert result.returncode == 0, result.stderr
    cfg = parse_config(result.stdout)
    assert cfg.robot == "khan_mini" and cfg.env.num_envs == num_envs
    assert cfg.data.npz_dir == root / "data/npz/khan_mini/lafan1"
    assert cfg.run_name == "khan_lafan1_dr_v3_only_4096" and cfg.log_root == root / "logs"
    assert cfg.data.clips is cfg.data.limit is cfg.data.manifest is cfg.data.library is None
    assert cfg.env.actuator_delay is False and cfg.env.merge_short_tail_bins is True
    assert cfg.env.delay_reset_first_step_no_delay is True
    assert cfg.env.tail_bin_merge_fraction == 1.0 and cfg.env.reset_min_remaining_steps == 32
    assert "actuator delay False" in result.stdout
    assert cfg.env.foot_tracking is False
    assert "--env.no-foot-tracking" in result.stdout
    assert "raw reward cap 0.112" in result.stdout
    assert "critic upper bound 11.2" in result.stdout
    baseline = TrainConfig(robot="khan_mini")
    baseline.env.actuator_delay = False
    resolve_training_observations(cfg)
    resolve_training_observations(baseline)
    for name in ("world_model", "wm_trainer", "policy", "critic", "policy_trainer", "schedule"):
        assert asdict(getattr(cfg, name)) == asdict(getattr(baseline, name)), name
    current_env, original_env = asdict(cfg.env), asdict(baseline.env)
    current_env.pop("num_envs")
    original_env.pop("num_envs")
    assert current_env.pop("anchor_obs") is False
    assert original_env.pop("anchor_obs") is False
    assert current_env == original_env
    assert not cfg.policy_trainer.activation_checkpoint
    assert cfg.policy_trainer.batch_size == 4096 and cfg.policy_trainer.horizon == 32
    assert cfg.schedule.updates_per_env_step == 0.4
    assert cfg.schedule.stop_after_iterations is None
    assert cfg.env.future_frames == cfg.policy.history == 5
    assert cfg.env.observation_layout == cfg.policy.observation_layout == "isaac_khan"
    assert cfg.policy.goal_dim == 155
    assert cfg.policy.action_mode == "default"
    assert cfg.env.reset_sampler == "window_value"
    assert cfg.env.return_horizon == 20 and cfg.env.return_memory == 256
    assert cfg.world_model.input_normalization and cfg.world_model.hidden_layer_norm
    assert cfg.world_model.hidden_activation == "gelu"
    assert cfg.world_model.termination_mode == "analytic"
    assert cfg.policy.running_normalization and cfg.policy_trainer.critic_value_bound
    assert "window_value sampler; horizon 20; memory 256" in result.stdout
    assert cfg.policy.reference_residual_beta == 1.0
    assert cfg.policy.reference_frame_offset == 1
    assert cfg.policy.goal_noise_z == 0.01
    assert cfg.policy.goal_noise_joint_pos == 0.05
    assert "future/history 5/5; goal 155; policy 465" in result.stdout
    # Optimization values are owned by TrainConfig, rather than copied into Bash.
    command_line = next(line for line in result.stdout.splitlines() if line.startswith("[COMMAND]"))
    flags = shlex.split(command_line.removeprefix("[COMMAND]"))[4:]
    assert not any(flag.startswith(("--world-model.", "--wm-trainer.", "--policy.",
                                    "--policy-trainer.", "--critic.", "--schedule."))
                   and flag not in {"--policy.history", "--policy.action-mode",
                                     "--policy.goal-noise-z", "--policy.goal-noise-joint-pos"}
                   for flag in flags)
    assert not (root / "logs").exists()


def test_legacy_anchor_flags_rejected_in_isaac_launcher(project):
    root, _ = project
    result = run_launcher(project, "--dry-run", "--no-anchor-obs")
    assert result.returncode != 0
    assert "remove the old anchor flag" in result.stderr
    assert not (root / "logs").exists()


def test_resume_and_short_execution_keep_original_lr_horizon(project):
    root, _ = project
    checkpoint = root / "trained Khan.pt"
    checkpoint.touch()
    result = run_launcher(project, "--dry-run", "--resume", str(checkpoint),
                          "--stop-after-iterations", "50", "--run-name", "resume_run")
    assert result.returncode == 0, result.stderr
    cfg = parse_config(result.stdout)
    assert cfg.resume == checkpoint
    assert cfg.schedule.stop_after_iterations == 50
    assert cfg.schedule.total_iterations == 320000
    assert cfg.schedule.lr_decay_fraction == 0.78125
    assert not (root / "logs").exists()


@pytest.mark.parametrize("args,expected", [
    (["--run-name", "../escape"], "one nonempty directory name"),
    (["--num-envs", "0"], "positive integer"),
    (["--num-envs", "-1"], "positive integer"),
    (["--stop-after-iterations", "0"], "positive integer"),
    (["--resume", "missing.pt"], "checkpoint does not exist"),
    (["--policy-trainer.batch-size", "128"], "Unknown option"),
    (["--run-name", "--dry-run"], "requires a value"),
])
def test_invalid_inputs_rejected_without_execution(project, args, expected):
    result = run_launcher(project, "--dry-run", *args)
    assert result.returncode != 0
    assert expected in result.stderr


def test_missing_data_and_existing_checkpoint_rejected(project):
    root, _ = project
    empty = root / "empty"
    empty.mkdir()
    result = run_launcher(project, "--dry-run", "--data-dir", str(empty))
    assert result.returncode != 0 and "No converted Khan NPZ" in result.stderr
    run = root / "logs/khan_lafan1_dr_v3_only_4096"
    run.mkdir(parents=True)
    (run / "ckpt_latest.pt").touch()
    result = run_launcher(project, "--dry-run")
    assert result.returncode != 0 and "Checkpoints already exist" in result.stderr


def test_missing_venv_points_to_portable_setup_command(project):
    root, _ = project
    (root / ".venv/bin/python").unlink()
    result = run_launcher(project, "--dry-run")
    assert result.returncode != 0
    assert "setup_server_env.sh" in result.stderr and "--project-dir" in result.stderr
    assert str(root / ".venv/bin/python") in result.stderr


def test_missing_robot_asset_rejected_before_execution(project):
    root, _ = project
    (root / "assets/robots/khan_mini/robot.xml").unlink()
    result = run_launcher(project, "--dry-run")
    assert result.returncode != 0 and "Missing Khan asset" in result.stderr


@pytest.mark.parametrize("flags,enabled", [
    (["--no-actuator-delay"], False), (["--actuator-delay"], True),
    (["--actuator-delay", "--no-actuator-delay"], False),
    (["--no-actuator-delay", "--actuator-delay"], True),
])
def test_delay_switch_keeps_h32_and_full_tail_merge(project, flags, enabled):
    result = run_launcher(project, "--dry-run", *flags)
    assert result.returncode == 0, result.stderr
    cfg = parse_config(result.stdout)
    assert cfg.env.actuator_delay is enabled
    assert cfg.env.delay_reset_first_step_no_delay is True
    assert cfg.env.reset_min_remaining_steps == 32
    assert cfg.env.tail_bin_merge_fraction == 1.0


@pytest.mark.parametrize("flags,action_mode,z_noise,q_noise", [
    ([], "default", 0.01, 0.05),
    (["--reference-residual"], "reference_residual", 0.01, 0.05),
    (["--no-reference-residual"], "default", 0.01, 0.05),
    (["--reference-residual", "--no-reference-residual"], "default", 0.01, 0.05),
    (["--no-goal-noise"], "default", 0.0, 0.0),
    (["--no-reference-residual", "--no-goal-noise"], "default", 0.0, 0.0),
    (["--no-reference-residual", "--reference-residual", "--no-goal-noise", "--goal-noise"],
     "reference_residual", 0.01, 0.05),
])
def test_action_space_and_encoder_noise_switches_are_independent(project, flags, action_mode, z_noise, q_noise):
    result = run_launcher(project, "--dry-run", *flags)
    assert result.returncode == 0, result.stderr
    cfg = parse_config(result.stdout)
    resolve_training_observations(cfg)
    assert cfg.policy.action_mode == action_mode
    assert cfg.policy.goal_noise_z == z_noise
    assert cfg.policy.goal_noise_joint_pos == q_noise
    assert cfg.policy_trainer.horizon == 32
    assert cfg.env.actuator_delay is False
    assert "critic/reward stay clean" in result.stdout
    assert "reference skip uses raw clean radians" in result.stdout
    if action_mode == "reference_residual":
        assert "PD target=raw clean q_ref(t+1) + 0.5*decoder_output; beta 1" in result.stdout
    else:
        assert "PD target=q_default + 0.5*actor_output; reference skip OFF" in result.stdout


@pytest.mark.parametrize("flags,enabled", [
    ([], False),
    (["--foot-tracking"], True),
    (["--no-foot-tracking"], False),
    (["--foot-tracking", "--no-foot-tracking"], False),
    (["--no-foot-tracking", "--foot-tracking"], True),
])
def test_foot_rewards_switch_does_not_disable_dr_v3(project, flags, enabled):
    result = run_launcher(project, "--dry-run", *flags)
    assert result.returncode == 0, result.stderr
    cfg = parse_config(result.stdout)
    resolve_training_observations(cfg)
    assert cfg.env.foot_tracking is enabled
    assert cfg.env.context_layout == "khan_dr_v3"
    assert cfg.policy.action_mode == "default"
    assert cfg.policy.goal_noise_z == 0.01 and cfg.policy.goal_noise_joint_pos == 0.05
    assert cfg.env.actuator_delay is False
    assert "xi 101; critic 615" in result.stdout
    assert "DR v3 remains enabled" in result.stdout
    assert f"raw reward cap {0.122 if enabled else 0.112}" in result.stdout
    assert f"critic upper bound {12.2 if enabled else 11.2}" in result.stdout
    assert ("--env.foot-tracking" if enabled else "--env.no-foot-tracking") in result.stdout


def test_previous_residual_and_foot_reward_modes_can_be_selected_together(project):
    result = run_launcher(project, "--dry-run", "--reference-residual", "--foot-tracking")
    assert result.returncode == 0, result.stderr
    cfg = parse_config(result.stdout)
    resolve_training_observations(cfg)
    assert cfg.policy.action_mode == "reference_residual"
    assert cfg.env.foot_tracking is True
    assert cfg.env.context_layout == "khan_dr_v3"
    assert cfg.policy.goal_noise_z == 0.01 and cfg.policy.goal_noise_joint_pos == 0.05
    assert "raw reward cap 0.122" in result.stdout
