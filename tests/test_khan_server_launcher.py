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
    assert cfg.run_name == "khan_lafan1_dr_v6" and cfg.log_root == root / "logs"
    assert cfg.data.clips is cfg.data.limit is cfg.data.manifest is cfg.data.library is None
    baseline = TrainConfig(robot="khan_mini")
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
    assert "future/history 5/5; clean goal 155; policy 465" in result.stdout
    # Optimization values are owned by TrainConfig, rather than copied into Bash.
    command_line = next(line for line in result.stdout.splitlines() if line.startswith("[COMMAND]"))
    flags = shlex.split(command_line.removeprefix("[COMMAND]"))[4:]
    assert not any(flag.startswith(("--world-model.", "--wm-trainer.", "--policy.",
                                    "--policy-trainer.", "--critic.", "--schedule."))
                   and flag != "--policy.history"
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
    run = root / "logs/khan_lafan1_dr_v6"
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
