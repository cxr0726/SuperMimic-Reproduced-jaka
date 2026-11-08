"""Exercise the compact server wrapper without starting Python training or CUDA."""

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
SCRIPT = ROOT / "scripts/training/train_khan_compact_server.sh"
UNWANTED_ENV = (
    "PYTHONPATH", "PYTHONHOME", "RANK", "LOCAL_RANK", "WORLD_SIZE",
    "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "SM_GPU_PINNED",
)


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "portable project"
    script = root / "scripts/training/train_khan_compact_server.sh"
    script.parent.mkdir(parents=True)
    shutil.copyfile(SCRIPT, script)
    python = root / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    # Any accidental interpreter invocation makes a dry-run fail.
    python.write_text("#!/bin/sh\nexit 93\n")
    python.chmod(0o755)
    assets = root / "assets/robots/khan_mini"
    assets.mkdir(parents=True)
    for name in ("robot.xml", "profile.json"):
        (assets / name).touch()
    library = root / "data/pt/khan_combined_8452_compact.pt"
    library.parent.mkdir(parents=True)
    library.write_bytes(b"path validation fixture; dry-run must not deserialize this")
    return root, script


def run_launcher(project, *args, env_updates=None):
    root, script = project
    env = os.environ.copy()
    env.update({name: "unwanted" for name in UNWANTED_ENV})
    for name in ("OMP_NUM_THREADS", "TORCHINDUCTOR_COMPILE_THREADS"):
        env.pop(name, None)
    env.update(env_updates or {})
    return subprocess.run(
        ["bash", str(script), *args], cwd=root.parent,
        env=env, text=True, capture_output=True, check=False,
    )


def printed_command(output):
    line = next(line for line in output.splitlines() if line.startswith("[COMMAND]"))
    command = shlex.split(line.removeprefix("[COMMAND]"))
    assert command[1:4] == ["-u", "-m", "supermimic.scripts.train"]
    return command


def parse_config(output):
    return tyro.cli(TrainConfig, args=printed_command(output)[4:], console_outputs=False)


@pytest.mark.parametrize("num_envs", [4096, 1024])
def test_default_compact_run_preserves_original_training_configuration(project, num_envs):
    root, _ = project
    result = run_launcher(project, "--dry-run", "--num-envs", str(num_envs))
    assert result.returncode == 0, result.stderr
    cfg = parse_config(result.stdout)
    assert cfg.robot == "khan_mini" and cfg.env.num_envs == num_envs
    assert cfg.data.library == root / "data/pt/khan_combined_8452_compact.pt"
    assert cfg.data.npz_dir is None
    assert cfg.run_name == "khan_bones_pt" and cfg.log_root == root / "logs"
    assert cfg.data.clips is cfg.data.limit is cfg.data.manifest is None
    assert cfg.env.compact_library and not cfg.env.reference_fk_compile
    assert cfg.env.actuator_delay is False and cfg.env.merge_short_tail_bins is True
    assert "actuator delay OFF" in result.stdout
    baseline = TrainConfig(robot="khan_mini")
    resolve_training_observations(cfg)
    resolve_training_observations(baseline)
    for name in ("world_model", "wm_trainer", "policy", "critic", "policy_trainer", "schedule"):
        assert asdict(getattr(cfg, name)) == asdict(getattr(baseline, name)), name
    actual_env, default_env = asdict(cfg.env), asdict(baseline.env)
    for name in ("num_envs", "compact_library"):
        actual_env.pop(name)
        default_env.pop(name)
    assert actual_env == default_env
    assert cfg.env.observation_layout == "isaac_khan"
    assert cfg.env.future_frames == cfg.policy.history == 5
    assert cfg.policy.goal_dim == 155
    assert not cfg.policy_trainer.activation_checkpoint
    assert cfg.policy_trainer.horizon == 32 and cfg.policy_trainer.batch_size == 4096
    assert cfg.wm_trainer.batch_size == 2048
    assert cfg.schedule.total_iterations == 320000
    assert cfg.schedule.updates_per_env_step == 0.4
    # The wrapper owns no duplicate network or optimizer hyperparameter lists.
    assert not any(
        flag.startswith(("--world-model.", "--wm-trainer.", "--policy.",
                         "--policy-trainer.", "--critic.", "--schedule."))
        and flag != "--policy.history" for flag in printed_command(result.stdout)[4:]
    )
    assert not (root / "data/npz").exists()
    assert not (root / "logs").exists()


@pytest.mark.parametrize("flags,expected", [
    (["--reference-fk-compile"], True),
    (["--no-reference-fk-compile"], False),
    (["--reference-fk-compile", "--no-reference-fk-compile"], False),
    (["--no-reference-fk-compile", "--reference-fk-compile"], True),
])
def test_reference_fk_switch_reaches_real_training_cli(project, flags, expected):
    result = run_launcher(project, "--dry-run", *flags)
    assert result.returncode == 0, result.stderr
    assert parse_config(result.stdout).env.reference_fk_compile is expected


def test_custom_relative_library_and_run_name_with_spaces(project):
    root, _ = project
    library = root.parent / "custom motions.pt"
    library.write_bytes(b"packed fixture")
    result = run_launcher(
        project, "--dry-run", "--library", library.name, "--run-name", "new Khan run",
    )
    assert result.returncode == 0, result.stderr
    cfg = parse_config(result.stdout)
    assert cfg.data.library == library and cfg.run_name == "new Khan run"


def test_existing_checkpoint_requires_explicit_resume_and_keeps_full_lr_schedule(project):
    root, _ = project
    run = root / "logs/khan_bones_pt"
    run.mkdir(parents=True)
    checkpoint = run / "ckpt_latest.pt"
    checkpoint.write_bytes(b"checkpoint fixture")
    result = run_launcher(project, "--dry-run")
    assert result.returncode != 0 and "Checkpoints already exist" in result.stderr
    result = run_launcher(
        project, "--dry-run", "--resume", str(checkpoint), "--stop-after-iterations", "50",
    )
    assert result.returncode == 0, result.stderr
    cfg = parse_config(result.stdout)
    assert cfg.resume == checkpoint
    assert cfg.schedule.stop_after_iterations == 50
    assert cfg.schedule.total_iterations == 320000
    assert cfg.schedule.lr_decay_fraction == 0.78125


@pytest.mark.parametrize("args,expected", [
    (["--run-name", "../escape"], "one nonempty directory name"),
    (["--num-envs", "0"], "positive integer"),
    (["--num-envs", "-1"], "positive integer"),
    (["--stop-after-iterations", "0"], "positive integer"),
    (["--library", "--dry-run"], "requires a value"),
    (["--resume", "missing.pt"], "checkpoint does not exist"),
    (["--data-dir", "data/npz"], "--library"),
    (["--policy-trainer.batch-size", "128"], "Unknown option"),
])
def test_invalid_arguments_rejected_before_training(project, args, expected):
    result = run_launcher(project, "--dry-run", *args)
    assert result.returncode != 0
    assert expected in result.stderr


@pytest.mark.parametrize("kind", ["missing", "directory", "empty", "wrong_extension"])
def test_invalid_library_paths_rejected_without_loading(project, kind):
    root, _ = project
    path = root / ("invalid.bin" if kind == "wrong_extension" else "invalid.pt")
    if kind == "directory":
        path.mkdir()
    elif kind == "empty":
        path.touch()
    elif kind == "wrong_extension":
        path.write_bytes(b"not a pt library")
    result = run_launcher(project, "--dry-run", "--library", str(path))
    assert result.returncode != 0
    assert "library" in result.stderr.lower()
    assert not (root / "logs").exists()


def test_empty_resume_checkpoint_is_rejected(project):
    root, _ = project
    checkpoint = root / "empty.pt"
    checkpoint.touch()
    result = run_launcher(project, "--dry-run", "--resume", str(checkpoint))
    assert result.returncode != 0 and "checkpoint" in result.stderr.lower()


@pytest.mark.parametrize("missing", [".venv/bin/python", "assets/robots/khan_mini/robot.xml"])
def test_missing_runtime_files_fail_before_execution(project, missing):
    root, _ = project
    (root / missing).unlink()
    result = run_launcher(project, "--dry-run")
    assert result.returncode != 0
    assert "Missing" in result.stderr
    if missing.endswith("python"):
        assert "setup_server_env.sh" in result.stderr and "--project-dir" in result.stderr


def test_actual_exec_cleans_environment_preserves_gpu_selection_and_changes_directory(project):
    root, _ = project
    capture = root / "launch capture"
    capture.mkdir()
    python = root / ".venv/bin/python"
    python.write_text(
        '#!/bin/bash\n'
        'pwd -P > "$KHAN_LAUNCH_CAPTURE/cwd"\n'
        'printf "%s\\0" "$@" > "$KHAN_LAUNCH_CAPTURE/argv"\n'
        '/usr/bin/env -0 > "$KHAN_LAUNCH_CAPTURE/env"\n'
        'exit 37\n'
    )
    result = run_launcher(
        project, "--reference-fk-compile", "--run-name", "exec_probe",
        env_updates={"KHAN_LAUNCH_CAPTURE": str(capture), "CUDA_VISIBLE_DEVICES": "3"},
    )
    assert result.returncode == 37, result.stderr
    assert (capture / "cwd").read_text().strip() == str(root)
    argv = (capture / "argv").read_bytes().decode().split("\0")[:-1]
    assert argv == printed_command(result.stdout)[1:]
    env = dict(item.split("=", 1) for item in
               (capture / "env").read_bytes().decode().split("\0") if item)
    assert not set(UNWANTED_ENV).intersection(env)
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["MUJOCO_GL"] == "egl"
    assert env["OMP_NUM_THREADS"] == env["TORCHINDUCTOR_COMPILE_THREADS"] == "1"
    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    assert not (root / "logs").exists()
