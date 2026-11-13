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


def test_learning_execution_flags_can_disable_capture_without_changing_batches(project):
    result = run_launcher(project, "--dry-run", "--no-update-capture", "--no-batch-reference-encoder")
    assert result.returncode == 0, result.stderr
    cfg = parse_config(result.stdout)
    assert not cfg.wm_trainer.cuda_graph and not cfg.policy_trainer.cuda_graph
    assert not cfg.policy_trainer.batch_reference_encoder
    assert cfg.wm_trainer.batch_size == 2048 and cfg.policy_trainer.batch_size == 4096
    assert cfg.policy_trainer.horizon == 32 and cfg.schedule.updates_per_env_step == .4


def test_fusion_and_cdf_controls_keep_original_budget(project):
    result = run_launcher(project, "--dry-run", "--no-learner-fusion", "--replay-sampling", "multinomial")
    assert result.returncode == 0, result.stderr
    cfg = parse_config(result.stdout)
    assert not cfg.wm_trainer.compile_full_window and not cfg.policy_trainer.compile_networks
    assert cfg.replay_sampling == "multinomial"
    assert cfg.wm_trainer.batches_per_update == 10 and cfg.policy_trainer.critic_epochs == 5
    assert cfg.wm_trainer.cuda_graph and cfg.policy_trainer.cuda_graph
    invalid = run_launcher(project, "--dry-run", "--replay-sampling", "invalid")
    assert invalid.returncode != 0 and "cdf or multinomial" in invalid.stderr


@pytest.mark.parametrize("num_envs", [4096, 1024])
def test_default_compact_run_preserves_current_training_configuration(project, num_envs):
    root, _ = project
    result = run_launcher(project, "--dry-run", "--num-envs", str(num_envs))
    assert result.returncode == 0, result.stderr
    cfg = parse_config(result.stdout)
    assert cfg.robot == "khan_mini" and cfg.env.num_envs == num_envs
    assert cfg.data.library == root / "data/pt/khan_combined_8452_compact.pt"
    assert cfg.data.npz_dir is None
    assert cfg.run_name == "khan_8452_official_residual_norm" and cfg.log_root == root / "logs"
    assert cfg.data.clips is cfg.data.limit is cfg.data.manifest is None
    assert cfg.env.compact_library and not cfg.env.reference_fk_compile
    assert cfg.env.actuator_delay is False and cfg.env.merge_short_tail_bins is True
    assert cfg.env.delay_reset_first_step_no_delay is True
    assert cfg.env.tail_bin_merge_fraction == 1.0 and cfg.env.reset_min_remaining_steps == 32
    assert "actuator delay False" in result.stdout
    baseline = TrainConfig(robot="khan_mini")
    baseline.env.actuator_delay = False
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
    assert cfg.policy.action_mode == "reference_residual"
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
    assert not cfg.policy_trainer.activation_checkpoint
    assert cfg.policy_trainer.horizon == 32 and cfg.policy_trainer.batch_size == 4096
    assert cfg.wm_trainer.batch_size == 2048
    assert cfg.schedule.total_iterations == 320000
    assert cfg.schedule.updates_per_env_step == 0.4
    # Numeric hyperparameters remain owned by TrainConfig; Bash selects the algorithm.
    assert not any(
        flag.startswith(("--world-model.", "--wm-trainer.", "--policy.",
                         "--policy-trainer.", "--critic.", "--schedule."))
        and flag not in {"--policy.history", "--policy.action-mode",
                          "--policy.goal-noise-z", "--policy.goal-noise-joint-pos",
                          "--wm-trainer.optimizer", "--policy-trainer.optimizer",
                          "--policy-trainer.critic-optimizer"}
        for flag in printed_command(result.stdout)[4:]
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
    run = root / "logs/khan_8452_official_residual_norm"
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
    (["--optimizer", "sgd"], "muon or adam"),
    (["--optimizer", "--dry-run"], "requires a value"),
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


@pytest.mark.parametrize("flags,expected", [
    ([], "muon"),
    (["--optimizer", "muon"], "muon"),
    (["--optimizer", "adam"], "adam"),
    (["--optimizer", "muon", "--optimizer", "adam"], "adam"),
    (["--optimizer", "adam", "--optimizer", "muon"], "muon"),
])
def test_optimizer_switch_reaches_all_three_networks_without_changing_budget(project, flags, expected):
    root, _ = project
    result = run_launcher(project, "--dry-run", *flags)
    assert result.returncode == 0, result.stderr
    cfg = parse_config(result.stdout)
    assert cfg.wm_trainer.optimizer == cfg.policy_trainer.optimizer == expected
    assert cfg.policy_trainer.critic_optimizer == expected
    assert cfg.wm_trainer.muon_weight_decay == cfg.policy_trainer.muon_weight_decay == .1
    assert cfg.policy_trainer.critic_muon_weight_decay == .1
    assert cfg.wm_trainer.batch_size == 2048 and cfg.wm_trainer.batches_per_update == 10
    assert cfg.policy_trainer.batch_size == 4096 and cfg.policy_trainer.horizon == 32
    assert cfg.policy_trainer.critic_epochs == 5 and cfg.policy_trainer.critic_minibatches == 4
    assert cfg.schedule.total_iterations == 320000 and cfg.schedule.updates_per_env_step == .4
    assert cfg.policy_trainer.cuda_graph and cfg.wm_trainer.cuda_graph
    assert cfg.env.compact_library and not cfg.env.reference_fk_compile
    assert cfg.env.actuator_delay is False and cfg.policy.action_mode == "reference_residual"
    label = "optimizer: hybrid Muon" if expected == "muon" else "optimizer: Adam"
    assert label in result.stdout
    assert not (root / "logs").exists()


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
    ([], "reference_residual", 0.01, 0.05),
    (["--reference-residual"], "reference_residual", 0.01, 0.05),
    (["--no-reference-residual"], "default", 0.01, 0.05),
    (["--reference-residual", "--no-reference-residual"], "default", 0.01, 0.05),
    (["--no-goal-noise"], "reference_residual", 0.0, 0.0),
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
