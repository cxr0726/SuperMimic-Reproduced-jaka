#!/usr/bin/env bash
# Single-GPU Khan training with the original single-GPU LAFAN optimization values.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"
NUM_ENVS=4096
RUN_NAME=khan_lafan1_clean_goal
DATA_DIR="$PROJECT_DIR/data/npz/khan_mini/lafan1"
RESUME=''
STOP_AFTER=''
DRY_RUN=0

usage() {
    cat <<'HELP'
Usage: bash scripts/training/train_khan_server.sh [options]

Single GPU, Khan_mini, original single-GPU LAFAN training parameters.
Networks, optimization, H=32, UTD=0.4, randomization and LR schedule reuse
the repository defaults. Khan uses the Isaac observation layout: 5 reference
frames, 5 history frames, clean goal 155 and policy input 465. Activation
recomputation remains disabled. No dataset is downloaded.

Options:
  --num-envs N                Parallel robot environments (default: 4096)
  --run-name NAME             Directory under project logs/ (default: khan_lafan1_clean_goal)
  --data-dir PATH             Converted Khan NPZ directory (default: project data/npz/khan_mini/lafan1)
  --resume PATH               Restore a Khan checkpoint with the same data/configuration
  --stop-after-iterations N   Optional execution limit; retains the 320k LR schedule
  --dry-run                  Validate paths and print the command without importing CUDA or training
  -h, --help                 Show help

The default run executes up to 320000 vectorized control steps. Use a new
run-name for a fresh run; existing checkpoints are never silently overwritten.
HELP
}

die() { printf '[ERROR] %s\n' "$*" >&2; exit 1; }
need_value() { (($# >= 2)) && [[ -n "$2" && "$2" != --* ]] || die "$1 requires a value"; }
positive_integer() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }

while (($#)); do
    case "$1" in
        --num-envs) need_value "$@"; NUM_ENVS="$2"; shift 2 ;;
        --run-name) need_value "$@"; RUN_NAME="$2"; shift 2 ;;
        --data-dir) need_value "$@"; DATA_DIR="$2"; shift 2 ;;
        --resume) need_value "$@"; RESUME="$2"; shift 2 ;;
        --stop-after-iterations) need_value "$@"; STOP_AFTER="$2"; shift 2 ;;
        --no-anchor-obs|--anchor-obs) die 'The Isaac layout already includes 5-frame anchor orientation in policy observations; remove the old anchor flag' ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "Unknown option: $1 (use --help)" ;;
    esac
done

positive_integer "$NUM_ENVS" || die '--num-envs must be a positive integer'
[[ -z "$STOP_AFTER" ]] || positive_integer "$STOP_AFTER" || \
    die '--stop-after-iterations must be a positive integer'
[[ -n "$RUN_NAME" && "$RUN_NAME" != . && "$RUN_NAME" != .. && "$RUN_NAME" != */* \
    && "$RUN_NAME" != *$'\n'* && "$RUN_NAME" != *$'\r'* ]] || \
    die '--run-name must be one nonempty directory name without path separators'

VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
    printf '[ERROR] Missing project Python: %s\nRun environment setup first:\n  bash %q --project-dir %q\n' \
        "$VENV_PYTHON" "$PROJECT_DIR/scripts/setup/setup_server_env.sh" "$PROJECT_DIR" >&2
    exit 1
fi
for asset in robot.xml profile.json; do
    [[ -f "$PROJECT_DIR/assets/robots/khan_mini/$asset" ]] || \
        die "Missing Khan asset: $PROJECT_DIR/assets/robots/khan_mini/$asset; copy the project assets/ directory"
done
[[ -d "$DATA_DIR" ]] || die "Khan NPZ directory does not exist: $DATA_DIR"
DATA_DIR="$(cd -- "$DATA_DIR" && pwd -P)"
shopt -s nullglob
NPZ_FILES=("$DATA_DIR"/*.npz)
((${#NPZ_FILES[@]})) || die "No converted Khan NPZ files in: $DATA_DIR"
RUN_DIR="$PROJECT_DIR/logs/$RUN_NAME"
EXISTING_CHECKPOINTS=("$RUN_DIR"/ckpt*.pt)
if [[ -z "$RESUME" && ${#EXISTING_CHECKPOINTS[@]} -gt 0 ]]; then
    die "Checkpoints already exist in $RUN_DIR; use --resume or choose another --run-name"
fi
if [[ -n "$RESUME" ]]; then
    [[ -f "$RESUME" ]] || die "Resume checkpoint does not exist: $RESUME"
    RESUME="$(cd -- "$(dirname -- "$RESUME")" && pwd -P)/$(basename -- "$RESUME")"
fi
shopt -u nullglob

# Avoid ROS/Conda imports and accidental torchrun configuration in a single-GPU launch.
unset PYTHONPATH PYTHONHOME
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT SM_GPU_PINNED
export PYTHONNOUSERSITE=1
export MUJOCO_GL=egl
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
cd -- "$PROJECT_DIR"

# TrainConfig owns the original network/optimization defaults. Observation
# windows and input layout are explicitly matched to Khan's Isaac policy.
command=(
    "$VENV_PYTHON" -u -m supermimic.scripts.train
    --robot khan_mini --run-name "$RUN_NAME" --log-root "$PROJECT_DIR/logs"
    --data.npz-dir "$DATA_DIR" --env.num-envs "$NUM_ENVS"
    --env.observation-layout isaac_khan --env.future-frames 5 --policy.history 5
)
if [[ -n "$STOP_AFTER" ]]; then
    command+=(--schedule.stop-after-iterations "$STOP_AFTER")
fi
if [[ -n "$RESUME" ]]; then
    command+=(--resume "$RESUME")
fi
printf '[INFO] Khan clips: %d; parallel environments: %s; learning batch: 4096; H: 32\n' \
    "${#NPZ_FILES[@]}" "$NUM_ENVS"
printf '[INFO] Khan observations: future/history 5/5; clean goal 155; policy 465; activation recomputation: OFF\n'
printf '[COMMAND]'
printf ' %q' "${command[@]}"
printf '\n'
if ((DRY_RUN)); then
    exit 0
fi
exec "${command[@]}"
