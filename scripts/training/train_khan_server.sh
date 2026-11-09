#!/usr/bin/env bash
# Single-GPU Khan training with the original single-GPU LAFAN optimization values.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"
NUM_ENVS=4096
RUN_NAME=khan_lafan1_residual_goal
DATA_DIR="$PROJECT_DIR/data/npz/khan_mini/lafan1"
RESUME=''
STOP_AFTER=''
DRY_RUN=0
ACTUATOR_DELAY=False
ACTION_MODE=reference_residual
GOAL_NOISE_Z=0.01
GOAL_NOISE_JOINT_POS=0.05

usage() {
    cat <<'HELP'
Usage: bash scripts/training/train_khan_server.sh [options]

Single GPU, Khan_mini, original single-GPU LAFAN training parameters.
Networks, optimization, H=32, UTD=0.4 and LR schedule reuse
the repository defaults. Khan uses the Isaac observation layout: 5 reference
frames, 5 history frames, goal 155 and policy input 465. The encoder goal has
uniform z +/-0.01 m and q +/-0.05 rad noise; the reference action skip stays clean.
Actions use q_ref(t+1) + 0.5 * decoder_residual. Activation
recomputation remains disabled. Actuator delay is disabled by default for this
server experiment; --actuator-delay enables held 0–20 ms delay with an immediate first step.
Khan uses full CAD convex feet, actual root reset clearance 0.05 m and a 605-dim
critic. Dynamics DR (PD, ankle armature, waist mass) retains the 91-dim context
including delay. Every incomplete final bin is merged into the preceding bin.
Physical reset requires at least 32 remaining reference steps.
No dataset is downloaded. Use a fresh run for the changed action space.

Options:
  --num-envs N                Parallel robot environments (default: 4096)
  --run-name NAME             Directory under project logs/ (default: khan_lafan1_residual_goal)
  --data-dir PATH             Converted Khan NPZ directory (default: project data/npz/khan_mini/lafan1)
  --no-reference-residual    Use the legacy default-pose action space
  --reference-residual      Use the next-reference residual action space (default)
  --no-goal-noise            Disable encoder goal augmentation; reference skip stays clean
  --goal-noise               Enable Isaac encoder goal noise (default)
  --no-actuator-delay        Disable actuator delay (default for this server experiment)
  --actuator-delay           Enable held 0–20 ms delay; first control step is immediate
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
        --no-reference-residual) ACTION_MODE=default; shift ;;
        --reference-residual) ACTION_MODE=reference_residual; shift ;;
        --no-goal-noise) GOAL_NOISE_Z=0.0; GOAL_NOISE_JOINT_POS=0.0; shift ;;
        --goal-noise) GOAL_NOISE_Z=0.01; GOAL_NOISE_JOINT_POS=0.05; shift ;;
        --no-actuator-delay) ACTUATOR_DELAY=False; shift ;;
        --actuator-delay) ACTUATOR_DELAY=True; shift ;;
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
    --policy.action-mode "$ACTION_MODE"
    --policy.goal-noise-z "$GOAL_NOISE_Z" --policy.goal-noise-joint-pos "$GOAL_NOISE_JOINT_POS"
    --env.actuator-delay "$ACTUATOR_DELAY" --env.delay-reset-first-step-no-delay True
    --env.merge-short-tail-bins True --env.tail-bin-merge-fraction 1.0
    --env.reset-min-remaining-steps 32
)
if [[ -n "$STOP_AFTER" ]]; then
    command+=(--schedule.stop-after-iterations "$STOP_AFTER")
fi
if [[ -n "$RESUME" ]]; then
    command+=(--resume "$RESUME")
    printf '[INFO] Resume requires matching action mode and goal noise. Legacy clean-goal checkpoints need --no-reference-residual --no-goal-noise.\n'
fi
printf '[INFO] Khan clips: %d; parallel environments: %s; learning batch: 4096; H: 32\n' \
    "${#NPZ_FILES[@]}" "$NUM_ENVS"
printf '[INFO] Khan observations: future/history 5/5; goal 155; policy 465; activation recomputation: OFF\n'
printf '[INFO] Actions: %s; scale 0.5; residual mode uses beta 1 and reference offset 1\n' "$ACTION_MODE"
printf '[INFO] Encoder goal noise: z +/- %s m, q +/- %s rad; reference skip and critic stay clean\n' "$GOAL_NOISE_Z" "$GOAL_NOISE_JOINT_POS"
printf '[INFO] Khan: xi 91; critic 605; Kp/Kd and ankle armature x0.8-1.2; waist mass -2..3 kg; actuator delay %s (configured 0–20 ms; applied only when enabled)\n' "$ACTUATOR_DELAY"
printf '[INFO] Reset bins: merge every incomplete final bin; require >=32 remaining reference steps; first-step action-rate unchanged\n'
printf '[INFO] Nominal ankle armature pitch 0.02 / roll 0.01; ankle pitch Kp/Kd 80/2, roll 20/1; reset root clearance 0.05 m; full convex feet\n'
printf '[COMMAND]'
printf ' %q' "${command[@]}"
printf '\n'
if ((DRY_RUN)); then
    exit 0
fi
exec "${command[@]}"
