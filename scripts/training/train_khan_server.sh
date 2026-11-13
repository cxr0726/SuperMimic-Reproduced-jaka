#!/usr/bin/env bash
# Single-GPU Khan training with current actor/EMA and original optimization values.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"
NUM_ENVS=4096
RUN_NAME=khan_lafan1_official_residual_norm
DATA_DIR="$PROJECT_DIR/data/npz/khan_mini/lafan1"
RESUME=''
STOP_AFTER=''
DRY_RUN=0
UPDATE_CAPTURE=1
BATCH_REFERENCE_ENCODER=1
ACTUATOR_DELAY=False
ACTION_MODE=reference_residual
GOAL_NOISE_Z=0.01
GOAL_NOISE_JOINT_POS=0.05

usage() {
    cat <<'HELP'
Usage: bash scripts/training/train_khan_server.sh [options]

Single GPU, Khan_mini, current repository training parameters.
Encoder hidden layers are 1024/512/256; decoder is 2048/1024/512/256.
Actor EMA decay is 0.995.
Optimization, H=32, UTD=0.4 and LR schedule reuse the repository defaults.
Khan uses the Isaac observation layout: 5 reference
frames, 5 history frames, goal 155 and policy input 465. The encoder goal has
uniform z +/-0.01 m and q +/-0.05 rad noise; critic/reward stay clean.
Residual actions use raw clean q_ref(t+1) + 0.5 * decoder_output (beta 1).
Only the encoder goal is noise-augmented and running-normalized; the reference
skip uses raw joint positions in radians. Activation recomputation remains disabled.
Actuator delay is disabled by default for this server experiment; --actuator-delay
enables held 0–20 ms delay with an immediate first step.
Khan uses full CAD convex feet, actual root reset clearance 0.05 m and a 615-dim
critic. Dynamics DR uses a 101-dim context: gains per reset, 12 leg armatures
at startup, waist mass and COM at startup, including the delay parameter. Every incomplete final bin is merged into the preceding bin.
Physical reset requires at least 32 remaining reference steps.
No dataset is downloaded. This branch uses window-value sampling every 20
physical steps (memory 256), WM input running normalization + LayerNorm/GELU,
analytic soft termination, actor obs/goal running normalization, actor/critic
LayerNorm/GELU hidden blocks and critic upper bounds (12.2). Bootstrap successors
stay within their source clip; physical end timing is unchanged. Use a fresh run.

Options:
  --num-envs N                Parallel robot environments (default: 4096)
  --run-name NAME             Directory under project logs/ (default: khan_lafan1_official_residual_norm)
  --data-dir PATH             Converted Khan NPZ directory (default: project data/npz/khan_mini/lafan1)
  --no-reference-residual   Opt out to default-pose actions for comparison
  --reference-residual      Use clean next-reference residual actions (default)
  --no-goal-noise           Disable encoder goal augmentation
  --goal-noise               Enable Isaac encoder goal noise (default)
  --no-actuator-delay        Disable actuator delay (default for this server experiment)
  --actuator-delay           Enable held 0–20 ms delay; first control step is immediate
  --update-capture          Capture full WM and Actor/Critic updates (default; single GPU)
  --no-update-capture       Use ordinary learner updates for comparison
  --batch-reference-encoder Batch reference-only goals across H32 (default)
  --no-batch-reference-encoder  Encode each imagined step separately
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
        --update-capture) UPDATE_CAPTURE=1; shift ;;
        --no-update-capture) UPDATE_CAPTURE=0; shift ;;
        --batch-reference-encoder) BATCH_REFERENCE_ENCODER=1; shift ;;
        --no-batch-reference-encoder) BATCH_REFERENCE_ENCODER=0; shift ;;
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
    --env.observation-layout isaac_khan --env.context-layout khan_dr_v3 --env.future-frames 5 --policy.history 5
    --policy.action-mode "$ACTION_MODE"
    --policy.goal-noise-z "$GOAL_NOISE_Z" --policy.goal-noise-joint-pos "$GOAL_NOISE_JOINT_POS"
    --env.actuator-delay "$ACTUATOR_DELAY" --env.delay-reset-first-step-no-delay True
    --env.merge-short-tail-bins True --env.tail-bin-merge-fraction 1.0
    --env.reset-min-remaining-steps 32
)
# TrainConfig owns the enabled defaults; only append explicit opt-outs.
if ((!UPDATE_CAPTURE)); then
    command+=(--wm-trainer.no-cuda-graph --policy-trainer.no-cuda-graph)
fi
if ((!BATCH_REFERENCE_ENCODER)); then
    command+=(--policy-trainer.no-batch-reference-encoder)
fi
if [[ -n "$STOP_AFTER" ]]; then
    command+=(--schedule.stop-after-iterations "$STOP_AFTER")
fi
if [[ -n "$RESUME" ]]; then
    command+=(--resume "$RESUME")
    printf '[INFO] Resume requires matching window sampler, normalization, architecture and action semantics; pre-port checkpoints require a fresh run.\n'
fi
printf '[INFO] Khan clips: %d; parallel environments: %s; learning batch: 4096; H: 32\n' \
    "${#NPZ_FILES[@]}" "$NUM_ENVS"
printf '[INFO] Learner execution: full-update capture=%s; batched reference encoder=%s; collection/reset unchanged\n' "$UPDATE_CAPTURE" "$BATCH_REFERENCE_ENCODER"
printf '[INFO] Khan observations: future/history 5/5; goal 155; policy 465; activation recomputation: OFF\n'
printf '[INFO] Actor hidden defaults: encoder 1024/512/256; decoder 2048/1024/512/256; EMA decay 0.995; training uses live, inference auto-selects EMA\n'
printf '[INFO] Actor/Critic hidden blocks: Linear -> LayerNorm -> GELU; bootstrap successor: source-clip clamped\n'
printf '[INFO] Official port defaults: window_value sampler; horizon 20; memory 256; WM input running norm + LayerNorm/GELU; analytic soft termination; actor obs/goal running norm; critic upper bound 12.2\n'
if [[ "$ACTION_MODE" == default ]]; then
    printf '[INFO] Actions: default; PD target=q_default + 0.5*actor_output; reference skip OFF\n'
else
    printf '[INFO] Actions: reference_residual; PD target=raw clean q_ref(t+1) + 0.5*decoder_output; beta 1\n'
fi
printf '[INFO] Encoder goal noise: z +/- %s m, q +/- %s rad; then running normalization; critic/reward stay clean; reference skip uses raw clean radians\n' "$GOAL_NOISE_Z" "$GOAL_NOISE_JOINT_POS"
printf '[INFO] Khan: xi 101; critic 615; Kp/Kd x0.8-1.2 per reset; 12 leg armatures x0.8-1.2 at startup; waist mass -2..3 kg; actuator delay %s (configured 0–20 ms; applied only when enabled)\n' "$ACTUATOR_DELAY"
printf '[INFO] Foot payload: independent left/right U(0,0.2) kg at startup, authored COM/inertia retained; foot pos/ori reward weights 0.25/0.25; raw reward cap 0.122\n'
printf '[INFO] Waist local COM offset at startup: x +/-0.035 m, y/z +/-0.07 m\n'
printf '[INFO] Reset bins: merge every incomplete final bin; require >=32 remaining reference steps; first-step action-rate unchanged\n'
printf '[INFO] Nominal ankle armature pitch 0.02 / roll 0.01; ankle pitch Kp/Kd 80/2, roll 20/1; reset root clearance 0.05 m; full convex feet\n'
printf '[COMMAND]'
printf ' %q' "${command[@]}"
printf '\n'
if ((DRY_RUN)); then
    exit 0
fi
exec "${command[@]}"
