#!/usr/bin/env bash
# Single-GPU Khan training from a packed compact reference library.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"
NUM_ENVS=4096
RUN_NAME=khan_bones_pt
LIBRARY="$PROJECT_DIR/data/pt/khan_combined_8452_compact.pt"
REFERENCE_FK_COMPILE=0
RESUME=''
STOP_AFTER=''
DRY_RUN=0
ACTUATOR_DELAY=True

usage() {
    cat <<'HELP'
Usage: bash scripts/training/train_khan_compact_server.sh [options]

Single GPU, Khan_mini, packed compact .pt motion library.
Uses the same training entry point and original network/optimization defaults
as train_khan_server.sh: H=32, batch=2048/4096, UTD=0.4, 320k LR schedule,
Isaac observation layout with future/history 5/5, clean goal 155, policy 465.
Actuator delay is held at 0–20 ms per episode; the first control step is immediate.
Reset clearance remains 0.05 m and first-step action-rate is unchanged. Every
incomplete final bin is merged into the previous bin. Physical reset requires
at least 32 remaining reference steps. Activation recomputation remains disabled.
Reference FK compilation is opt-in;
its effect on total training throughput depends on the target GPU.

Options:
  --library PATH             Packed compact .pt file (default: project data/pt/khan_combined_8452_compact.pt)
  --num-envs N               Parallel robot environments (default: 4096)
  --run-name NAME            Directory under project logs/ (default: khan_bones_pt)
  --reference-fk-compile     Compile current/imagination reference FK on CUDA
  --no-reference-fk-compile  Use eager reference FK (default)
  --no-actuator-delay        Disable delay for an otherwise identical H32/full-tail ablation
  --actuator-delay           Enable held 0–20 ms delay (default); first control step is immediate
  --resume PATH              Restore a Khan checkpoint with matching data/configuration/assets
  --stop-after-iterations N  Optional execution limit; retains the 320k LR schedule
  --dry-run                 Check paths and print command without executing Python or training
  -h, --help                Show help

Use a new run-name for a fresh run; existing checkpoints are never silently
overwritten. Dry-run checks file paths, not the tensors or metadata inside .pt.
Training validates library/robot metadata and checkpoint compatibility.
Use train_khan_server.sh --data-dir PATH for a directory of full NPZ motions.
HELP
}

die() { printf '[ERROR] %s\n' "$*" >&2; exit 1; }
need_value() { (($# >= 2)) && [[ -n "$2" && "$2" != --* ]] || die "$1 requires a value"; }
positive_integer() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }

while (($#)); do
    case "$1" in
        --library) need_value "$@"; LIBRARY="$2"; shift 2 ;;
        --num-envs) need_value "$@"; NUM_ENVS="$2"; shift 2 ;;
        --run-name) need_value "$@"; RUN_NAME="$2"; shift 2 ;;
        --reference-fk-compile) REFERENCE_FK_COMPILE=1; shift ;;
        --no-reference-fk-compile) REFERENCE_FK_COMPILE=0; shift ;;
        --resume) need_value "$@"; RESUME="$2"; shift 2 ;;
        --stop-after-iterations) need_value "$@"; STOP_AFTER="$2"; shift 2 ;;
        --data-dir) die 'Use --library PATH for a compact .pt file; for NPZ use train_khan_server.sh --data-dir PATH' ;;
        --no-anchor-obs|--anchor-obs) die 'The Isaac layout already includes 5-frame anchor orientation in policy observations; remove the old anchor flag' ;;
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
[[ "$LIBRARY" == *.pt ]] || die "Compact library must be a .pt file: $LIBRARY"
[[ -f "$LIBRARY" ]] || die "Compact library file does not exist: $LIBRARY"
[[ -r "$LIBRARY" && -s "$LIBRARY" ]] || die "Compact library must be readable and nonempty: $LIBRARY"
LIBRARY="$(cd -- "$(dirname -- "$LIBRARY")" && pwd -P)/$(basename -- "$LIBRARY")"

RUN_DIR="$PROJECT_DIR/logs/$RUN_NAME"
shopt -s nullglob
EXISTING_CHECKPOINTS=("$RUN_DIR"/ckpt*.pt)
if [[ -z "$RESUME" && ${#EXISTING_CHECKPOINTS[@]} -gt 0 ]]; then
    die "Checkpoints already exist in $RUN_DIR; use --resume or choose another --run-name"
fi
shopt -u nullglob
if [[ -n "$RESUME" ]]; then
    [[ -f "$RESUME" ]] || die "Resume checkpoint does not exist: $RESUME"
    [[ -r "$RESUME" && -s "$RESUME" ]] || die "Resume checkpoint must be readable and nonempty: $RESUME"
    RESUME="$(cd -- "$(dirname -- "$RESUME")" && pwd -P)/$(basename -- "$RESUME")"
fi

# Keep single-GPU launches independent of ROS, Conda and stale torchrun settings.
unset PYTHONPATH PYTHONHOME
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT SM_GPU_PINNED
export PYTHONNOUSERSITE=1
export MUJOCO_GL=egl
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
cd -- "$PROJECT_DIR"

# TrainConfig owns the network and optimization values, just as in the NPZ launcher.
command=(
    "$VENV_PYTHON" -u -m supermimic.scripts.train
    --robot khan_mini --run-name "$RUN_NAME" --log-root "$PROJECT_DIR/logs"
    --data.library "$LIBRARY" --env.num-envs "$NUM_ENVS" --env.compact-library
    --env.observation-layout isaac_khan --env.future-frames 5 --policy.history 5
    --env.actuator-delay "$ACTUATOR_DELAY" --env.delay-reset-first-step-no-delay True
    --env.merge-short-tail-bins True --env.tail-bin-merge-fraction 1.0
    --env.reset-min-remaining-steps 32
)
REFERENCE_FK_MODE=eager
if ((REFERENCE_FK_COMPILE)); then
    command+=(--env.reference-fk-compile)
    REFERENCE_FK_MODE=compiled
else
    command+=(--env.no-reference-fk-compile)
fi
if [[ -n "$STOP_AFTER" ]]; then
    command+=(--schedule.stop-after-iterations "$STOP_AFTER")
fi
if [[ -n "$RESUME" ]]; then
    command+=(--resume "$RESUME")
fi

printf '[INFO] Khan compact library: %s\n' "$LIBRARY"
printf '[INFO] Reference FK: %s; parallel environments: %s; learning batch: 4096; H: 32\n' \
    "$REFERENCE_FK_MODE" "$NUM_ENVS"
printf '[INFO] Khan observations: future/history 5/5; clean goal 155; policy 465; activation recomputation: OFF\n'
printf '[INFO] Khan: xi 91; critic 605; Kp/Kd and ankle armature x0.8-1.2; waist mass -2..3 kg; actuator delay %s (0–20 ms, first control step immediate)\n' "$ACTUATOR_DELAY"
printf '[INFO] Reset bins: merge every incomplete final bin; require >=32 remaining reference steps; first-step action-rate unchanged\n'
printf '[INFO] Nominal ankle armature pitch 0.02 / roll 0.01; ankle pitch Kp/Kd 80/2, roll 20/1; reset root clearance 0.05 m; full convex feet\n'
printf '[COMMAND]'
printf ' %q' "${command[@]}"
printf '\n'
if ((DRY_RUN)); then
    exit 0
fi
exec "${command[@]}"
