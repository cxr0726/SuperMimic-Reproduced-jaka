#!/usr/bin/env bash
# Single-GPU Khan training with clean physical actions and compiled learner updates.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"
NUM_ENVS=4096
RUN_NAME=khan_lafan1_adam_clean_dr_v4_4096
DATA_DIR="$PROJECT_DIR/data/npz/khan_mini/lafan1"
RESUME=''
STOP_AFTER=''
DRY_RUN=0
UPDATE_CAPTURE=1
BATCH_REFERENCE_ENCODER=1
LEARNER_FUSION=1
REPLAY_SAMPLING=''
OPTIMIZER=adam
ACTUATOR_DELAY=True
ANKLE_ACTION_SECOND_DIFF_WEIGHT=-0.02
ACTION_MODE=default
FOOT_TRACKING=0
CONTEXT_LAYOUT=khan_dr_v4
ANKLE_ENCODER_BIAS_HALF_RANGE=0.02
GOAL_NOISE_Z=0.01
GOAL_NOISE_JOINT_POS=0.05

usage() {
    cat <<'HELP'
Usage: bash scripts/training/train_khan_server.sh [options]

Single GPU, Khan_mini, current repository training parameters.
Encoder hidden layers are 1024/512/256; decoder is 2048/1024/512/256.
Actor EMA decay is 0.995.
H=32, UTD=0.4 and the LR schedule reuse the repository defaults.
WM, Actor and Critic use Adam. Whole-update CUDA Graph capture, compiler
fusion, batched reference encoding and the CDF replay sampler are enabled.
Physical collection passes the Actor action through with zero execution noise.
Use --optimizer muon to select hybrid Muon for a separate run.
Khan uses the Isaac observation layout: 5 reference
frames, 5 history frames, goal 155 and policy input 465. The encoder goal has
uniform z +/-0.01 m and q +/-0.05 rad noise; critic/reward stay clean.
The default run uses q_default + 0.5 * actor_output actions and
disables the extra foot pos/ori rewards (raw reward cap 0.112).
--reference-residual opts in to raw clean q_ref(t+1) + 0.5 * decoder_output
(beta 1); --foot-tracking restores extra foot rewards. Only the encoder goal
is noise-augmented and running-normalized; the optional reference skip uses
raw joint positions in radians. Activation recomputation remains disabled.
By default actuator delay is sampled once per episode from 0, 5, 10, 15, 20 ms.
The first control step is immediate. Use --no-actuator-delay for a delay ablation.
The four ankle axes have a squared action second-difference penalty (weight -0.02),
starting at the third control step in each episode.
Khan uses measured rear and forefoot colliders and a knee soft lower limit
of -0.03 rad. All 27 hinges have a nominal frictionloss of 0.1 N m. DR v4 adds
independent startup U(0,0.15) N m to each ankle axis and the same startup
U(0,0.15) kg to both feet. Episode-held encoder bias is +/-0.02 rad on the ankles and
+/-0.01 rad elsewhere. DR v4 gives xi 105 and critic 619. DR v3 gives 101 and
615, with nominal ankle frictionloss and startup foot payload U(0,0.2) kg.
Every incomplete final bin is merged into the preceding bin.
Physical reset requires at least 32 remaining reference steps.
No dataset is downloaded. This branch uses window-value sampling every 20
physical steps (memory 256), WM input running normalization + LayerNorm/GELU,
analytic soft termination, actor obs/goal running normalization, actor/critic
LayerNorm/GELU hidden blocks and critic upper bounds (11.2 by default). Bootstrap successors
stay within their source clip; physical end timing is unchanged. Use a fresh run.

Options:
  --num-envs N                Parallel robot environments (default: 4096)
  --run-name NAME             Directory under project logs/ (default: khan_lafan1_adam_clean_dr_v4_4096)
  --data-dir PATH             Converted Khan NPZ directory (default: project data/npz/khan_mini/lafan1)
  --no-reference-residual    Use default-pose actions (default)
  --reference-residual       Opt in to clean next-reference residual actions
  --no-foot-tracking         Disable extra foot pos/ori rewards (default)
  --foot-tracking            Enable extra foot pos/ori rewards, weights 0.25/0.25
  --ankle-frictionloss       Randomize four ankle axes with DR v4 (default)
  --no-ankle-frictionloss    Use DR v3 with nominal ankle frictionloss and U(0,0.2) kg foot payload
  --ankle-encoder-bias RAD   Ankle encoder bias half-range (default: 0.02 rad)
  --no-goal-noise            Disable encoder goal augmentation
  --goal-noise               Enable Isaac encoder goal noise (default)
  --no-actuator-delay        Disable actuator delay
  --actuator-delay           Enable held 0–20 ms delay (default); first control step is immediate
  --no-ankle-action-second-diff  Disable the ankle penalty for matching older runs
  --update-capture          Capture full WM and Actor/Critic updates (default; single GPU)
  --no-update-capture       Use ordinary learner updates for comparison
  --batch-reference-encoder Batch reference-only goals across H32 (default)
  --no-batch-reference-encoder  Encode each imagined step separately
  --learner-fusion          Compile full WM window and Actor/Critic blocks (default)
  --no-learner-fusion       Use the d98f4df single-step compilation boundary
  --replay-sampling MODE    cdf (default) or multinomial; same legal-pair distribution
  --optimizer MODE          adam (default) or muon; selects WM, Actor and Critic together
  --resume PATH               Restore a Khan checkpoint with the same data/configuration
  --stop-after-iterations N   Optional execution limit; retains the 320k LR schedule
  --dry-run                  Validate paths and print the command without importing CUDA or training
  -h, --help                 Show help

The default run executes up to 320000 vectorized control steps. Use a new
run-name for a fresh run; existing checkpoints are never silently overwritten.
Use matching model assets, action noise and optimizer settings for resume.
Start a fresh run when changing any of those settings.
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
        --no-foot-tracking) FOOT_TRACKING=0; shift ;;
        --foot-tracking) FOOT_TRACKING=1; shift ;;
        --no-ankle-frictionloss) CONTEXT_LAYOUT=khan_dr_v3; shift ;;
        --ankle-frictionloss) CONTEXT_LAYOUT=khan_dr_v4; shift ;;
        --ankle-encoder-bias) need_value "$@"; ANKLE_ENCODER_BIAS_HALF_RANGE="$2"; shift 2 ;;
        --no-goal-noise) GOAL_NOISE_Z=0.0; GOAL_NOISE_JOINT_POS=0.0; shift ;;
        --goal-noise) GOAL_NOISE_Z=0.01; GOAL_NOISE_JOINT_POS=0.05; shift ;;
        --no-actuator-delay) ACTUATOR_DELAY=False; shift ;;
        --actuator-delay) ACTUATOR_DELAY=True; shift ;;
        --no-ankle-action-second-diff) ANKLE_ACTION_SECOND_DIFF_WEIGHT=0.0; shift ;;
        --update-capture) UPDATE_CAPTURE=1; shift ;;
        --no-update-capture) UPDATE_CAPTURE=0; shift ;;
        --batch-reference-encoder) BATCH_REFERENCE_ENCODER=1; shift ;;
        --no-batch-reference-encoder) BATCH_REFERENCE_ENCODER=0; shift ;;
        --learner-fusion) LEARNER_FUSION=1; shift ;;
        --no-learner-fusion) LEARNER_FUSION=0; shift ;;
        --replay-sampling) need_value "$@"; REPLAY_SAMPLING="$2"; shift 2 ;;
        --optimizer) need_value "$@"; OPTIMIZER="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "Unknown option: $1 (use --help)" ;;
    esac
done

positive_integer "$NUM_ENVS" || die '--num-envs must be a positive integer'
[[ "$ANKLE_ENCODER_BIAS_HALF_RANGE" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || \
    die '--ankle-encoder-bias must be a nonnegative decimal in radians'
[[ -z "$REPLAY_SAMPLING" || "$REPLAY_SAMPLING" == cdf || "$REPLAY_SAMPLING" == multinomial ]] || die '--replay-sampling must be cdf or multinomial'
[[ "$OPTIMIZER" == muon || "$OPTIMIZER" == adam ]] || die '--optimizer must be muon or adam'
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
    --env.observation-layout isaac_khan --env.context-layout "$CONTEXT_LAYOUT" --env.future-frames 5 --policy.history 5
    --policy.action-mode "$ACTION_MODE"
    --policy.goal-noise-z "$GOAL_NOISE_Z" --policy.goal-noise-joint-pos "$GOAL_NOISE_JOINT_POS"
    --env.actuator-delay "$ACTUATOR_DELAY" --env.delay-reset-first-step-no-delay True
    --env.action-noise-std 0.0 --env.ankle-encoder-bias-half-range "$ANKLE_ENCODER_BIAS_HALF_RANGE"
    --env.ankle-action-second-diff-weight "$ANKLE_ACTION_SECOND_DIFF_WEIGHT"
    --env.merge-short-tail-bins True --env.tail-bin-merge-fraction 1.0
    --env.reset-min-remaining-steps 32
    --wm-trainer.optimizer "$OPTIMIZER"
    --policy-trainer.optimizer "$OPTIMIZER" --policy-trainer.critic-optimizer "$OPTIMIZER"
)
# TrainConfig owns the enabled defaults; only append explicit opt-outs.
if ((!UPDATE_CAPTURE)); then
    command+=(--wm-trainer.no-cuda-graph --policy-trainer.no-cuda-graph)
fi
if ((!BATCH_REFERENCE_ENCODER)); then
    command+=(--policy-trainer.no-batch-reference-encoder)
fi
if ((!LEARNER_FUSION)); then
    command+=(--wm-trainer.no-compile-full-window --policy-trainer.no-compile-networks)
fi
if [[ -n "$REPLAY_SAMPLING" ]]; then
    command+=(--replay-sampling "$REPLAY_SAMPLING")
fi
# Additional foot rewards are independent of foot payload randomization.
RAW_REWARD_CAP=0.112
CRITIC_UPPER_BOUND=11.2
if ((FOOT_TRACKING)); then
    command+=(--env.foot-tracking)
    RAW_REWARD_CAP=0.122
    CRITIC_UPPER_BOUND=12.2
else
    command+=(--env.no-foot-tracking)
fi
if [[ -n "$STOP_AFTER" ]]; then
    command+=(--schedule.stop-after-iterations "$STOP_AFTER")
fi
if [[ -n "$RESUME" ]]; then
    command+=(--resume "$RESUME")
    printf '[INFO] Resume requires matching window sampler, normalization, architecture, action, reward and context layout; pre-port checkpoints require a fresh run.\n'
fi
printf '[INFO] Khan clips: %d; parallel environments: %s; learning batch: 4096; H: 32\n' \
    "${#NPZ_FILES[@]}" "$NUM_ENVS"
printf '[INFO] Learner execution: full-update capture=%s; batched reference encoder=%s\n' "$UPDATE_CAPTURE" "$BATCH_REFERENCE_ENCODER"
printf '[INFO] Learner fusion=%s; replay sampling=%s; FP32 model/state tensors and original learning budget retained\n' "$LEARNER_FUSION" "${REPLAY_SAMPLING:-cdf}"
if [[ "$OPTIMIZER" == muon ]]; then
    printf '[INFO] WM/Actor/Critic optimizer: hybrid Muon; hidden matrices Muon (decay 0.1); input/output weights and vectors fused capturable Adam (decay 0); live LR follows cosine schedule\n'
    printf '[INFO] Muon Newton-Schulz internally uses BF16; network autocast remains OFF\n'
else
    printf '[INFO] WM/Actor/Critic optimizer: Adam; zero weight decay; CUDA Graph-compatible when capture is enabled\n'
fi
printf '[INFO] Khan observations: future/history 5/5; goal 155; policy 465; activation recomputation: OFF\n'
printf '[INFO] Actor hidden defaults: encoder 1024/512/256; decoder 2048/1024/512/256; EMA decay 0.995; training uses live, inference auto-selects EMA\n'
printf '[INFO] Actor/Critic hidden blocks: Linear -> LayerNorm -> GELU; bootstrap successor: source-clip clamped\n'
printf '[INFO] Official port defaults: window_value sampler; horizon 20; memory 256; WM input running norm + LayerNorm/GELU; analytic soft termination; actor obs/goal running norm; critic upper bound %s\n' "$CRITIC_UPPER_BOUND"
if [[ "$ACTION_MODE" == default ]]; then
    printf '[INFO] Actions: default; PD target=q_default + 0.5*actor_output; reference skip OFF\n'
else
    printf '[INFO] Actions: reference_residual; PD target=raw clean q_ref(t+1) + 0.5*decoder_output; beta 1\n'
fi
printf '[INFO] Encoder goal noise: z +/- %s m, q +/- %s rad; then running normalization; critic/reward stay clean\n' "$GOAL_NOISE_Z" "$GOAL_NOISE_JOINT_POS"
printf '[INFO] Physical collection execution noise: 0 on all 27 joints\n'
printf '[INFO] Four-ankle action second-difference weight: %s (scored from step 3)\n' "$ANKLE_ACTION_SECOND_DIFF_WEIGHT"
printf '[INFO] Episode encoder bias: four ankle axes +/- %s rad; other 23 axes +/- 0.01 rad\n' "$ANKLE_ENCODER_BIAS_HALF_RANGE"
printf '[INFO] Knee soft lower -0.03 rad; nominal frictionloss 0.1 N m on all 27 hinges\n'
if [[ "$CONTEXT_LAYOUT" == khan_dr_v4 ]]; then
    printf '[INFO] Khan DR v4: xi 105; critic 619; four ankles add independent startup U(0,0.15) N m\n'
    FOOT_PAYLOAD="shared startup payload U(0,0.15) kg, same addition on both feet"
else
    printf '[INFO] Khan DR v3: xi 101; critic 615; ankle frictionloss: nominal\n'
    FOOT_PAYLOAD="independent startup payload U(0,0.2) kg per foot"
fi
printf '[INFO] Kp/Kd x0.8-1.2 per reset; 12 leg armatures x0.8-1.2 at startup; waist mass -2..3 kg; actuator delay %s (configured 0–20 ms; applied only when enabled)\n' "$ACTUATOR_DELAY"
printf '[INFO] Feet: nominal ankle_roll_Link mass 0.58502 kg; %s, authored COM/inertia retained\n' "$FOOT_PAYLOAD"
if ((FOOT_TRACKING)); then
    printf '[INFO] Extra foot pos/ori rewards: ON, weights 0.25/0.25; raw reward cap %s\n' "$RAW_REWARD_CAP"
else
    printf '[INFO] Extra foot pos/ori rewards: OFF; original 15-body/neck rewards retained; raw reward cap %s\n' "$RAW_REWARD_CAP"
fi
printf '[INFO] Waist local COM offset at startup: x +/-0.035 m, y/z +/-0.07 m\n'
printf '[INFO] Reset bins: merge every incomplete final bin; require >=32 remaining reference steps; first-step action-rate unchanged\n'
printf '[INFO] Nominal ankle armature pitch 0.02 / roll 0.01; ankle pitch Kp/Kd 80/2, roll 20/1; reset root clearance 0.05 m; measured rear and forefoot colliders\n'
printf '[COMMAND]'
printf ' %q' "${command[@]}"
printf '\n'
if ((DRY_RUN)); then
    exit 0
fi
exec "${command[@]}"
