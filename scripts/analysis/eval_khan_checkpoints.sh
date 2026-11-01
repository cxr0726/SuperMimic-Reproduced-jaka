#!/usr/bin/env bash
# Evaluate every provided checkpoint with the repository's unchanged benchmark.
# Usage: bash scripts/analysis/eval_khan_checkpoints.sh [run_dir] [npz_dir] [num_envs]
#   [--legacy-revision fcad4a5f69b3b3c896b559314d1cff430b81990f]
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$PROJECT_DIR"
LEGACY_REVISION=''
positionals=()
while (($#)); do
    case "$1" in
        --legacy-revision)
            [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || {
                echo '--legacy-revision requires a Git revision' >&2; exit 1;
            }
            LEGACY_REVISION="$2"; shift 2 ;;
        -h|--help)
            echo 'Usage: bash scripts/analysis/eval_khan_checkpoints.sh [run_dir] [npz_dir] [num_envs] [--legacy-revision REV]'
            echo 'Legacy 14-body/waist-anchor checkpoints require --legacy-revision fcad4a5f69b3b3c896b559314d1cff430b81990f'
            exit 0 ;;
        --*) echo "Unknown option: $1" >&2; exit 1 ;;
        *) positionals+=("$1"); shift ;;
    esac
done
((${#positionals[@]} <= 3)) || { echo 'Too many positional arguments' >&2; exit 1; }
RUN_DIR=${positionals[0]:-logs/Khan_lafan1_4096}
DATA_DIR=${positionals[1]:-data/npz/khan_mini/lafan1}
NUM_ENVS=${positionals[2]:-36}
PYTHON_BIN=${VENV_PYTHON:-"$PROJECT_DIR/.venv/bin/python"}
DEVICE=${DEVICE:-cuda:0}
OUT_DIR="$RUN_DIR/analysis"
mkdir -p "$OUT_DIR"

shopt -s nullglob
checkpoints=()
for candidate in "$RUN_DIR"/ckpt_*.pt; do
    name=$(basename -- "$candidate" .pt)
    if [[ ${name#ckpt_} =~ ^[0-9]+$ ]]; then
        checkpoints+=("$candidate")
    fi
done
if (( ${#checkpoints[@]} == 0 )); then
    echo "No numerically named ckpt_*.pt files found in $RUN_DIR" >&2
    exit 1
fi

for checkpoint in "${checkpoints[@]}"; do
    stem=$(basename -- "$checkpoint" .pt)
    result="$OUT_DIR/benchmark_$stem.json"
    logfile="$OUT_DIR/benchmark_$stem.log"
    if [[ ${SKIP_EXISTING:-0} == 1 && -f "$result" ]]; then
        echo "Using existing $result"
        continue
    fi
    echo "Evaluating $checkpoint"
    benchmark=(scripts/eval_benchmark.py)
    if [[ -n "$LEGACY_REVISION" ]]; then
        benchmark=(scripts/analysis/eval_legacy_khan_benchmark.py --revision "$LEGACY_REVISION")
    fi
    if ! env -u PYTHONPATH "$PYTHON_BIN" -u "${benchmark[@]}" \
        --ckpt "$checkpoint" --npz-dir "$DATA_DIR" \
        --num-envs "$NUM_ENVS" --device "$DEVICE" --out "$result" \
        > "$logfile" 2>&1; then
        tail -n 60 "$logfile" >&2
        exit 1
    fi
    rg 'clips \| failed|^saved' "$logfile"
done

env -u PYTHONPATH "$PYTHON_BIN" scripts/analysis/analyze_khan_training_logs.py \
    --run-dir "$RUN_DIR" --out-dir "$OUT_DIR"
