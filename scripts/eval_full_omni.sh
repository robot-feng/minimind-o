#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHT="${EVAL_WEIGHT:-sft_omni}"
PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/envs/minimind/bin/python}"
CONDA_BIN_DIR="${CONDA_BIN_DIR:-/data/miniconda3/bin}"
RESULTS_DIR="${EVAL_RESULTS_DIR:-$ROOT/out/eval_intermediate}"
AUDIO_DIR="${EVAL_AUDIO_DIR:-$ROOT/out/eval_full_audio}"
LOG_FILE="${EVAL_LOG:-$ROOT/out/eval_full_omni.log}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    printf 'Evaluation Python is not executable: %s\n' "$PYTHON_BIN" >&2
    exit 127
fi

mkdir -p "$RESULTS_DIR" "$AUDIO_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

eval_python() {
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        printf '[dry-run]'
        printf ' %q' "$PYTHON_BIN" "$@"
        printf '\n'
        return
    fi
    PATH="$CONDA_BIN_DIR:$PATH" "$PYTHON_BIN" "$@"
}

eval_visual_metric() {
    local source="$1" output="$2"
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        eval_python "$ROOT/eval_visual_metrics.py" "$source"
        printf '[dry-run] metrics output: %s\n' "$output"
    else
        eval_python "$ROOT/eval_visual_metrics.py" "$source" | tee "$output"
    fi
}

echo "$(date '+evaluation_started_at=%F %T %Z')"
echo "weight=$WEIGHT"
echo "dry_run=${DRY_RUN:-0}"

echo "=== Text, audio-input, and video evaluation ==="
eval_python "$ROOT/eval_omni.py" \
    --weight "$WEIGHT" --mode 0,2,6 --prompt_lang 2 --max_new_tokens 128 \
    --seed 42 --output_dir "$AUDIO_DIR" \
    --audio_dir "$ROOT/dataset/eval_omni" \
    --video_dir "$ROOT/out/eval_video" --video_frames 4 \
    --results_jsonl "$RESULTS_DIR/${WEIGHT}_video.jsonl"

echo "=== English image-to-text-and-audio evaluation ==="
eval_python "$ROOT/eval_omni.py" \
    --weight "$WEIGHT" --mode 4 --prompt_lang 0 --max_new_tokens 80 \
    --temperature 0 --seed 42 \
    --image_dir "$ROOT/dataset/eval_omni" --output_dir "$AUDIO_DIR" \
    --results_jsonl "$RESULTS_DIR/${WEIGHT}_image_audio_en.jsonl"

echo "=== Chinese text-only image evaluation ==="
eval_python "$ROOT/eval_omni.py" \
    --weight "$WEIGHT" --mode 4 --text_only --prompt_lang 1 --max_new_tokens 80 \
    --temperature 0 --seed 42 --image_dir "$ROOT/dataset/eval_omni" \
    --results_jsonl "$RESULTS_DIR/${WEIGHT}_image_text_zh.jsonl"

echo "=== Visual concept metrics ==="
eval_visual_metric "$RESULTS_DIR/${WEIGHT}_image_audio_en.jsonl" \
    "$RESULTS_DIR/${WEIGHT}_image_audio_en_metrics.json"
eval_visual_metric "$RESULTS_DIR/${WEIGHT}_image_text_zh.jsonl" \
    "$RESULTS_DIR/${WEIGHT}_image_text_zh_metrics.json"

echo "evaluation_exit=0"
