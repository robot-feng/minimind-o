#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHT="${EVAL_WEIGHT:-sft_omni}"
BASELINE_WEIGHT="${EVAL_BASELINE_WEIGHT:-sft_i2t_mini}"
REFERENCE_MODEL="${EVAL_REFERENCE_MODEL:-jingyaogong/minimind-3o}"
REFERENCE_VISION="${EVAL_REFERENCE_VISION:-jingyaogong/siglip2-base-p32-256-ve}"
REFERENCE_LABEL="${EVAL_REFERENCE_LABEL:-minimind-3o-release-siglip2}"
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

eval_visual_comparison() {
    local before="$1" after="$2" output="$3"
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        eval_python "$ROOT/eval_visual_metrics.py" "$before" --compare "$after"
        printf '[dry-run] comparison output: %s\n' "$output"
    else
        eval_python "$ROOT/eval_visual_metrics.py" "$before" --compare "$after" | tee "$output"
    fi
}

echo "$(date '+evaluation_started_at=%F %T %Z')"
echo "weight=$WEIGHT"
echo "baseline_weight=$BASELINE_WEIGHT"
echo "reference_model=$REFERENCE_MODEL"
echo "reference_vision=$REFERENCE_VISION"
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

echo "=== English text-only baseline and final image evaluation ==="
eval_python "$ROOT/eval_omni.py" \
    --weight "$BASELINE_WEIGHT" --mode 4 --text_only --prompt_lang 0 \
    --max_new_tokens 80 --temperature 0 --seed 42 \
    --image_dir "$ROOT/dataset/eval_omni" \
    --results_jsonl "$RESULTS_DIR/${BASELINE_WEIGHT}_image_text_en.jsonl"
eval_python "$ROOT/eval_omni.py" \
    --weight "$WEIGHT" --mode 4 --text_only --prompt_lang 0 --max_new_tokens 80 \
    --temperature 0 --seed 42 --image_dir "$ROOT/dataset/eval_omni" \
    --results_jsonl "$RESULTS_DIR/${WEIGHT}_image_text_en.jsonl"
echo "=== Upstream released model reference (SigLIP2, one image frame) ==="
eval_python "$ROOT/eval_omni.py" \
    --load_from "$REFERENCE_MODEL" --vision_dir "$REFERENCE_VISION" --video_frames 1 \
    --mode 4 --text_only --prompt_lang 0 --max_new_tokens 80 \
    --temperature 0 --seed 42 --image_dir "$ROOT/dataset/eval_omni" \
    --results_jsonl "$RESULTS_DIR/${REFERENCE_LABEL}_image_text_en.jsonl"

echo "=== Chinese text-only image evaluation ==="
eval_python "$ROOT/eval_omni.py" \
    --weight "$WEIGHT" --mode 4 --text_only --prompt_lang 1 --max_new_tokens 80 \
    --temperature 0 --seed 42 --image_dir "$ROOT/dataset/eval_omni" \
    --results_jsonl "$RESULTS_DIR/${WEIGHT}_image_text_zh.jsonl"

echo "=== Visual concept metrics ==="
eval_visual_metric "$RESULTS_DIR/${WEIGHT}_image_audio_en.jsonl" \
    "$RESULTS_DIR/${WEIGHT}_image_audio_en_metrics.json"
eval_visual_metric "$RESULTS_DIR/${WEIGHT}_image_text_en.jsonl" \
    "$RESULTS_DIR/${WEIGHT}_image_text_en_metrics.json"
eval_visual_metric "$RESULTS_DIR/${WEIGHT}_image_text_zh.jsonl" \
    "$RESULTS_DIR/${WEIGHT}_image_text_zh_metrics.json"
eval_visual_metric "$RESULTS_DIR/${REFERENCE_LABEL}_image_text_en.jsonl" \
    "$RESULTS_DIR/${REFERENCE_LABEL}_image_text_en_metrics.json"
eval_visual_comparison "$RESULTS_DIR/${BASELINE_WEIGHT}_image_text_en.jsonl" \
    "$RESULTS_DIR/${WEIGHT}_image_text_en.jsonl" \
    "$RESULTS_DIR/${WEIGHT}_vs_${BASELINE_WEIGHT}_image_text_en.json"
eval_visual_comparison "$RESULTS_DIR/${REFERENCE_LABEL}_image_text_en.jsonl" \
    "$RESULTS_DIR/${WEIGHT}_image_text_en.jsonl" \
    "$RESULTS_DIR/${WEIGHT}_vs_${REFERENCE_LABEL}_image_text_en.json"
eval_python "$ROOT/scripts/plot_visual_comparison.py" \
    "$RESULTS_DIR/${WEIGHT}_vs_${BASELINE_WEIGHT}_image_text_en.json" \
    --output "$RESULTS_DIR/${WEIGHT}_visual_comparison.png" \
    --markdown-output "$RESULTS_DIR/${WEIGHT}_visual_comparison.md" \
    --before-label "$BASELINE_WEIGHT" --after-label "$WEIGHT"
eval_python "$ROOT/scripts/plot_visual_comparison.py" \
    "$RESULTS_DIR/${WEIGHT}_vs_${REFERENCE_LABEL}_image_text_en.json" \
    --output "$RESULTS_DIR/${WEIGHT}_vs_${REFERENCE_LABEL}_visual_comparison.png" \
    --markdown-output "$RESULTS_DIR/${WEIGHT}_vs_${REFERENCE_LABEL}_visual_comparison.md" \
    --before-label "$REFERENCE_LABEL" --after-label "$WEIGHT" \
    --title "MiniMind-O visual evaluation: release reference vs TIPSv2 reproduction"

echo "evaluation_exit=0"
