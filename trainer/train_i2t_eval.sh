#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_ENV="${CONDA_ENV:-minimind}"
DATA_PATH="${DATA_PATH:-../dataset/sft_i2t_mini.parquet}"
SAVE_DIR="${SAVE_DIR:-../out}"
BASE_WEIGHT="${BASE_WEIGHT:-sft_full_a2a}"
PROJ_WEIGHT="${PROJ_WEIGHT:-sft_i2t_mini_proj}"
FINAL_WEIGHT="${FINAL_WEIGHT:-sft_i2t_mini}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29561}"
VISION_DIR="${VISION_DIR:-google/tipsv2-b14}"
PROJ_BATCH_SIZE="${PROJ_BATCH_SIZE:-2}"
PROJ_ACCUMULATION_STEPS="${PROJ_ACCUMULATION_STEPS:-2}"
ALL_BATCH_SIZE="${ALL_BATCH_SIZE:-2}"
ALL_ACCUMULATION_STEPS="${ALL_ACCUMULATION_STEPS:-4}"
DRY_RUN="${DRY_RUN:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

if ! [[ "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
    echo "NPROC_PER_NODE must be a positive integer" >&2
    exit 2
fi

if [[ "$NPROC_PER_NODE" == 1 ]]; then
    TORCHRUN=()
    PYTHON_RUNNER=("$PYTHON_BIN")
elif command -v torchrun >/dev/null 2>&1; then
    TORCHRUN=(torchrun)
    PYTHON_RUNNER=("$PYTHON_BIN")
elif [[ "$DRY_RUN" == 1 ]]; then
    TORCHRUN=("$PYTHON_BIN" -m torch.distributed.run)
    PYTHON_RUNNER=("$PYTHON_BIN")
else
    CONDA_EXE="${CONDA_EXE:-$(command -v conda || true)}"
    if [[ -z "$CONDA_EXE" && -x /data/miniconda3/bin/conda ]]; then
        CONDA_EXE=/data/miniconda3/bin/conda
    fi
    if [[ -z "$CONDA_EXE" || ! -x "$CONDA_EXE" ]]; then
        echo "torchrun not found and conda executable unavailable" >&2
        exit 127
    fi
    TORCHRUN=("$CONDA_EXE" run --no-capture-output -n "$CONDA_ENV" torchrun)
    PYTHON_RUNNER=("$CONDA_EXE" run --no-capture-output -n "$CONDA_ENV" "$PYTHON_BIN")
fi

if [[ "$DRY_RUN" != 1 ]]; then
    [[ -s "$DATA_PATH" ]] || { echo "Missing I2T dataset: $DATA_PATH" >&2; exit 1; }
    [[ -s "$SAVE_DIR/${BASE_WEIGHT}_768.pth" ]] || {
        echo "Missing base checkpoint: $SAVE_DIR/${BASE_WEIGHT}_768.pth" >&2
        exit 1
    }
fi

run() {
    if [[ "$DRY_RUN" == 1 ]]; then
        printf '[dry-run]'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

train_stage() {
    local mode="$1" from_weight="$2" save_weight="$3"
    local batch_size="$4" accumulation_steps="$5" learning_rate="$6"
    local -a command=()
    if [[ "$NPROC_PER_NODE" == 1 ]]; then
        command=("${PYTHON_RUNNER[@]}" train_sft_omni.py)
    else
        command=("${TORCHRUN[@]}" --standalone --nproc_per_node "$NPROC_PER_NODE"
                 --master_port "$MASTER_PORT" train_sft_omni.py)
    fi
    command+=(--vision_only --mode "$mode" --data_path "$DATA_PATH"
              --from_weight "$from_weight" --save_weight "$save_weight"
              --save_dir "$SAVE_DIR" --vision_dir "$VISION_DIR"
              --epochs 1 --batch_size "$batch_size"
              --accumulation_steps "$accumulation_steps"
              --learning_rate "$learning_rate" --max_seq_len 768 --use_compile 0)
    run "${command[@]}"
}

eval_checkpoint() {
    local weight="$1" results="$2"
    local -a command=("${PYTHON_RUNNER[@]}" ../eval_omni.py --save_dir "$SAVE_DIR"
                      --weight "$weight" --text_only --mode 4 --prompt_lang 1
                      --image_dir ../dataset/eval_omni --max_new_tokens 80
                      --temperature 0 --seed 42 --results_jsonl "$results")
    run "${command[@]}"
}

train_stage vision_proj "$BASE_WEIGHT" "$PROJ_WEIGHT" \
    "$PROJ_BATCH_SIZE" "$PROJ_ACCUMULATION_STEPS" 5e-5
train_stage all "$PROJ_WEIGHT" "$FINAL_WEIGHT" \
    "$ALL_BATCH_SIZE" "$ALL_ACCUMULATION_STEPS" 5e-6

if [[ "$DRY_RUN" != 1 ]]; then
    mkdir -p ../out/eval_intermediate
fi
eval_checkpoint "$BASE_WEIGHT" ../out/eval_intermediate/sft_full_a2a.jsonl
eval_checkpoint "$FINAL_WEIGHT" ../out/eval_intermediate/sft_i2t_mini.jsonl
run "${PYTHON_RUNNER[@]}" ../eval_visual_metrics.py \
    ../out/eval_intermediate/sft_full_a2a.jsonl --compare \
    ../out/eval_intermediate/sft_i2t_mini.jsonl
