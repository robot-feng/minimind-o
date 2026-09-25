#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

DATASET_DIR="${DATASET_DIR:-../dataset}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MASTER_PORT="${MASTER_PORT:-29560}"
VISION_DIR="${VISION_DIR:-google/tipsv2-b14}"
LOG_FILE="${LOG_FILE:-../out/sft_full_dense.log}"
CONDA_ENV="${CONDA_ENV:-minimind}"
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

# systemd services do not inherit the interactive shell's activated conda env.
if command -v torchrun >/dev/null 2>&1; then
    TORCHRUN=(torchrun)
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
fi

declare -A expected_sha256=(
    [sft_t2a.parquet]=2e5eb373e8853109f7d2d4e032a2a6e47088cd61873af0896e9fc32d1464597e
    [sft_a2a.parquet]=48cd60f4dbc987fee459968637758b27858daab52dfe6f21ac257cd4af182965
    [sft_i2t.parquet]=712f4026cd0e21b369feddca7334b1e465cb8182b5f298006f3f4f877f926643
)
for file in sft_t2a.parquet sft_a2a.parquet sft_i2t.parquet; do
    if [[ ! -s "$DATASET_DIR/$file" ]]; then
        printf 'Missing full training dataset: %s/%s\n' "$DATASET_DIR" "$file" >&2
        exit 1
    fi
done
if [[ "${DRY_RUN:-0}" != 1 ]]; then
    for file in sft_t2a.parquet sft_a2a.parquet sft_i2t.parquet; do
        actual_sha256="$(sha256sum "$DATASET_DIR/$file" | awk '{print $1}')"
        if [[ "$actual_sha256" != "${expected_sha256[$file]}" ]]; then
            printf 'Dataset checksum mismatch for %s: expected %s, got %s\n' \
                "$DATASET_DIR/$file" "${expected_sha256[$file]}" "$actual_sha256" >&2
            exit 1
        fi
    done
fi
if [[ "${DRY_RUN:-0}" != 1 ]]; then
    for file in ../out/llm_768.pth ../model/SenseVoiceSmall/model.pt; do
        if [[ ! -s "$file" ]]; then
            printf 'Missing training resource: %s\n' "$file" >&2
            exit 1
        fi
    done
fi

run_stage() {
    local save_weight="$1" from_weight="$2" data_file="$3"
    local epochs="$4" learning_rate="$5" mode="$6" use_compile="$7" max_seq_len="$8"
    local -a command=(
        "${TORCHRUN[@]}" --standalone --nproc_per_node "$NPROC_PER_NODE" --master_port "$MASTER_PORT"
        train_sft_omni.py
        --learning_rate "$learning_rate" --data_path "$DATASET_DIR/$data_file"
        --epochs "$epochs" --batch_size "$BATCH_SIZE" --use_compile "$use_compile"
        --from_weight "$from_weight" --save_weight "$save_weight"
        --max_seq_len "$max_seq_len" --mode "$mode" --use_moe 0 --from_resume 1
        --vision_dir "$VISION_DIR"
    )

    printf '\n=== %s: %s, %s epoch(s), %s ===\n' "$save_weight" "$data_file" "$epochs" "$mode"
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        printf '[dry-run]'
        printf ' %q' "${command[@]}"
        printf '\n'
    else
        "${command[@]}"
    fi
}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# Keep each stage's optimizer state separate so --from_resume can safely continue it.
run_stage sft_full_t2a llm sft_t2a.parquet 6 5e-4 all 1 512
run_stage sft_full_a2a_proj sft_full_t2a sft_a2a.parquet 1 5e-4 audio_proj 0 1024
run_stage sft_full_a2a sft_full_a2a_proj sft_a2a.parquet 3 5e-5 all 0 1024
run_stage sft_full_i2t_proj sft_full_a2a sft_i2t.parquet 1 5e-5 vision_proj 1 768
run_stage sft_full_i2t sft_full_i2t_proj sft_i2t.parquet 1 5e-6 all 1 768
run_stage sft_full_a2a_final sft_full_i2t sft_a2a.parquet 1 5e-6 all 0 1024
run_stage sft_omni sft_full_a2a_final sft_i2t.parquet 1 5e-6 vision_proj 1 768

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '\nDry run complete; no training was started.\n'
else
    printf '\nFull Dense training complete: ../out/sft_omni_768.pth\n'
fi
