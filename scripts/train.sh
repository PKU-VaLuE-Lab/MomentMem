#!/usr/bin/env bash
# VMBench example: train memorizer, predict memories, then train executor.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_PATH="${DATA_PATH:-"{data_path}"}"
OUTPUT_PATH="${OUTPUT_PATH:-"{output_path}"}"
MEMORIZER_BASE_PATH="${MEMORIZER_BASE_PATH:-"{memorizer_base_path}"}"
EXECUTOR_BASE_PATH="${EXECUTOR_BASE_PATH:-"{executor_base_path}"}"
for name in DATA_PATH OUTPUT_PATH MEMORIZER_BASE_PATH EXECUTOR_BASE_PATH; do
  value="${!name}"
  if [[ -z "$value" || "$value" == *'{'* ]]; then
    echo "Set $name to a local path; see README.md." >&2
    exit 2
  fi
done
# Resolve paths before OpenPI changes the working directory.
DATA_PATH="$(.venv-memorizer/bin/python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$DATA_PATH")"
OUTPUT_PATH="$(.venv-memorizer/bin/python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$OUTPUT_PATH")"
MEMORIZER_BASE_PATH="$(.venv-memorizer/bin/python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$MEMORIZER_BASE_PATH")"
EXECUTOR_BASE_PATH="$(.venv-memorizer/bin/python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$EXECUTOR_BASE_PATH")"
mkdir -p "$OUTPUT_PATH"

.venv-memorizer/bin/torchrun --nproc_per_node=8 \
  memorizer/sft/train_qwen3_vl_sft.py --config configs/memorizer.json \
  --model-name-or-path "$MEMORIZER_BASE_PATH" --dataset-root "$DATA_PATH" \
  --output-dir "$OUTPUT_PATH/memorizer"

.venv-memorizer/bin/python memorizer/sft/generate_keyframe_labels_ar.py \
  --model-name-or-path "$OUTPUT_PATH/memorizer" --dataset-root "$DATA_PATH" \
  --threshold 0.5 --output "$OUTPUT_PATH/keyframes.json"
.venv-memorizer/bin/python memorizer/sft/tools/export_vlm_keyframes_to_lerobot.py \
  --dataset-root "$DATA_PATH" --raw-predictions "$OUTPUT_PATH/keyframes.json" \
  --threshold 0.5 --output-root "$OUTPUT_PATH/predicted_memory"

export HF_LEROBOT_HOME="$OUTPUT_PATH/predicted_memory/lerobot_format"
export MOMENTMEM_BASE_PARAMS="$EXECUTOR_BASE_PATH"
export MOMENTMEM_ASSETS_PATH="$OUTPUT_PATH/policy_assets"
export MOMENTMEM_CHECKPOINT_PATH="$OUTPUT_PATH/executor"
cd executor
uv run scripts/compute_norm_stats.py --config-name pi05_memory
uv run scripts/train.py pi05_memory --exp-name vmbench --fsdp-devices 8

# Export only inference files; retain raw checkpoints locally for training resume.
cd ..
.venv-memorizer/bin/python scripts/export_checkpoint.py vlm \
  --source "$OUTPUT_PATH/memorizer" --output "$OUTPUT_PATH/release/memorizer"
.venv-memorizer/bin/python scripts/export_checkpoint.py vla \
  --source "$OUTPUT_PATH/executor/pi05_memory/vmbench/60000" \
  --output "$OUTPUT_PATH/release/executor"
