#!/usr/bin/env bash
# Run each stage in its own terminal: memorizer, executor, evaluate.
set -euo pipefail
cd "$(dirname "$0")/.."

MEMORIZER_PATH="${MEMORIZER_PATH:-"{memorizer_path}"}"
EXECUTOR_PATH="${EXECUTOR_PATH:-"{executor_path}"}"
RESULT_PATH="${RESULT_PATH:-"{result_path}"}"
require_path() {
  local name="$1" value="${!1}"
  if [[ -z "$value" || "$value" == *'{'* ]]; then
    echo "Set $name to a local path; see README.md." >&2
    exit 2
  fi
}
case "${1:-}" in
  memorizer)
    require_path MEMORIZER_PATH
    exec .venv-serving/bin/vllm serve "$MEMORIZER_PATH" \
      --served-model-name momentmem-memorizer --port 8000 \
      --limit-mm-per-prompt '{"image":64}' --max-model-len 16384
    ;;
  executor)
    require_path MEMORIZER_PATH
    require_path EXECUTOR_PATH
    exec executor/.venv/bin/python memory_agent/agent_server.py \
      --vlm-model momentmem-memorizer --vlm-base-url http://127.0.0.1:8000/v1 \
      --vlm-prompt-config "$MEMORIZER_PATH/keyframe_prompt_config.json" \
      --openpi-config pi05_memory --openpi-checkpoint "$EXECUTOR_PATH" \
      --threshold 0.5 --retrieval-stride 4 --port 8001
    ;;
  evaluate)
    require_path MEMORIZER_PATH
    require_path EXECUTOR_PATH
    require_path RESULT_PATH
    .venv-benchmark/bin/python maniskill_env/evaluate_closed_loop.py prepare \
      --output-root "$RESULT_PATH" --qwen-model "$MEMORIZER_PATH" \
      --vla-checkpoint "$EXECUTOR_PATH" --vla-config pi05_memory \
      --trials-per-task 100 --execution-horizon 10 --shader rt --render-backend cuda \
      --episode-seed-master 202608100001 --policy-seed-master 202608100002
    .venv-benchmark/bin/python maniskill_env/evaluate_closed_loop.py run \
      --output-root "$RESULT_PATH" --worker-index 0 --worker-count 1 \
      --port 8001 --shader rt --render-backend cuda --max-steps 2000
    .venv-benchmark/bin/python maniskill_env/evaluate_closed_loop.py summarize \
      --output-root "$RESULT_PATH"
    ;;
  *) echo "Usage: bash scripts/eval.sh {memorizer|executor|evaluate}" >&2; exit 2 ;;
esac
