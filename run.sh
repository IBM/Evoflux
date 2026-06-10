#!/bin/bash
# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_LOCAL_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" #"meta-llama/Llama-3.2-3B-Instruct" ##"nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16" ##"Qwen/Qwen3.5-4B" ##"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" ##"HuggingFaceTB/SmolLM3-3B" ##"ibm-granite/granite-4.0-h-micro" #"google/gemma-4-E2B-it" ##"Qwen/Qwen3.5-4B" #"meta-llama/Llama-3.2-3B-Instruct"   # planner  (port 8000)
DEFAULT_JUDGE_MODEL="openai/gpt-oss-20b"            # judge    (port 8001)
DEFAULT_LOCAL_PORT=8000
DEFAULT_JUDGE_PORT=8001

PROVIDER=""
MODEL=""
LOCAL_MODEL=""
LOCAL_BASE_URL=""
LOCAL_JUDGE_MODEL=""
LOCAL_JUDGE_BASE_URL=""
JUDGE_PROVIDER=""
JUDGE_MODEL=""

if [ -z "$1" ]; then
  echo "Usage: ./run.sh [local | openrouter | bedrock/aws]"
  echo "  local  [planner-model] [judge-model] [react-port]  — two vLLM servers (ports 8000 + 8001; react-port overrides planner port when STAGES=react)"
  echo "  openrouter                                          — OpenRouter for planner + judge"
  echo "  bedrock / aws                                       — AWS Bedrock for planner + judge"
  exit 1

elif [ "$1" = "local" ]; then
  LOCAL_MODEL="${2:-$DEFAULT_LOCAL_MODEL}"
  LOCAL_JUDGE_MODEL="${3:-$DEFAULT_JUDGE_MODEL}"
  if [ "${STAGES}" = "react" ] && [ -n "$4" ]; then
    DEFAULT_LOCAL_PORT="$4"
  fi
  LOCAL_BASE_URL="http://localhost:${DEFAULT_LOCAL_PORT}/v1"
  LOCAL_JUDGE_BASE_URL="http://localhost:${DEFAULT_JUDGE_PORT}/v1"

  mkdir -p ./log

  # ── Start planner vLLM server (port 8000) if not already running ─────────
  if ! curl -s "http://localhost:${DEFAULT_LOCAL_PORT}/health" > /dev/null 2>&1; then
    echo "Starting planner vLLM server (port ${DEFAULT_LOCAL_PORT}): $LOCAL_MODEL"
    python -m vllm.entrypoints.openai.api_server \
      --model "$LOCAL_MODEL" \
      --port  "$DEFAULT_LOCAL_PORT" \
      --api-key "EMPTY" \
      > ./log/vllm_planner.log 2>&1 &
    echo "  Planner PID: $!"
  else
    echo "Planner vLLM server already running on port ${DEFAULT_LOCAL_PORT}."
  fi

  # ── Start judge vLLM server (port 8001) if not already running ───────────
  if ! curl -s "http://localhost:${DEFAULT_JUDGE_PORT}/health" > /dev/null 2>&1; then
    echo "Starting judge vLLM server (port ${DEFAULT_JUDGE_PORT}): $LOCAL_JUDGE_MODEL"
    python -m vllm.entrypoints.openai.api_server \
      --model "$LOCAL_JUDGE_MODEL" \
      --port  "$DEFAULT_JUDGE_PORT" \
      --api-key "EMPTY" \
      > ./log/vllm_judge.log 2>&1 &
    echo "  Judge PID: $!"
  else
    echo "Judge vLLM server already running on port ${DEFAULT_JUDGE_PORT}."
  fi

  # ── Wait for both servers to be ready ────────────────────────────────────
  echo "Waiting for vLLM servers to be ready..."
  for PORT in "$DEFAULT_LOCAL_PORT" "$DEFAULT_JUDGE_PORT"; do
    READY=0
    for i in $(seq 1 60); do
      if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "  Port ${PORT} ready."
        READY=1
        break
      fi
      sleep 2
    done
    if [ "$READY" = "0" ]; then
      echo "ERROR: vLLM server on port ${PORT} did not start in time. Check ./log/vllm_*.log"
      exit 1
    fi
  done

elif [ "$1" = "openrouter" ]; then
  PROVIDER="openrouter"
  MODEL="qwen/qwen-2.5-72b-instruct"
  JUDGE_PROVIDER="openrouter"
  JUDGE_MODEL="o4-mini"

elif [ "$1" = "aws" ] || [ "$1" = "bedrock" ]; then
  PROVIDER="bedrock" ##""
  MODEL="us.anthropic.claude-haiku-4-5-20251001-v1:0" #"us.meta.llama3-3-70b-instruct-v1:0" #"us.anthropic.claude-sonnet-4-5-20250929-v1:0" #"us.meta.llama3-3-70b-instruct-v1:0"  #"us.meta.llama4-scout-17b-instruct-v1:0" #"us.meta.llama4-maverick-17b-instruct-v1:0" ##"us.anthropic.claude-sonnet-4-5-20250929-v1:0" #"us.meta.llama4-maverick-17b-instruct-v1:0" # #"us.anthropic.claude-haiku-4-5-20251001-v1:0" #"us.anthropic.claude-sonnet-4-20250514-v1:0" # haiku 3.5
  JUDGE_PROVIDER="bedrock"
  JUDGE_MODEL="us.anthropic.claude-sonnet-4-20250514-v1:0"

else
  echo "Error. Argument should be [local | openrouter | bedrock/aws]"
  exit 1
fi

mkdir -p ./log

MODEL_VERSION="${MODEL_VERSION:-original}"
if [[ "$MODEL_VERSION" != "original" && "$MODEL_VERSION" != "sft" && "$MODEL_VERSION" != "sft+dpo" ]]; then
  echo "Error: MODEL_VERSION must be one of: original, sft, sft+dpo (got '$MODEL_VERSION')"
  exit 1
fi
STAGES="${STAGES:-validate}"
EVOLVE_ON_VAL_FLAG=""
if [[ "$MODEL_VERSION" == "original" ]]; then
  EVOLVE_ON_VAL_FLAG="--evolve-on-val"
fi
OUTPUT_DIR="results/${MODEL_VERSION}"
MODEL_NAME="${LOCAL_MODEL##*/}"
if [ "$1" = "local" ]; then
  echo "Running pipeline: planner=$LOCAL_MODEL  judge=$LOCAL_JUDGE_MODEL  stages=$STAGES"
  PYTHONPATH=. python3 -m scripts.run_pipeline --benchmark mcpbench \
    --stages               "$STAGES" \
    --local-model          "$LOCAL_MODEL" \
    --local-backend        "vllm" \
    --local-base-url       "$LOCAL_BASE_URL" \
    --local-judge-model    "$LOCAL_JUDGE_MODEL" \
    --local-judge-base-url "$LOCAL_JUDGE_BASE_URL" \
    --output-dir           "$OUTPUT_DIR" \
    $EVOLVE_ON_VAL_FLAG \
    > "./log/$STAGES/${STAGES}_${MODEL_NAME}_test.txt" ##2>&1
else
  echo "Running pipeline: provider=$PROVIDER  model=$MODEL  judge=$JUDGE_MODEL  stages=$STAGES"
  PYTHONPATH=. python3 -m scripts.run_pipeline --benchmark mcpbench \
    --stages         "$STAGES" \
    --provider       "$PROVIDER" \
    --model          "$MODEL" \
    --judge-provider "$JUDGE_PROVIDER" \
    --judge-model    "$JUDGE_MODEL" \
    --output-dir     "$OUTPUT_DIR" \
    $EVOLVE_ON_VAL_FLAG \
    > "./log/$STAGES/${STAGES}_${MODEL_NAME}_test.txt" ##2>&1
fi
