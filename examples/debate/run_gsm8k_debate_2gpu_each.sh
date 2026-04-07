#!/usr/bin/env bash
# Real multi-GPU smoke run: 4 GPUs total (2 for model A, 2 for model B), TP=2 each.
#
# GPU layout (default): physical GPUs 0,1 for party A and 6,7 for party B.
#   We set CUDA_VISIBLE_DEVICES=0,1,6,7 so Ray remaps them to logical cuda:0..3;
#   global_pool (first) takes the first two visible devices, debate_pool_b the last two.
#
# To use four consecutive GPUs instead:  CUDA_VISIBLE_DEVICES=0,1,2,3
# To customize order:                      CUDA_VISIBLE_DEVICES=... (party A first pair, party B second pair)
#
# Prerequisites:
#   - GSM8K parquet with data_source=openai/gsm8k (standard verl preprocessing).
#   - Full verl install (Ray, vLLM, FSDP engine, etc.) matching your cluster.
#
# Usage:
#   cd /path/to/verl_repo
#   export MODEL_A=Qwen/Qwen2.5-0.5B-Instruct
#   export MODEL_B=Qwen/Qwen2.5-1.5B-Instruct   # optional; defaults to MODEL_A
#   bash examples/debate/run_gsm8k_debate_2gpu_each.sh
#
# JSONL + Weights & Biases (see examples/debate/config/ppo_debate_gsm8k_2gpu_each.yaml):
#   - Trajectories append to trainer.debate_trajectory_jsonl (under repo cwd).
#   - Default: W&B metrics only (party_a/*, party_b/*, …); debate_wandb_log_trajectories=false.
#   - To also log debate/trajectories table in W&B: trainer.debate_wandb_log_trajectories=true
#   - To disable W&B entirely: trainer.logger=[console]
# Data defaults to ${ROOT}/data/gsm8k/{train,test}.parquet unless you set GSM8K_TRAIN / GSM8K_TEST.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
# Prepend so this repo wins over any pip-installed verl; Ray workers get the same path via runtime_env.
export PYTHONPATH="${ROOT}${PYTHONPATH:+:}${PYTHONPATH:-}"

export GSM8K_TRAIN="${GSM8K_TRAIN:-${ROOT}/data/gsm8k/train.parquet}"
export GSM8K_TEST="${GSM8K_TEST:-${ROOT}/data/gsm8k/test.parquet}"

for _f in "${GSM8K_TRAIN}" "${GSM8K_TEST}"; do
  if [[ ! -f "${_f}" ]]; then
    echo "Missing parquet: ${_f}" >&2
    echo "From repo root create GSM8K data, for example:" >&2
    echo "  python examples/data_preprocess/gsm8k.py --local_save_dir ${ROOT}/data/gsm8k" >&2
    echo "Or set GSM8K_TRAIN and GSM8K_TEST to existing train.parquet / test.parquet paths." >&2
    exit 1
  fi
done

# Default: party A on physical 0,1; party B on physical 6,7 (see header comment).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,6,7}"

CFG_DIR="${ROOT}/examples/debate/config"
CFG_NAME="ppo_debate_gsm8k_2gpu_each"

echo "Using config: ${CFG_DIR}/${CFG_NAME}.yaml"
echo "GSM8K_TRAIN=${GSM8K_TRAIN}"
echo "GSM8K_TEST=${GSM8K_TEST}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "MODEL_A=${MODEL_A:-Qwen/Qwen2.5-0.5B-Instruct} MODEL_B=${MODEL_B:-same as MODEL_A unless set}"

exec python3 -m verl.trainer.main_ppo_debate \
    --config-path="${CFG_DIR}" \
    --config-name="${CFG_NAME}" \
    "$@"
