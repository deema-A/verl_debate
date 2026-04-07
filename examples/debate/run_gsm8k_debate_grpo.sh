#!/usr/bin/env bash
# GSM8K two-model debate (main_ppo_debate.py):
#   - Model A: dataset prompt + party_a_suffix (CoT, #### line).
#   - Model B: sees A's full reply, answers with #### using only that reasoning.
#   - Reward: GSM8K score on B's response vs ground_truth, copied to BOTH parties (shared team reward).
#
# Prereqs: Parquet rows with data_source=openai/gsm8k and reward_model.ground_truth.
# GPU layout: global_pool (model A) + debate_pool_b (model B); sizes must match your parallel plan.
#
# Define ``debate.actor_rollout_ref_b`` in YAML (duplicate ``actor_rollout_ref`` subtree, change ``model.path``),
# or pass Hydra overrides on the command line.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:}${PYTHONPATH:-}"

exec python3 -m verl.trainer.main_ppo_debate \
    data.train_files="${GSM8K_TRAIN:-${ROOT}/data/gsm8k/train.parquet}" \
    data.val_files="${GSM8K_TEST:-${ROOT}/data/gsm8k/test.parquet}" \
    reward_model.enable=false \
    critic.enable=false \
    algorithm.adv_estimator=grpo \
    debate.gsm8k.enable=true \
    debate.pool_b.n_gpus_per_node="${DEBATE_B_GPUS_PER_NODE:-4}" \
    debate.pool_b.nnodes="${DEBATE_B_NNODES:-1}" \
    "$@"
