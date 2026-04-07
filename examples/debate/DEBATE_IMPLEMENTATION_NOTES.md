# Debate training — implementation notes (what was fixed / how it works)

This document summarizes the **engineering steps** that made two-model GSM8K debate + async vLLM training reliable in this fork. It is meant for maintainers; user-facing run instructions are in [README.md](./README.md).

---

## 1. End-to-end flow (mental model)

1. Dataloader yields a batch; `uid` is attached per row (`len(batch)` so it works when tensor batch is only a dummy / agent-loop layout).
2. `gen_batch = _get_gen_batch(batch)` — tensor prompts stay on **`batch`**; `gen_batch` carries **non-tensor** fields (e.g. `raw_prompt`) for the agent loop. So `gen_batch.batch` is often **`None`**.
3. `gen_batch.repeat(rollout.n)` duplicates rows for multiple samples per prompt.
4. **GSM8K:** `append_party_a_suffix_to_gen_batch` edits **`non_tensor_batch["raw_prompt"]`** (chat messages), not `input_ids`, because async rollout never put token tensors on `gen_batch`.
5. Party A `generate_sequences` → tensors (`prompts`, `responses`, …) on output.
6. `build_gen_batch_for_party_b` decodes with tokenizer A, builds text for B, encodes with tokenizer B → new `input_ids` for B’s `generate_sequences`, and stores **`debate_precomputed_prompt_ids`** in `non_tensor_batch`. **Why:** async `single_turn_agent` ignores batch `input_ids` and retokenizes **`raw_prompt` only**; without this key, party B would not see A’s reply (JSONL would show identical A/B “prompts”). `SingleTurnAgentLoop` uses `debate_precomputed_prompt_ids` when present.
7. Shared GSM8K score from B’s text → `inject_shared_rm_scores` on both aligned batches (writes `rm_scores`, so party PPO skips `_compute_reward_colocate` — same as any batch that already has `rm_scores`).
8. Each party: old log-prob, advantage, update, checkpoint engine pushes weights to that party’s vLLM stack.

---

## 2. Bug: `TypeError: 'NoneType' object is not subscriptable` in `append_party_a_suffix_to_gen_batch`

**Symptom:** Crash at `out.batch["input_ids"]` when `debate.gsm8k.enable=true`.

**Cause:** `_get_gen_batch` uses `DataProto.pop` with **empty** `batch_keys`, so the returned `gen_batch` has **`batch=None`**. The agent loop consumes `raw_prompt` from `non_tensor_batch`, not pre-tokenized `input_ids` on `gen_batch`.

**Fix:** Implement party-A suffix by appending to the **last user message** in `raw_prompt` (string or multimodal `content` list), with a small helper `_append_suffix_to_chat_messages`. Removed the unused token round-trip branch later, since debate + agent loop never populated `input_ids` on `gen_batch`.

**Related:** `batch.non_tensor_batch["uid"]` was updated to use **`len(batch)`** instead of **`len(batch.batch)`** so batch size is correct when `batch` is indexed via non-tensor fields.

---

## 3. Config / worker plumbing

- **`debate.actor_rollout_ref_b`** must merge into the **full** `actor_rollout_ref` tree (not only `model.path`), so Hydra `_target_` and nested keys (e.g. `hf_model`) stay valid for party B. Implemented via `_merged_actor_rollout_ref_b()` (OmegaConf merge A then B).
- **`Role.ActorRolloutB`** — Python 3.12 disallows extending `enum.Enum` with new members; extra role value lives on the same `Role` enum in `utils.py`, with `DebateRole = Role` in `utils_debate.py` for readable imports.
- **Worker asserts** — `actor_rollout_b` allowed where worker groups validate role names (`engine_workers`, `fsdp_workers`, `megatron_workers` as applicable).
- **Duplicate vLLM Ray actor names** — Second HTTP server stack needs a unique `http_ray_server_actor_name_prefix` (`vllm_b_`, `sglang_b_`, `vllm_omni_b_`) set in merged B rollout config.

---

## 4. YAML / `validate_config`

Example `ppo_debate_gsm8k_2gpu_each.yaml` sets batch sizes that satisfy `validate_config` when dynamic batching is off, e.g. `ppo_mini_batch_size`, `ppo_micro_batch_size_per_gpu`, `log_prob_micro_batch_size_per_gpu`.

---

## 5. Ray / `PYTHONPATH`

- `constants_ppo.py` runtime env propagates **`PYTHONPATH`** so Ray workers import the **same** repo as the driver.
- Debate shell scripts **prepend** the repo root to `PYTHONPATH` before launching `python -m verl.trainer.main_ppo_debate`.

---

## 6. Hybrid engine: CUDA cache RPC and vLLM `gpu_memory_utilization`

**Colocated `WorkerDict` (non-fused):** Remote methods are exposed as `{role_key}_{method}` on the Ray actor (e.g. `actor_rollout_ref_empty_training_cuda_cache`). The `RayWorkerGroup` rebinds them to short names like `empty_training_cuda_cache()`. Calling `execute_all_sync("empty_training_cuda_cache")` goes through `getattr(ActorHandle, "empty_training_cuda_cache")`, which does **not** exist on the actor — use **`worker_group.empty_training_cuda_cache()`** (same pattern as `init_model()`).

**vLLM v1 KV vs preflight:** `gpu_memory_utilization` is applied **per GPU** in the party’s pool (not as a fraction of total cluster VRAM). Extremely low values (e.g. **0.015**) can pass “free memory ≥ utilization × total” checks after FSDP offload but leave **too little budget for KV cache blocks**. Raise into the **~0.35–0.55** range when `param_offload` + cache flush leave most of each card free; reduce if the **preflight** path reports insufficient free memory before KV init.

---

## 7. Environment gotchas (outside verl code)

- **NumPy vs Numba** — Some stacks pin NumPy (e.g. 2.2.x) so vLLM/numba imports do not fail; fix is environment-level.
- **Party B checkpoint** — If B’s checkpoint path is missing at step 0, training can start B from init weights (warning in logs).

---

## 8. Shutdown / UX (optional hardening)

- **`_shutdown_dataloaders()`** on `RayPPOTrainer` — Before exit on the last step, explicitly shut down `StatefulDataLoader` worker processes to reduce `Exception ignored in __del__` / worker **Killed** noise when Ray and loaders tear down together.
- **Final validation message** — If `trainer.test_freq <= 0`, print that validation was skipped instead of `Final validation metrics: None`.

These do not change training math; they only clarify logs and teardown.

---

## 9. Code churn summary (files touched in this effort)

| Area | Files (representative) |
|------|-------------------------|
| Trainer loop | `verl/trainer/ppo/ray_trainer_debate.py`, `verl/trainer/main_ppo_debate.py` |
| GSM8K debate helpers | `verl/trainer/ppo/debate_gsm8k.py` |
| B prompt construction | `verl/trainer/ppo/debate_batch_utils.py` |
| Base trainer teardown / logging | `verl/trainer/ppo/ray_trainer.py` |
| Roles / mapping | `verl/trainer/ppo/utils.py`, `utils_debate.py` |
| Workers / rollout naming | `verl/workers/engine_workers.py`, `fsdp_workers.py`, `megatron_workers.py`, rollout config usage |
| Examples | `examples/debate/*.sh`, `examples/debate/config/*.yaml` |
| Ray env | `verl/trainer/constants_ppo.py` |

---

## 10. Joint trajectory JSONL dump

`trainer.debate_trajectory_jsonl` (path) makes `RayPPODebateTrainer` append one JSON object per row after each step’s PPO updates, with decoded A/B prompts and responses and scalar `reward_party_a` / `reward_party_b` (sum of `token_level_scores`). File is **append** mode; delete the path between runs if you want a clean file, or use `jq -s '.'` to build one JSON array.

Optional W&B: `debate_wandb_log_trajectories: true` with `wandb` or `vemlp_wandb` in `trainer.logger` logs `debate/trajectories` as a `wandb.Table` (row/cell caps configurable).

---

## 11. How to verify quickly

- Short run: `total_training_steps: 2`, small `train_max_samples`, `bash examples/debate/run_gsm8k_debate_2gpu_each.sh`.
- Expect metrics for **both** parties under **`party_a/`** and **`party_b/`** (e.g. `party_a/critic/score/mean`, `party_b/actor/pg_loss`), plus `debate/gsm8k/*` when enabled.
- Exit-time vLLM / `ResourceTracker` messages can still appear; distinguish them from **exceptions before** 100% progress.

---

*This notes file reflects the implementation state in this repository; upstream verl may differ.*
