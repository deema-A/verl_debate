# Two-model debate training (PPO)

This example is for when you want **two different policies** to learn together in a single training loop—like a debate where model A speaks first and model B responds to what A actually said.

Party A rolls out first. Party B’s prompt is built from A’s **decoded** text (a deliberate “sync point”), so you can use **different tokenizers or checkpoints** for A and B without forcing a shared vocabulary. Each side then gets a normal PPO-style update on **its own** GPU pool and rollout server.

**How to launch:** `python -m verl.trainer.main_ppo_debate` with Hydra configs (see below).

## What one training step looks like

1. **Party A generates** — The usual async agent loop plus vLLM or SGLang on the `global_pool` GPUs. If you turn on GSM8K debate mode, extra instructions are stitched into the chat `raw_prompt` before sampling.

2. **Party B’s prompt** — We decode A’s prompt and reply with tokenizer A, wrap the opponent text with `debate.opponent_header` and `debate.second_turn_suffix`, then re-encode with tokenizer B. We set **`debate_precomputed_prompt_ids`** so the async single-turn agent sends that exact prefix to vLLM (otherwise it would only retokenize `raw_prompt` and you’d lose the careful handoff).

3. **Party B generates** — Second agent loop on `debate_pool_b`.

4. **Reward (optional GSM8K)** — With `debate.gsm8k.enable=true`, a simple rule-based GSM8K score compares **B’s** formatted answer to `reward_model.ground_truth`. The **same** scalar goes into both parties’ `rm_scores`—a shared team reward. You’ll want `reward_model.enable=false` for this path.

5. **PPO twice** — `_execute_party_ppo` runs for A, then for B: old log-probs, advantages, actor update, and weight sync into each side’s inference stack.

## Before you run

**Config you’ll care about**

- `debate.pool_b`: `{ n_gpus_per_node, nnodes }` for B’s GPUs—should match how Ray actually sees devices on your machine.
- `debate.actor_rollout_ref_b`: layered on top of `actor_rollout_ref`. If B is a different model from A, set at least `model.path` for B. You can mirror everything else from A with YAML anchors (see `config/ppo_debate_gsm8k_2gpu_each.yaml`).

**Data**

- Same style of RLHF parquet as elsewhere in verl. For GSM8K debate, rows should look like other GSM8K recipes (`data_source=openai/gsm8k`, `reward_model.ground_truth`, etc.). The example config points at `${REPO_ROOT}/data/gsm8k/{train,test}.parquet` unless you override `GSM8K_TRAIN` / `GSM8K_TEST`. From repo root you can build those files with: `python examples/data_preprocess/gsm8k.py --local_save_dir data/gsm8k`.

**Not wired up in this trainer (misconfig will error)**

Reference-policy KL, REMAX, `rollout.skip`, teacher distillation, and using a **reward model** at the same time as `debate.gsm8k.enable=true`.

## Quick start

From the **repository root** so `verl` imports cleanly:

```bash
export PYTHONPATH="/path/to/verl/repo${PYTHONPATH:+:$PYTHONPATH}"
```

### Smoke run: two GPUs per model (four GPUs total, TP=2 each)

```bash
export CUDA_VISIBLE_DEVICES=0,1,6,7   # example: first pair → A, second pair → B
export GSM8K_TRAIN=/path/to/train.parquet
export GSM8K_TEST=/path/to/test.parquet
bash examples/debate/run_gsm8k_debate_2gpu_each.sh
```

Optional env vars: `MODEL_A`, `MODEL_B` for checkpoints.

### Lighter recipe (GRPO + GSM8K debate)

```bash
bash examples/debate/run_gsm8k_debate_grpo.sh
```

You can still override pools, paths, or models on the command line, e.g. `debate.pool_b.n_gpus_per_node=2`.

## GPU layout

Ray orders devices the way CUDA does **after** `CUDA_VISIBLE_DEVICES`. The trainer builds:

- **`global_pool`** — from `trainer.n_gpus_per_node` × `trainer.nnodes` (party A).
- **`debate_pool_b`** — from `debate.pool_b` (party B).

Example: `CUDA_VISIBLE_DEVICES=0,1,6,7` becomes logical `cuda:0`–`cuda:3`; if each pool asks for two GPUs, A uses the first two and B uses the last two.

**Important:** vLLM does **not** merge VRAM across all four cards. Each party only uses **its** pair. On **each** GPU, `actor_rollout_ref.rollout.gpu_memory_utilization` is a fraction of **that** card for weights plus KV cache. Tiny values like `0.015` can pass a preflight check but leave **no room for KV blocks** (“No available memory for the cache blocks”) on vLLM v1. With `actor.fsdp_config.param_offload: true` and the trainer flushing cache before vLLM starts, use a normal colocation number—the example YAML is around **0.45**; try **0.35–0.55** and nudge down if vLLM still complains about free memory.

## Handy `debate.*` knobs

| Key | What it does |
|-----|----------------|
| `debate.pool_b` | GPU pool for party B. |
| `debate.actor_rollout_ref_b` | Party B’s `actor_rollout_ref` overlay (merged with A). |
| `debate.gsm8k.enable` | CoT / `####` formatting for A, shared GSM8K reward from B’s answer. |
| `debate.gsm8k.party_a_suffix` | Extra text on A’s last user turn (after defaults merge). |
| `debate.opponent_header` / `debate.second_turn_suffix` | Framing around A’s reply inside B’s prompt. |

Async vLLM/SGLang HTTP servers register **named** Ray actors. Party B’s config sets a different `http_ray_server_actor_name_prefix` (`vllm_b_`, `sglang_b_`, …) so the two stacks never step on each other.

## Where the code lives

| Path | Role |
|------|------|
| `verl/trainer/main_ppo_debate.py` | Hydra entry, `TaskRunner`, pools, tokenizer B. |
| `verl/trainer/ppo/ray_trainer_debate.py` | Main `fit` loop: dual rollouts + PPO. |
| `verl/trainer/ppo/debate_batch_utils.py` | Builds B’s batch from A’s output. |
| `verl/trainer/ppo/debate_gsm8k.py` | GSM8K defaults, prompt suffix, shared `rm_scores`. |
| `verl/trainer/ppo/utils_debate.py` | `DebateRole` → `Role` (includes `ActorRolloutB`). |

## Saving full debate trajectories (JSONL)

Set **`trainer.debate_trajectory_jsonl`** to a path (see the example YAML). After each step the trainer **appends** one JSON object per rollout row, including:

- `global_step`, `uid`, `data_source`, `ground_truth`
- `party_a_prompt`, `party_a_response`, `party_b_prompt`, `party_b_response` (plain text)
- `party_a` / `party_b` rewards (token score sums; identical under shared GSM8K reward)

It’s **JSONL** (one JSON object per line) so you can stream long runs. To merge into one array:

```bash
jq -s '.' outputs/debate_trajectories/two_gpu_each.jsonl > all_trajectories.json
```

Clear the key or set it to `null` to turn logging off.

### Weights & Biases

**Metrics only (no trajectory table in W&B)** — what you want if trajectories live in JSONL:

1. Put **`wandb`** in `trainer.logger`, e.g. `logger: [console, wandb]`.
2. Leave **`trainer.debate_wandb_log_trajectories: false`** (default in `ppo_trainer`).
3. Set **`trainer.debate_trajectory_jsonl`** to your JSONL path; transcripts append there each step, full length.

W&B still gets scalars and curves: **`party_a/`** and **`party_b/`** prefixes (e.g. `party_a/critic/score/mean`, `party_b/actor/pg_loss`), plus shared keys (`training/*`, `debate/*`, timing, throughput, …). **`vemlp_wandb`** works the same if it’s in `logger`.

**Optional: trajectory table in W&B** — set **`debate_wandb_log_trajectories: true`**. Each step also logs **`debate/trajectories`** as a `wandb.Table` (same columns as JSONL but **capped** for the UI). Tune **`debate_wandb_trajectory_max_rows`** (default 32) and **`debate_wandb_trajectory_max_chars`** (default 4000 per cell). For analysis, prefer **`debate_trajectory_jsonl`** (no cell truncation).

**If W&B prints `teardown_atexit` / `BrokenPipeError` at exit:** make sure the run reaches **`logger.finish()`** (the trainer calls this so `wandb.finish()` and `wandb.teardown()` can run before Ray tears everything down). A **`ResourceTracker` / `_recursion_count`** line during Python 3.12 shutdown often comes from **`multiprocess`** and is noisy but harmless—update the package when a fix exists, or ignore.

## When training “ends” but the logs look scary

Seeing vLLM `Engine core ... died unexpectedly` or multiprocessing `ResourceTracker` lines **after** the progress bar hits 100% is fairly common: Ray is shutting actors down and processes exit in a rough order. That’s usually **teardown noise**, not a sign the last step failed.

For validation numbers at the end, set `trainer.test_freq` to something positive (e.g. `1`).

## Deeper notes and debugging history

See [DEBATE_IMPLEMENTATION_NOTES.md](./DEBATE_IMPLEMENTATION_NOTES.md) in this folder.
