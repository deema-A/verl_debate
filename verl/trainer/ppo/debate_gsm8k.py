# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""GSM8K two-party debate: shared reward from party B's formatted answer vs ground truth."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.reward_score.gsm8k import compute_score as gsm8k_compute_score


DEFAULT_PARTY_A_SUFFIX = (
    "\n\nLet's think step by step. Show your reasoning, then put the final numeric answer alone "
    "on the last line exactly in the form #### <integer>.\n"
)

DEFAULT_OPPONENT_HEADER = "\n\n[First model — reasoning / chain-of-thought]\n"

DEFAULT_SECOND_TURN_SUFFIX = (
    "\n\nThe line above is another model's reasoning (not necessarily correct). "
    "Based on that explanation, give your own final numeric answer on the last line "
    "exactly as #### <integer>.\n"
)


def merge_debate_cfg_with_gsm8k_defaults(debate_cfg: DictConfig) -> DictConfig:
    """When ``debate.gsm8k.enable`` is true, fill default prompt/reward strings unless overridden."""
    if not OmegaConf.select(debate_cfg, "gsm8k.enable", default=False):
        return debate_cfg
    defaults = OmegaConf.create(
        {
            "opponent_header": DEFAULT_OPPONENT_HEADER,
            "second_turn_suffix": DEFAULT_SECOND_TURN_SUFFIX,
            "gsm8k": {
                "party_a_suffix": DEFAULT_PARTY_A_SUFFIX,
                "method": "strict",
                "format_score": 0.0,
                "score": 1.0,
            },
        }
    )
    return OmegaConf.merge(defaults, debate_cfg)


def _append_suffix_to_chat_messages(messages: list[Any], suffix: str) -> list[Any]:
    """Append ``suffix`` to the last user-visible text in a chat ``messages`` list (in-place safe via deepcopy)."""
    msgs = copy.deepcopy(messages)
    if not msgs or not suffix:
        return msgs
    last = msgs[-1]
    if not isinstance(last, dict):
        return msgs
    content = last.get("content")
    if isinstance(content, str):
        last["content"] = content + suffix
    elif isinstance(content, list):
        for part in reversed(content):
            if isinstance(part, dict) and part.get("type") == "text":
                part["text"] = part.get("text", "") + suffix
                break
        else:
            content.append({"type": "text", "text": suffix.lstrip("\n") or suffix})
    return msgs


def append_party_a_suffix_to_gen_batch(gen_batch: DataProto, suffix: str) -> DataProto:
    """Append party-A instructions to each ``raw_prompt`` (async agent loop / RLHFDataset).

    ``_get_gen_batch`` leaves tensors on the trainer batch; the generation batch only carries
    non-tensor fields, so we edit chat messages in place (deep copy per row).
    """
    if not suffix:
        return gen_batch
    if "raw_prompt" not in gen_batch.non_tensor_batch:
        raise ValueError(
            "debate GSM8K party_a_suffix requires non_tensor_batch['raw_prompt'] "
            "(async rollout with RLHFDataset)."
        )
    out = copy.deepcopy(gen_batch)
    bs = len(out)
    rp = out.non_tensor_batch["raw_prompt"]
    new_rp = np.empty(bs, dtype=object)
    for i in range(bs):
        new_rp[i] = _append_suffix_to_chat_messages(rp[i], suffix)
    out.non_tensor_batch["raw_prompt"] = new_rp
    return out


def _valid_response_length(batch: DataProto) -> torch.Tensor:
    prompt_len = batch.batch["prompts"].size(1)
    return batch.batch["attention_mask"][:, prompt_len:].sum(dim=-1)


def build_shared_gsm8k_rm_scores(
    batch_party_b: DataProto,
    tokenizer_b: PreTrainedTokenizer,
    *,
    method: str = "strict",
    format_score: float = 0.0,
    score: float = 1.0,
) -> list[float]:
    """Score each row using party B's decoded response vs ``reward_model.ground_truth`` (GSM8K)."""
    valid_lens = _valid_response_length(batch_party_b)
    scores: list[float] = []
    for i in range(len(batch_party_b)):
        vl = int(valid_lens[i].item())
        resp_ids = batch_party_b.batch["responses"][i][:vl]
        text = tokenizer_b.decode(resp_ids.tolist(), skip_special_tokens=True)
        gt = batch_party_b[i].non_tensor_batch["reward_model"].get("ground_truth", None)
        if gt is None:
            scores.append(0.0)
            continue
        scores.append(
            float(gsm8k_compute_score(text, str(gt), method=method, format_score=format_score, score=score))
        )
    return scores


def inject_shared_rm_scores(batch: DataProto, scores: list[float]) -> None:
    """Write identical per-row scalars into ``rm_scores`` at the last valid response token (in-place)."""
    assert len(scores) == len(batch)
    valid_lens = _valid_response_length(batch)
    rm = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)
    for i, s in enumerate(scores):
        vl = int(valid_lens[i].item())
        if vl > 0:
            rm[i, vl - 1] = float(s)
    batch.batch["rm_scores"] = rm
    batch.meta_info.setdefault("reward_extra_keys", [])


def shared_gsm8k_metrics(scores: list[float]) -> dict[str, float]:
    if not scores:
        return {}
    arr = torch.tensor(scores, dtype=torch.float32)
    return {
        "debate/gsm8k/shared_reward_mean": float(arr.mean().item()),
        "debate/gsm8k/shared_acc": float((arr >= 1.0 - 1e-6).float().mean().item()),
    }
