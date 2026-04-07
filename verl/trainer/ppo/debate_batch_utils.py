# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build model-B prompts after model-A generation (cross-tokenizer text sync point)."""

from __future__ import annotations

import numpy as np
import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask

# Async ``single_turn_agent`` tokenizes ``raw_prompt`` only; it ignores ``input_ids`` on the batch.
# Party B must pass A's reply via this key so the agent loop sends the true debate prefix to vLLM.
DEBATE_PRECOMPUTED_PROMPT_IDS_KEY = "debate_precomputed_prompt_ids"


def _decode_prompt_response_row(
    tokenizer: PreTrainedTokenizer,
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
) -> tuple[str, str]:
    p = tokenizer.decode(prompt_ids.tolist(), skip_special_tokens=True)
    r = tokenizer.decode(response_ids.tolist(), skip_special_tokens=True)
    return p, r


def build_gen_batch_for_party_b(
    *,
    output_party_a: DataProto,
    tokenizer_a: PreTrainedTokenizer,
    tokenizer_b: PreTrainedTokenizer,
    debate_cfg: DictConfig,
    max_prompt_length: int,
    device: torch.device,
) -> DataProto:
    """Create a generation batch for party B: prompts include A's decoded reply (sync point).

    Uses plain text round-trip so A and B can use different tokenizers/vocabs.
    """
    opponent_header = debate_cfg.get("opponent_header", "\n\n[Opponent]\n")
    second_turn_suffix = debate_cfg.get("second_turn_suffix", "\n\nYour response:\n")

    batch_size = len(output_party_a)
    input_ids_list = []
    attention_mask_list = []

    precomputed_prompt_ids = np.empty(batch_size, dtype=object)

    for i in range(batch_size):
        prompt_ids = output_party_a.batch["prompts"][i]
        response_ids = output_party_a.batch["responses"][i]
        p_txt, r_txt = _decode_prompt_response_row(tokenizer_a, prompt_ids, response_ids)
        full_text = f"{p_txt}{opponent_header}{r_txt}{second_turn_suffix}"

        enc = tokenizer_b(
            full_text,
            padding=False,
            truncation=True,
            max_length=max_prompt_length,
            return_tensors="pt",
            add_special_tokens=True,
        )
        row_ids = enc["input_ids"].squeeze(0)
        precomputed_prompt_ids[i] = row_ids.detach().cpu().tolist()
        input_ids_list.append(row_ids)
        attention_mask_list.append(enc["attention_mask"].squeeze(0))

    max_len = max(x.shape[0] for x in input_ids_list)
    pad_id = tokenizer_b.pad_token_id
    if pad_id is None:
        pad_id = tokenizer_b.eos_token_id

    padded_ids = []
    padded_mask = []
    for ids, mask in zip(input_ids_list, attention_mask_list, strict=True):
        pad_len = max_len - ids.shape[0]
        if pad_len > 0:
            ids = torch.cat([torch.full((pad_len,), pad_id, dtype=ids.dtype), ids])
            mask = torch.cat([torch.zeros(pad_len, dtype=mask.dtype), mask])
        padded_ids.append(ids)
        padded_mask.append(mask)

    input_ids = torch.stack(padded_ids).to(device)
    attention_mask = torch.stack(padded_mask).to(device)
    position_ids = compute_position_id_with_mask(attention_mask)

    batch = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
        batch_size=batch_size,
    )

    non_tensor = {k: v.copy() if hasattr(v, "copy") else v for k, v in output_party_a.non_tensor_batch.items()}
    non_tensor[DEBATE_PRECOMPUTED_PROMPT_IDS_KEY] = precomputed_prompt_ids
    meta = dict(output_party_a.meta_info)

    return DataProto(batch=batch, non_tensor_batch=non_tensor, meta_info=meta)


def align_batch_repeat_with_output(base_batch: DataProto, gen_output: DataProto, repeat_times: int) -> DataProto:
    """Repeat base_batch to match rollout.n before union with gen_output (same as RayPPOTrainer.fit)."""
    repeated = base_batch.repeat(repeat_times=repeat_times, interleave=True)
    return repeated.union(gen_output)
