# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""CPU tests for debate prompt stitching (no Ray)."""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from transformers import GPT2TokenizerFast

from verl import DataProto
from verl.trainer.ppo.debate_batch_utils import (
    DEBATE_PRECOMPUTED_PROMPT_IDS_KEY,
    build_gen_batch_for_party_b,
)

pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def gpt2_tokenizer():
    return GPT2TokenizerFast.from_pretrained("gpt2")


def _fake_party_a_output(tokenizer, batch_size: int = 2) -> DataProto:
    tokenizer.pad_token = tokenizer.eos_token
    prompts = ["Q1 hello", "Q2 world"]
    responses = ["A1 reply", "B1 counter"]
    prompt_toks = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    resp_toks = [tokenizer.encode(r, add_special_tokens=False) for r in responses]
    max_p = max(len(x) for x in prompt_toks)
    max_r = max(len(x) for x in resp_toks)
    pad = tokenizer.pad_token_id
    pid = torch.full((batch_size, max_p), pad, dtype=torch.long)
    rid = torch.full((batch_size, max_r), pad, dtype=torch.long)
    for i in range(batch_size):
        pt = prompt_toks[i]
        pid[i, max_p - len(pt) :] = torch.tensor(pt)
        rt = resp_toks[i]
        rid[i, : len(rt)] = torch.tensor(rt)
    batch = TensorDict(
        {
            "prompts": pid,
            "responses": rid,
            "input_ids": torch.cat([pid, rid], dim=1),
            "attention_mask": torch.ones(batch_size, max_p + max_r, dtype=torch.long),
        },
        batch_size=batch_size,
    )
    uid = np.array([f"u{i}" for i in range(batch_size)], dtype=object)
    return DataProto(
        batch=batch,
        non_tensor_batch={"uid": uid},
        meta_info={"eos_token_id": tokenizer.eos_token_id, "pad_token_id": pad},
    )


def test_build_gen_batch_for_party_b_shape_and_text(gpt2_tokenizer):
    tok_a = gpt2_tokenizer
    tok_b = gpt2_tokenizer
    out_a = _fake_party_a_output(tok_a, batch_size=2)
    debate_cfg = OmegaConf.create({})
    gen_b = build_gen_batch_for_party_b(
        output_party_a=out_a,
        tokenizer_a=tok_a,
        tokenizer_b=tok_b,
        debate_cfg=debate_cfg,
        max_prompt_length=256,
        device=torch.device("cpu"),
    )
    assert len(gen_b) == 2
    assert gen_b.batch["input_ids"].shape[0] == 2
    assert gen_b.batch["attention_mask"].shape == gen_b.batch["input_ids"].shape
    assert gen_b.batch["position_ids"].shape == gen_b.batch["input_ids"].shape
    text0 = tok_b.decode(gen_b.batch["input_ids"][0].tolist(), skip_special_tokens=True)
    assert "Q1 hello" in text0 and "A1 reply" in text0
    assert DEBATE_PRECOMPUTED_PROMPT_IDS_KEY in gen_b.non_tensor_batch
    pre0 = gen_b.non_tensor_batch[DEBATE_PRECOMPUTED_PROMPT_IDS_KEY][0]
    assert pre0 == gen_b.batch["input_ids"][0][gen_b.batch["attention_mask"][0].bool()].tolist()
