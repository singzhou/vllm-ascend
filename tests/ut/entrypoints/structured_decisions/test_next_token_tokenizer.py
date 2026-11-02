# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real tokenizer contracts from v0.32.0 test_strategies.py.

Requires the Qwen tokenizer files and a complete vLLM Python environment.
No model weights or accelerator are needed for these tokenizer checks.
"""

import pytest

from vllm_ascend.entrypoints.structured_decisions.question_types import (
    CHOICE_LABELS_128,
    LABELS,
)
from vllm_ascend.entrypoints.structured_decisions.strategies import reply_label_ids


@pytest.fixture(scope="module")
def qwen():
    from vllm.tokenizers import get_tokenizer

    tokenizer = get_tokenizer("Qwen/Qwen3-0.6B")
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "state"}],
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_dict=False,
    )
    return tokenizer, prompt_ids


def test_labels_start_the_reply(qwen):
    tokenizer, prompt_ids = qwen
    tail, ids = reply_label_ids(tokenizer, prompt_ids)
    assert tokenizer.decode(tail) == "\n\n"
    assert [tokenizer.decode([i]) for i in ids] == list(LABELS)
    _, ids = reply_label_ids(tokenizer, prompt_ids, ("yes", "no", *"0123456789"))
    assert len(set(ids)) == 12
    with pytest.raises(ValueError, match="not one distinct token"):
        reply_label_ids(tokenizer, tokenizer.encode("team:"))


def test_extended_choice_codebook_starts_the_real_reply(qwen):
    tokenizer, prompt_ids = qwen
    tail, ids = reply_label_ids(tokenizer, prompt_ids, CHOICE_LABELS_128)
    assert len(ids) == len(set(ids)) == 128
    assert not set(ids) & set(tokenizer.all_special_ids)
    assert not set(ids) & set(tokenizer.get_added_vocab().values())
    assert [tokenizer.decode([token]) for token in ids] == list(CHOICE_LABELS_128)
    text = tokenizer.decode(tail)
    for label, token in zip(CHOICE_LABELS_128, ids):
        assert tokenizer.encode(text + label, add_special_tokens=False) == [
            *tail,
            token,
        ]
