# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Translate one SystemOne request into independent causal question rows."""

from dataclasses import dataclass
from itertools import pairwise

from vllm_ascend.decision.encoding import SPECIAL, encode, rows_of
from vllm_ascend.entrypoints.systemone.protocol import to_record, with_date_facts


@dataclass(frozen=True)
class ReadoutSpec:
    option_ends: tuple[int, ...]
    decide: int
    state_length: int

    @classmethod
    def parse(cls, data, prompt_length):
        if not isinstance(data, dict) or data.get("version") != 2:
            raise ValueError("Missing or unsupported Kev readout metadata")
        ends, decide = data.get("option_ends"), data.get("decide")
        if not isinstance(ends, (tuple, list)) or not 1 <= len(ends) <= 255:
            raise ValueError("Kev readout needs 1..255 option boundaries")
        if type(decide) is not int or decide != prompt_length - 1:
            raise ValueError("Kev decide must be the last prompt token")
        if any(type(p) is not int or p < 0 or p >= decide for p in ends):
            raise ValueError("Invalid option boundary")
        if any(a >= b for a, b in pairwise(ends)):
            raise ValueError("Option boundaries must be strictly increasing")
        state_length = data.get("state_length")
        if type(state_length) is not int or not 1 <= state_length <= ends[0]:
            raise ValueError("Invalid Kev state-prefix length")
        return cls(tuple(ends), decide, state_length)

    def as_dict(self):
        return {
            "version": 2,
            "state_length": self.state_length,
            "option_ends": list(self.option_ends),
            "decide": self.decide,
        }


@dataclass(frozen=True)
class QuestionRow:
    token_ids: list[int]
    readout: ReadoutSpec


@dataclass(frozen=True)
class DecisionPlan:
    rows: list[QuestionRow]
    metadata: list[dict]
    input_tokens: int
    state_truncated: bool


def validate_tokenizer(tokenizer, expected_ids):
    actual = [tokenizer.convert_tokens_to_ids(t) for t in SPECIAL]
    if actual != expected_ids or len(set(actual)) != len(SPECIAL):
        raise ValueError("Kev delimiter token IDs do not match the exported tokenizer")


def plan_request(request, tokenizer, config, max_model_len):
    if config.date_facts:
        request = request.model_copy(update={"state": with_date_facts(request.state)})
    record, metadata = to_record(request)
    context = min(config.max_context, max_model_len)
    encoded = encode(
        tokenizer,
        record,
        max_state=context,
        max_branch=context,
        strict=config.strict_length,
    )
    state, _, branches = rows_of(encoded)
    rows = []
    for branch in branches:
        tokens = state + branch["ids"]
        spec = ReadoutSpec(tuple(len(state) + p for p in branch["opts"]), len(state) + branch["decide"], len(state))
        ReadoutSpec.parse(spec.as_dict(), len(tokens))
        rows.append(QuestionRow(tokens, spec))
    return DecisionPlan(rows, metadata, len(encoded["ids"]), encoded["state_truncated"])
