#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
# This file is a part of the vllm-ascend project.
#
"""Unit tests for the dspark speculative-decoding proposer."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer
from vllm_ascend.spec_decode.dspark_proposer import AscendDSparkProposer
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer

# 0 = single-DP (no padding); >0 = multi-DP where num_input_tokens >
# num_query_total, the out-of-bounds regime.
MULTI_DP_PADDING_SIZES = [0, 8, 32]
_NUM_SPECULATIVE_TOKENS = 3
_MAX_BATCH_SIZE = 2
_MAX_NUM_TOKENS = 8
_HIDDEN_SIZE = 16


class _DSparkProposerTestBase:
    """Shared helpers for ``AscendDSparkProposer`` tests."""

    @staticmethod
    def _make_vllm_config(hf_config: SimpleNamespace) -> SimpleNamespace:
        """Build the minimal config consumed by the DSpark initializer."""
        draft_model_config = SimpleNamespace(hf_config=hf_config, get_hidden_size=lambda: _HIDDEN_SIZE)
        return SimpleNamespace(
            speculative_config=SimpleNamespace(draft_sample_method="greedy", draft_model_config=draft_model_config),
            parallel_config=SimpleNamespace(data_parallel_size=1, use_sequence_parallel_moe=False, is_moe_model=False),
            compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE),
            model_config=SimpleNamespace(enforce_eager=True),
            scheduler_config=SimpleNamespace(max_num_batched_tokens=_MAX_NUM_TOKENS),
        )

    @classmethod
    def _make_proposer(
        cls,
        *,
        max_num_tokens: int,
        num_reqs: int,
        block_size: int,
        hf_config: SimpleNamespace | None = None,
    ):
        device = torch.device("cpu")
        vllm_config = cls._make_vllm_config(hf_config or SimpleNamespace())

        def mock_parent_init(
            proposer: AscendDSparkProposer,
            vllm_config: SimpleNamespace,
            device: torch.device,
            runner: object | None = None,
        ) -> None:
            del runner
            proposer.draft_model_config = vllm_config.speculative_config.draft_model_config
            proposer.num_speculative_tokens = block_size
            proposer.max_batch_size = num_reqs
            proposer.max_num_tokens = max_num_tokens
            proposer.dtype = torch.float32
            proposer.device = device
            proposer.hidden_size = _HIDDEN_SIZE
            proposer.hidden_states = torch.empty(0)
            proposer._dflash_hidden_states = torch.empty(0)

        with patch.object(AscendDSparkProposer.__base__, "__init__", mock_parent_init):
            proposer = AscendDSparkProposer(vllm_config, device)

        num_query_total = num_reqs * proposer.num_query_per_req
        proposer.positions = torch.zeros(max_num_tokens, dtype=torch.int32, device=device)
        proposer.positions[:num_query_total] = torch.arange(num_query_total, dtype=torch.int32)
        proposer.parallel_drafting_token_id = 0
        proposer.kv_cache_gid = 0
        proposer._dflash_num_context = 0

        proposer.input_ids = torch.zeros(max_num_tokens, dtype=torch.int64, device=device)
        proposer._context_positions_buffer = torch.zeros(max_num_tokens, dtype=torch.int32, device=device)
        proposer._slot_mapping_buffer = torch.zeros(max_num_tokens, dtype=torch.int32, device=device)
        proposer._dspark_seed_buffer = torch.zeros(max_num_tokens, dtype=torch.int64, device=device)
        proposer._dflash_hidden_states = torch.zeros((max_num_tokens, 8), dtype=torch.float32, device=device)
        proposer.arange_dflash = torch.arange(max_num_tokens + 1, dtype=torch.int32, device=device)
        proposer.token_arange_np = np.arange(max_num_tokens + 1, dtype=np.int32)

        gid = 0
        proposer.draft_attn_groups = [
            SimpleNamespace(
                kv_cache_group_id=gid,
                kv_cache_spec=SimpleNamespace(block_size=block_size),
                layer_names=["L0"],
            )
        ]
        proposer._layer_group_idx = [gid]
        block_table = torch.zeros((num_reqs, 16), dtype=torch.int32, device=device)
        proposer._per_group_block_tables = {gid: block_table}
        proposer._per_group_block_table_buffers = {gid: block_table}
        slot = torch.zeros(max_num_tokens, dtype=torch.int32, device=device)
        proposer._per_group_slot_mappings = {gid: slot}
        proposer._per_group_query_slot_mapping_buffers = {gid: slot.clone()}
        proposer._per_group_context_slot_mapping_buffers = {gid: slot.clone()}
        # Normally set by load_model / check_gdn_layer.
        proposer.kernel_block_size = block_size
        proposer.has_gdn = False
        return proposer

    # fmt: off
    @staticmethod
    def _invoke_set_inputs_first_pass(
        proposer,
        *,
        num_reqs,
        block_size,
        seq_len=128,
        context=None,
        num_rejected=None,
        with_optional_attrs=False,
    ):
        """Drive ``set_inputs_first_pass`` with a configurable cad.

        ``context`` sets ``query_start_loc_cpu[num_reqs]`` so the proposer
        copies ``context`` rows of target hidden states (0 by default).
        Returns ``(num_query_total, token_indices, cad, extra,
        next_token_ids, target_hidden_states)``.
        """
        next_token_ids = torch.arange(1, num_reqs + 1, dtype=torch.int64)
        target_hidden_states = torch.arange(
            num_reqs * 8, dtype=torch.float32
        ).reshape(num_reqs, 8)
        query_start_loc_cpu = torch.zeros(num_reqs + 1, dtype=torch.int32)
        if context is not None:
            query_start_loc_cpu[num_reqs] = context
        cad = SimpleNamespace(
            num_reqs=num_reqs,
            query_start_loc=torch.arange(num_reqs + 1, dtype=torch.int32) * block_size,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=torch.full((num_reqs,), seq_len, dtype=torch.int32),
            max_seq_len=seq_len,
        )
        if with_optional_attrs:
            cad.actual_seq_lengths_q = [0] * num_reqs
            cad.decode_token_per_req = 0
        num_query_total, token_indices, cad, extra = proposer.set_inputs_first_pass(
            target_token_ids=torch.zeros(num_reqs, dtype=torch.int64),
            next_token_ids=next_token_ids,
            target_positions=torch.zeros(num_reqs, dtype=torch.int32),
            target_hidden_states=target_hidden_states,
            token_indices_to_sample=None,
            cad=cad,
            num_rejected_tokens_gpu=num_rejected,
        )
        return num_query_total, token_indices, cad, extra, next_token_ids, target_hidden_states


# fmt: on


class TestDSparkPositionsFullUnderMultiDp(_DSparkProposerTestBase):
    """Guard: under multi-DP the dspark draft proposer must hand DSA attention a
    full-length positions buffer so ``positions[:num_input_tokens]`` never reads
    out of bounds (the slice is DP-padded and may exceed the local query size)."""

    @staticmethod
    def _call_set_inputs_first_pass(proposer, *, num_reqs, block_size):
        # query_start_loc_cpu[num_reqs] is 0 so _dflash_num_context becomes 0.
        cad = SimpleNamespace(
            num_reqs=num_reqs,
            query_start_loc=torch.arange(num_reqs + 1, dtype=torch.int32) * block_size,
            query_start_loc_cpu=torch.zeros(num_reqs + 1, dtype=torch.int32),
            seq_lens=torch.full((num_reqs,), 128, dtype=torch.int32),
            max_seq_len=128,
        )
        proposer.set_inputs_first_pass(
            target_token_ids=torch.zeros(num_reqs, dtype=torch.int64),
            next_token_ids=torch.zeros(num_reqs, dtype=torch.int64),
            target_positions=torch.zeros(num_reqs, dtype=torch.int32),
            target_hidden_states=torch.zeros((num_reqs, 8), dtype=torch.float32),
            token_indices_to_sample=None,
            cad=cad,
            num_rejected_tokens_gpu=None,
        )
        return cad

    @pytest.mark.parametrize("dp_padding", MULTI_DP_PADDING_SIZES)
    def test_positions_not_pre_sliced(self, monkeypatch, dp_padding):
        """``cad.positions`` must be the full buffer, not ``[:num_query_total]``."""
        monkeypatch.setattr(
            "vllm_ascend.spec_decode.dspark_proposer.copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid",
            MagicMock(),
        )
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        num_query_total = num_reqs * block_size
        num_input_tokens = num_query_total + dp_padding

        proposer = self._make_proposer(max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size)
        cad = self._call_set_inputs_first_pass(proposer, num_reqs=num_reqs, block_size=block_size)

        # DSA attention slices positions[:num_input_tokens] (DP-padded); a
        # pre-slice to num_query_total reads out of bounds under multi-DP.
        assert cad.positions.shape[0] == max_num_tokens
        assert cad.positions[:num_input_tokens].shape[0] == num_input_tokens

    @pytest.mark.parametrize("dp_padding", [8, 32])
    def test_positions_full_and_padded_for_dsa(self, monkeypatch, dp_padding):
        """After set_inputs_first_pass + _pad_draft_buffers, positions[:num_input]
        is full-length and zero-padded in the DP region."""
        monkeypatch.setattr(
            "vllm_ascend.spec_decode.dspark_proposer.copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid",
            MagicMock(),
        )
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        num_query_total = num_reqs * block_size
        num_input_tokens = num_query_total + dp_padding

        proposer = self._make_proposer(max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size)
        proposer.positions[num_query_total:num_input_tokens] = -999
        cad = self._call_set_inputs_first_pass(proposer, num_reqs=num_reqs, block_size=block_size)
        proposer._pad_draft_buffers(num_query_total, num_input_tokens)

        dsa_slice = cad.positions[:num_input_tokens]
        assert dsa_slice.shape[0] == num_input_tokens
        assert torch.all(dsa_slice[num_query_total:] == 0)


class TestPadDraftBuffersBeforeBuild(_DSparkProposerTestBase):
    """Guard: ``_pad_draft_buffers`` must zero the DP-padding region of positions
    and run before ``build_draft_attn_metadata``, so the attention backend reads
    valid (zero) padding instead of stale values."""

    def test_zeros_dp_padding_region(self):
        """``_pad_draft_buffers`` zeros positions / input_ids / slot_mapping in
        the DP-padding region."""
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        num_actual = num_reqs * block_size
        num_input = num_actual + 16

        proposer = self._make_proposer(max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size)
        proposer.positions[num_actual:num_input] = -999
        proposer.input_ids[num_actual:num_input] = -999
        proposer._slot_mapping_buffer[num_actual:num_input] = -999
        for buf in proposer._per_group_query_slot_mapping_buffers.values():
            buf[num_actual:num_input] = -999

        proposer._pad_draft_buffers(num_actual, num_input)

        assert torch.all(proposer.positions[num_actual:num_input] == 0)
        assert torch.all(proposer.input_ids[num_actual:num_input] == proposer.parallel_drafting_token_id)
        assert torch.all(proposer._slot_mapping_buffer[num_actual:num_input] == -1)
        for buf in proposer._per_group_query_slot_mapping_buffers.values():
            assert torch.all(buf[num_actual:num_input] == -1)
        assert torch.all(proposer.positions[:num_actual] != -999)

    def test_noop_without_dp_padding(self):
        """Single-DP (num_input <= num_actual) leaves buffers untouched."""
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        num_actual = num_reqs * block_size

        proposer = self._make_proposer(max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size)
        snapshot = proposer.positions.clone()
        proposer._pad_draft_buffers(num_actual, num_actual)
        assert torch.equal(proposer.positions, snapshot)

    def test_must_precede_build(self):
        """build_draft_attn_metadata reads positions but does not zero it, so
        _pad_draft_buffers must run first."""
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        num_actual = num_reqs * block_size
        num_input = num_actual + 16

        def capture_build():
            captured = {}

            def fake_build(common_attn_metadata, num_input_tokens, num_actual_tokens):
                captured["region"] = common_attn_metadata.positions[num_actual:num_input].clone()
                return None, common_attn_metadata

            return captured, fake_build

        ok = self._make_proposer(max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size)
        ok.positions[num_actual:num_input] = -999
        cap_ok, build_ok = capture_build()
        ok.build_draft_attn_metadata = build_ok
        ok._pad_draft_buffers(num_actual, num_input)
        ok.build_draft_attn_metadata(SimpleNamespace(positions=ok.positions), num_input, num_actual)
        assert torch.all(cap_ok["region"] == 0)

        bug = self._make_proposer(max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size)
        bug.positions[num_actual:num_input] = -999
        cap_bug, build_bug = capture_build()
        bug.build_draft_attn_metadata = build_bug
        bug.build_draft_attn_metadata(SimpleNamespace(positions=bug.positions), num_input, num_actual)
        bug._pad_draft_buffers(num_actual, num_input)
        assert torch.all(cap_bug["region"] == -999)

    def test_called_before_build_in_propose(self):
        """In ``_propose`` the ``_pad_draft_buffers`` call must precede
        ``build_draft_attn_metadata``."""
        src = inspect.getsource(AscendSpecDecodeBaseProposer._propose)
        pad_idx = src.find("self._pad_draft_buffers(")
        build_idx = src.find("self.build_draft_attn_metadata(")
        # Only assert when both calls live directly in _propose; a refactor that
        # extracts them elsewhere leaves this guard inert rather than brittle.
        if pad_idx != -1 and build_idx != -1:
            assert pad_idx < build_idx, (
                "_pad_draft_buffers must be called before build_draft_attn_metadata "
                "in _propose, otherwise the attention backend reads un-zeroed "
                "positions in the DP-padding region."
            )


class TestDSparkInitialization(_DSparkProposerTestBase):
    """Tests for DSpark initialization configuration."""

    @pytest.mark.parametrize(
        ("hf_config", "expected_sample_from_anchor", "expected_num_query_per_req"),
        [
            pytest.param(SimpleNamespace(), True, _NUM_SPECULATIVE_TOKENS),
            pytest.param(SimpleNamespace(dspark_bonus_anchor=True), False, 1 + _NUM_SPECULATIVE_TOKENS),
        ],
    )
    def test_configures_anchor_sampling(
        self,
        hf_config: SimpleNamespace,
        expected_sample_from_anchor: bool,
        expected_num_query_per_req: int,
    ) -> None:
        """Verify the bonus-anchor flag selects the expected query layout."""
        proposer = self._make_proposer(
            max_num_tokens=_MAX_NUM_TOKENS,
            num_reqs=_MAX_BATCH_SIZE,
            block_size=_NUM_SPECULATIVE_TOKENS,
            hf_config=hf_config,
        )
        # max_query_tokens is sized for capture buckets (1+N step), not
        # num_query_per_req alone, to accommodate ACLGraph padding.
        expected_max_query_tokens = _MAX_BATCH_SIZE * (1 + _NUM_SPECULATIVE_TOKENS)
        assert proposer.sample_from_anchor is expected_sample_from_anchor
        assert proposer.num_query_per_req == expected_num_query_per_req
        assert proposer.max_query_tokens == expected_max_query_tokens


# fmt: off
class TestSetPerGroupAttnMetadata(_DSparkProposerTestBase):
    """``set_per_group_attn_metadata`` stores the runner-provided per-group
    block table / slot mapping into the read-only dicts the proposer consults
    during ``set_inputs_first_pass``."""

    def test_stores_block_table_and_slot_mapping(self):
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        proposer = self._make_proposer(
            max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size
        )
        # a gid not pre-populated by _make_proposer (which only seeds gid=0)
        gid = 7
        block_table = torch.zeros((num_reqs, 16), dtype=torch.int32)
        slot_mapping = torch.full((max_num_tokens,), 42, dtype=torch.int32)

        proposer.set_per_group_attn_metadata(gid, block_table, slot_mapping)

        assert proposer._per_group_block_tables[gid] is block_table
        assert proposer._per_group_slot_mappings[gid] is slot_mapping

    def test_overwrites_existing_gid(self):
        num_reqs, block_size, max_num_tokens = 2, 5, 256
        proposer = self._make_proposer(
            max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size
        )
        gid = 0  # already populated by _make_proposer
        old_block_table = proposer._per_group_block_tables[gid]
        new_block_table = torch.ones((num_reqs, 16), dtype=torch.int32)
        new_slot_mapping = torch.ones(max_num_tokens, dtype=torch.int32)

        proposer.set_per_group_attn_metadata(gid, new_block_table, new_slot_mapping)

        assert proposer._per_group_block_tables[gid] is new_block_table
        assert proposer._per_group_slot_mappings[gid] is new_slot_mapping
        assert proposer._per_group_block_tables[gid] is not old_block_table


class TestDSparkInitValidation:
    """``AscendDSparkProposer.__init__`` rejects probabilistic draft sampling
    (unsupported on the v1 model runner) and, for the greedy path, allocates
    the DSpark-specific draft/seed buffers and overrides the DFlash
    query-token / cudagraph defaults."""

    @staticmethod
    def _make_vllm_config(
        *,
        num_speculative_tokens,
        max_batch_size,
        max_num_tokens,
        draft_sample_method,
        hidden_size=8,
    ):
        speculative_config = SimpleNamespace(
            num_speculative_tokens=num_speculative_tokens,
            draft_sample_method=draft_sample_method,
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(),
                get_hidden_size=lambda: hidden_size
            ),
        )
        return SimpleNamespace(speculative_config=speculative_config)

    @staticmethod
    def _stub_dflash_init(
        monkeypatch,
        *,
        num_speculative_tokens,
        max_batch_size,
        max_num_tokens,
        dtype,
        device,
    ):
        """Replace the heavy DFlash/Eagle base init with a stub that only sets
        the attributes DSpark's ``__init__`` subsequently reads."""

        def _stub(self, vllm_config, device, runner=None):
            self.num_speculative_tokens = num_speculative_tokens
            self.max_batch_size = max_batch_size
            self.max_num_tokens = max_num_tokens
            self.dtype = dtype
            self.device = device
            self.draft_model_config = vllm_config.speculative_config.draft_model_config
            # present so the ``del`` in DSpark.__init__ succeeds
            self.hidden_size = 0
            self.hidden_states = None
            self._dflash_hidden_states = None

        monkeypatch.setattr(AscendDflashProposer, "__init__", _stub)

    def test_probabilistic_rejected(self, monkeypatch):
        device = torch.device("cpu")
        self._stub_dflash_init(
            monkeypatch,
            num_speculative_tokens=5,
            max_batch_size=16,
            max_num_tokens=256,
            dtype=torch.float32,
            device=device,
        )
        vllm_config = self._make_vllm_config(
            num_speculative_tokens=5,
            max_batch_size=16,
            max_num_tokens=256,
            draft_sample_method="probabilistic",
        )
        with pytest.raises(ValueError, match="probabilistic"):
            AscendDSparkProposer(vllm_config, device)

    def test_greedy_allocates_dspark_buffers(self, monkeypatch):
        device = torch.device("cpu")
        num_spec, max_batch, max_num_tokens, hidden = 5, 16, 256, 8
        self._stub_dflash_init(
            monkeypatch,
            num_speculative_tokens=num_spec,
            max_batch_size=max_batch,
            max_num_tokens=max_num_tokens,
            dtype=torch.float32,
            device=device,
        )
        vllm_config = self._make_vllm_config(
            num_speculative_tokens=num_spec,
            max_batch_size=max_batch,
            max_num_tokens=max_num_tokens,
            draft_sample_method="greedy",
            hidden_size=hidden,
        )
        proposer = AscendDSparkProposer(vllm_config, device)

        blk = 1 + num_spec
        # max_query_tokens is sized for capture buckets (1+N step).
        max_query_tokens = max_batch * (1 + num_spec)
        # DSpark-specific draft / seed buffers.
        assert proposer._dspark_draft_buffer.shape == (max_batch, blk)
        assert proposer._dspark_draft_buffer.dtype == torch.int64
        assert proposer._dspark_seed_buffer.shape == (max_batch,)
        assert proposer._dspark_seed_buffer.dtype == torch.int64
        # hidden_size / hidden states come from the draft model config.
        assert proposer.hidden_size == hidden
        assert proposer.hidden_states.shape == (max_num_tokens, hidden)
        assert proposer._dflash_hidden_states.shape == (max_num_tokens, hidden)
        # DSpark ACLGraph is now supported; use_cuda_graph follows the same
        # logic as AscendSpecDecodeBaseProposer (runner._use_aclgraph() and
        # speculative_config.enforce_eager).  The stub parent init does not set
        # use_cuda_graph, so it defaults to False in this unit test (no real
        # runner).  A real runner with ACLGraph enabled would yield True.
        assert not hasattr(proposer, "use_cuda_graph") or proposer.use_cuda_graph is False
        # anchor-first: N query tokens per request, no bonus token (unlike
        # DFlash's 1+N).
        assert proposer.max_query_tokens == max_query_tokens
        assert proposer.positions.shape == (max_query_tokens,)
        assert proposer.positions.dtype == torch.int32
        assert proposer._slot_mapping_buffer.shape == (max_query_tokens,)
        # per-group bookkeeping dicts start empty / None.
        assert proposer._per_group_block_tables == {}
        assert proposer._per_group_slot_mappings == {}
        assert proposer._context_slot_mapping_buffers is None


class TestSetInputsFirstPassOutputs(_DSparkProposerTestBase):
    """``set_inputs_first_pass`` returns the anchor-first query budget and
    rewrites the common attention metadata into the DSpark cross-attention
    shape (N query tokens per request, non-causal, chunked-prefill state)."""

    @pytest.fixture(autouse=True)
    def _mock_kernel(self, monkeypatch):
        monkeypatch.setattr(
            "vllm_ascend.spec_decode.dspark_proposer."
            "copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid",
            MagicMock(),
        )

    def test_return_value_and_token_indices(self):
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        proposer = self._make_proposer(
            max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size
        )
        num_query_total, token_indices, _cad, extra = (
            self._invoke_set_inputs_first_pass(
                proposer, num_reqs=num_reqs, block_size=block_size
            )[:4]
        )
        assert num_query_total == num_reqs * block_size
        assert token_indices.shape == (num_reqs * block_size,)
        assert token_indices.dtype == torch.int32
        # 4th return slot is unused (no per-group attn metadata tuple here).
        assert extra is None

    def test_seed_buffer_copied_from_next_tokens(self):
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        proposer = self._make_proposer(
            max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size
        )
        self._invoke_set_inputs_first_pass(
            proposer, num_reqs=num_reqs, block_size=block_size
        )
        expected = torch.arange(1, num_reqs + 1, dtype=torch.int64)
        assert torch.equal(proposer._dspark_seed_buffer[:num_reqs], expected)
        assert torch.all(proposer._dspark_seed_buffer[num_reqs:] == 0)

    def test_context_hidden_states_copied(self):
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        proposer = self._make_proposer(
            max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size
        )
        self._invoke_set_inputs_first_pass(
            proposer, num_reqs=num_reqs, block_size=block_size, context=num_reqs
        )
        assert proposer._dflash_num_context == num_reqs
        expected = torch.arange(num_reqs * 8, dtype=torch.float32).reshape(num_reqs, 8)
        assert torch.equal(proposer._dflash_hidden_states[:num_reqs], expected)

    def test_cad_rewritten_to_cross_attention_shape(self):
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        proposer = self._make_proposer(
            max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size
        )
        num_query_total, _, cad, _ = self._invoke_set_inputs_first_pass(
            proposer, num_reqs=num_reqs, block_size=block_size, with_optional_attrs=True
        )[:4]
        # token budgets reflect anchor-first (N per request, no bonus).
        assert cad.num_actual_tokens == num_query_total
        assert cad.num_input_tokens == num_query_total
        assert cad.max_query_len == block_size
        assert cad.max_seq_len == 128 + block_size
        # attention is non-causal cross-attention over the draft query block.
        assert cad.causal is False
        assert cad.attn_mask is None
        assert cad.attn_state == AscendAttentionState.ChunkedPrefill
        # positions is the full buffer (DSA slices it), not a pre-slice.
        assert cad.positions is proposer.positions
        # slot mapping is a slice of the primary group's query buffer (shares
        # storage from offset 0); a fresh slice is not identity-equal, so check
        # the underlying storage and length instead.
        assert (
            cad.slot_mapping.data_ptr()
            == proposer._per_group_query_slot_mapping_buffers[0].data_ptr()
        )
        assert cad.slot_mapping.shape[0] == num_query_total
        # optional attrs the proposer rewrites when present.
        assert cad.actual_seq_lengths_q == [block_size] * num_reqs
        assert cad.decode_token_per_req == block_size

    def test_cad_query_start_loc_and_seq_lens(self):
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        proposer = self._make_proposer(
            max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size
        )
        _nqt, _ti, cad, _extra = self._invoke_set_inputs_first_pass(
            proposer, num_reqs=num_reqs, block_size=block_size
        )[:4]
        expected_qsl = torch.arange(num_reqs + 1, dtype=torch.int32) * block_size
        assert torch.equal(cad.query_start_loc, expected_qsl)
        assert torch.equal(cad.query_start_loc_cpu, expected_qsl)
        # seq_lens grow by block_size when no tokens were rejected.
        assert torch.equal(cad.seq_lens, torch.full((num_reqs,), 128 + block_size, dtype=torch.int32))


class TestSetInputsFirstPassRejectedTokens(_DSparkProposerTestBase):
    """The ``has_num_rejected`` branch must shrink ``seq_lens`` by the rejected
    token count before adding the draft block size, and flag the kernel."""

    def test_seq_lens_subtracts_rejected(self, monkeypatch):
        monkeypatch.setattr(
            "vllm_ascend.spec_decode.dspark_proposer."
            "copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid",
            MagicMock(),
        )
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        proposer = self._make_proposer(
            max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size
        )
        rejected = torch.full((num_reqs,), 2, dtype=torch.int32)
        _nqt, _ti, cad, _extra = self._invoke_set_inputs_first_pass(
            proposer, num_reqs=num_reqs, block_size=block_size, num_rejected=rejected
        )[:4]
        # effective = seq_lens(128) - rejected(2) = 126; then + block_size(5) = 131.
        assert torch.equal(
            cad.seq_lens, torch.full((num_reqs,), 128 - 2 + block_size, dtype=torch.int32)
        )

    def test_kernel_called_with_has_num_rejected(self, monkeypatch):
        kernel = MagicMock()
        monkeypatch.setattr(
            "vllm_ascend.spec_decode.dspark_proposer."
            "copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid",
            kernel,
        )
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        proposer = self._make_proposer(
            max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size
        )
        rejected = torch.full((num_reqs,), 2, dtype=torch.int32)
        self._invoke_set_inputs_first_pass(
            proposer, num_reqs=num_reqs, block_size=block_size, num_rejected=rejected
        )
        # The proposer calls the kernel as ``kernel[1,](...)`` (Triton-style
        # grid indexing), so the call lands on the indexed sub-mock.
        sub = kernel[1,]
        assert sub.called
        kwargs = sub.call_args.kwargs
        assert kwargs["HAS_NUM_REJECTED"] is True
        assert kwargs["num_rejected_tokens_ptr"] is rejected
        assert kwargs["SAMPLE_FROM_ANCHOR"] is True


class TestInitializeAttnBackendErrors(_DSparkProposerTestBase):
    """``initialize_attn_backend`` raises clearly when the draft model does not
    expose the DSpark layer-name API, or when no draft attention groups can be
    built from the kv-cache groups."""

    @staticmethod
    def _make_proposer_for_init():
        proposer = AscendDSparkProposer.__new__(AscendDSparkProposer)
        proposer.vllm_config = SimpleNamespace()
        proposer.device = torch.device("cpu")
        return proposer

    def test_model_without_draft_layer_names_raises(self, monkeypatch):
        # get_layers_from_vllm_config is called first; stub it so the model
        # check is what actually fails.
        monkeypatch.setattr(
            "vllm_ascend.spec_decode.dspark_proposer.get_layers_from_vllm_config",
            lambda *a, **k: {},
        )
        proposer = self._make_proposer_for_init()
        # model lacks get_draft_kv_cache_layer_names entirely.
        proposer.model = SimpleNamespace()

        kv_cache_config = SimpleNamespace(kv_cache_groups=[])
        with pytest.raises(RuntimeError, match="get_draft_kv_cache_layer_names"):
            proposer.initialize_attn_backend(kv_cache_config)

    def test_no_draft_attn_groups_raises(self, monkeypatch):
        monkeypatch.setattr(
            "vllm_ascend.spec_decode.dspark_proposer.get_layers_from_vllm_config",
            lambda *a, **k: {},
        )
        proposer = self._make_proposer_for_init()
        # draft layer names exist, but no kv-cache group names overlap them.
        proposer.model = SimpleNamespace(get_draft_kv_cache_layer_names=lambda: {"L0"})

        non_overlapping_group = SimpleNamespace(layer_names=["OTHER_LAYER"])
        kv_cache_config = SimpleNamespace(kv_cache_groups=[non_overlapping_group])
        with pytest.raises(RuntimeError, match="registered draft attention groups"):
            proposer.initialize_attn_backend(kv_cache_config)


class TestKernelBlockSizeResolution(_DSparkProposerTestBase):
    """Slot ids must be derived from the *kernel* block size.

    On hybrid (Gated-DeltaNet) targets such as Qwen3.5/Qwen3.6, vLLM enlarges the
    attention ``kv_cache_spec.block_size`` so the attention page size matches the
    mamba page size. ``BlockTable`` then splits each manager block into kernel
    blocks, so block table entries and slot mappings are addressed in kernel
    blocks. Deriving slot ids from the manager block size writes the draft query
    block's K/V to out-of-range slots; the draft then attends over uninitialized
    KV cache and its hidden states become NaN.
    """

    @pytest.mark.parametrize(("has_gdn", "expected"), [(True, 128), (False, 512)])
    def test_inputs_kernel_block_size_follows_has_gdn(self, monkeypatch, has_gdn, expected):
        num_reqs, block_size = 1, _NUM_SPECULATIVE_TOKENS
        proposer = self._make_proposer(
            max_num_tokens=_MAX_NUM_TOKENS, num_reqs=num_reqs, block_size=block_size
        )
        # Split-block group: kv manager 512, kernel 128. GDN must use the kernel
        # block size; everything else keeps the group's own spec block size.
        proposer.draft_attn_groups[0].kv_cache_spec = SimpleNamespace(block_size=512)
        proposer.kernel_block_size = 128
        proposer.has_gdn = has_gdn

        kernel = MagicMock()
        monkeypatch.setattr(
            "vllm_ascend.spec_decode.dspark_proposer."
            "copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid",
            kernel,
        )
        self._invoke_set_inputs_first_pass(
            proposer, num_reqs=num_reqs, block_size=block_size
        )

        assert kernel[1,].call_args.kwargs["block_size"] == expected
# fmt: on


class TestDSparkDummyRunACLGraph(_DSparkProposerTestBase):
    """Tests for DSpark dummy_run with ACLGraph FULL mode enabled.

    Verifies that when use_cuda_graph=True and aclgraph_runtime_mode=FULL,
    the dummy_run constructs proper drafting metadata for graph capture.
    """

    @pytest.mark.parametrize(
        ("uniform", "num_reqs", "expected_tokens"),
        [
            pytest.param(True, 2, 14, id="uniform-native-dspark-width"),
            pytest.param(False, 2, 16, id="nonuniform-preserves-descriptor-width"),
            pytest.param(True, None, 16, id="missing-request-count-fallback"),
        ],
    )
    def test_graph_token_count_is_decoupled_from_target_descriptor(
        self,
        uniform: bool,
        num_reqs: int | None,
        expected_tokens: int,
    ):
        proposer = self._make_proposer_with_graph_support(
            max_num_tokens=256,
            num_reqs=2,
            block_size=7,
        )
        descriptor = BatchDescriptor(
            num_tokens=16,
            num_reqs=num_reqs,
            uniform=uniform,
        )

        assert proposer.get_graph_num_input_tokens(descriptor) == expected_tokens

    def test_bonus_anchor_keeps_target_graph_width(self):
        proposer = self._make_proposer_with_graph_support(
            max_num_tokens=256,
            num_reqs=1,
            block_size=7,
            hf_config=SimpleNamespace(dspark_bonus_anchor=True),
        )
        descriptor = BatchDescriptor(num_tokens=8, num_reqs=1, uniform=True)

        assert proposer.num_query_per_req == 8
        assert proposer.get_graph_num_input_tokens(descriptor) == 8

    def test_runtime_padded_request_uses_positive_kv_length(self):
        """A whole-request DSpark graph padding entry cannot use KV length 0."""
        proposer = self._make_proposer_with_graph_support(
            max_num_tokens=256,
            num_reqs=1,
            block_size=7,
        )
        seq_lens = torch.tensor([40], dtype=torch.int32)

        padded = proposer._adjust_parallel_draft_seq_lens_for_graph(seq_lens, 2)

        assert torch.equal(padded, torch.tensor([40, 1], dtype=torch.int32))

    def test_dflash_graph_seq_lens_keeps_zero_padding(self):
        """The DSpark-only fix must not change DFlash graph metadata."""
        proposer = self._make_proposer_with_graph_support(
            max_num_tokens=256,
            num_reqs=1,
            block_size=7,
        )
        proposer.method = "dflash"
        seq_lens = torch.tensor([40], dtype=torch.int32)

        padded = proposer._adjust_parallel_draft_seq_lens_for_graph(seq_lens, 2)

        assert torch.equal(padded, torch.tensor([40, 0], dtype=torch.int32))

    @staticmethod
    def _make_proposer_with_graph_support(
        *,
        max_num_tokens: int,
        num_reqs: int,
        block_size: int,
        hf_config: SimpleNamespace | None = None,
        use_cuda_graph: bool = True,
    ):
        """Create a DSpark proposer with use_cuda_graph set to the given value."""
        proposer = _DSparkProposerTestBase._make_proposer(
            max_num_tokens=max_num_tokens,
            num_reqs=num_reqs,
            block_size=block_size,
            hf_config=hf_config,
        )
        proposer.use_cuda_graph = use_cuda_graph
        # Attributes that the real llm_base_proposer.__init__ sets but the
        # stub mock_parent_init skips.
        vllm_config = _DSparkProposerTestBase._make_vllm_config(hf_config or SimpleNamespace())
        if not hasattr(proposer, "vllm_config"):
            proposer.vllm_config = vllm_config
        if not hasattr(proposer, "_runnable"):
            proposer._runnable = lambda **kw: None
        if not hasattr(proposer, "token_indices_to_sample"):
            proposer.token_indices_to_sample = torch.zeros(max_num_tokens, dtype=torch.int32)
        # _get_positions checks uses_mrope, uses_xdrope_dim, etc.
        if not hasattr(proposer, "uses_mrope"):
            proposer.uses_mrope = False
        if not hasattr(proposer, "uses_xdrope_dim"):
            proposer.uses_xdrope_dim = 0
        if not hasattr(proposer, "draft_uses_xdrope_dim"):
            proposer.draft_uses_xdrope_dim = 0
        # query_start_loc buffer (CpuGpuBuffer-like) needed by dummy_run's
        # _pad_query_start_loc_for_fia call.
        if not hasattr(proposer, "query_start_loc"):
            buf_size = num_reqs + 2  # max_num_reqs + 2 for padding
            cpu_buf = torch.zeros(buf_size, dtype=torch.int32, pin_memory=True)
            gpu_buf = torch.zeros(buf_size, dtype=torch.int32)
            buf = SimpleNamespace(
                cpu=cpu_buf,
                gpu=gpu_buf,
                np=cpu_buf.numpy(),
                copy_to_gpu=lambda: gpu_buf.copy_(cpu_buf, non_blocking=True),
            )
            proposer.query_start_loc = buf
        return proposer

    @staticmethod
    def _make_runner_mock(num_reqs: int):
        """Create a minimal runner mock for dummy_run."""
        runner = MagicMock()
        runner._sync_metadata_across_dp = MagicMock(side_effect=lambda n, **kw: (n, None, None))
        runner.optimistic_seq_lens_cpu = torch.ones(num_reqs, dtype=torch.int32)
        runner.seq_lens = torch.ones(num_reqs, dtype=torch.int32) * 128
        # input_batch.block_table[gid].get_device_tensor()[:num_reqs]
        block_table_tensor = torch.zeros((num_reqs, 16), dtype=torch.int32)
        mock_bt = MagicMock()
        mock_bt.get_device_tensor.return_value = block_table_tensor
        runner.input_batch = MagicMock()
        runner.input_batch.block_table = {0: mock_bt}
        runner.attn_groups = [MagicMock()]
        # _pad_query_start_loc_for_fia: mirrors the real implementation for
        # FULL mode (adds a dummy request if last qsl entry < num_tokens).
        runner.uniform_decode_query_len = 1  # not used in FULL branch
        runner.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL)

        def _pad_query_start_loc_for_fia(
            query_start_loc,
            num_tokens_padded,
            num_reqs_padded,
            num_reqs,
            cudagraph_runtime_mode=None,
            batch_desc_num_reqs=None,
        ):
            if cudagraph_runtime_mode == CUDAGraphMode.FULL:
                num_reqs_padded = num_reqs
            else:
                num_reqs_padded = batch_desc_num_reqs if batch_desc_num_reqs is not None else num_reqs
            if query_start_loc.np[num_reqs_padded] < num_tokens_padded:
                query_start_loc.np[num_reqs_padded + 1] = num_tokens_padded
                num_reqs_padded = num_reqs_padded + 1
            query_start_loc.copy_to_gpu()
            return num_reqs_padded

        runner._pad_query_start_loc_for_fia = _pad_query_start_loc_for_fia
        return runner

    def test_eager_mode_downgrades_to_none(self):
        """When use_cuda_graph=False, aclgraph_runtime_mode is downgraded to NONE."""
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        proposer = self._make_proposer_with_graph_support(
            max_num_tokens=max_num_tokens,
            num_reqs=num_reqs,
            block_size=block_size,
            use_cuda_graph=False,
        )
        proposer.runner = self._make_runner_mock(num_reqs)

        # Override _runnable to track how it's called
        captured_kwargs = {}

        def capture_runnable(**kwargs):
            captured_kwargs.update(kwargs)

        proposer._runnable = capture_runnable
        proposer._dflash_num_context = 0

        with (
            patch("vllm_ascend.spec_decode.dspark_proposer.set_ascend_forward_context"),
            patch("vllm_ascend.spec_decode.dspark_proposer.get_forward_context") as mock_gfc,
        ):
            mock_gfc.return_value = MagicMock(cudagraph_runtime_mode=CUDAGraphMode.NONE)
            proposer.dummy_run(
                num_tokens=num_reqs * block_size,
                num_reqs=num_reqs,
                aclgraph_runtime_mode=CUDAGraphMode.FULL,
            )

        # In eager mode, _runnable should receive empty metadata
        assert captured_kwargs["multi_steps_attn_metadata"] == []

    def test_full_mode_builds_drafting_metadata(self):
        """When use_cuda_graph=True and FULL mode, drafting metadata is built."""
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        proposer = self._make_proposer_with_graph_support(
            max_num_tokens=max_num_tokens,
            num_reqs=num_reqs,
            block_size=block_size,
            use_cuda_graph=True,
        )
        proposer.runner = self._make_runner_mock(num_reqs)

        # Mock the metadata builder
        mock_metadata = MagicMock()
        mock_metadata.attn_mask = MagicMock()
        mock_builder = MagicMock()
        mock_builder.build_for_graph_capture.return_value = mock_metadata
        proposer.draft_attn_groups[0].get_metadata_builder = MagicMock(return_value=mock_builder)

        captured_kwargs = {}

        def capture_runnable(**kwargs):
            captured_kwargs.update(kwargs)

        proposer._runnable = capture_runnable
        proposer._dflash_num_context = 0
        # Need _update_full_graph_params to not fail
        proposer.update_stream = MagicMock()

        with (
            patch("vllm_ascend.spec_decode.dspark_proposer.set_ascend_forward_context"),
            patch("vllm_ascend.spec_decode.dspark_proposer.get_forward_context") as mock_gfc,
            patch("vllm_ascend.spec_decode.dspark_proposer._EXTRA_CTX", capturing=True),
        ):
            mock_gfc.return_value = MagicMock(cudagraph_runtime_mode=CUDAGraphMode.FULL)
            proposer.dummy_run(
                num_tokens=num_reqs * block_size,
                num_reqs=num_reqs,
                aclgraph_runtime_mode=CUDAGraphMode.FULL,
            )

        # build_for_graph_capture should have been called (DSpark dense GQA
        # uses the same capture entry point as DFlash).
        mock_builder.build_for_graph_capture.assert_called_once()
        call_args = mock_builder.build_for_graph_capture.call_args
        # Second positional arg should be AscendAttentionState.ChunkedPrefill
        assert len(call_args.args) > 1
        assert call_args.args[1] == AscendAttentionState.ChunkedPrefill

        # _runnable should receive non-empty metadata
        assert len(captured_kwargs["multi_steps_attn_metadata"]) == 1

    def test_full_mode_uses_num_query_per_req(self):
        """DSpark uses self.num_query_per_req (not 1+N like dFlash)."""
        num_reqs, block_size, max_num_tokens = 4, 3, 256
        # sample_from_anchor=True => num_query_per_req = N = block_size
        proposer = self._make_proposer_with_graph_support(
            max_num_tokens=max_num_tokens,
            num_reqs=num_reqs,
            block_size=block_size,
            use_cuda_graph=True,
        )
        assert proposer.num_query_per_req == block_size  # anchor-first: N

        proposer.runner = self._make_runner_mock(num_reqs)

        mock_metadata = MagicMock()
        mock_builder = MagicMock()
        mock_builder.build_for_graph_capture.return_value = mock_metadata
        proposer.draft_attn_groups[0].get_metadata_builder = MagicMock(return_value=mock_builder)

        proposer._runnable = lambda **kw: None
        proposer._dflash_num_context = 0

        num_tokens = num_reqs * block_size  # = 12, matches num_reqs * num_query_per_req
        with (
            patch("vllm_ascend.spec_decode.dspark_proposer.set_ascend_forward_context"),
            patch("vllm_ascend.spec_decode.dspark_proposer.get_forward_context"),
        ):
            proposer.dummy_run(
                num_tokens=num_tokens,
                num_reqs=num_reqs,
                aclgraph_runtime_mode=CUDAGraphMode.FULL,
            )

        # Check that build_for_graph_capture received common_attn_metadata with
        # DSpark's num_query_per_req geometry.
        # When num_tokens == num_reqs * num_query_per_req (no padding gap),
        # query_start_loc is arange * num_query_per_req (no dummy request).
        call_args = mock_builder.build_for_graph_capture.call_args
        common = call_args.args[0] if call_args.args else call_args.kwargs.get("common_attn_metadata")
        expected_qsl = torch.arange(num_reqs + 1, dtype=torch.int32) * block_size
        assert torch.equal(common.query_start_loc, expected_qsl)

    def test_bonus_anchor_uses_N_plus_1(self):
        """With bonus-anchor, num_query_per_req = N + 1."""
        num_reqs, block_size, max_num_tokens = 4, 3, 256
        proposer = self._make_proposer_with_graph_support(
            max_num_tokens=max_num_tokens,
            num_reqs=num_reqs,
            block_size=block_size,
            hf_config=SimpleNamespace(dspark_bonus_anchor=True),
            use_cuda_graph=True,
        )
        assert proposer.num_query_per_req == 1 + block_size

        proposer.runner = self._make_runner_mock(num_reqs)

        mock_metadata = MagicMock()
        mock_builder = MagicMock()
        mock_builder.build_for_graph_capture.return_value = mock_metadata
        proposer.draft_attn_groups[0].get_metadata_builder = MagicMock(return_value=mock_builder)

        proposer._runnable = lambda **kw: None
        proposer._dflash_num_context = 0

        num_tokens = num_reqs * (1 + block_size)  # = 16, matches num_query_total
        with (
            patch("vllm_ascend.spec_decode.dspark_proposer.set_ascend_forward_context"),
            patch("vllm_ascend.spec_decode.dspark_proposer.get_forward_context"),
        ):
            proposer.dummy_run(
                num_tokens=num_tokens,
                num_reqs=num_reqs,
                aclgraph_runtime_mode=CUDAGraphMode.FULL,
            )

        call_args = mock_builder.build_for_graph_capture.call_args
        common = call_args.args[0] if call_args.args else call_args.kwargs.get("common_attn_metadata")
        # When num_tokens == num_reqs * num_query_per_req, no padding gap
        expected_qsl = torch.arange(num_reqs + 1, dtype=torch.int32) * (1 + block_size)
        assert torch.equal(common.query_start_loc, expected_qsl)

    def test_full_mode_sets_chunked_prefill_state(self):
        """Capture metadata must use ChunkedPrefill state."""
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        proposer = self._make_proposer_with_graph_support(
            max_num_tokens=max_num_tokens,
            num_reqs=num_reqs,
            block_size=block_size,
            use_cuda_graph=True,
        )
        proposer.runner = self._make_runner_mock(num_reqs)

        mock_metadata = MagicMock()
        mock_builder = MagicMock()
        mock_builder.build_for_graph_capture.return_value = mock_metadata
        proposer.draft_attn_groups[0].get_metadata_builder = MagicMock(return_value=mock_builder)

        proposer._runnable = lambda **kw: None
        proposer._dflash_num_context = 0

        with (
            patch("vllm_ascend.spec_decode.dspark_proposer.set_ascend_forward_context"),
            patch("vllm_ascend.spec_decode.dspark_proposer.get_forward_context"),
        ):
            proposer.dummy_run(
                num_tokens=num_reqs * block_size,
                num_reqs=num_reqs,
                aclgraph_runtime_mode=CUDAGraphMode.FULL,
            )

        # attn_state should be set to ChunkedPrefill on the metadata
        assert mock_metadata.attn_state == AscendAttentionState.ChunkedPrefill
        # attn_mask should be cleared (DSpark is non-causal)
        assert mock_metadata.attn_mask is None

    def test_per_group_metadata_for_multiple_groups(self):
        """With multiple draft KV groups, each group gets its own metadata."""
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        proposer = self._make_proposer_with_graph_support(
            max_num_tokens=max_num_tokens,
            num_reqs=num_reqs,
            block_size=block_size,
            use_cuda_graph=True,
        )
        proposer.runner = self._make_runner_mock(num_reqs)

        # Add a second draft attention group
        gid1 = proposer.draft_attn_groups[0].kv_cache_group_id
        gid2 = gid1 + 1
        second_group = SimpleNamespace(
            kv_cache_group_id=gid2,
            kv_cache_spec=SimpleNamespace(block_size=block_size),
            layer_names=["L1"],
        )
        mock_builder2 = MagicMock()
        mock_metadata2 = MagicMock()
        mock_builder2.build_for_graph_capture.return_value = mock_metadata2
        second_group.get_metadata_builder = MagicMock(return_value=mock_builder2)

        proposer.draft_attn_groups = [proposer.draft_attn_groups[0], second_group]
        proposer._per_group_query_slot_mapping_buffers[gid2] = torch.zeros(max_num_tokens, dtype=torch.int32)
        proposer._per_group_block_table_buffers[gid2] = torch.zeros((num_reqs, 16), dtype=torch.int32)

        # Setup first builder mock too
        mock_metadata1 = MagicMock()
        mock_builder1 = MagicMock()
        mock_builder1.build_for_graph_capture.return_value = mock_metadata1
        proposer.draft_attn_groups[0].get_metadata_builder = MagicMock(return_value=mock_builder1)

        captured_kwargs = {}

        def capture_runnable(**kwargs):
            captured_kwargs.update(kwargs)

        proposer._runnable = capture_runnable
        proposer._dflash_num_context = 0

        with (
            patch("vllm_ascend.spec_decode.dspark_proposer.set_ascend_forward_context"),
            patch("vllm_ascend.spec_decode.dspark_proposer.get_forward_context"),
        ):
            proposer.dummy_run(
                num_tokens=num_reqs * block_size,
                num_reqs=num_reqs,
                aclgraph_runtime_mode=CUDAGraphMode.FULL,
            )

        # Both builders should have been called
        mock_builder1.build_for_graph_capture.assert_called_once()
        mock_builder2.build_for_graph_capture.assert_called_once()

        # The per_layer_attn_metadata should contain layers from both groups
        metadata = captured_kwargs["multi_steps_attn_metadata"][0]
        # Should have entries for layers from both groups
        assert "L0" in metadata
        assert "L1" in metadata

    def test_full_mode_separates_context_and_query_graph_widths(self):
        """The target captures 1+N context tokens while DSpark captures N queries."""
        num_reqs, num_spec, max_num_tokens = 2, 7, 256
        # sample_from_anchor=True => num_query_per_req = N = 7
        # num_query_total = 2*7 = 14, but capture bucket = 16 (1+N=8 per req * 2)
        proposer = self._make_proposer_with_graph_support(
            max_num_tokens=max_num_tokens,
            num_reqs=num_reqs,
            block_size=num_spec,
            use_cuda_graph=True,
        )
        assert proposer.num_query_per_req == num_spec

        proposer.runner = self._make_runner_mock(num_reqs)

        mock_metadata = MagicMock()
        mock_builder = MagicMock()
        mock_builder.build_for_graph_capture.return_value = mock_metadata
        proposer.draft_attn_groups[0].get_metadata_builder = MagicMock(return_value=mock_builder)

        captured_kwargs = {}

        def capture_runnable(**kwargs):
            captured_kwargs.update(kwargs)

        proposer._runnable = capture_runnable
        proposer._dflash_num_context = 0

        # The target descriptor contains 16 verification/context tokens, but
        # the native anchor-first DSpark graph contains 14 query tokens.
        target_num_tokens = num_reqs * (1 + num_spec)  # = 16
        descriptor = BatchDescriptor(
            num_tokens=target_num_tokens,
            num_reqs=num_reqs,
            uniform=True,
        )
        with (
            patch("vllm_ascend.spec_decode.dspark_proposer.set_ascend_forward_context"),
            patch("vllm_ascend.spec_decode.dspark_proposer.get_forward_context"),
        ):
            proposer.dummy_run(
                num_tokens=target_num_tokens,
                num_reqs=num_reqs,
                aclgraph_runtime_mode=CUDAGraphMode.FULL,
                batch_descriptor=descriptor,
            )

        call_args = mock_builder.build_for_graph_capture.call_args
        common = call_args.args[0] if call_args.args else call_args.kwargs.get("common_attn_metadata")
        assert common.query_start_loc.tolist() == [0, 7, 14]
        assert common.num_reqs == num_reqs
        assert common.num_input_tokens == num_reqs * num_spec
        assert captured_kwargs["num_input_tokens"] == num_reqs * num_spec
        assert captured_kwargs["num_tokens"] == num_reqs * num_spec
        assert proposer._dflash_num_context == target_num_tokens


class TestBuildDraftAttnMetadataQueryStartLocPadding(_DSparkProposerTestBase):
    """If a larger batch/DP bucket pads ``query_start_loc`` before
    ``build_draft_attn_metadata`` is invoked, the DSpark branch must pass the
    resulting layout through without modification."""

    def test_query_start_loc_passed_through_in_graph_mode(self):
        """An already padded query_start_loc is preserved."""
        num_reqs, num_spec = 2, 7
        proposer = self._make_proposer(
            max_num_tokens=256,
            num_reqs=num_reqs,
            block_size=num_spec,
        )
        proposer.use_cuda_graph = True
        proposer.use_compress = False
        proposer.method = "dspark"

        mock_metadata = MagicMock()
        mock_builder = MagicMock()
        mock_builder.build_for_drafting.return_value = mock_metadata
        proposer.draft_attn_groups[0].get_metadata_builder = MagicMock(return_value=mock_builder)

        num_input_tokens = 16  # padded to capture bucket
        num_actual_tokens = num_reqs * num_spec  # = 14

        # Simulate an externally padded fallback layout with one extra request.
        num_reqs_padded = num_reqs + 1
        qsl = torch.tensor([0, 7, 14, 16], dtype=torch.int32)
        qsl_cpu = torch.tensor([0, 7, 14, 16], dtype=torch.int32)
        common = SimpleNamespace(
            num_reqs=num_reqs_padded,
            query_start_loc=qsl,
            query_start_loc_cpu=qsl_cpu,
            block_table_tensor=torch.zeros((num_reqs_padded, 16), dtype=torch.int32),
            num_input_tokens=num_input_tokens,
        )

        proposer.build_draft_attn_metadata(common, num_input_tokens, num_actual_tokens)

        call_args = mock_builder.build_for_drafting.call_args
        meta_common = call_args.args[0] if call_args.args else call_args.kwargs.get("common_attn_metadata")
        # query_start_loc is passed through unchanged (already padded).
        assert meta_common.query_start_loc[num_reqs_padded].item() == num_input_tokens

    def test_query_start_loc_not_modified_when_already_matches(self):
        """When query_start_loc already matches, no modification occurs."""
        num_reqs, num_spec = 2, 7
        proposer = self._make_proposer(
            max_num_tokens=256,
            num_reqs=num_reqs,
            block_size=num_spec,
        )
        proposer.use_cuda_graph = True
        proposer.use_compress = False
        proposer.method = "dspark"

        mock_metadata = MagicMock()
        mock_builder = MagicMock()
        mock_builder.build_for_drafting.return_value = mock_metadata
        proposer.draft_attn_groups[0].get_metadata_builder = MagicMock(return_value=mock_builder)

        # Simulate bonus_anchor where num_query_per_req = N+1 = 8,
        # and num_input_tokens = 16 = 2*8 (no padding gap, no dummy request).
        num_query_per_req = num_spec + 1
        num_input_tokens = num_reqs * num_query_per_req  # = 16
        num_actual_tokens = num_input_tokens

        qsl = torch.arange(num_reqs + 1, dtype=torch.int32) * num_query_per_req  # [0, 8, 16]
        qsl_cpu = qsl.clone()
        common = SimpleNamespace(
            num_reqs=num_reqs,
            query_start_loc=qsl,
            query_start_loc_cpu=qsl_cpu,
            block_table_tensor=torch.zeros((num_reqs, 16), dtype=torch.int32),
            num_input_tokens=num_input_tokens,
        )

        proposer.build_draft_attn_metadata(common, num_input_tokens, num_actual_tokens)

        call_args = mock_builder.build_for_drafting.call_args
        meta_common = call_args.args[0] if call_args.args else call_args.kwargs.get("common_attn_metadata")
        # No modification needed — last element already equals num_input_tokens.
        assert meta_common.query_start_loc[num_reqs].item() == num_input_tokens

    def test_no_padding_in_eager_mode(self):
        """In eager mode, query_start_loc is not padded."""
        num_reqs, num_spec = 2, 7
        proposer = self._make_proposer(
            max_num_tokens=256,
            num_reqs=num_reqs,
            block_size=num_spec,
        )
        proposer.use_cuda_graph = False
        proposer.use_compress = False
        proposer.method = "dspark"

        mock_metadata = MagicMock()
        mock_builder = MagicMock()
        mock_builder.build_for_drafting.return_value = mock_metadata
        proposer.draft_attn_groups[0].get_metadata_builder = MagicMock(return_value=mock_builder)

        num_input_tokens = 16
        num_actual_tokens = 14

        qsl = torch.arange(num_reqs + 1, dtype=torch.int32) * num_spec  # [0, 7, 14]
        qsl_cpu = qsl.clone()
        common = SimpleNamespace(
            num_reqs=num_reqs,
            query_start_loc=qsl,
            query_start_loc_cpu=qsl_cpu,
            block_table_tensor=torch.zeros((num_reqs, 16), dtype=torch.int32),
            num_input_tokens=num_input_tokens,
        )

        proposer.build_draft_attn_metadata(common, num_input_tokens, num_actual_tokens)

        call_args = mock_builder.build_for_drafting.call_args
        meta_common = call_args.args[0] if call_args.args else call_args.kwargs.get("common_attn_metadata")
        # In eager mode, query_start_loc should NOT be padded.
        assert meta_common.query_start_loc[num_reqs].item() == num_reqs * num_spec  # = 14
