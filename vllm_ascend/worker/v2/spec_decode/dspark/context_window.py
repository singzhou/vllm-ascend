#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Windowed draft-context reads for the DSpark speculator.

The draft keeps writing its KV at absolute positions, so the cache still holds
every drafted token (full KV retention). Only the *reads* are restricted: the
draft block attends to an aligned window that ends at the block itself, which
is the context the draft was trained with.

Window definition, shared token-for-token with the SpecForge training side:

* ``anchor`` is the absolute position of the first query slot of the draft
  block, and ``B`` the block length. The draft input-preparation kernel writes
  ``seq_len = last_valid_pos + 1 + B``, hence ``anchor = seq_len - B``.
* ``start = floor(max(anchor - W, 0) / align) * align``.
* The draft reads ``[start, seq_len)``: the context part ``[start, anchor)``
  plus the ``B`` positions of its own block. Intra-block attention stays
  bidirectional.

``align`` must be a multiple of the draft KV group's kernel block size so that
``start`` always falls on a block boundary and the block table can simply be
re-based at ``start // kernel_block_size``.

The one place where ``anchor`` is approximate is a request that has reached
``max_model_len``: the preparation kernel clamps ``seq_len`` there, so the
anchor derived from it is *smaller* than the real one and the window can
start up to one ``align`` step early. Such a request then reads a little more
context than training did -- never unwritten or evicted KV, since everything
below ``seq_len`` has been written -- and only in the last block before the
model length. Recovering the exact anchor would mean carrying the unclamped
position out of the preparation kernel; until then this is a known, bounded
deviation from the training definition.
"""

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class DraftContextWindowConfig:
    """``dflash_config.context_window`` / ``context_window_align`` of a draft."""

    context_window: int
    context_window_align: int


def _nested_config_value(section: object, name: str) -> object:
    if isinstance(section, Mapping):
        return section.get(name)
    return getattr(section, name, None)


def read_context_window_config(draft_hf_config: object) -> DraftContextWindowConfig | None:
    """Read the draft's context-window settings, or None when it has none.

    A checkpoint without both fields must behave exactly as before, so the
    absence of the whole pair disables the feature; a half-configured
    checkpoint is rejected instead of silently falling back to a default
    window that the draft was never trained with.
    """
    section = getattr(draft_hf_config, "dflash_config", None)
    window = _nested_config_value(section, "context_window")
    align = _nested_config_value(section, "context_window_align")
    if window is None and align is None:
        return None
    if window is None or align is None:
        raise ValueError(
            "DSpark context windowing needs both dflash_config.context_window and "
            f"dflash_config.context_window_align: got {window=}, {align=}."
        )
    if isinstance(window, bool) or isinstance(align, bool):
        raise ValueError("dflash_config.context_window and context_window_align must be integers.")
    window, align = int(window), int(align)
    if window <= 0 or align <= 0:
        raise ValueError(
            f"dflash_config.context_window ({window}) and context_window_align ({align}) must be positive."
        )
    return DraftContextWindowConfig(window, align)


def window_start(
    anchor: torch.Tensor, context_window: int, align: int, *, out: torch.Tensor | None = None
) -> torch.Tensor:
    """First absolute position the draft block may read, aligned down.

    Pure arithmetic on the device tensor: no host synchronization. The scalar
    equivalent is ``max(anchor - W, 0) // align * align``; both sides of the
    training/inference pair must agree on it exactly.
    """
    shifted = torch.sub(anchor, context_window, out=out)
    shifted.clamp_(min=0)
    return shifted.div_(align, rounding_mode="floor").mul_(align)


class DraftContextWindow:
    """Resident windowed block tables and KV lengths for the draft layers.

    Every buffer here is allocated once and written in place, because FULL
    graph replay reads the captured addresses.
    """

    def __init__(
        self,
        config: DraftContextWindowConfig,
        *,
        group_ids: Sequence[int],
        kernel_block_sizes: Sequence[int],
        full_block_tables: Sequence[torch.Tensor],
        block_length: int,
        device: torch.device,
    ) -> None:
        if not group_ids:
            raise ValueError("DSpark context windowing requires at least one draft KV cache group.")
        self.config = config
        self.block_length = block_length
        self.group_ids = tuple(group_ids)
        self.kernel_block_sizes = {gid: int(kernel_block_sizes[gid]) for gid in self.group_ids}
        for gid, kernel_block_size in self.kernel_block_sizes.items():
            if kernel_block_size <= 0 or config.context_window_align % kernel_block_size:
                raise ValueError(
                    f"dflash_config.context_window_align ({config.context_window_align}) must be an integer "
                    f"multiple of the draft KV cache group {gid} kernel block size ({kernel_block_size}); "
                    "otherwise the window start is not a block boundary."
                )

        # An aligned start sits at most one ``align`` step below ``anchor - W``,
        # so a windowed length never exceeds W + align + B.
        widest_window = config.context_window + config.context_window_align + block_length

        max_num_reqs = full_block_tables[self.group_ids[0]].shape[0]
        self.windowed_seq_lens = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self._anchors = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self._starts = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self._full_block_tables = {gid: full_block_tables[gid] for gid in self.group_ids}
        self.block_table_clones: dict[int, torch.Tensor] = {}
        self._column_offsets: dict[int, torch.Tensor] = {}
        clone_capacity = widest_window
        for gid in self.group_ids:
            full_block_table = self._full_block_tables[gid]
            kernel_block_size = self.kernel_block_sizes[gid]
            # One spare column so a window that starts mid-table still covers
            # the widest window in full.
            num_cols = -(-widest_window // kernel_block_size) + 1
            # A table that cannot even hold the window (a short max_model_len)
            # caps the clone, and with it every length handed to the kernels.
            num_cols = min(num_cols, full_block_table.shape[1])
            clone_capacity = min(clone_capacity, num_cols * kernel_block_size)
            self.block_table_clones[gid] = torch.zeros(
                (max_num_reqs, num_cols), dtype=full_block_table.dtype, device=device
            )
            self._column_offsets[gid] = torch.arange(num_cols, dtype=torch.int32, device=device).unsqueeze(0)

        # Nothing downstream may claim more KV than the narrowest clone holds:
        # this bounds the device lengths, the host upper bound, the batch
        # maximum and the length graph capture is tiled for.
        self.max_windowed_seq_len = min(widest_window, clone_capacity)
        # What the *context* (pre-block) part of a windowed length can reach;
        # the draft block's own B positions are added on top of it upstream.
        self.max_windowed_context_len = max(self.max_windowed_seq_len - block_length, 0)

        # The draft groups' tables are replaced; every other group keeps its own.
        self.windowed_input_block_tables = [
            self.block_table_clones.get(gid, table) for gid, table in enumerate(full_block_tables)
        ]

    @contextmanager
    def swap_in(self, input_buffers: Any, block_tables: Any) -> Iterator[None]:
        """Point the draft's metadata inputs at the windowed buffers.

        Recomputes the window from the draft's current KV lengths, then swaps
        the two tensors the upstream metadata build reads -- the device
        ``seq_lens`` and the per-group block tables -- for the resident
        windowed ones. Slot mappings are deliberately left alone: the draft
        still writes its KV at absolute positions.
        """
        self.apply(input_buffers.seq_lens)
        original_seq_lens = input_buffers.seq_lens
        original_block_tables = block_tables.input_block_tables
        input_buffers.seq_lens = self.windowed_seq_lens
        block_tables.input_block_tables = self.windowed_input_block_tables
        try:
            yield
        finally:
            input_buffers.seq_lens = original_seq_lens
            block_tables.input_block_tables = original_block_tables

    def windowed_seq_lens_cpu_upper_bound(self, seq_lens_cpu_upper_bound: torch.Tensor) -> torch.Tensor:
        """Cap the host-side *context* upper bound to what the window exposes.

        The caller's buffer belongs to the input batch, so this returns a new
        tensor. The bound stays an upper bound (never shorter than the exact
        windowed length), which is what FIA needs when it falls back to host
        lengths; DSpark's exact per-request lengths come from
        :attr:`windowed_seq_lens`.
        """
        return torch.clamp(seq_lens_cpu_upper_bound, max=self.max_windowed_context_len)

    def apply(self, seq_lens: torch.Tensor) -> None:
        """Recompute the windowed lengths and block tables from ``seq_lens``.

        ``seq_lens`` is the draft's own absolute KV length per request, as
        written by the input-preparation kernel; padded request slots hold 0
        and stay at 0 here.
        """
        config = self.config
        anchors = self._anchors
        torch.sub(seq_lens, self.block_length, out=anchors)
        anchors.clamp_(min=0)
        starts = window_start(anchors, config.context_window, config.context_window_align, out=self._starts)
        torch.sub(seq_lens, starts, out=self.windowed_seq_lens)
        self.windowed_seq_lens.clamp_(min=0)

        for gid in self.group_ids:
            full_block_table = self._full_block_tables[gid]
            clone = self.block_table_clones[gid]
            num_full_cols = full_block_table.shape[1]
            src_cols = (
                torch.div(starts, self.kernel_block_sizes[gid], rounding_mode="floor").unsqueeze(1)
                + self._column_offsets[gid]
            )
            # A window whose start block sits near the end of the table has
            # fewer real columns than the clone: clamp the gather and blank the
            # columns past the table so no stale block id is read.
            in_range = src_cols < num_full_cols
            gathered = torch.gather(full_block_table, 1, src_cols.clamp_(max=num_full_cols - 1).to(torch.int64))
            clone.copy_(torch.where(in_range, gathered, torch.zeros_like(gathered)))
