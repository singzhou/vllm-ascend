# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual pure-Python planner without importing NPU dependencies.

Extract only the layout/planning functions; the stand-ins model the cache
descriptor interfaces, not the allocation algorithm under test.
"""

import ast
import unittest
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

SOURCE = Path(__file__).resolve().parents[4] / "vllm_ascend/patch/platform/patch_kv_cache_utils.py"


@dataclass
class UniformSpec:
    kv_cache_specs: dict
    pages: int

    def get_max_layers_per_page_size(self):
        return max(Counter(spec.page_size_bytes for spec in self.kv_cache_specs.values()).values())

    def max_memory_usage_pages(self, config):
        return self.pages


@dataclass
class TensorDescriptor:
    size: int
    layers: list
    offset: int
    layer_stride: int
    block_stride: int


def load_planner():
    names = {
        "_page_sizes",
        "_get_max_layers_per_page_size",
        "_get_deepseek_v4_cache_layout",
        "_get_kv_cache_config_deepseek_v4_main",
        "_ascend_pool_bytes_per_block",
        "_ascend_max_memory_usage_bytes_from_groups",
    }
    tree = ast.parse(SOURCE.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(functions) == len(names)
    namespace = {
        "defaultdict": defaultdict,
        "UniformTypeKVCacheSpecs": UniformSpec,
        "KVCacheTensor": TensorDescriptor,
        "tq_cache": SimpleNamespace(uses_turboquant_groups=lambda groups: False),
        "is_deepseek_v41_cache": lambda groups: False,
        "_get_glm5_next_cache_layout": lambda groups: None,
        "_is_deepseek_v4_groups": lambda groups: True,
        "may_override_num_blocks": lambda config, blocks: blocks,
    }
    # Postpone production annotations, whose runtime types need torch/vLLM.
    tree = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *functions],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), str(SOURCE), "exec"), namespace)
    return namespace


def make_group(layers, pages):
    return SimpleNamespace(
        layer_names=list(layers),
        kv_cache_spec=UniformSpec(
            {name: SimpleNamespace(page_size_bytes=size) for name, size in layers.items()}, pages
        ),
    )


def groups_with_draft(prefix="mtp"):
    full = {f"target.indexer.{i}": 64 for i in range(8)}
    full.update({f"target.attn.{i}": 128 for i in range(4)})
    full.update({f"{prefix}.{i}.attn": 128 for i in range(3)})
    return [make_group(full, 8), make_group({f"target.swa.{i}": 128 for i in range(6)}, 5)]


class TestDSV4TupleCapacity(unittest.TestCase):
    def setUp(self):
        self.planner = load_planner()

    def test_startup_budget_materializes_enough_blocks(self):
        groups = groups_with_draft()
        required_blocks = sum(group.kv_cache_spec.pages for group in groups)
        budget = self.planner["_ascend_max_memory_usage_bytes_from_groups"](None, groups)
        # The old layout reserved 8 normal + 3 MTP tuples and admitted this
        # budget at startup, yet produced only 9 blocks for a 13-block request.
        self.assertLess(budget // ((64 + 128) * (8 + 3)), required_blocks)
        blocks, tensors = self.planner["_get_kv_cache_config_deepseek_v4_main"](None, groups, budget)
        self.assertEqual(blocks, required_blocks)
        self.assertTrue(all(tensor.size == budget for tensor in tensors))
        # Upstream reserves a null block before checking capacity.
        stride = self.planner["_ascend_pool_bytes_per_block"](groups)
        blocks, _ = self.planner["_get_kv_cache_config_deepseek_v4_main"](None, groups, budget + stride)
        self.assertEqual(blocks - 1, required_blocks)

    def test_layer_addresses_are_disjoint_for_live_group_blocks(self):
        groups = groups_with_draft()
        budget = self.planner["_ascend_max_memory_usage_bytes_from_groups"](None, groups)
        blocks, tensors = self.planner["_get_kv_cache_config_deepseek_v4_main"](None, groups, budget)
        views = {name: (tensor, index) for tensor in tensors for index, name in enumerate(tensor.layers)}
        self.assertEqual(set(views), {name for group in groups for name in group.layer_names})
        regions = []
        first_block = 0
        for group in groups:
            for name in group.layer_names:
                tensor, index = views[name]
                size = group.kv_cache_spec.kv_cache_specs[name].page_size_bytes
                for block in range(first_block, first_block + group.kv_cache_spec.pages):
                    start = tensor.offset + index * tensor.layer_stride + block * tensor.block_stride
                    self.assertGreaterEqual(start, 0)
                    self.assertLessEqual(start + size, tensor.size)
                    regions.append((start, start + size, name))
            first_block += group.kv_cache_spec.pages
        self.assertEqual(first_block, blocks)
        regions.sort()
        for left, right in zip(regions, regions[1:]):
            self.assertLessEqual(left[1], right[0], (left, right))

    def test_draft_namespace_does_not_change_capacity(self):
        mtp = groups_with_draft("mtp")
        draft = groups_with_draft("draft")
        size = self.planner["_ascend_pool_bytes_per_block"]
        self.assertEqual(size(mtp), size(draft))

    def test_draft_can_grow_a_full_bucket(self):
        groups = [make_group({f"target.{i}": 64 for i in range(8)} | {"mtp.0": 64, "mtp.1": 64}, 4)]
        size = self.planner["_ascend_pool_bytes_per_block"](groups)
        self.assertEqual(size, 10 * 64)

    def test_target_only_layout_keeps_existing_capacity(self):
        groups = [make_group({"target.a": 64, "target.b": 128}, 4)]
        size = self.planner["_ascend_pool_bytes_per_block"](groups)
        self.assertEqual(size, 64 + 128)
        blocks, _ = self.planner["_get_kv_cache_config_deepseek_v4_main"](None, groups, 10 * size)
        self.assertEqual(blocks, 10)


if __name__ == "__main__":
    unittest.main()
