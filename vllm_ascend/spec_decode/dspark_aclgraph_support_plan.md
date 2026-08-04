# Qwen3.6 DSpark ACLGraph 支持开发计划

## 1. 目标与范围

当前 `vllm_ascend/spec_decode/dspark_proposer.py` 无条件设置 `self.use_cuda_graph = False`，因此 Qwen3.6 DSpark 小模型只能 eager 执行。本计划的目标是让 **Qwen3.6 主模型和 DSpark 小模型同时入图**。

本次范围严格限定为：

- Qwen3.6 DSpark；
- Qwen3.6 主模型可能包含 GDN 的 hybrid KV cache 场景；
- v1 `AscendDSparkProposer` 路径；
- ACLGraph `FULL`/`FULL_DECODE_ONLY` 中实际派发给 dFlash/DSpark 的 full-graph 路径；
- greedy DSpark（继续不支持 probabilistic sampling）。

本次不考虑：

- DeepSeek V4 DSpark；
- 为 DSpark 新增独立 dispatcher、独立配置体系或额外环境变量；
- 主/小模型 eager/graph 的四种排列组合；
- 与 Qwen3.6 无关的多 backend、DSA、SFA-CP 或其他模型扩展；
- 重构通用 speculative decoding 框架。

## 2. 总体原则

### 2.1 配置和开关与 dFlash 完全对齐

DSpark 不新增任何 graph 配置或开关，直接复用 dFlash/基类现有逻辑：

```python
self.use_cuda_graph = (
    self.runner._use_aclgraph()
    and not self.speculative_config.enforce_eager
)
```

实现只删除 DSpark 对 `self.use_cuda_graph = False` 的强制覆盖，不重新计算、不改写基类判断。

因此用户配置、capture sizes、dispatcher、`BatchDescriptor`、DP padding、`ACLGraphWrapper` 和 draft graph params 初始化方式都与 dFlash 保持一致。

具体沿用 dFlash 已有开关语义：`compilation_config.mode`、`compilation_config.cudagraph_mode`、目标模型 `enforce_eager`、`speculative_config.enforce_eager` 和 `disable_padded_drafter_batch`；DSpark 不增加特例。

### 2.2 执行流程与 dFlash 完全对齐

DSpark 按 dFlash 的既有时序执行：

```text
启动/capture：
runner dummy_run
  -> drafter.dummy_run（沿用 runner 派发的 mode/descriptor）
  -> 构造小模型 capture metadata
  -> set_ascend_forward_context
  -> _runnable（ACLGraphWrapper capture/replay）
  -> 非 capturing 时更新 draft full-graph params

实际推理：
set_inputs_first_pass
  -> 基类现有 padding/dispatcher/DP sync
  -> 构造 draft attention metadata
  -> set_ascend_forward_context
  -> _runnable（ACLGraphWrapper replay）
  -> 更新 draft full-graph params
  -> 返回 draft tokens
```

不为 DSpark 增加独立 dispatcher 或 capture 生命周期，也不修改主模型图执行路径。

### 2.3 只保留 DSpark 必需差异

DSpark 与 dFlash 的实现差异仅限于：

1. 每请求 query 数使用 `self.num_query_per_req`：
   - `sample_from_anchor=True`：`N`；
   - bonus-anchor：`N + 1`。
2. Qwen3.6 hybrid KV manager 下，draft attention layer 可能需要按现有 KV cache group 取得 block table/slot mapping。
3. 小模型本身没有 GDN；`has_gdn` 仅用于处理 Qwen3.6 全局 BlockTable 中 manager block size 与 draft attention kernel block size 的差异。
4. DSpark 正常路径使用 `build_for_drafting(..., draft_index=1)`。capture metadata 也必须保持相同 drafting 语义，不能机械照抄 dFlash 的普通 `build_for_graph_capture`。
5. DSpark 的 MarkovHead/argmax/in-place token 更新保留现有实现，不做额外抽象或重构。

## 3. 最小实现方案

### 阶段一：启用图开关并补齐 dummy capture 流程

主要修改 `vllm_ascend/spec_decode/dspark_proposer.py`：

1. 删除以下强制 eager 逻辑：

   ```python
   self.use_cuda_graph = False
   ```

2. 参照 `AscendDflashProposer.dummy_run`，在 DSpark `dummy_run` 中补齐：
   - `multi_steps_attn_metadata`；
   - FULL 模式下的 capture metadata 构造；
   - 将首步 metadata 传给 `set_ascend_forward_context`；
   - 将完整 metadata 列表传给 `_runnable` 和 `draft_attn_metadatas`；
   - `_runnable` 后在非 capturing replay 阶段调用 `_update_full_graph_params`。
3. 继续复用 dFlash/基类已有的：
   - `runner._sync_metadata_across_dp`；
   - `aclgraph_runtime_mode` 和 `batch_descriptor`；
   - `ACLGraphWrapper`；
   - `update_stream`；
   - draft graph params namespace。
4. 保留现有 `_pad_draft_buffers(num_query_total, num_input_tokens)`，确保 capture bucket 的 padding 区在 metadata 构造前已清零。

capture metadata 使用现有预分配 buffer，并与真实 DSpark 路径保持一致：

- `query_start_loc = arange * self.num_query_per_req`；
- `max_query_len = self.num_query_per_req`；
- `positions = self.positions`；
- `causal = False`；
- `attn_state = ChunkedPrefill`；
- query/context slot mapping 使用 DSpark 已有 per-group buffer；
- block table 使用对应 draft cache group 的 device tensor。

metadata builder 优先直接复用：

```python
builder.build_for_drafting(common_attn_metadata, draft_index=1)
```

只有在 Qwen3.6 实际 capture 证明该接口缺少必要的 capture 初始化时，才对 Qwen3.6 使用的 builder 增加一个很薄的 capture wrapper；不预先引入新的通用接口。

### 阶段二：保留 Qwen3.6 GDN 场景所需的 KV 映射

不重构现有 DSpark per-group bookkeeping，只检查并复用以下逻辑：

1. `initialize_attn_backend` 只筛选 DSpark draft layer。
2. 每个 draft group 继续使用自己的 block table、query slot mapping 和 context slot mapping。
3. capture metadata 仅绑定到该 group 的 draft layer names。
4. slot id 的 block size 沿用现有 Qwen3.6 修复：
   - 目标包含 GDN 时使用 draft attention 的 `kernel_block_size`；
   - 非 GDN 时使用 group KV spec 的 block size。
5. 不把 GDN metadata/backend 传入 DSpark 小模型图；这里只处理全局 BlockTable 的地址单位。

为了控制修改范围：

- 不修改 KV cache group 的生成规则；
- 不新增独立 block table；
- 不修改主模型 attention metadata；
- 不扩展 DSV4/DSA 分支；
- 不处理运行时多个不同 draft backend 的泛化问题。

### 阶段三：最小测试和验证

单元测试集中放在 `tests/ut/spec_decode/test_dspark_proposer.py`：

1. 初始化时不再强制 `use_cuda_graph=False`，结果与 dFlash/基类开关一致。
2. `use_cuda_graph=False` 时 `dummy_run` 仍降级为 `CUDAGraphMode.NONE`。
3. FULL 模式下：
   - 构造一份 DSpark drafting metadata；
   - forward context 收到相同 metadata；
   - `_runnable` 收到相同 `multi_steps_attn_metadata`；
   - capture 阶段不更新 graph params；
   - replay 阶段调用一次 `_update_full_graph_params`。
4. `sample_from_anchor=True/False` 分别验证 `N` 和 `N + 1` query shape。
5. Qwen3.6 GDN 场景验证使用 `kernel_block_size` 计算 slot mapping。
6. 有多个 draft KV group 时，各 group 使用自己的 block table/slot mapping，且 layer metadata 映射正确。
7. eager 现有测试继续通过，防止图支持破坏原路径。

NPU E2E 只验证用户要求的组合：

- Qwen3.6 主模型入图；
- Qwen3.6 DSpark 小模型入图；
- 使用与 dFlash 相同的 compilation/speculative graph 配置方式；
- 固定 prompt 下输出正确，无 NaN、KV 越界或重复 capture；
- 至少覆盖一个小 batch 和一个较大 capture bucket；
- 记录 draft graph capture/replay 日志，确认不是“主模型入图、小模型 eager”。

建议验证命令：

```bash
pytest -sv tests/ut/spec_decode/test_dspark_proposer.py
ruff check vllm_ascend/spec_decode/dspark_proposer.py tests/ut/spec_decode/test_dspark_proposer.py
bash format.sh ci
```

## 4. 预计修改文件

默认只修改：

- `vllm_ascend/spec_decode/dspark_proposer.py`
- `tests/ut/spec_decode/test_dspark_proposer.py`

仅当 Qwen3.6 NPU capture 明确证明 `build_for_drafting` 缺少 capture 初始化时，才最小修改 Qwen3.6 实际使用的 metadata builder。默认不修改：

- `vllm_ascend/spec_decode/llm_base_proposer.py`
- `vllm_ascend/worker/model_runner_v1.py`
- `vllm_ascend/compilation/acl_graph.py`
- DSV4/DSA 相关文件。

## 5. 完成定义

- Qwen3.6 主模型和 DSpark 小模型在同一 graph 配置下均发生 capture/replay。
- DSpark 的配置、开关、dispatcher 和执行时序与 dFlash 对齐。
- 小模型图不执行 GDN；Qwen3.6 GDN 仅影响已有 BlockTable block-size 映射。
- 修改集中在 DSpark proposer 和单测，没有不必要的框架重构。
- eager 回归、DSpark 图单测、Qwen3.6 NPU 图验证、lint 和格式检查通过。
