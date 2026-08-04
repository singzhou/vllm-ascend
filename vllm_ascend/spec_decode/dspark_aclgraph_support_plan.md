# DSpark 独立 ACLGraph 支持开发计划

## 1. 当前结论

`vllm_ascend/spec_decode/dspark_proposer.py` 对应的 v1 DSpark proposer 当前不支持 ACLGraph：

1. `AscendDSparkProposer.__init__` 无条件执行 `self.use_cuda_graph = False`。
2. `dummy_run` 因此会把 `CUDAGraphMode.FULL` 降级为 `NONE`。
3. `dummy_run` 没有构造 DSpark drafting metadata，向 forward context 传入的是 `None`/空列表，也没有完成 draft graph params 的 capture/replay 更新。
4. 现有单元测试明确断言 DSpark 只能 eager。

但原计划把目标模型的 GDN/hybrid cache、主模型 dispatcher 和主模型 capture size 带入了 DSpark 图设计，这个方向不正确。本计划按“**DSpark 草稿图是独立执行单元**”重新设计。

## 2. 架构边界

### 2.1 必须独立的内容

DSpark 草稿图必须独立拥有：

- draft model 权重和算子图；
- draft attention backend 和 metadata builder；
- draft graph dispatcher/capture descriptor；
- draft capture sizes；
- draft graph entry、workspace、task handles 和 graph params；
- draft 输入/输出持久化 buffer；
- draft KV cache layer 到 block table/slot mapping 的映射。

DSpark 图编译和 replay 不得依赖：

- 主模型使用 FIA、MLA、GDN、Mamba 或其他 backend；
- `runner.attn_backend` 或 `runner.attn_groups` 中的主模型 layer/backend；
- 主模型 graph entry、workspace、task handles 或 graph params；
- 主模型的 `BatchDescriptor` 作为 DSpark graph key；
- 目标模型 `has_gdn` 之类的结构判断；
- 主模型 token 数作为 DSpark capture token 数。

### 2.2 合理且不可消除的接口关系

推测解码算法决定了 DSpark 仍需接收：

- 目标模型输出的 hidden states、positions、next token；
- rejected-token 信息和当前请求数；
- scheduler/DP 提供的批次同步结果；
- 全局 KV cache allocator 分配给 **draft cache layer** 的物理 cache tensor、block table 和 slot mapping。

这些是数据输入或基础设施句柄，不表示草稿图与主模型计算图耦合。进入 `_runnable` 前，所有动态输入应复制到 DSpark 自己的持久化 buffer；ACLGraph 只捕获对这些 draft-owned buffer 的读取和写入。

## 3. 复查发现的现有耦合点

### 3.1 图能力开关错误地继承主模型状态

基类当前使用：

```python
self.use_cuda_graph = (
    self.runner._use_aclgraph()
    and not self.speculative_config.enforce_eager
)
```

`runner._use_aclgraph()` 包含目标模型 `model_config.enforce_eager` 等判断。DSpark 是否入图应由以下条件独立决定：

- compilation mode/ACLGraph runtime 可用；
- draft backend 声明支持 full graph；
- `speculative_config.enforce_eager` 为 false；
- DSpark 的 padded batch/capture shape 条件满足。

目标模型是否 eager 不应直接关闭 DSpark 图。若目标 eager 时现有 runner 不执行 capture warmup，应为 drafter 增加独立 capture 入口或允许首次调用安全 lazy capture。

### 3.2 实际 propose 复用了主模型 dispatcher

`AscendSpecDecodeBaseProposer._propose` 当前调用 `runner.cudagraph_dispatcher.dispatch(...)`。这会让 draft graph 的 bucket 选择受主模型 dispatcher 配置约束。

DSpark 应使用自己的 dispatcher，以：

```text
num_draft_tokens = padded_num_reqs * num_query_per_req
```

生成 DSpark 自己的 `BatchDescriptor`。其中 `num_query_per_req` 由 `sample_from_anchor` 决定是 `N` 或 `N + 1`。

### 3.3 dummy capture 复用了主模型 BatchDescriptor

runner 当前完成主模型 dummy forward 后，把主模型 `batch_desc` 直接传给 drafter。对 DSpark 来说，该 descriptor 的 token shape 通常不是 draft query shape。

DSpark `dummy_run` 应只把传入的 `num_reqs` 当作调度输入，然后使用自己的 dispatcher 重新计算 draft descriptor。传入的主模型 descriptor 不得进入 DSpark `ACLGraphWrapper` 的 key。

### 3.4 draft graph params 使用了主模型 capture sizes

runner 当前从主模型 dispatcher 收集 `capture_sizes`，再用同一列表初始化 `set_draft_graph_params(capture_sizes)`。这不保证包含 `num_reqs * num_query_per_req` 对应的 DSpark token sizes。

应从 DSpark dispatcher 单独收集 `draft_capture_sizes`，并只用它初始化 draft graph params。主图和草稿图的 params bucket 必须分离。

### 3.5 block size 来源错误地依赖目标模型 `has_gdn`

DSpark 草稿模型本身没有 GDN。当前 `has_gdn` 检查的是目标 model config，只是在全局 KV manager page-size 对齐后间接修正 slot mapping；它不应成为 DSpark 图的结构分支。

正确做法是：

- 对每个 draft cache layer/group，从其 draft attention backend 获取支持的 kernel block size；
- 从 KV allocator 获取该 draft cache group 的 manager block size；
- 由通用 BlockTable/kernel-block 映射规则决定 slot id；
- 不检查目标模型类型，也不出现 `if self.has_gdn`。

runner 目前只把首个 `kernel_block_sizes` 项传给 drafter，DSpark 又忽略了该参数。应改为传递完整的逐 group kernel block size，并像 `step3p5.py` 一样按 draft layer 所在 gid 创建 metadata builder。

## 4. metadata 设计

### 4.1 不能直接复制 dFlash

dFlash 的 graph capture 使用 `builder.build_for_graph_capture(..., ChunkedPrefill)`，但 DSpark 普通执行路径实际调用：

```python
builder.build_for_drafting(
    common_attn_metadata,
    draft_index=1,
    ...,
)
```

`build_for_drafting` 包含 DSpark/SWA 所需的 slot mapping、RoPE cache、稀疏索引等语义。直接替换为 dFlash 的 `build_for_graph_capture` 可能得到错误 metadata；DSA builder 的 capture 接口当前也不接受 `ChunkedPrefill`。

### 4.2 推荐接口

优先增加 backend drafting-capture 接口：

```python
build_for_graph_capture_drafting(
    common_attn_metadata,
    draft_index=1,
    kernel_block_size=...,
)
```

该接口应复用 `build_for_drafting` 的语义，同时保证：

- 输入 shape 按 DSpark capture bucket 固定；
- metadata 中的 device tensor 地址跨 replay 不变；
- padding slot 使用 `PADDING_SLOT_ID`；
- query/sample index 的 padding 行指向合法位置；
- 不通过 device tensor `.item()`/`.tolist()` 决定图内结构；
- capture 和 replay 使用相同 metadata 对象结构；
- 动态 seq_lens、slot mapping、block table 内容通过持久 buffer 和 graph-param update 更新。

如果现有 `build_for_drafting` 经审计已经满足以上约束，可以直接复用并补充测试，不为接口命名而增加无意义封装。

### 4.3 draft KV cache group 的含义

DSpark 可以有多个 draft cache layer。例如 DSV4 DSpark 有多个 draft decoder layer，每层暴露自己的 SWA cache layer。全局 KV allocator 可能把这些 draft cache layer 放入一个或多个 group。

多 group 只表示 block table/slot mapping 的路由可能不同，不表示草稿模型包含 GDN，也不表示需要读取主模型 attention metadata。

实现要求：

- 从全局 `KVCacheConfig` 中只筛选 `get_draft_kv_cache_layer_names()` 返回的 layer；
- 每个 draft layer 使用自身声明的 backend/spec；
- group id 只用于找到 allocator 分配给该 draft layer 的 cache/block table；
- 为每个 draft group 构造独立 metadata，并仅绑定到该 group 的 draft layers；
- 不把某个 group 的 metadata 复用给其他 group。

## 5. 分阶段实施

### 阶段 A：建立独立 draft graph dispatcher

1. 为 `AscendDSparkProposer` 初始化独立的 cudagraph dispatcher/key 集合。
2. capture key 至少包含 draft padded token 数，以及 dispatcher 已有的 LoRA/uniform 等必要维度。
3. DSpark `_propose` 和 `dummy_run` 均使用同一个 draft dispatcher。
4. `dummy_run` 根据 `num_reqs * num_query_per_req` 重新 dispatch，不使用主模型 descriptor。
5. 从 draft dispatcher 单独生成 `draft_capture_sizes`。
6. 独立初始化 draft graph params/workspace bucket。

验收：相同主模型 descriptor 下，`N` 与 `N + 1` 两种 DSpark query geometry 得到不同且正确的 draft descriptor；draft graph entry 不与主图共享。

### 阶段 B：独立判断 DSpark 图能力

1. 删除 `AscendDSparkProposer.__init__` 中无条件 `self.use_cuda_graph = False`。
2. 新增 DSpark 自身 capability check，不调用主模型 backend capability。
3. 仅由 draft enforce-eager、draft backend 支持度、ACLGraph runtime 和 padded-batch 条件决定是否入图。
4. 支持以下组合并明确测试：
   - 主模型 graph + DSpark graph；
   - 主模型 eager + DSpark graph；
   - 主模型 graph + DSpark eager；
   - 主模型 eager + DSpark eager。
5. 如果当前全局 compilation config 无法表达第二/第三种组合，先重构配置解析边界，不能静默继承主模型结果。

验收：切换目标模型 eager/graph 不会改变 DSpark 自身 capability check 的结果。

### 阶段 C：按 draft layer 初始化 attention/cache metadata

1. runner 向 drafter 传入完整的逐 KV group kernel block sizes，而不是首组值。
2. DSpark 按 `draft_layer_name -> gid -> draft backend/spec` 建立映射。
3. metadata builder 使用 draft backend 和对应 gid 的 kernel block size。
4. 删除 graph 路径对 `has_gdn`、`runner.attn_groups` 和主模型 backend 的引用。
5. 为每个 draft group 预分配固定地址的：
   - query slot mapping；
   - context slot mapping；
   - query start locations；
   - seq lens；
   - block-table view/copy buffer（仅在原地址不能稳定 replay 时需要）。

验收：将目标模型 backend mock 成任意类型都不影响 DSpark metadata；只改变 draft backend/spec 时 metadata 才变化。

### 阶段 D：实现 graph-safe drafting metadata

1. 审计 DSA/SFA 等 DSpark 实际 draft backend 的 `build_for_drafting`。
2. 实现或复用 `build_for_graph_capture_drafting`。
3. 按 draft descriptor 的 padded request/token shape 构造 metadata。
4. 保持普通 DSpark 路径的 attention state、non-causal/SWA 和 sparse-index 语义，不套用 dFlash 的固定 `ChunkedPrefill` 假设。
5. capture/replay 只更新 tensor 内容，不改变 tensor 地址、Python 容器结构或循环次数。

验收：同一组持久 buffer 上，普通 drafting metadata 与 capture-drafting metadata 对有效 token 产生相同 attention 参数。

### 阶段 E：捕获独立 DSpark 执行图

图边界建议从 DSpark 持久输入 buffer 开始，包含：

1. `precompute_and_store_context_kv`；
2. draft model forward；
3. logits/MarkovHead bias；
4. 固定 `num_speculative_tokens` 次数的 argmax/in-place token 更新；
5. draft token 输出 view/copy。

图外输入准备负责把 target hidden states、positions、next tokens 和 rejected-token 信息复制到 DSpark 持久 buffer。图内不得直接捕获主模型输出 tensor 的临时地址。

capture/replay 生命周期：

1. DSpark forward context 使用 draft descriptor 和 draft per-layer metadata。
2. `_runnable` 是 DSpark 独立 `ACLGraphWrapper`。
3. capture 时登记 draft backend 的 task handles/workspace。
4. replay 前后按 backend 协议更新 **draft graph params**。
5. `_update_full_graph_params` 只接收 draft backend 和 draft metadata。
6. 若多个 draft groups 使用同一个 backend，只更新一次；若运行时发现不同 draft backend，按 draft backend 去重更新。不得引入主模型 backend。

验收：关闭主模型 graph 或改变主模型 backend 后，DSpark graph capture count、key、workspace 地址和 replay 路径不变。

### 阶段 F：capture 调度与 warmup

1. 为 DSpark 增加独立 capture/warmup 入口。
2. 可以由 runner 负责调用该入口，但 runner 只承担生命周期编排，不向 DSpark 传主图 descriptor/metadata/backend。
3. 若复用主模型 capture 循环触发 DSpark warmup，DSpark 必须在入口处重新 dispatch 自己的 descriptor；这种复用仅是启动时序复用。
4. 主模型完全 eager 时仍需执行 DSpark capture warmup，或明确采用首次 draft step lazy capture，并记录一次性延迟。

验收：主模型没有任何 graph entry 时，DSpark 仍可预捕获并 replay。

## 6. 预计修改文件

核心修改：

- `vllm_ascend/spec_decode/dspark_proposer.py`
- `vllm_ascend/spec_decode/llm_base_proposer.py`（抽取独立 drafter dispatch/capture 能力；避免 DSpark 继续调用 runner dispatcher）
- `vllm_ascend/worker/model_runner_v1.py`（仅做 drafter 生命周期编排、传递完整 allocator/kernel-block 信息）
- `vllm_ascend/compilation/acl_graph.py`（若需独立 draft capture-size/params namespace）
- DSpark 实际使用的 metadata builder，例如 `vllm_ascend/attention/dsa_v1.py`、`sfa_v1.py` 或其 CP 版本。

测试修改：

- `tests/ut/spec_decode/test_dspark_proposer.py`
- 对应 ACLGraph/metadata builder 单元测试；
- DSpark NPU E2E graph accuracy/performance 用例。

## 7. 测试计划

### 7.1 独立性单元测试

1. DSpark graph capability 不受目标模型 `enforce_eager` 和 backend 类型影响。
2. DSpark `_propose`/`dummy_run` 不调用 `runner.cudagraph_dispatcher`。
3. DSpark graph metadata 不访问 `runner.attn_groups`、`runner.attn_backend` 或 `has_gdn`。
4. 传入主模型 descriptor 后，DSpark 仍生成自己的 draft descriptor。
5. draft graph params 的 key 只来自 draft capture sizes。
6. target 临时 hidden-state tensor 地址变化，但复制后的 DSpark graph input buffer 地址稳定。

### 7.2 metadata 单元测试

1. `sample_from_anchor=true/false` 的 `N`、`N + 1` query geometry。
2. 单 draft KV group。
3. 多 draft KV group，各自使用正确 block table、slot mapping、builder 和 layer mapping。
4. manager block size 与 draft kernel block size不同时，按通用映射计算，不检查目标模型类型。
5. graph padding slot/index 安全。
6. eager `build_for_drafting` 与 capture drafting metadata 的有效区参数一致。
7. capture 时不更新 replay params，replay 时只更新 draft params。

### 7.3 NPU E2E

1. eager DSpark 与 graph DSpark 的 draft token、最终 token、接受长度一致。
2. 主 eager/draft graph 与主 graph/draft graph 输出一致。
3. batch size 跨 1、2、4、8 等 draft capture bucket，确认不重复 capture。
4. rejected tokens、请求加入退出、不同 sequence length。
5. Qwen DSpark 和 DSV4 DSpark 各验证其自身 draft backend；不把目标 GDN 作为 DSpark 功能分支。
6. TP/DP 场景下 draft dispatcher 和 collective 顺序一致，无死锁。
7. 长稳检查 KV 越界、NaN、graph 地址变化和内存增长。

性能记录：draft capture 时间、首次 replay 延迟、TPOT、吞吐、NPU 内存峰值和重复 capture 次数。

## 8. 建议提交拆分

1. `refactor(spec_decode): decouple dspark graph dispatch from target model`
2. `test(spec_decode): add dspark graph independence coverage`
3. `feat(spec_decode): add graph-safe dspark drafting metadata`
4. `feat(spec_decode): enable independent aclgraph for dspark`
5. `test(spec_decode): add dspark aclgraph npu regression`

所有提交使用 `git commit -s`。

## 9. 完成定义

- DSpark 不再无条件 eager。
- DSpark 使用独立 dispatcher、descriptor、capture sizes、graph params 和 workspace。
- DSpark 图不读取主模型 attention backend/metadata/graph 状态。
- 目标模型 eager 时 DSpark 仍能 capture/replay。
- draft KV cache 仅按 draft layer/backend/spec 建模；不存在“DSpark GDN”分支。
- eager 与 graph 正确性一致，UT、NPU E2E、lint 和格式检查通过。
