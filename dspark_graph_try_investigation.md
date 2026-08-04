# DSpark ACLGraph 实现精度问题调查与解决方案

## 1. 调查结论

对 `dspark_graph_try.diff` 做静态代码审查后，结论是：这份实现已经补齐了 DSpark 图的基本拉起链路，但当前还不满足正确 replay 的条件。接受率和 eager 严重偏离，不像普通 BF16/算子数值误差，更符合“图内注意力持续使用 capture 阶段的错误或陈旧 metadata”的特征。

当前至少存在两个可独立造成严重精度下降的 P0 问题：

1. **Qwen3.6 DSpark 的 `dspark_swa_indices` 不是 graph-stable tensor。**
   `build_for_drafting()` 在图外根据当前 `block_table/query_start_loc/seq_lens` 新建索引 tensor，但 ACLGraph replay 不会使用新建的 metadata 对象；图内继续读取 capture dummy-run 时的索引地址。因此实际请求很可能始终访问 dummy block 或上一次 capture 的 KV slot。
2. **把主模型 `(N+1) * B` 的图桶直接当成 DSpark `N * B` 的执行形状，并让最后一个真实请求吞掉 padding。**
   例如 `N=7, B=2` 时，eager 的 `query_start_loc` 是 `[0, 7, 14]`，当前 graph 实现改成 `[0, 7, 16]`，第二个请求被解释成 9 个 query token。`slot_mapping=-1` 只禁止 padding token 写 KV，不能阻止它们改变 query 分段、SWA 可见区间和稀疏索引。

此外还有 capture/replay metadata 地址不一致、graph 分支覆盖 DSpark eager `seq_lens` 语义、测试全部 mock 掉真实 DSA builder 等问题。它们会叠加放大上述错误。

因此，这份 diff 当前的评估是：

| 项目 | 结论 |
|---|---|
| 服务拉起与 graph key 初始化 | 基本完成 |
| DFlash 风格的 capture/replay 调用时序 | 基本对齐 |
| Qwen3.6 GDN 目标下的 kernel block size | 修改方向正确 |
| DSpark 图输入的语义正确性 | 不正确 |
| capture/replay tensor 地址稳定性 | 不完整 |
| eager/graph 精度等价 | 不成立 |
| 当前是否适合合入 | 不适合，应先修复 P0 |

> 本报告是静态调查结果。当前环境未启动真实 Qwen3.6 权重服务，也没有 NPU 接受率日志，因此最终结论仍需按第 7 节在 NPU 上验证。但代码证据已经足以解释“服务可运行、接受率严重下降”的现象。

## 2. 本次实现做了什么

`dspark_graph_try.diff` 共涉及 5 个文件，核心改动如下：

- `dspark_proposer.py`
  - 删除 `self.use_cuda_graph = False`，允许继承基类 graph 开关；
  - 把 `max_query_tokens` 扩到 `max_batch_size * (1 + N)`；
  - 在 `dummy_run()` 中构造 DSpark drafting metadata；
  - 将 metadata 传入 `_runnable` 和 forward context；
  - replay 后调用 `_update_full_graph_params()`；
  - 保留 Qwen3.6/GDN 场景下的 draft kernel block size。
- `llm_base_proposer.py`
  - graph 模式下把 DSpark 的 `query_start_loc[-1]` 改成 `num_input_tokens`。
- `model_runner_v1.py`
  - 让 DSpark 初始化 cudagraph keys。
- `mamba/postprocess.py`
  - 同步上游 Mamba postprocess kernel 的新签名和状态索引逻辑。
- `test_dspark_proposer.py`
  - 增加大量 dummy-run、分组 metadata 和 padding 单测。

其中 graph 开关、key 初始化、per-group metadata、padding buffer 初始化和 replay 后更新调用的方向是合理的。问题主要不在“有没有调用 graph”，而在“graph 捕获了哪些地址和语义，以及 replay 前是否更新了这些地址里的内容”。

## 3. P0 根因一：`dspark_swa_indices` 在 replay 中是陈旧的

### 3.1 代码链路

diff 在 `dspark_graph_try.diff:778-781` 对每个 draft attention group 调用：

```python
attn_metadata = builder.build_for_drafting(
    common_attn_metadata,
    draft_index=1,
)
```

对 Qwen3.6 DSpark，drafting metadata 会进入 DSA 的 `build_decode_metadata_for_drafting()`。该函数在 `vllm_ascend/attention/dsa_v1.py:1358-1367` 调用 `build_dspark_swa_indices()`，依据当前请求的：

- `block_table`；
- `query_start_loc`；
- `seq_lens`；
- draft block size；

动态创建 `dspark_swa_indices`。这个 tensor 最终在 `vllm_ascend/attention/dsa_v1.py:2494-2496` 作为 `ori_sparse_indices` 直接传入稀疏注意力算子。

关键在于：`build_for_drafting()` 发生在 `_runnable()` 之外，也就是 ACLGraph 捕获区域之外。它生成的新 tensor 只是本次 Python metadata 对象的一部分。

而 `ACLGraphWrapper` 已明确约定：wrapper 不负责维护 persistent buffer，也不会在 replay 时把新输入复制到 capture buffer，见 `vllm_ascend/compilation/acl_graph.py:76-82`。真正 replay 时直接执行 `entry.aclgraph.replay()`，新传入的 metadata 对象不会重新绑定图内地址。

同时，DSA 的 `update_graph_params()` 当前是空实现。因此没有其他机制把本轮新算出的 `dspark_swa_indices` 更新到 capture 地址。

### 3.2 为什么会严重影响接受率

`dspark_swa_indices` 决定每个 draft query 能看到哪些 SWA KV slot。若图内一直读取 dummy-run 的索引：

- query 会访问 dummy block table 对应的 KV；
- 不同请求可能访问同一批错误 slot；
- context 长度变化后可见窗口仍不变化；
- draft hidden states 从第一层稀疏注意力开始就与 eager 分叉；
- Markov head/LM head 输出自然会大幅变化，最终表现为接受率显著下降。

该问题与“图能否成功 capture/replay”无冲突：地址合法、shape 合法时服务完全可以正常运行，只是读取的内容在语义上错误。这与用户观察高度一致。

### 3.3 为什么现有测试没有发现

新增单测把 `get_metadata_builder()` 和 `build_for_drafting()` 替换成 `MagicMock`，只验证：

- builder 被调用；
- metadata 被传给 `_runnable`；
- `draft_index=1`；
- layer/group 映射存在。

测试没有运行真实 DSA builder，因此没有覆盖：

- `dspark_swa_indices` 的具体数值；
- capture 与 replay 的 `data_ptr()` 是否一致；
- block table 改变后 graph 输出是否改变；
- eager 与 graph 的 draft logits/token 是否一致。

## 4. P0 根因二：最后一个真实请求吞掉 graph padding

### 4.1 当前实现的错误假设

diff 的 `dspark_graph_try.diff:743-758` 和 `:828-849` 都采用了同一策略：当 DSpark 实际 query 数小于图桶 token 数时，直接把 `query_start_loc` 的最后一个边界改成图桶大小。

注释认为 padding token 的 `slot_mapping=-1` 且输出会被忽略，所以这样是安全的。这个判断不成立，因为 `query_start_loc` 不只控制 KV 写入，还定义了每个请求的 query 分段。

以 `sample_from_anchor=True, N=7, B=2` 为例：

| 路径 | `query_start_loc` | 每请求 query 长度 |
|---|---|---|
| eager | `[0, 7, 14]` | `[7, 7]` |
| 当前 graph | `[0, 7, 16]` | `[7, 9]` |

Qwen3.6 DSpark 的 DSA builder 会在 `vllm_ascend/attention/dsa_v1.py:1334-1365` 使用该边界计算：

- `max_seqlen_q`；
- `cu_seqlens_q`；
- 每请求 `query_lens`；
- `prefix_lens = seq_lens - query_lens`；
- 每个 token 的 SWA visible slots。

因此 padding 会改变最后一个真实请求的 attention，而不只是产生几行无用输出。

单请求场景同样有问题：`N=7, B=1` 会从 `[0, 7]` 变为 `[0, 8]`。这意味着每个 graph batch 都可能污染真实请求，而不是只影响某个边角 batch size。

### 4.2 这也违反了 DSpark 与主模型执行形状独立的原则

`dspark_graph_try.diff:668-673` 和 `:712-717` 明确把 DSpark 的 buffer/capture token 数绑定到主模型的 `uniform_decode_query_len = 1 + N`。

但在 `sample_from_anchor=True` 时：

- 主模型验证步长：`1 + N`；
- DSpark 草稿模型 query 步长：`N`。

两者可以复用相同的用户开关和 capture 生命周期，但不能假设拥有相同的模型输入 shape。当前实现为了复用主模型 `BatchDescriptor.num_tokens`，反过来改变了 DSpark 的请求语义。

仓库里的 v2 DSpark 路径已经体现了正确关系：draft padded token 数由 `num_reqs_padded * self.num_query_per_req` 计算，而不是直接使用 target 的 `(1 + N) * B`。

### 4.3 正确的 padding 方式

优先方案是让 DSpark 图的物理输入长度直接按自己的执行几何计算：

```text
draft_num_tokens = padded_num_reqs * dspark.num_query_per_req
```

也就是 graph key/capture 生命周期可以与 DFlash 对齐，但 draft runnable 的 token shape 必须由 DSpark 自己决定。

若 v1 dispatcher 暂时必须使用更大的物理图桶，则 padding 必须建模成独立的虚拟请求，例如：

```text
[0, 7, 14, 16]  # 第三个 segment 才是 padding request
```

而不能改成 `[0, 7, 16]`。同时需要为虚拟请求补齐安全的 `seq_lens`、block-table row 和其他 per-request metadata。该方案改动面和维护成本都高于使用 DSpark 自身的 `N * B` 图形状，不建议作为最终设计。

## 5. 其他高风险问题

### 5.1 capture/replay 使用的 metadata buffer 地址不一致

真实 `_propose()` 会把 metadata 写入基类预分配的稳定 buffer：

- `self.query_start_loc_group[0]`；
- `self.seq_lens_group[0]`；
- `self.slot_mapping_group[0]`。

见 `vllm_ascend/spec_decode/llm_base_proposer.py:892-898`。

但新 `dummy_run()`：

- 用乘法/`clone()` 新建 `qsl`；
- 直接引用 `runner.seq_lens`；
- 没有使用 `query_start_loc_group[0]` 和 `seq_lens_group[0]`。

因此 capture 与 replay 构造的 Python metadata 即使数值相同，tensor 地址也不是同一组。ACLGraph replay 不会重新绑定 metadata 中的新地址。

特别需要注意：`ACLGraphWrapper` 的 debug 地址检查只枚举位置参数中的 tensor；当前 `_runnable` 主要使用 kwargs，且 metadata 嵌套在 list/dict 中，所以这个问题不一定会触发现有地址断言。

### 5.2 FULL graph 分支覆盖了 DSpark eager 的 `seq_lens` 语义

DSpark 的 `set_inputs_first_pass()` 会执行：

```python
cad.seq_lens = effective_seq_lens + self.num_query_per_req
```

这是 eager 路径当前使用的语义。但进入 FULL graph 后，基类只为 `method == "dflash"` 保留这份 `seq_lens`；其他方法会改用 `runner.seq_lens`，见 `vllm_ascend/spec_decode/llm_base_proposer.py:840-845`。

DSpark 原本被强制 eager，所以历史代码没有暴露此分支。现在直接开启 DSpark graph 后，它会进入“非 dflash”分支，可能丢失：

- `+ num_query_per_req`；
- `num_rejected_tokens_gpu` 修正后的 effective length；
- 与 eager 相同的 SWA prefix/visible length。

这会进一步导致 `build_dspark_swa_indices()` 的窗口起点和可见长度错误。修复时 DSpark 应与 DFlash 一样保留 `set_inputs_first_pass()` 已构造的 `seq_lens`，然后复制到 graph-stable buffer。

### 5.3 `num_actual_tokens` 在 capture 时被设置为 padded token 数

新 `dummy_run()` 把 `num_actual_tokens` 和 `num_input_tokens` 都设置成 `num_query_tokens`。对于 `N=7, B=2`，两者都是 16；真实 DSpark 的 actual token 数则是 14。

DSA forward 在 `vllm_ascend/attention/dsa_v1.py:1786-1794` 会读取 metadata 中的 Python 整数 `num_decode_tokens/num_actual_tokens` 来切分 hidden states。这些值在 graph capture 后也是静态的。只有在 padding 被建模成安全的独立 segment，或者 DSpark 使用自己的 `N * B` capture shape 时，处理 padded 行才不会破坏真实请求。

### 5.4 `query_start_loc` 判断会引入 NPU 同步

`llm_base_proposer.py` 新增逻辑在真实 hot path 中执行：

```python
if qsl[num_reqs] != num_input_tokens:
```

当 `qsl` 在 NPU 上时，Python 条件判断需要读取 device scalar，会隐式触发 NPU 到 CPU 的同步。它不是本次精度根因，但会造成每轮 speculative decode 的同步开销，也违反项目对 hot path 避免 device scalar 判断的要求。

### 5.5 `mamba/postprocess.py` 不属于 DSpark 小模型图路径

该改动用于同步 target GDN/Mamba 状态后处理 kernel 的接口，包括 `idx_mapping` 和 `PRECOMPUTED_NEW_COMPUTED`。从静态代码看，新旧公式在对应开关下是代数一致的；若服务已经能运行，它不像本次 graph-vs-eager draft 精度差异的首要根因。

但它属于主模型 GDN 状态维护路径，而不是 DSpark 小模型图编译本身。建议拆成独立补丁、独立回归：

- 同一份 Mamba patch 下比较 DSpark eager 与 graph；
- 单独验证 target GDN state copy 的多轮一致性；
- 避免把主模型兼容性修复与 DSpark graph 精度问题混在一个提交中。

### 5.6 GDN kernel block size 修复应保留

`dspark_proposer.py` 中根据 `has_gdn` 保留 draft backend 的 `kernel_block_size`，并用它计算 query slot id，修改方向正确。

这里的 `has_gdn` 只说明目标模型的全局 KV manager 因 GDN page size 做了放大；DSpark 草稿模型本身并没有 GDN layer。该改动是在修正 draft attention 对全局 BlockTable 的地址单位，不是在给草稿模型增加 GDN metadata。

## 6. 推荐解决方案

### 6.1 推荐的最小正确设计

目标是保留 DFlash 的配置和执行时序，但让 DSpark 的图输入与 metadata 完全由自己的 `num_query_per_req` 决定。

#### 修改一：恢复 DSpark 独立执行形状

- 保持 graph 开关、capture 生命周期和用户参数与 DFlash 一致；
- 不再把 `BatchDescriptor.num_tokens = (N+1)*B` 直接作为 DSpark runnable 的输入长度；
- 从 padded request count 推导：

```python
dspark_num_input_tokens = num_reqs_padded * self.num_query_per_req
```

- capture 和 replay 都使用相同的 DSpark token 数；
- `max_query_tokens` 重新按 `max_batch_size * num_query_per_req` 定义，除非还有 DP/SP 对齐需求；
- 删除两处“改写最后一个真实请求边界”的逻辑。

这一步从根源上消除 `N` 与 `N+1` 的语义冲突，也符合“草稿模型的图编译和执行路径不应依赖主模型输入 shape”的要求。

#### 修改二：为 DSA drafting metadata 提供稳定地址

至少要把以下会被图内算子直接读取的 tensor 做成 capture/replay 共用的持久 buffer：

- `query_start_loc`；
- `seq_lens`；
- per-group `block_table`；
- per-group `slot_mapping`；
- `dspark_swa_indices`；
- DSA sparse-attention metadata workspace；
- RoPE `cos/sin`（现有 draft-index cache 可继续复用）。

最关键的缺口是 `dspark_swa_indices`。建议在 DSA builder 中按 draft step 预分配，例如 `spec_dspark_swa_indices[draft_index - 1]`：

1. 每轮在图外按当前请求计算新的 indices；
2. `copy_()` 到固定 buffer；
3. metadata 始终返回固定 buffer 的同 shape slice；
4. capture 和 replay 断言 `data_ptr()` 相同。

不要仅把新 tensor 放进新的 metadata 对象；ACLGraph replay 不会消费这个新对象。

#### 修改三：capture metadata 直接复用真实路径的稳定 buffer

`dummy_run()` 不应自己用 `clone()` 创建一套近似 metadata。应当：

- 使用 `query_start_loc_group[0]`；
- 使用 `seq_lens_group[0]`；
- 使用现有 per-group block-table/slot-mapping buffer；
- 使用与真实 `_propose()` 相同的 DSpark metadata helper；
- capture 前写 dummy 值，replay 前写真实值；
- 保证传给图内算子的所有 tensor 地址不变。

#### 修改四：保留 DSpark 自己构造的 `seq_lens`

FULL graph padding分支不能用 `runner.seq_lens` 覆盖 DSpark `set_inputs_first_pass()` 的结果。建议将判断改为保留所有 parallel-drafting 方法的语义，例如 DSpark 与 DFlash走同一分支，再复制到 `seq_lens_group[0]`。

### 6.2 不推荐的方案

- **继续让最后一个真实请求吞 padding：** 请求分段必然错误。
- **只调用 `_update_full_graph_params()`：** Qwen3.6 对应 DSA backend 的实现当前是 no-op，无法更新 `dspark_swa_indices`。
- **只把 padding 的 slot mapping 设为 -1：** 只能控制 KV 写入，不能修复 query segmentation/SWA indices。
- **只增加 mock 单测：** 无法验证 graph 实际读取的地址与数值。
- **把整个 metadata 对象每轮重新创建后传入 `_runnable`：** replay 不重新执行 Python forward，也不会重新绑定对象里的 tensor。

## 7. 建议验证顺序

### 7.1 第一组：不用跑完整 benchmark 的定点验证

在同一个 capture size 下记录 capture 和两次 replay：

1. `query_start_loc` 的值与 `data_ptr()`；
2. `seq_lens` 的值与 `data_ptr()`；
3. 每个 group 的 block table/slot mapping 的 `data_ptr()`；
4. `dspark_swa_indices` 的前若干值、checksum 和 `data_ptr()`；
5. 第一层 DSpark attention 输出的最大绝对误差；
6. Markov head 前 hidden states、logits 和 draft token ids。

修复后的要求：

- capture/replay 中图所读取 tensor 的 `data_ptr()` 不变；
- replay 前 persistent buffer 内容随请求变化；
- eager 与 graph 的 `query_start_loc/seq_lens/dspark_swa_indices` 对真实 token 完全一致；
- 第一层 attention 输出在允许的 NPU 数值误差内一致。

### 7.2 第二组：高辨识度 A/B 用例

建议固定 `N=7`：

| 用例 | 目的 |
|---|---|
| `B=1`, anchor-first | 当前实现会把 `[0,7]` 变成 `[0,8]`，最容易观察分段污染 |
| `B=2`, anchor-first | 检查 `[0,7,14]` 是否被错误改成 `[0,7,16]` |
| 相同 batch shape、不同 block table | 检查 graph 是否仍使用 capture dummy 的 SWA indices |
| 相同请求连续多轮 decode | 检查 seq_lens/visible window 是否随轮次更新 |
| bonus-anchor (`num_query_per_req=N+1`) | 图桶无 N/N+1 gap，可隔离 padding 问题与 stale indices 问题 |
| GDN target 下 DSpark eager vs graph | 保持同一份 kernel-block-size/Mamba patch，隔离 draft graph 差异 |

预期诊断信号：

- bonus-anchor 明显好转但仍不完全恢复：padding 分段和 stale indices 两者都存在；
- 更换 block table 后 graph indices/checksum 不变：直接确认 persistent-buffer 缺失；
- 第一层 attention 已明显分叉：无需继续怀疑 LM head 或 sampler。

### 7.3 第三组：接受率回归

修复后至少记录：

- 同一模型、同一 prompt 集、同一随机/greedy 配置；
- eager draft 接受率；
- graph draft 接受率；
- 每个 speculative position 的接受率；
- draft token 完全一致比例；
- graph capture/replay 次数以及是否发生 eager fallback。

目标不应只定义为“服务不报错”，而应要求 graph 与 eager 的 draft token/接受率在合理数值误差范围内一致。

## 8. 单测修改建议

当前以下测试实际上固化了错误行为，应删除或改写：

- `test_full_mode_query_start_loc_padded_to_num_query_tokens`；
- `test_query_start_loc_padded_in_graph_mode`。

新测试应断言：

1. DSpark 真实请求边界永远保持 `arange * num_query_per_req`；
2. 若存在 graph padding，padding 是独立虚拟 segment，不能扩展最后一个真实请求；
3. 更推荐直接断言 DSpark graph token 数为 `num_reqs_padded * num_query_per_req`；
4. capture/replay 的 `query_start_loc/seq_lens/dspark_swa_indices` 使用同一预分配 buffer；
5. 修改 block table/seq lens 后，persistent `dspark_swa_indices` 内容随之改变但地址不变；
6. 使用真实 DSA builder 做小尺寸 eager/graph metadata 等价测试，不要把 builder 全部 mock 掉；
7. 增加 NPU e2e：固定输入比较 eager 与 graph 的 draft tokens 和接受率。

## 9. 建议拆分提交

为了降低定位成本，建议拆成三组独立修改：

1. **Qwen3.6/GDN block size 修复**
   - 保留现有 `kernel_block_size` 修改及对应 UT。
2. **DSpark ACLGraph 正确性**
   - DSpark 自身 capture shape；
   - persistent metadata buffers；
   - capture/replay 同路径；
   - eager/graph 等价测试。
3. **Mamba postprocess 上游接口同步**
   - 独立验证 target GDN state 更新，不与 draft graph 精度变更混合。

这样可以确保接受率变化能明确归因，也符合“尽量少改代码、让 DSpark 草稿图路径独立于主模型”的目标。

## 10. 最终判断

这份实现中，graph 调度骨架已经基本搭起来，但目前把“metadata 对象在 replay 前重新构造”误当成了“图内 metadata 已更新”。ACLGraph 真正需要维护的是：**capture 时图读取的固定 tensor 地址，以及每次 replay 前写入这些地址的正确内容**。

对 Qwen3.6 DSpark，最关键的内容不只是 input ids、positions 和 slot mapping，还包括动态生成的 `dspark_swa_indices`、正确的 `query_start_loc` 请求分段和 DSpark 自己的 `seq_lens`。当前三者都没有完整满足，因此严重精度问题是必然风险，而不是偶发波动。

建议优先按以下顺序修复：

1. 将 DSpark graph 输入长度从主模型 `(N+1)*B` 解耦为 `N*B`；
2. 为 `dspark_swa_indices` 增加 graph-stable buffer；
3. capture 与 replay 统一使用 `query_start_loc_group/seq_lens_group`；
4. 保留 DSpark `set_inputs_first_pass()` 的 seq-lens 语义；
5. 用真实 DSA builder 和 NPU eager/graph 对比验证，再评估接受率。
