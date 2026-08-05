# DSpark ACLGraph 整体设计

## 1. 文档目的

本文描述 vLLM Ascend V1 Model Runner 中 DSpark draft model 的 ACLGraph 设计，重点说明：

- DSpark 与 DFlash 不同的 token 几何语义；
- target graph descriptor 与 draft graph 实际执行宽度的解耦；
- target hidden states 到 draft context KV-cache 的入图方式；
- per-KV-group attention metadata、slot mapping 和静态 buffer 生命周期；
- memory profile、graph capture、graph replay 三个阶段的边界；
- 当前实现必须保持的正确性不变量和回归测试。

当前实机验证配置为 Qwen3.6-27B target、TP=4、DSpark dense GQA draft、5 层
full-attention（非 SWA）、`N=7`、单请求、`FULL_DECODE_ONLY`。

## 2. 算法语义

### 2.1 两种宽度

设每个请求投机生成 `N` 个 draft token。

target model 需要验证当前 token 加 `N` 个 draft token，因此验证宽度为：

```text
V = N + 1
```

默认 anchor-first DSpark 的 draft backbone 只处理 `N` 个 query：

```text
Q = N
```

但 draft model 的 context KV 由 target verification hidden states 预计算，必须消费完整
验证宽度：

```text
C = N + 1
```

当前 `N=7` 时：

```text
target verification width = 8
draft query width         = 7
draft context-KV width    = 8
```

如果 draft checkpoint 配置 `dspark_bonus_anchor=True`，则：

```text
Q = N + 1
```

此时 draft query 与 target verification 宽度重新相等。

### 2.2 与 DFlash 的差异

| 项目 | DFlash | 默认 DSpark |
|---|---:|---:|
| target 验证宽度 | `N+1` | `N+1` |
| draft query 宽度 | `N+1` | `N` |
| context-KV 预计算宽度 | `N+1` | `N+1` |
| query graph 与 target descriptor 是否同宽 | 是 | 否 |

因此不能直接把 DFlash 的 target capture size 同时作为 DSpark query backbone 的执行宽度。

## 3. 总体数据流

```text
target model verification hidden states（每请求 N+1 行）
                         |
                         v
              DSpark input preparation
          +--------------+---------------+
          |                              |
          v                              v
context hidden/positions/slots      N 个 draft query
          |                    input_ids/positions/slots
          v                              |
fused context K/V projection             |
K norm + RoPE                            |
          |                              |
          v                              v
5 层 draft KV-cache update ------> draft attention backbone
                                         |
                                         v
                              sample hidden states / raw logits
                                         |
                                         v
                               autoregressive Markov bias
                                         |
                                         v
                                  N 个 draft token
```

一次 draft 执行同时包含两条不同宽度的子路径：

1. context-KV 子路径使用 `C=N+1`；
2. query backbone 子路径使用 `Q=N`。

两条路径共享同一个 `_runnable` 和 ACLGraph，但不能共享 token 数语义。

## 4. Graph descriptor 与实际 draft 宽度解耦

### 4.1 descriptor 仍使用 target 语义

调度器和 target model 以验证宽度 `N+1` 构造 `BatchDescriptor`。该 descriptor 继续作为：

- graph dispatch 的 bucket 描述；
- target/draft graph 对应关系的稳定标识；
- uniform batch、请求数和 LoRA 等属性的载体。

不能把 descriptor 本身改写为 `N`，否则 target graph 与 draft graph 的调度关系会失配。

### 4.2 draft 执行宽度单独计算

`AscendSpecDecodeBaseProposer.get_graph_num_input_tokens()` 默认返回 descriptor 的
`num_tokens`。DSpark 覆盖该方法：

```python
if batch_descriptor.uniform and batch_descriptor.num_reqs is not None:
    return batch_descriptor.num_reqs * self.num_query_per_req
return batch_descriptor.num_tokens
```

uniform decode 下，draft graph 实际 token 数为：

```text
draft_graph_tokens = batch_size * Q
```

非 uniform 或外部 padding 无法直接推导请求几何时，保留 descriptor token 数，并由
尾部虚拟请求逻辑保证 FIA metadata 闭合。

### 4.3 draft graph 参数空间

`NPUModelRunner` 初始化 graph params 时分别维护：

- target `capture_sizes`；
- 通过 drafter `get_graph_num_input_tokens()` 转换后的 `draft_capture_sizes`。

以单请求 `N=7` 为例：

```text
target graph params key width = 8
draft graph params key width  = 7
```

这样 draft attention graph 的 handles、events、workspace 和 attention params 均按真实
query 宽度注册，不会用 8-token 容器重放 7-token 计算。

## 5. 静态 buffer 设计

ACLGraph 要求捕获与重放期间设备 tensor 地址稳定。DSpark 为此预分配以下 buffer：

| Buffer | 容量 | 用途 |
|---|---:|---|
| `input_ids` | capture 上限 | draft query token IDs |
| `positions` | `B_max*(N+1)` | draft query position IDs |
| `_dflash_hidden_states` | `max_num_tokens` | target context hidden states |
| `_context_positions_buffer` | `max_num_tokens` | context position IDs |
| per-group query slot buffer | `max_query_tokens` | query K/V 写入位置 |
| per-group context slot buffer | `max_num_tokens` | target context K/V 写入位置 |
| `_dspark_seed_buffer` | `B_max` | MarkovHead 的起始 token |
| `_dspark_draft_buffer` | `B_max*(N+1)` | seed 加 N 个 draft token |

`max_query_tokens` 保留 `B_max*(N+1)` 容量。默认 DSpark 实际只使用 `B* N`，但该容量
还要兼容 bonus-anchor、非 uniform descriptor 和 capture padding。

真实请求只更新这些 buffer 的内容，不替换 tensor 对象。

## 6. KV group 与 slot mapping

### 6.1 attention group 建立

`initialize_attn_backend()` 从 draft model 的
`get_draft_kv_cache_layer_names()` 获取 draft attention 层，并与 target KV cache config
中的 group 做交集，为每一种 backend/cache spec 创建 `AttentionGroup`。

实现同时维护：

```text
layer name -> KV group id
KV group id -> query slot buffer
KV group id -> context slot buffer
KV group id -> block table
```

当前 5 个 dense GQA full-attention 层属于同一标准 attention group，但实现不依赖这一
特例，可以支持 draft 层分布在多个 KV group。

### 6.2 per-group 到 per-layer 视图

Qwen context-KV precompute 按模型层顺序写 cache，因此模型侧接收的是 per-layer slot
列表：

```python
self._context_slot_mapping_buffers = [
    self._per_group_context_slot_mapping_buffers[group_idx]
    for group_idx in self._layer_group_idx
]
```

该列表不复制 tensor。同一 group 中的多个层引用同一个持久 slot tensor；不同 group
引用各自 buffer。

### 6.3 为什么必须在捕图前绑定

Qwen context-KV patch 在 slot mapping 为 `None` 时会跳过 cache update：

```python
if context_slot_mapping is None:
    return
```

如果 per-layer 列表直到第一个真实请求才创建，capture 只会记录 K/V 投影，不会记录
任何层的 `do_kv_cache_update`。真实请求之后创建 Python 列表无法修改已经捕获的图。

因此初始化顺序必须是：

```text
创建 per-group 持久 tensor
  -> 建立 per-layer 引用列表
  -> ACLGraph capture
```

capture 时 slot tensor 初始内容可以是 0；图依赖的是 tensor 地址和 cache-update 算子。
replay 前 input-preparation kernel 会原位写入当前请求的真实 slot IDs。

## 7. 生命周期

### 7.1 构造阶段

- 分配与具体 KV group 无关的通用 buffer；
- `_layer_group_idx` 初始化为空列表；
- `_context_slot_mapping_buffers` 初始化为 `None`。

### 7.2 memory profile

可用显存探测会在 KV cache 配置前调用 `dummy_run(is_profile=True)`。此时：

- KV group 尚未创建；
- `_layer_group_idx` 仍为空；
- 不应绑定 context slot list；
- profile 保持原来的 no-cache 行为。

该阶段不能使用“属性必然已存在且非空”作为前置条件。

### 7.3 KV backend 初始化

`initialize_attn_backend()` 完成：

1. draft layer/group 解析；
2. per-group query/context slot tensor 分配；
3. `_layer_group_idx` 建立；
4. per-layer context slot list 首次绑定。

### 7.4 ACLGraph capture

`dummy_run()` 在正式 capture 时：

1. 从 target descriptor 计算 draft query 宽度 `Q`；
2. 独立保留原始 context 宽度 `C`；
3. 防御性重绑 per-layer context slots；
4. 为每个 draft attention group 构造 graph-capture metadata；
5. 执行 context-KV precompute 和所有层 cache update；
6. 执行 N-token query backbone、LM head 和 MarkovHead；
7. 将完整计算捕入 draft ACLGraph。

### 7.5 graph replay

每个真实请求执行：

1. `set_inputs_first_pass()` 把 target hidden states 复制到持久 context buffer；
2. Triton input-preparation kernel 按 KV group 原位填写：
   - query input IDs；
   - query/context positions；
   - query/context slot mappings；
   - token sample indices；
3. 更新 runtime attention metadata 和 graph params；
4. replay draft graph；
5. 返回 N 个 draft token 给 target model 验证。

## 8. Input preparation

DSpark 复用 DFlash 输入准备 kernel，但每个 KV group 单独调用。关键输入包括：

- target `next_token_ids`；
- target positions；
- target context slot mapping；
- group block table；
- query start locations 和 sequence lengths；
- rejected-token 计数；
- `num_query_per_req` 和 `num_speculative_tokens`。

关键输出包括：

- draft `input_ids`；
- context/query positions；
- per-group context/query slot mapping；
- `token_indices_to_sample`。

对于 Qwen3.6 等 GDN/hybrid target，KV manager block size 可能为了匹配 mamba page 而被
放大。slot ID 必须使用 kernel block size；非 GDN 场景使用 group cache spec 的 block
size。混用 manager block size 会把 query K/V 写到错误或越界位置。

如果存在 rejected tokens，effective sequence length 先减去 rejected 数量，再增加
DSpark query 宽度。

## 9. Draft attention metadata

DSpark query attention 使用：

```text
attn_state = ChunkedPrefill
causal     = False
attn_mask  = None
max_query_len = Q
```

uniform anchor-first 场景的 query start locations 为：

```text
[0, Q, 2Q, ..., BQ]
```

正常情况下最后一个位置恰好等于 draft graph token 数。对于非 uniform、DP padding 或
更大 bucket，如果最后一个位置小于 graph token 数，则追加一个尾部虚拟请求，使：

```text
query_start_loc[-1] == num_input_tokens
```

虚拟请求的 KV length 必须为正数，当前使用 1。FIA 即使忽略 padding token 的输出，
也不能接收 0 长 KV 请求。DFlash 不存在默认 N/N+1 宽度差，继续使用原有 0 padding
语义。

每个 group 的 capture metadata 最终按 layer name 展开，使 draft 的全部 attention 层
都能在 capture/replay 时取到正确 metadata。

## 10. Context-KV 预计算

Qwen draft model 对全部 draft attention 层执行一次融合 context-KV 预计算：

1. 对 target context hidden states 做 hidden norm；
2. 用融合权重一次性投影所有层的 K/V；
3. 每层执行 K norm；
4. 合并层和 context 维度执行 RoPE；
5. 根据 per-layer context slot mapping 写入每层 KV cache。

输入宽度始终为 `C=N+1`，与 query graph 的 `Q=N` 独立。这里的 cache update 是 draft
attention 正确性的前置条件；如果 replay 中缺失更新，draft query 会读取上一轮或错误
位置的 context K/V，接受率会显著下降。

## 11. Draft token 生成

query backbone 输出的 sample hidden states 经 LM head 得到 raw logits，并 reshape 为：

```text
[batch, N, vocab]
```

`_dspark_draft_buffer[:, 0]` 保存 target 提供的 seed。随后按位置执行：

```text
markov embedding
  -> markov bias
  -> raw logits + bias
  -> argmax
  -> 写入下一个 draft token
```

最终返回 buffer 的 `[:, 1:]`，即 N 个 draft token。当前 V1 实现只支持 greedy draft
sampling，配置 probabilistic sampling 会在初始化阶段直接报错。

## 12. 正确性不变量

实现和后续修改必须保持以下不变量：

1. target descriptor 仍表达 `N+1` 验证语义；
2. 默认 DSpark query graph 实际执行 `N` token/request；
3. context-KV precompute 始终消费 `N+1` hidden states/request；
4. query、context、sample buffer 的设备地址在 capture/replay 间稳定；
5. per-layer slot list 在正式 capture 前非空，并引用持久 per-group tensor；
6. memory profile 早于 KV 初始化时不得强制绑定；
7. 每个 draft attention 层都必须捕获 context KV-cache update；
8. `query_start_loc[-1]` 必须等于 graph query token 数；
9. padding request 的 DSpark KV length 必须大于 0；
10. runtime metadata、draft graph params 和实际 query tensor 宽度必须一致；
11. eager 与 ACLGraph 在相同输入下应产生一致 draft tokens 和接受率。

## 13. 代码职责

| 文件 | 职责 |
|---|---|
| `vllm_ascend/spec_decode/dspark_proposer.py` | DSpark buffer、group、输入准备、双宽度和 capture metadata |
| `vllm_ascend/spec_decode/llm_base_proposer.py` | graph dispatch、draft 宽度扩展点、padding、graph replay、MarkovHead |
| `vllm_ascend/worker/model_runner_v1.py` | target/draft capture size 注册和 model capture 调度 |
| `vllm_ascend/patch/worker/patch_qwen3_dflash.py` | Qwen 融合 context-KV 投影和 per-layer cache update |
| `vllm_ascend/attention/attention_v1.py` | attention graph params 捕获和 replay 更新 |
| `tests/ut/spec_decode/test_dspark_proposer.py` | DSpark 图宽、metadata、padding、group 和生命周期回归测试 |

## 14. 回归测试覆盖

重点测试包括：

- uniform descriptor 的 target/draft 图宽解耦；
- bonus-anchor 保持 `N+1` query 宽度；
- context width=`N+1`、query width=`N`；
- layer-to-group context slot 引用顺序和 tensor identity；
- pre-KV memory profile 跳过 slot binding；
- FULL graph capture 构造 per-layer metadata；
- eager 不执行 graph padding；
- graph query-start-location 的尾部闭合；
- DSpark padding request 使用正 KV length；
- DFlash 继续保持原有 padding 行为；
- multi-group metadata 构造；
- rejected-token、GDN kernel block size 和 context hidden-state copy。

## 15. 实机验证结果

验证日志：

```text
dspark_analysis_log/dspark_eager_try_5.info
```

文件名包含 `eager`，但日志中的实际启动配置是：

```text
cudagraph_mode=FULL_DECODE_ONLY
cudagraph_capture_sizes=[8]
```

验证结果：

- target capture width：8；
- draft query capture width：7；
- context-KV width：8；
- per-layer context slot mapping：5 层已绑定；
- graph capture 成功并发生 ACLGraph replay；
- replay 中每一步、每一层 projected K/V 与写回 cache 的 K/V 一致；
- 服务启动完成且真实请求执行成功；
- `Accepted=14`、`Drafted=14`；
- 平均 draft acceptance rate：`100.0%`。

该运行使用真实 target/draft checkpoint，不是 dummy weights。

## 16. 当前适用范围与限制

- 当前实机验证为 V1 Model Runner、TP=4、单请求、`N=7`、FULL decode graph；
- draft model 为 dense GQA、5 个 full-attention 层，不经过 SWA 分支；
- probabilistic draft sampling 暂不支持；
- 多请求、非 uniform 和 DP padding 已有单元级形状覆盖，仍建议补充对应 NPU 实机回归；
- dense model 不涉及 expert parallel 或 flashcomm1；
- 临时精度探针和专用诊断日志已全部删除，生产代码只保留正常框架日志。

## 17. 后续维护建议

1. 修改 token geometry 时，同时检查 descriptor、draft graph params、query metadata 和
   context width，不能只改其中一处；
2. 修改 buffer 生命周期时，分别检查 pre-KV profile、capture 和 replay 三个阶段；
3. 新增 KV group 拓扑时，验证 layer order 与 per-layer slot list 的对应关系；
4. 新增 capture bucket 或 DP padding 策略时，验证 query-start-location 尾部和正 KV
   length；
5. 精度回归优先比较 eager/graph draft tokens 和接受率；只有需要定位时才使用临时
   设备 probe，问题关闭后不应保留在生产热路径中。
