# DSpark ACLGraph 精度问题调查与解决方案（Dense GQA 修订版）

## 1. 修订说明

当前 Qwen3.6 DSpark 草稿模型已确认是普通 dense GQA，而不是 DSA/SFA/SWA：

- `dflash_config.attention_mode = "gqa"`；
- 5 个 draft layer 全部是 `full_attention`；
- `num_attention_heads = 32`；
- `num_key_value_heads = 8`；
- `sliding_window = null`；
- `use_sliding_window = false`。

证据来自 `/opt/zsy/SpecForge/configs/qwen3.6-27b-dspark.json`。

因此，上一版报告中关于 `dspark_swa_indices` 的分析不适用于当前模型，应撤销。当前路径应按：

```text
Qwen3DSpark dense GQA
  -> AscendAttentionMetadataBuilder
  -> AscendAttentionBackendImpl
  -> full_graph_fia / update_graph_params
```

重新评估。

重新检查后的首要结论是：**FULL graph 分支覆盖了 DSpark 在 `set_inputs_first_pass()` 中构造好的 `seq_lens`，导致 graph 与 eager 使用不同的 KV 有效长度。** 这会直接破坏 DSpark 非因果并行草稿注意力，是当前最可信、最直接的严重精度根因。

## 2. 修订后的结论摘要

| 优先级 | 问题 | 对当前 dense GQA 的影响 |
|---|---|---|
| P0 | FULL graph 用 `runner.seq_lens` 覆盖 DSpark 的 `effective_seq_lens + N` | 直接改变实际 KV 可见长度，足以造成严重接受率下降 |
| P1 | dummy capture 与 real replay 的 padding request 结构可能因 graph mode 不同而不一致 | 可能造成 graph task 更新时 batch/list 结构不一致 |
| P1 | `query_start_loc` padding 逻辑放在公共基类且通过 NPU scalar 做 Python 判断 | 当前 dense GQA 下未必损害有效 token，但设计脆弱且有同步开销 |
| P1 | 单测全部 mock metadata builder，没有验证真实 FIA graph params | 无法发现 KV length、q-length 和 block table 的真实差异 |
| 非根因 | `dspark_swa_indices` | 当前模型没有 SWA/DSA，不适用 |
| 应保留 | GDN target 下使用 draft kernel block size | 修复 BlockTable 地址单位，方向正确 |
| 应拆分 | `mamba/postprocess.py` 修改 | 属于主模型 GDN 状态路径，不是 dense DSpark 小模型图路径 |

当前 diff 已经能完成 graph key 初始化、capture 和 replay，但还没有做到 eager/graph attention metadata 等价，不建议以当前状态合入。

> 本报告为静态调查。没有在当前环境启动真实 Qwen3.6 服务或执行 NPU 接受率测试。最终根因应按第 8 节通过 tensor 对比和接受率 A/B 进行确认。

## 3. Dense GQA 图执行时真正维护什么

当前 dense GQA 使用普通 Ascend Attention/FIA。它与 DSA 路径有一个关键区别：metadata 并非完全依赖固定 tensor 地址。

### 3.1 Capture 阶段

`dummy_run()` 构造 attention metadata，然后进入 `_runnable()`。图捕获期间，`full_graph_fia()` 会：

1. 读取 `actual_seq_lengths_q[-1]` 作为 graph params 的 token-size key；
2. 捕获 FIA query/KV/output 等 tensor 地址；
3. 保存 FIA graph task handle、event、workspace 和算子参数。

### 3.2 Replay 阶段

真实 `_propose()` 会重新构造本轮 metadata。随后 `_update_full_graph_params()` 调用 `AscendAttentionBackendImpl.update_graph_params()`，从新的 draft metadata 中取出：

- `actual_seq_lengths_q`；
- `seq_lens_list`；
- `block_tables`；
- `causal` 对应的 sparse mode。

然后用 `torch.npu.graph_task_update_*` 更新已经捕获的 FIA task。外部 event 负责协调 graph replay 与参数更新。

因此，对当前 dense GQA 来说：

- input ids、positions、hidden states、slot mapping 等模型输入仍需要稳定 buffer；
- q-length、KV-length、block table 可以通过 graph task update 更新；
- 但传入 `update_graph_params()` 的 metadata 数值必须与 eager 完全一致；
- graph params 的 key 和 capture/replay metadata 结构必须一致。

上一版报告所说“每轮新 metadata 对象完全不会被 graph 使用”不适用于这条普通 FIA 路径。

## 4. P0 根因：FULL graph 丢失 DSpark 自己的 `seq_lens`

### 4.1 eager 路径的正确语义

DSpark 在 `vllm_ascend/spec_decode/dspark_proposer.py:286-304` 中构造第一轮 draft attention 输入：

```python
effective_seq_lens = cad.seq_lens
if has_num_rejected:
    effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu

cad.seq_lens = effective_seq_lens + self.num_query_per_req
```

对于当前 anchor-first 配置：

```text
num_query_per_req = N = 7
```

也就是说，eager 传给 dense GQA attention 的 KV length 是：

```text
原始 seq_len - 本轮 rejected token 数 + 7 个并行 draft query
```

这是 DSpark parallel drafting 的核心语义。7 个 query 的 K/V 会先写入 draft KV cache，随后非因果注意力需要在正确的 `seq_lens` 范围内读取它们。

### 4.2 graph 路径如何把它覆盖掉

进入 FULL graph 后，基类执行 `vllm_ascend/spec_decode/llm_base_proposer.py:840-845`：

```python
if self.method == "dflash":
    common_attn_metadata.seq_lens = self._adjust_tensor(
        common_attn_metadata.seq_lens, num_reqs_padded
    )
else:
    common_attn_metadata.seq_lens = self._adjust_tensor(
        self.runner.seq_lens, num_reqs_padded
    )
```

DSpark 原本被强制 eager，所以历史代码没有让它进入这里。现在 graph 被打开后，DSpark 落入 `else`：

- `set_inputs_first_pass()` 的 `+ N` 被丢弃；
- `num_rejected_tokens_gpu` 对 effective length 的修正被丢弃；
- graph metadata 重新使用 target runner 的原始 `seq_lens`。

这是一个明确的 eager/graph 数据差异，不依赖任何模型推测。

### 4.3 为什么会严重影响 dense GQA 精度

普通 Ascend metadata builder 在 parallel drafting 下会明确使用 device `common_attn_metadata.seq_lens`：

```python
elif self.speculative_config and self.speculative_config.parallel_drafting:
    seq_lens = common_attn_metadata.seq_lens
```

随后生成 `seq_lens_list`，并在 graph replay 更新时作为 FIA 的：

```text
actual_seq_lengths_kv
```

假设 eager 应使用：

```text
[L1 - R1 + 7, L2 - R2 + 7]
```

当前 graph 却可能使用：

```text
[L1, L2]
```

这会让 draft query 看到错误的 KV 尾部范围。对 parallel non-causal drafting 来说，这不是轻微数值差异，而是注意力输入集合发生变化。第一层 draft attention 就会与 eager 分叉，最终 Markov head logits 和 draft token 接受率都会明显下降。

该问题应列为第一修复项。

## 5. 重新评估 `query_start_loc` padding

### 5.1 当前实现

anchor-first DSpark 每请求只有 `N` 个 query，而 target graph bucket 使用 `N+1`。例如 `N=7, B=2`：

```text
实际 DSpark query: 14
物理 graph bucket: 16
```

diff 把：

```text
[0, 7, 14]
```

改为：

```text
[0, 7, 16]
```

即让最后一个请求额外承载两个 padding query。

### 5.2 对当前 full non-causal GQA 的影响

在当前严格限定下：

- full attention；
- `causal=False`；
- 无 sliding window；
- padding query 位于最后；
- padding slot mapping 为 `-1`；
- padding 输出不会参与采样；
- 每个有效 query 都使用同一个 request-level KV length。

额外的尾部 query 一般不会改变前 7 个有效 query 的 attention 结果。它们只是在同一个 request segment 后面增加无效 query 行，不会像 SWA 那样通过 query length 重算窗口。

因此，**仅凭 `[0,7,16]` 不能认定是当前 dense GQA 接受率严重下降的主因。** 上一版报告在这里把 SWA 语义错误套到了 full GQA，需要纠正。

### 5.3 它仍然是一个兼容性风险

当前策略只有在上述全部不变量成立时才安全。以下场景不能直接复用：

- causal attention；
- sliding-window attention；
- query length 参与稀疏索引或 mask 构造的 backend；
- padding 不在尾部；
- padding token 会写 KV；
- padding hidden state 会参与后续采样或归约。

另外，runtime 的 `_pad_query_start_loc_for_fia()` 在某些 `CUDAGraphMode.FULL` 分支会追加虚拟 padding request，而新 `dummy_run()` 总是采用“最后请求吸收 padding”。这可能导致 capture 与 replay 的 request count/list length 不一致。

所以当前 padding 方案可以作为 dense full-GQA 的最小实现，但必须：

1. 明确 gated 到 `causal=False + no sliding_window + ordinary FIA`；
2. capture 和 replay 使用完全相同的 padding policy；
3. 不把该策略放成所有 DSpark backend 的公共默认行为。

## 6. 其他实现问题

### 6.1 `query_start_loc` 的 NPU scalar 判断会同步

diff 在真实 hot path 中执行：

```python
if qsl[num_reqs] != num_input_tokens:
```

当 `qsl` 位于 NPU 时，Python 条件判断会读取 device scalar，产生 NPU 到 CPU 同步。它不是当前精度主因，但会影响 speculative decode 性能。

这里已经有 `query_start_loc_cpu`，应基于 CPU metadata 或已知的 `num_actual_tokens/num_input_tokens` 判断，不应读取 NPU scalar。

### 6.2 padding 逻辑修改了公共基类

新增逻辑位于 `AscendSpecDecodeBaseProposer.build_draft_attn_metadata()`，虽然通过 `method == "dspark"` 分支间接限定，但仍把一种 dense-GQA 特有的 padding 策略放到了公共 metadata 构造流程。

更精简的做法是：

- 在 DSpark proposer 中准备好完整 common metadata；
- 基类只负责按 group 选择 block table/slot mapping 和调用 builder；
- 不让基类猜测 DSpark 的 graph layout。

### 6.3 capture builder API 应按普通 FIA 路径对齐 DFlash

当前 dummy capture 调用 `build_for_drafting(..., draft_index=1)`。这对拥有特殊 drafting metadata 的 backend 有意义，但当前 dense GQA 使用普通 Ascend attention builder；其 `build_for_graph_capture()` 已支持 `ChunkedPrefill`。

当前目标模型建议使用 DFlash 相同流程：

```python
builder.build_for_graph_capture(
    common_attn_metadata,
    AscendAttentionState.ChunkedPrefill,
)
```

真实 replay metadata 仍走正常 drafting/build 路径。这样 capture 入口、graph workspace 和 graph-param 注册行为与 DFlash 更一致。

为了兼容未来特殊 backend，可以保留一个小的 backend capability 分支，而不是默认所有 DSpark 都强制 `build_for_drafting()`。

### 6.4 新单测没有覆盖真实 builder 和 graph update

diff 新增的大部分测试使用 `MagicMock` builder，因此只能证明：

- 调用了 builder；
- metadata list 被传入 runnable；
- group/layer 映射存在。

它们没有验证：

- graph metadata 的 `seq_lens_list` 是否等于 eager；
- `actual_seq_lengths_q` 是否与 graph params key 一致；
- `update_graph_params()` 实际收到什么 KV length；
- rejection 后 graph 是否仍使用 effective seq length；
- capture/replay request count 是否一致。

### 6.5 GDN block-size 修复应保留

目标 Qwen3.6 带 GDN 时，全局 KV manager 会因为 Mamba page size 放大 attention page。DSpark 小模型本身没有 GDN，但其 dense GQA KV slot 仍通过全局 BlockTable 寻址。

因此：

- `has_gdn=True` 时继续使用 draft backend 的 kernel block size；
- 不把 manager block size误当成 kernel slot 地址单位。

这部分修改方向正确，并同时改善 eager 和 graph，不是本次 graph 精度回退的来源。

### 6.6 Mamba postprocess 应拆分

`mamba/postprocess.py` 修改属于主模型 GDN state copy/acceptance 后处理路径，不属于 dense DSpark 小模型的 graph forward。

建议用相同 Mamba patch 分别测试 DSpark eager 与 graph，避免把主模型状态兼容修改误判为草稿图精度修复。最终也应拆成独立提交。

### 6.7 已检查但不列为当前首要根因的部分

- `_dflash_num_context`：dummy-run 在进入 `_runnable()` 前已经设置为 capture token 数，真实运行则由 `set_inputs_first_pass()` 更新实际 context 数；graph 使用固定物理 shape，context slot 的尾部 padding 会被填成 `-1`。
- query slot mapping：使用 per-group 预分配 buffer，`_pad_draft_buffers()` 会把实际 query 之后的 slot 填成 `-1`。
- token sampling indices：真实 `_propose()` 会把本轮 indices 复制到 `self.token_indices_to_sample` 固定 buffer，graph 捕获的是该稳定地址。
- Markov seed/output：`_dspark_seed_buffer` 和 `_dspark_draft_buffer` 都是预分配 buffer，真实请求会在 replay 前更新。
- block table：普通 FIA 的 `update_graph_params()` 会从本轮 draft metadata 重新绑定 `block_tables`，并非只能使用 dummy capture 的表内容。
- q-length/KV-length 更新通道：普通 FIA 已存在 graph-task update；当前问题不是“没有更新接口”，而是送进更新接口的 DSpark `seq_lens` 已在基类中被错误覆盖。

这些部分仍应通过 NPU trace 验证，但从静态路径看，它们的优先级低于 `seq_lens` 覆盖问题。

## 7. 推荐修改方案

### 7.1 第一阶段：只修当前 dense full-GQA，最小改动

#### 修改一：保留 DSpark 的 `seq_lens`

FULL graph 分支应让 parallel-drafting 方法保留 `set_inputs_first_pass()` 已经构造的 seq length：

```python
if self.method in ("dflash", "dspark"):
    common_attn_metadata.seq_lens = self._adjust_tensor(
        common_attn_metadata.seq_lens,
        num_reqs_padded,
    )
else:
    ...
```

更兼容的写法是基于能力而不是方法名，例如：

```python
if self.parallel_drafting:
    preserve proposer-built seq_lens
```

但应先确认其他 parallel-drafting proposer 是否与 DFlash/DSpark 语义一致，避免无意扩大行为范围。

必须覆盖 rejection 场景，确保保留下来的是：

```text
effective_seq_lens + num_query_per_req
```

而不是简单的 `runner.seq_lens + N`。

#### 修改二：统一 capture/replay padding policy

当前 dense full-GQA 可以继续使用 tail-absorb 策略，但要把它封装在 DSpark proposer 中，并明确校验：

```text
causal == false
sliding_window is None
backend == ordinary AscendAttention/FIA
```

capture 和 replay 必须得到相同的：

- `num_reqs`；
- `actual_seq_lengths_q` list 长度；
- block table row 数；
- `seq_lens_list` 长度；
- graph params key。

不要让 dummy 固定采用 tail-absorb，而 runtime 某些模式采用虚拟 request。

#### 修改三：capture 使用普通 graph-capture builder

对当前 dense GQA，直接对齐 DFlash：

```python
builder.build_for_graph_capture(
    common_attn_metadata,
    AscendAttentionState.ChunkedPrefill,
)
```

这样实现最小，也避免引入当前模型不需要的 DSA/SFA drafting 分支。

#### 修改四：移除 hot-path device scalar 判断

使用以下已知 host 值判断 padding：

```text
num_actual_tokens != num_input_tokens
```

或者读取 `query_start_loc_cpu`，不要读取 NPU tensor 元素。

### 7.2 第二阶段：兼容性设计，但不阻塞当前修复

建议只增加一个很小的 graph-layout policy，而不是为不同模型堆积硬编码：

```text
DSparkGraphLayout
  - num_actual_tokens
  - num_input_tokens
  - num_query_per_req
  - padding_policy: tail_absorb | virtual_request | exact_shape
```

当前 Qwen3.6 dense full-GQA：

```text
padding_policy = tail_absorb
```

未来若出现 causal/SWA/特殊稀疏 backend：

```text
padding_policy = virtual_request
或 exact_shape（draft graph 独立按 B*N 捕获）
```

这样可以满足：

- 当前修改尽量少；
- 配置开关和 capture 生命周期仍与 DFlash 对齐；
- 不把主模型结构传入 DSpark 小模型；
- 不让 dense-GQA 的安全假设污染其他 attention backend。

这里的“兼容性”应体现为清晰的 layout/padding 能力边界，而不是现在就实现 DSV4、DSA 或 SWA。

## 8. 最快确认根因的验证方法

### 8.1 四点 metadata dump

对同一轮请求分别记录：

1. 进入 `set_inputs_first_pass()` 前的 `cad.seq_lens`；
2. `set_inputs_first_pass()` 返回后的 `cad.seq_lens`；
3. FULL graph padding分支后的 `common_attn_metadata.seq_lens`；
4. `AscendAttentionBackendImpl.update_graph_params()` 实际使用的 `seq_lens_list`。

预期当前代码会观察到：

```text
步骤 2: effective_seq_lens + 7
步骤 3/4: runner.seq_lens
```

若成立，就直接确认 graph 覆盖了 DSpark KV length。

### 8.2 最小一行 A/B

只把 FULL graph 判断从：

```python
if self.method == "dflash":
```

临时改成：

```python
if self.method in ("dflash", "dspark"):
```

保持其他 diff、模型权重、prompt 和 graph 配置不变，重新比较接受率。这是成本最低、辨识度最高的验证。

如果接受率大幅恢复，基本可以确认 `seq_lens` 是主因。

### 8.3 layer-wise 对比

依次对比 eager/graph：

1. embedding/输入 hidden states；
2. 第一层 Q/K/V；
3. 第一层 attention output；
4. 最后一层 hidden states；
5. raw logits；
6. Markov bias 后 logits；
7. draft token ids。

若输入一致、第一层 attention 开始分叉，同时 metadata KV length 不一致，不需要继续怀疑 LM head 或 sampler。

### 8.4 高辨识度用例

| 用例 | 目的 |
|---|---|
| `N=7, B=1, rejected=0` | 检查固定 `+7` 是否在 graph 被丢失 |
| `N=7, B=2, rejected=[0,2]` | 检查 rejection 修正是否被 graph 覆盖 |
| bonus-anchor，query step=`N+1` | 消除 N/N+1 padding gap，单独观察 seq-lens 问题 |
| 相同 graph bucket 连续多轮 | 检查每轮 graph task update 的 KV length 是否变化 |
| 修复前后第一层 attention | 比完整接受率更快定位 |

bonus-anchor 如果仍然存在严重差距，而保留 `seq_lens` 后恢复，会进一步说明 padding 不是当前主因。

## 9. 单测和 E2E 建议

### 9.1 必须增加的 UT

1. `set_inputs_first_pass()` 后 `seq_lens = base - rejected + N`；
2. FULL graph padding后该值仍被保留；
3. 普通真实 `AscendAttentionMetadataBuilder` 生成的 `seq_lens_list` 与 eager 一致；
4. `actual_seq_lengths_q[-1] == num_input_tokens`，确保 graph params key 正确；
5. capture/replay 的 request count 和 list 长度一致；
6. `causal=False` 时 graph update 把 sparse mode 更新为 0；
7. padding query 的 slot mapping 为 `-1`；
8. device tensor 上不发生 Python scalar 判断。

### 9.2 修改现有 padding 测试

现有测试只断言 `query_start_loc[-1]` 被扩到图桶大小，信息不足。应同时断言当前策略的安全前提：

- backend 为普通 full attention；
- `causal=False`；
- 无 sliding window；
- 有效请求的起始边界不变；
- padding 只附加在尾部；
- KV slot 为 `-1`；
- `seq_lens` 仍是 DSpark 构造的值。

### 9.3 NPU E2E

现有 `tests/e2e/pull_request/one_card/spec_decode/test_dspark.py` 使用 `PIECEWISE`，不能证明 DSpark full graph 精度。

需要新增 full graph 用例，至少输出：

- eager acceptance per position；
- graph acceptance per position；
- graph replay 日志证据；
- draft token 一致率；
- `N=7` 下无 rejection 和有 rejection 两种场景。

## 10. 建议保留、修改和拆分的代码

| 文件/改动 | 建议 |
|---|---|
| 删除 DSpark 强制 eager | 保留 |
| `model_runner_v1.py` 初始化 DSpark graph keys | 保留 |
| GDN target 的 kernel block size 修复 | 保留 |
| per-group block table/slot mapping | 保留 |
| dummy-run 构造 graph metadata | 保留方向，改用普通 graph-capture builder |
| `max_query_tokens=(N+1)*B` | 当前 tail-absorb 方案下可保留 |
| 基类覆盖 DSpark `seq_lens` | 必须修复 |
| 基类中的 NPU scalar padding 判断 | 移到 DSpark 并改为 host 判断 |
| Mamba postprocess 修改 | 与 DSpark graph 拆分提交和测试 |
| DSA/SWA 兼容代码 | 当前不增加，只保留明确扩展点 |

## 11. 最终判断

在当前 dense full-GQA、非因果、无 SWA 的真实模型结构下：

1. `dspark_swa_indices` 与当前问题无关；
2. tail padding 被最后请求吸收不一定改变有效 query 输出，不能继续列为首要精度根因；
3. 普通 FIA backend 会通过 `update_graph_params()` 消费本轮 metadata，因此 metadata 更新机制本身是存在的；
4. **最明确的错误是 FULL graph 把 DSpark 的 `effective_seq_lens + N` 覆盖成 `runner.seq_lens`；**
5. 该错误直接改变 FIA 的 `actual_seq_lengths_kv`，足以解释严重接受率下降；
6. 修复应先保持 DSpark seq-lens 语义，再统一 capture/replay padding 结构并补真实 builder/E2E 测试。

建议先做第 8.2 节的一行 A/B。若接受率恢复，再完成正式的小范围实现：

```text
保留 proposer-built seq_lens
+ dense-GQA gated padding policy
+ DFlash 风格 graph-capture builder
+ 真实 FIA metadata/graph-update 测试
```

这是当前修改最少、根因最直接、同时为其他 attention 结构保留兼容边界的方案。
