# DSpark 通过 FIA v2 规避 SeqLens 同步的设计说明

日期：2026-08-10  
仓库：`vllm-ascend`  
适用范围：model runner v1、DSpark parallel drafting、普通 `AscendAttentionBackend`  
目标：消除构造 DSpark draft attention metadata 时，由 NPU `seq_lens.tolist()` 引入的 Host 等待

---

## 1. 结论

本改动不是为了宣称推理链路“绝对零同步”，而是消除下面这一个会破坏异步算子下发的关键等待点：

```text
主模型在 NPU 上产生/修正 seq_lens
        ↓
CPU 构造 DSpark metadata 时执行 seq_lens.tolist()
        ↓
必须等待 seq_lens 之前的主模型 NPU 任务完成
        ↓
CPU 无法继续下发 sample tokens 和 draft model 算子
```

解决办法是：

1. DSpark rejection 修正后的精确 `seq_lens` 始终保留为 NPU Tensor；
2. metadata builder 只接收一个 CPU `seq_lens`，用于生成 host 侧结构信息；
3. builder 返回后，把 attention metadata 中的 `seq_lens` 恢复为精确 NPU Tensor，并设置 DSpark 专用标记；
4. ACLGraph 捕获、ACLGraph replay 参数更新和 eager attention 都改用 FIA v2；
5. FIA v2 的 `actual_seq_kvlen` 直接接收 Tensor，不再要求把 NPU 长度转换成 Python `list[int]`。

因此，CPU 可以在主模型仍在 NPU 上执行时继续构造 metadata，并继续下发 sample tokens/draft model 相关算子。`seq_lens` 的数据依赖由 NPU stream 自己维护，不再通过 Host D2H 读取强制等待。

---

## 2. 原始同步是如何产生的

### 2.1 DSpark 的精确长度只能在设备侧及时得到

异步 speculative decoding 中，上一轮 draft tokens 的接受数量由设备侧采样/校验结果决定。DSpark 在 `set_inputs_first_pass()` 中计算本轮 draft attention 使用的长度：

```python
effective_seq_lens = cad.seq_lens
if num_rejected_tokens_gpu is not None:
    effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu

cad.seq_lens = effective_seq_lens + self.num_query_per_req
```

其中：

- `cad.seq_lens` 是 NPU Tensor；
- `num_rejected_tokens_gpu` 是 NPU Tensor；
- 减去 rejected tokens、再加本轮 query 数量的运算都在设备侧执行；
- 得到的 `cad.seq_lens` 才是本轮 attention 应使用的精确 KV 长度。

如果要求 CPU 在同一时刻得到完全相同的值，就必须等待上述 NPU 运算完成并执行 D2H。这正是当前异步调度要避免的等待。

### 2.2 generic builder 把 device Tensor 转成了 Python list

`AscendAttentionMetadataBuilder.build()` 在 parallel drafting 下会选择：

```python
seq_lens = common_attn_metadata.seq_lens
```

随后执行：

```python
seq_lens_list = seq_lens.tolist()
```

当 `seq_lens` 位于 NPU 时，`.tolist()` 不是普通 Python 数据转换。它必须：

1. 等待生成 `seq_lens` 的设备任务完成；
2. 将长度从 NPU 拷贝到 Host；
3. 将 Host Tensor 展开为 Python list。

由于 `seq_lens` 依赖上一轮主模型校验结果，这个等待会把 CPU 卡在 metadata 构造阶段。原本可以和主模型尾部执行重叠的 sample tokens、draft model 算子下发因此被推迟。

---

## 3. 为什么“只用 CPU 长度绕过 builder”还不够

一种不完整的方案是：builder 内临时使用 CPU `seq_lens`，builder 返回后仍让普通 FIA v1 消费 `seq_lens_list`。

这个方案虽然能让 builder 的 `.tolist()` 不再读取 NPU，但存在正确性问题：

- CPU 长度可能还没有包含最新 rejected-token 修正；
- FIA v1 的 `actual_seq_lengths_kv` 仍消费 Host `list[int]`；
- attention 实际计算会使用过期 KV 长度；
- 随着请求连续执行，长度误差可能积累，表现为草稿接受率越来越低。

另一种看似直接的方案是把 NPU Tensor 填进 `seq_lens_list` 字段，但普通 FIA v1 的接口语义仍是 Host IntArray。即使 Python 层没有显式 `.tolist()`，框架或算子适配层也可能拒绝 Tensor，或者在内部把 Tensor 回读到 Host。因此不能用这种方式保证消除关键等待。

完整方案必须同时满足：

```text
builder 不读取 device seq_lens
                +
attention 算子原生消费 device seq_lens
```

FIA v2 提供了第二个条件。

---

## 4. FIA v1 与 FIA v2 在本问题中的差异

### 4.1 FIA v1

普通 FIA v1 调用形式使用：

```python
torch_npu.npu_fused_infer_attention_score(
    ...,
    actual_seq_lengths_kv=seq_lens_list,
)
```

`seq_lens_list` 是 Host `list[int]`。因此只要精确长度当前只存在于 NPU，就需要一次 NPU → Host 可见性等待。

### 4.2 FIA v2

FIA v2 调用形式使用：

```python
torch_npu.npu_fused_infer_attention_score_v2(
    ...,
    actual_seq_kvlen=attn_metadata.seq_lens,
)
```

ACLGraph 捕获/更新使用 `.out` 版本：

```python
torch_npu.npu_fused_infer_attention_score_v2.out(
    ...,
    actual_seq_kvlen=attn_metadata.seq_lens,
)
```

这里的 `actual_seq_kvlen` 可以是 NPU Tensor。Host 只传递 Tensor 描述和地址，不需要先获得每个请求的长度值。

设备侧仍然必须保证 attention 在长度计算完成后执行，但这通过 stream/event 的设备依赖完成。它不会迫使 Host 在下发 sample tokens 算子之前读取长度结果。

---

## 5. 修改后的长度流转

### 5.1 DSpark 生成精确 device `seq_lens`

`vllm_ascend/spec_decode/dspark_proposer.py`：

```text
target seq_lens (NPU)
        - num_rejected_tokens_gpu
        + num_query_per_req
        ↓
current DSpark seq_lens (NPU, authoritative)
```

这部分保持原有设备计算，不引入 Host 读取。

### 5.2 builder 只获得 CPU 结构占位

`vllm_ascend/spec_decode/llm_base_proposer.py::build_draft_attn_metadata()` 在 DSpark 分支中：

```python
device_seq_lens = common_attn_metadata.seq_lens
```

先保存精确的设备 Tensor。然后仅在以下条件成立时启用新路径：

```python
self.method == "dspark"
and isinstance(builder, AscendAttentionMetadataBuilder)
and not enable_c8_quant
```

CPU builder 输入的选择顺序是：

1. `common_attn_metadata._seq_lens_cpu`；
2. `common_attn_metadata.seq_lens_cpu`；
3. 两者都不存在时，创建长度与 batch 相同的 CPU `ones` Tensor。

这个 CPU Tensor 的作用只有：

- 让 builder 生成 batch-size 对齐的 `seq_lens_list`；
- 参与 padding 等结构处理；
- 避免 builder 对 NPU Tensor 执行 `.tolist()`。

它不再决定 DSpark attention 的实际 KV 边界。

### 5.3 builder 返回后恢复 device Tensor

builder 完成后执行：

```python
common_attn_metadata.seq_lens = device_seq_lens
attn_metadata.seq_lens = device_seq_lens
attn_metadata.use_device_seq_lens = True
```

三个动作分别用于：

- 恢复当前 group 的 common metadata，避免多 KV-cache group 时下一个 group 继承 CPU 占位；
- 保证当前 attention metadata 持有精确 NPU 长度；
- 用 `use_device_seq_lens` 标记选择 FIA v2，而不是根据全局 speculative 配置猜测。

### 5.4 ACLGraph 捕获使用 FIA v2

`AscendAttentionBackendImpl.forward_fused_infer_attention()` 在 capture 阶段判断：

```python
if self.sinks is not None or use_device_seq_lens:
    self.full_graph_fia_v2(...)
```

对于 DSpark，`use_device_seq_lens=True`，因此进入 `full_graph_fia_v2()`，最终调用：

```python
torch_npu.npu_fused_infer_attention_score_v2.out(
    ...,
    actual_seq_kvlen=attn_metadata.seq_lens,
)
```

图中捕获的是 tensor-length 版本的 FIA v2，而不是 Host-list 版本的 FIA v1。

### 5.5 ACLGraph replay 更新继续传 device Tensor

图 replay 前，`update_graph_params()` 会检查 draft attention metadata 中是否存在：

```python
metadata.use_device_seq_lens is True
```

若存在，则选择 FIA v2 graph-task 更新分支，并使用：

```python
seq_lens = metadata.seq_lens
```

更新已捕获算子时仍传：

```python
actual_seq_kvlen=seq_lens
```

因此不是只有第一次 capture 使用 device Tensor；每一次真实 batch 的 replay 参数更新也都使用当前 batch 的精确 NPU `seq_lens`。

### 5.6 eager 同样使用 FIA v2

在非 capture 路径中，同一标记会选择：

```python
torch_npu.npu_fused_infer_attention_score_v2(
    ...,
    actual_seq_kvlen=attn_metadata.seq_lens,
)
```

所以 eager 模式可以使用 FIA v2，并且当前实现也确实这样做。这样可以避免 eager 下 builder 使用 CPU 占位后，又回到 FIA v1 消费过期 `seq_lens_list`。

---

## 6. DSpark attention 语义保持

DSpark parallel drafting 将：

```python
cad.causal = False
cad.attn_mask = None
```

因此 FIA v2 不能直接沿用 sinks 模型的默认 causal 参数。DSpark 标记路径显式保持与原 FIA v1 一致的非因果全注意力语义：

```text
sparse_mode = 0
pre_tokens  = SWA_INT_MAX
next_tokens = SWA_INT_MAX
atten_mask  = None
```

如果这里错误地使用 `sparse_mode=3` 或 `next_tokens=0`，每个并行 draft query 能看到的 KV 范围会改变，即使长度同步问题消失，也可能造成接受率下降。

现有 sinks 路径在没有 `use_device_seq_lens` 标记时仍保持原参数：

```text
sparse_mode = 3（无 sliding window 时）
next_tokens = 0
```

因此本次参数变化只作用于 DSpark device-length 路径。

---

## 7. 影响范围控制

新增 FIA v2 使能不是按 `parallel_drafting=True` 全局开启，而是由 DSpark 构造的 metadata 标记驱动。

| 路径 | 是否由本改动新增使用 FIA v2 | 原因 |
|---|---:|---|
| DSpark + 普通 AscendAttention + ACLGraph | 是 | 目标路径，使用 Tensor `actual_seq_kvlen` |
| DSpark + 普通 AscendAttention + eager | 是 | 保持 device-length 正确性 |
| DFlash | 否 | 不设置 DSpark 标记 |
| MTP | 否 | 不设置 DSpark 标记 |
| 普通 target attention | 否 | 不设置 DSpark draft 标记 |
| C8 KV-cache attention | 否 | 当前显式排除，避免绕过专用 C8 实现 |
| compressed/其他 metadata builder | 否 | 只接受 `AscendAttentionMetadataBuilder` |
| 带 attention sinks 的模型 | 不是新增 | 原代码已经使用 FIA v2 |

这个限定避免仅为 DSpark 优化而改变核心 attention 模块的默认行为。

---

## 8. 为什么这能恢复异步下发重叠

修改前的 Host 时序：

```text
提交主模型算子
    ↓
build draft metadata
    ↓
device seq_lens.tolist()  ← Host 在这里等待主模型相关任务完成
    ↓
提交 sample tokens / draft model 算子
```

修改后的 Host 时序：

```text
提交主模型算子
    ↓
build draft metadata（只读取 CPU 结构数据）
    ↓
提交 sample tokens / draft model / FIA v2 算子
    ↓
NPU stream 在设备侧按依赖顺序执行
```

关键变化不是取消设备上的真实依赖，而是取消 Host 为获得 Python `seq_lens_list` 而提前等待该依赖完成。这样 AsyncScheduler 才能继续积累并下发后续算子，让 Host 下发与主模型尾部执行产生重叠。

---

## 9. 临时打印与验证方式

当前代码暂时保留两个打印。

### 9.1 图捕获证明

```text
[DSpark][FIA v2] capturing ACLGraph with tensor seq_lens
```

出现位置：`full_graph_fia_v2()`。

它证明：

- DSpark metadata 标记已经传到 attention backend；
- capture 没有进入 `full_graph_fia()`；
- 捕获的是 `npu_fused_infer_attention_score_v2.out`。

图捕获按 graph size 和 layer 执行，因此该日志可能出现多次。

### 9.2 图回放更新证明

```text
[DSpark][FIA v2] updating ACLGraph with tensor seq_lens
```

出现位置：`update_graph_params()`。

它证明：

- 实际请求的 draft metadata 仍携带 device-length 标记；
- replay 参数更新选择 FIA v2 分支；
- 当前 batch 的 `metadata.seq_lens` 被作为 Tensor 传给 `actual_seq_kvlen`。

### 9.3 打印的使用限制

这两个 `print(..., flush=True)` 仅用于分支确认。`flush=True` 会引入 Host 标准输出开销，因此：

- 可以用于确认功能路径；
- 不应保留在最终吞吐/时延数据采集中；
- 完成验证后应删除两个 print，再做正式性能对比。

---

## 10. 建议的验证清单

### 10.1 路径验证

- 启动 DSpark ACLGraph 模式；
- 日志中看到 capture 打印；
- 首个真实 decode/replay 后看到 update 打印；
- DFlash/MTP 运行时不应出现 DSpark FIA v2 打印；
- `--enforce-eager` 下 DSpark 能正常推理。

### 10.2 同步验证

使用 NPU profiler 对比修改前后 Host timeline：

- 修改前：`seq_lens.tolist()` 附近存在等待主模型 stream 完成的空洞；
- 修改后：builder 阶段不再触发该 NPU → Host 长度读取；
- sample tokens/draft model 算子下发时间应前移；
- 不要求其他无关同步点全部消失。

仅看到两个 print 只能证明分支选择正确，不能代替 profiler 对重叠收益的证明。

### 10.3 正确性验证

- 对比修改前后的 target 最终输出；
- 连续运行足够多轮，观察 draft acceptance rate 是否稳定；
- 覆盖存在 rejected tokens 的请求；
- 覆盖 graph padding/batch bucket 切换；
- 检查 `actual_seq_kvlen` Tensor 的 batch 维与 `actual_seq_qlen`、block table 一致。

如果吞吐改善但接受率持续下降，优先检查：

1. replay 是否错误回到 `metadata.seq_lens_list`；
2. `attn_metadata.seq_lens` 是否被 CPU 占位覆盖；
3. 多 KV-cache group 间是否恢复了原 device Tensor；
4. DSpark 非因果参数是否仍为 `sparse_mode=0`、`next_tokens=SWA_INT_MAX`。

---

## 11. 已知边界

1. 本改动只消除 `seq_lens_list` 强制 Host 等待这一关键点，不承诺整个执行链绝对没有任何同步。
2. 其他模块中的 event synchronize、分布式 shape 读取或日志操作不属于本次修改范围。
3. C8 attention 暂不切换到此路径；其 forward/capture 有专用量化实现。
4. 最终是否完全消除 CANN/torch_npu 内部的隐式等待，需要以目标软件版本上的 profiler 结果为准。
5. FIA v2 与 FIA v1 可能有细微数值差异，应同时观察最终输出和 draft acceptance，而不能只看是否成功运行。

---

## 12. 代码位置

| 文件/符号 | 作用 |
|---|---|
| `vllm_ascend/spec_decode/dspark_proposer.py::set_inputs_first_pass` | 在设备侧计算 rejection 修正后的精确 `seq_lens` |
| `vllm_ascend/spec_decode/llm_base_proposer.py::build_draft_attn_metadata` | builder CPU 占位、恢复 device Tensor、设置专用标记 |
| `vllm_ascend/attention/attention_v1.py::forward_fused_infer_attention` | 根据标记选择 graph/eager FIA v2 |
| `vllm_ascend/attention/attention_v1.py::full_graph_fia_v2` | 捕获 Tensor `actual_seq_kvlen` 的 FIA v2 图任务 |
| `vllm_ascend/attention/attention_v1.py::update_graph_params` | replay 时用当前 device `seq_lens` 更新 FIA v2 参数 |

最终数据流可以概括为：

```text
DSpark device seq_lens
        ├── CPU mirror/placeholder → builder structural fields only
        │                              └── seq_lens_list 不参与 DSpark attention
        │
        └── attn_metadata.seq_lens (NPU Tensor)
                 ├── FIA v2 ACLGraph capture
                 ├── FIA v2 ACLGraph replay update
                 └── FIA v2 eager forward
```

这条“控制信息留 Host、精确动态长度留 Device”的分离，是规避本次 `seq_lens_list` 同步的核心。
