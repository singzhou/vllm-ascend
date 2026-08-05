# Qwen3.6 DSpark 草稿模型 ACLGraph 精度诊断说明

## 1. 问题与本轮目标

当前现象：同一套 Qwen3.6-27B 主模型和 DSpark 草稿模型，草稿模型 eager 时精度与接受率正常；草稿模型进入 ACLGraph 后服务可启动、可推理，但接受率显著下降。

本轮不直接改算法，而是先建立 eager/ACLGraph 可一一对齐的证据链，回答以下问题：

1. 图重放前，DSpark 的 seed、query、position、slot mapping 和 KV 长度是否已经与 eager 不同；
2. capture metadata 与真实 replay metadata 的 request 分段是否一致；
3. 若输入和 metadata 一致，数值最早在草稿模型 backbone、LM head 还是 Markov head 分叉；
4. 最终 draft token 是否来自本轮 graph replay，而不是 capture 值、上一轮值或错误的输出别名。

本报告基于静态代码分析。本机不能启动真实 NPU 服务，因此不能在收到真实权重、真实请求的 eager/graph 对照日志前宣称根因已经闭环。

## 2. 已确认的模型边界

本次只分析用户给定的模型组合：

- 主模型：Qwen3.6-27B；
- 草稿模型：Qwen3 DSpark；
- 草稿 attention：普通 dense GQA；
- 草稿层数：5 层；
- 5 层全部为 full attention；
- 无 SWA；
- DSpark anchor-first 时，投机 query 数为 `N = SpecStep`；
- 主模型验证长度和统一 decode 图桶步长为 `N + 1`。

因此，本轮可先排除草稿模型自身的 GDN、DSA、SFA、SWA 窗口和 MoE/EP 分支。主模型包含 GDN 时仍会影响全局 KV manager 的 block size，但当前代码已在生成 DSpark slot id 时使用 draft attention 的 kernel block size；该项不是“仅入图后精度下降”的首要嫌疑。

## 3. 关键执行链

真实推理的核心路径如下：

```text
AscendDSparkProposer.set_inputs_first_pass
  -> 构造每请求 N 个 DSpark query
  -> seq_lens = 原 seq_lens - rejected + N
  -> graph dispatcher 将 N*B pad 到 (N+1)*B 图桶
  -> FULL graph 增加尾部虚拟 request，使 q 累积长度覆盖图桶
  -> build_draft_attn_metadata(draft_index=1)
  -> set_ascend_forward_context
  -> ACLGraphWrapper replay / eager runnable
  -> DSpark backbone
  -> LM head raw logits
  -> N 次 Markov bias + argmax
  -> draft_token_ids[:, 1:]
```

启动 capture 路径由 `AscendDSparkProposer.dummy_run()` 构造 dummy metadata，再调用相同的 `_runnable` 捕图。Python 日志若写在 `_run_merged_draft()` 内，只会在 capture 或 eager Python 执行阶段出现，不能代表每次 graph replay。因此本轮将日志放在图外；必须观察的图内数值仅通过持久 device buffer 做小切片复制，graph 返回后再打印。

## 4. 静态分析结论与当前假设

### 4.1 已修复且必须继续验证：DSpark `seq_lens` 不能被 target runner 覆盖

DSpark 正确语义是：

```text
draft_seq_len[i] = target_seq_len[i] - rejected[i] + N
```

当前分支已把 FULL graph 的保留条件改为：

```python
if self.method in ("dflash", "dspark"):
    preserve proposer-built seq_lens
```

也就是说，上一轮最明确的 `seq_lens` 覆盖问题在当前 HEAD 已被修正。但仍需由 `[inputs]` 与 `[metadata]` 日志确认最终进入 FIA graph-task update 的 `actual_seq_kv` 确实等于上述公式，尤其要覆盖 rejected token 非零的轮次。

### 4.2 P0 候选：capture/replay 的虚拟 request KV 长度不同

anchor-first 下，真实 query 为 `N*B`，图桶为 `(N+1)*B`。当前实现将差值 `B` 作为一个尾部虚拟 request：

```text
N=7, B=2
真实 qsl:       [0, 7, 14]
入图后 qsl:     [0, 7, 14, 16]
虚拟 request q: 2
```

目前 capture 中虚拟 request 的 `seq_lens` 补 1，真实 replay 通过 `_adjust_tensor()` 补 0。两者都会进入普通 FIA metadata，并用于 graph task 的动态参数更新。即使虚拟输出不采样，capture/replay 的 KV 长度约束不同仍可能改变算子校验、workspace/任务参数或错误地复用图任务。

对应证据：`[metadata] capture_layout.capture_seq_lens` 与 `actual_seq_kv` 的最后一项。优先检查二者是否分别为 1 和 0，以及把 runtime padding KV length 临时统一为 1 后接受率是否恢复。

### 4.3 P0 候选：图桶 padding request 的分段结构影响普通 FIA 图任务

当前策略不是给每个真实请求补 1 个 query，而是把全部 `B` 个 padding query 聚合成一个虚拟 request。当 `B` 较大时，该虚拟 request 的 query length 可超过 `decode_threshold=N+1`，使 metadata 的 decode/prefill 分类或捕获的算子形态与小 batch 不同。

这不一定改变 full、non-causal GQA 的有效 query 数值，但必须通过不同 batch 的日志确认：

- `actual_seq_q[-1] == graph_tokens`；
- 前 B 个边界严格为 `N, 2N, ..., BN`；
- 只多一个虚拟 request；
- capture 和 replay 的 q 分段完全相同；
- batch 增大后，分叉是否首次出现在虚拟 request 长度超过阈值时。

### 4.4 P1 候选：FIA graph-task 更新映射或时序错误

普通 GQA FULL graph 并非只使用 capture 时的 q/KV 长度。`AscendAttentionBackendImpl.update_graph_params()` 会从本轮 `draft_attn_metadatas` 取：

- `actual_seq_lengths_q`；
- `seq_lens_list`；
- `block_tables`；
- `causal=False` 对应的 sparse mode。

如果 `[inputs]`、`[metadata]` 均与 eager 符合预期，但 `hidden_probe` 从第一轮就不同，问题应继续向 graph-task 参数更新、事件时序、捕获的 FIA handle 与 layer key 映射收敛，而不是继续怀疑 Markov head。

当前 5 层均为同一普通 GQA backend，降低了混合 backend layer-order 错配的概率，但不能排除 graph params 中 captured op 数与 5 个 draft layer key 数量不匹配。

### 4.5 P1 候选：持久输入/输出 buffer 地址或本轮数据未更新

ACLGraphWrapper 不替调用者复制动态输入，只要求调用者使用稳定 buffer。当前需要验证：

- seed buffer 每轮是否等于 target sampled next token；
- `input_ids`、`positions`、`token_indices_to_sample` 是否为本轮值；
- graph 输出是否随 seed 和上下文改变；
- 连续两轮是否出现 output/probe 完全重复，但输入已经变化。

若输入变化而所有输出探针保持上一轮值，优先怀疑 graph output alias、持久 buffer 更新或 replay/update 时序。

### 4.6 暂不列为首要根因

- 草稿模型 DSA/SWA 索引：当前模型没有这些层；
- Markov 算法本身：同权重 eager 接受率正常；
- draft LM head 权重加载：同权重 eager 正常，除非 graph probe 证明 backbone hidden 一致而 raw logits 不同；
- 主模型 GDN 语义：可能影响全局 KV 表，但无法单独解释草稿 eager 正常、graph 错误；
- `build_for_graph_capture()` 与普通 GQA 的 `build_for_drafting()`：普通 `AscendAttentionMetadataBuilder` 的 drafting 默认仍调用同一个 `build()`，`fast_build` 在此 builder 中没有改变字段。特殊 DSA/SFA backend 才需要重新评估该差异。

## 5. 打点清单

所有日志均使用 `logger.info`，统一前缀 `[dspark_diag]`，只在全局 rank 0 的前 6 个真实 proposer 调用输出。没有使用 `DEBUG` 级别。

### 5.1 `AscendDSparkProposer.__init__`：静态几何

文件：`vllm_ascend/spec_decode/dspark_proposer.py`

日志：`[dspark_diag][init]`

字段：

- `use_aclgraph`：草稿模型是否真正启用图；
- `sample_from_anchor`：是否走 anchor-first；
- `num_speculative_tokens`：N；
- `num_query_per_req`：应为 N，bonus-anchor 时才是 N+1；
- `max_query_tokens`：是否能覆盖 `(N+1)*max_batch`。

意义：先排除“以为草稿入图但实际 eager”以及 N/N+1 配置错误。

### 5.2 `set_inputs_first_pass()`：图外真实输入

日志：`[dspark_diag][step=k][inputs]`

该函数在 ACLGraphWrapper 之前执行，所以日志对应真实请求，不会只在 capture 阶段打印。

字段与目的：

| 字段 | 目的/预期 |
|---|---|
| `batch` | 真实请求数 B |
| `context_tokens` | 本轮 target hidden/context token 总数 |
| `num_query_total` | anchor-first 应为 `B*N` |
| `num_sample_total` | 应为 `B*N` |
| `rejected` | 上一轮每请求拒绝数 |
| `qsl_before` | target 原始 query 分段 |
| `seq_lens_before` | target 原始 KV 长度 |
| `next_token_ids` | Markov seed 来源 |
| `target_positions` | target 位置输入，检查错位 |
| `hidden_head/tail` | combined target hidden 的首/尾小切片，比较 eager/graph 输入是否一致 |
| `input_ids` | DSpark query token/mask 布局 |
| `query_positions` | DSpark N 个 query 的位置 |
| `sample_indices` | 从 N 个 backbone 输出中取 logits 的位置 |
| `query_slots` | 实际 query 的 KV 写槽 |
| `qsl_after` | 应为 `[0,N,2N,...,B*N]` |
| `seq_lens_after` | 应逐项等于 `before - rejected + N` |

该日志会从 NPU 拷少量 tensor 到 CPU，产生同步，只保留前 6 轮用于诊断。

### 5.3 `build_draft_attn_metadata()`：FIA 实际消费的 metadata

日志：`[dspark_diag][step=k][metadata]`

该函数同样在 graph replay 之前的 Python 路径执行，记录的 metadata 会被 `_update_full_graph_params()` 用于普通 FIA graph task 更新。

字段与目的：

| 字段 | 目的/预期 |
|---|---|
| `actual_tokens` | 真实 DSpark query，anchor-first 为 `B*N` |
| `graph_tokens` | FULL graph 图桶，通常为 `B*(N+1)` |
| `padded_reqs` | 有 padding gap 时通常为 `B+1` |
| `actual_seq_q` | 累积 q 边界；最后必须等于 `graph_tokens` |
| `actual_seq_kv` | FIA 的实际 KV 长度；前 B 项必须保持 DSpark 公式 |
| `block_table_shape` | 行数必须覆盖 `padded_reqs` |
| `slot_mapping_len` | 普通 builder 中通常只保留 `actual_tokens` 个真实写槽 |
| `causal/attn_state` | 当前模型应为 `False/ChunkedPrefill` |
| `capture_layout` | 相同 `graph_tokens` 图在 dummy capture 时的结构摘要 |

`capture_layout.capture_seq_lens` 是 dummy 值，不要求与真实请求的有效 KV 长度相同；重点是 list 长度、padding 项约定和 q 分段必须结构兼容。

### 5.4 `_run_merged_draft()`：图内 device 探针

文件：`vllm_ascend/spec_decode/llm_base_proposer.py`

图内没有 logger/print，只做三个很小的 device-to-device `copy_`：

1. `hidden_probe`：进入 LM head 的第一个 sampled hidden 的前 8 维；
2. `raw_logits_probe`：Markov bias 之前 raw logits 第一行的前 8 维；
3. `markov_bias_probe`：N 个 Markov step 各自第一行 bias 的前 8 维。

持久 buffer 会成为 captured graph 的可写目标，因此每次 replay 都能刷新。复制量很小，但属于临时诊断操作，根因确认后应删除。

### 5.5 `_propose()` 返回后：图外打印真实 replay 结果

日志：`[dspark_diag][step=k][output]`

字段：

- `seed`：本轮第一个请求的 Markov seed；
- `draft_token_ids`：最终返回 verifier 的 token；
- `hidden_probe`；
- `raw_logits_probe`；
- `markov_bias_probe`。

该日志在 `super()._propose()` 返回后执行。对 graph 模式而言，ACLGraph replay 已经提交完成；随后从 probe buffer 拷 CPU 会同步并拿到本轮图结果，避免把 capture 阶段的 Python 值误当成 replay 值。

## 6. 如何运行对照

### 6.1 两组服务必须只改变草稿模型是否入图

建议保持以下内容完全一致：

- 主模型权重与主模型图配置；
- 草稿模型权重；
- TP/DP、block size、max model len、capture sizes；
- prompt、chat template、temperature、seed 和 max tokens；
- 并发数，首轮先固定为单请求；
- 不更改 Mamba/GDN 相关 patch。

仅切换 speculative config 中草稿模型的 `enforce_eager`：

```text
对照 A：草稿 enforce_eager=true   （主模型仍可入图，草稿 eager）
对照 B：草稿 enforce_eager=false  （主模型与草稿均入图）
```

必须使用真实权重；dummy 权重只能验证 capture/执行路径，不能判断接受率或数值精度。

### 6.2 请求建议

先用固定单请求、greedy 解码：

```json
{
  "temperature": 0,
  "max_tokens": 32,
  "messages": [{"role": "user", "content": "使用两次运行完全相同的固定 prompt"}]
}
```

为了让 step 0～5 可对齐，建议每次重启服务后只发这一条请求。不要先发送健康检查之外的生成请求，否则日志中的前 6 个 proposer step 会被别的请求占用。

### 6.3 收集日志

分别保存完整 eager 和 graph 服务日志，例如：

```bash
rg '\[dspark_diag\]' eager_server.log > eager_dspark_diag.log
rg '\[dspark_diag\]' graph_server.log > graph_dspark_diag.log
```

请回传：

1. `eager_dspark_diag.log`；
2. `graph_dspark_diag.log`；
3. 两次运行的完整启动命令或 speculative config；
4. 相同请求体；
5. 两次接受率统计；
6. 若有报错，再附首个 traceback 前后至少 100 行。

## 7. 日志判读顺序

按同一 `step=k` 逐层比较，第一次出现差异的位置就是下一轮排查边界。

| 最早差异 | 结论/下一步 |
|---|---|
| `[init]` 的 N、query 数不同 | 配置或 bonus-anchor 语义错误 |
| `[inputs]` 的 seed/position/hidden 不同 | graph 配置间接改变了主模型输出或 input prep |
| `[inputs]` 相同，但 `seq_lens_after` 不满足公式 | rejection 或 DSpark KV length 构造错误 |
| `[inputs]` 相同，`[metadata]` 的前 B 个 KV length 不同 | FULL graph padding/metadata 覆盖错误 |
| capture/replay 的 q list 长度或尾边界不同 | graph task 的 TND 分段结构不一致 |
| 输入与 metadata 一致，`hidden_probe` 不同 | 草稿 backbone graph，优先查 FIA/KV cache/graph-task update |
| hidden 相同，`raw_logits_probe` 不同 | LM head 的 graph/TP 路径 |
| raw logits 相同，`markov_bias_probe` 不同 | seed、Markov embedding/head 或 TP gather |
| 所有 probe 相同，draft token 不同 | argmax、in-place logits、draft buffer 或输出 alias |
| 输入已变化，但连续 step 的所有 probe/output 不变 | graph replay 使用陈旧 buffer/上一轮参数 |

浮点 probe 应优先比较“是否明显不同”。BF16 下不要求字符串逐位完全一致；如果差异只在很小的舍入误差，而 argmax 与接受率一致，则不应误判为根因。

## 8. 收到日志后的优先 A/B

根据当前静态结论，建议按以下顺序做最小 A/B，每次只改一处：

1. 将 runtime 虚拟 request 的 `seq_lens` padding 从 0 改为与 capture 一致的 1；
2. 对 B=1 和 B=2 分别比较，确认尾部虚拟 query 长度变化是否触发分叉；
3. 若 metadata 完全一致但 hidden 分叉，在 `update_graph_params()` 增加 captured op 数、draft layer key 和每层 q/KV list 的 rank-0 INFO；
4. 若 hidden 一致但 raw logits 分叉，单独让 LM head eager，保持 backbone 入图；
5. 若只在 Markov 阶段分叉，将 Markov head 暂时移出大图做等价性 A/B，而不是先改算法。

bonus-anchor（query 数恰为 N+1）可作为高辨识度实验：它消除了 DSpark 与主模型图桶的 N/N+1 gap。若 bonus-anchor 图精度恢复，而 anchor-first 仍错误，根因高度集中在 padding request/layout；若二者都错，则更应检查 graph-task 更新或持久 buffer。

## 9. 已知代价与清理要求

- 前 6 个真实 proposer step 会把少量 NPU tensor 拷到 CPU，造成同步和性能下降；
- 图内额外包含三个小 probe buffer 的 `copy_`；
- 这些日志用于定位，不用于性能测试；
- 接受率/精度 A/B 完成后，必须移除 probe、INFO 打点和临时同步，再做正式吞吐测试；
- 本轮没有新增环境变量，避免把临时调试开关扩散到公共配置。

## 10. 当前验证状态

| 项目 | 状态 |
|---|---|
| 静态路径检查 | 已完成 |
| eager/graph 输入与 metadata 打点 | 已加入 |
| graph 内 backbone/LM head/Markov 数值探针 | 已加入 |
| Python 语法编译 | 已通过（字节码缓存重定向到 `/tmp`） |
| ruff | 当前本机没有 `ruff` 可执行文件，待容器运行 |
| DSpark UT | 当前宿主缺少可导入的 vLLM/torch_npu 运行环境，待服务容器运行 |
| 真实 NPU eager 接受率 | 待用户运行 |
| 真实 NPU ACLGraph 接受率 | 待用户运行 |
| 根因闭环 | 待 eager/graph 对照日志 |
