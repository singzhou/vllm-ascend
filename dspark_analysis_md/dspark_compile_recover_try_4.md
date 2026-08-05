# DSpark ACLGraph 精度恢复尝试 4：定位 context KV 与五层 backbone 的首个分叉点

## 1. Try 3 结论

Try 3 已正确形成 DSpark 原生图形状：

- target/context 宽度为 `N + 1 = 8`；
- draft query/FIA 图宽度为 `N = 7`；
- `query_start_loc=[0, 7]`，没有虚拟 request；
- 五个 full-attention FIA op 均在 `graph_tokens=7` 下更新。

但 try 2 和 try 3 的六轮草稿 token 序列逐轮完全相同，接受率也都为
`9/35 = 25.7%`。因此 N/N+1 图宽错误不是本次精度异常的直接根因。

现有日志还证明：相同 seed 的第一组 Markov bias 一致，而 final hidden 和
raw logits 已经分叉。错误位于 draft backbone 的 embedding、context KV 预写
或五层 decoder/FIA 中，不能继续从 sampler 或 MarkovHead 排查。

## 2. 新发现的高优先级嫌疑

静态检查 `dummy_run` 和运行时输入构造后发现：

- 运行时 `set_inputs_first_pass` 会把每个 KV group 的 context slot buffer
  组装成 `_context_slot_mapping_buffers` 列表；
- DSpark 初始化时该字段为 `None`；
- ACLGraph capture 的 `dummy_run` 没有显式组装这个列表；
- `precompute_and_store_context_kv` 在 slot mapping 为 `None` 时只计算 K/V，
  随后直接返回，不捕获 `reshape_and_cache`。

如果启动阶段没有其他路径提前填充该列表，draft graph 将捕获 context K/V
投影，却不会捕获把 8 个 context K/V 写入五层 cache 的算子。运行时即使
`set_inputs_first_pass` 已生成正确 slots，ACLGraph replay 也不会重新执行未被
捕获的 cache write。这与“step 0 eager 正确、第一次 graph replay 立即分叉”
完全吻合。

Try 4 暂不改变该行为，只增加证据点。这样可以避免在没有真实 NPU 证据前
把诊断假设直接变成修复。

## 3. 探针实现原则

所有数值探针遵循以下规则：

1. 在 NPU 上预分配持久 buffer；
2. eager forward 或 ACLGraph capture 时执行小切片 `copy_`；
3. ACLGraph replay 执行已捕获的 `copy_`，不依赖 Python hook 再次运行；
4. 只在 `_propose` 返回后由 TP0 使用 INFO 日志读取；
5. 图内不调用 logger、`print`、`.item()` 或 `.cpu()`；
6. 仅记录前 6 个 runtime step，每个向量只保留前 8 个通道。

probe row 数为 `min(num_query_per_req, 8)`。当前 `N=7`、batch=1 配置会记录
七个 query position，足够逐 position 对齐 eager 与 compile。

## 4. 代码修改和探针位置

### 4.1 捕图时的 context slot 生命周期

文件：`vllm_ascend/spec_decode/dspark_proposer.py`

位置：`AscendDSparkProposer.dummy_run`，进入 FULL ACLGraph capture、调用
`_runnable` 之前。

日志：

```text
[dspark_diag][capture_context]
query_tokens=7
context_tokens=8
slot_mapping_is_none=...
slot_mapping_layers=...
slot_edges=...
```

目的：直接确认捕图时是否向 context KV 预写传入了五层 slot mapping。

判读：

- `slot_mapping_is_none=True`：捕获图必然没有 context cache write，是当前
  第一根因候选；
- `False` 但 `slot_mapping_layers != 5`：层到 slot mapping 的绑定不完整；
- `False` 且五层 `slot_edges` 合理：继续比较 projected/cached K/V。

### 4.2 query embedding 与 final norm

文件：`vllm_ascend/spec_decode/dspark_proposer.py`

探针：

- `query_embedding`：挂在第一个 draft decoder layer 的 forward pre-hook；
- `final_norm`：挂在 draft backbone final RMSNorm 的 forward hook。

没有直接 hook `embed_tokens`，因为当前 checkpoint 没有自己的 embedding，
draft 与 target 共享同一个 embedding module。直接挂共享模块会被 target forward
覆盖，并可能把探针 copy 错误带入 target graph。第一层输入是完成 mask-token
替换后的 draft-only embedding，语义更准确。

日志：

```text
[dspark_diag][step=1][backbone]
query_embedding=[[...], ...]
final_norm=[[...], ...]
```

目的：

- embedding 已分叉：优先检查 graph 的 `input_ids`/共享 embedding 输入刷新；
- embedding 一致、final norm 分叉：错误位于 context KV 或五层 decoder；
- final norm 一致、既有 sampled hidden 不一致：检查 sample indices/unpad。

### 4.3 context projected K/V 与 cache readback

文件：`vllm_ascend/patch/worker/patch_qwen3_dflash.py`

位置：patched `precompute_and_store_context_kv`。

每层记录两个 context row：第一个 context token 和最后一个 context token；每行
记录前 8 个通道。

探针：

- `projected_k/projected_v`：K norm、RoPE 完成后，cache insert 之前；
- `cached_k/cached_v`：`do_kv_cache_update` 完成后，根据同一 slot mapping 从
  实际 KV cache 读回。

`projected_*` 的 copy 位于 `context_slot_mapping is None` 判断之前。因此即使
capture 确实漏掉 slot mapping，graph replay 仍应刷新 projected K/V，而
`cached_*` 不会被 graph 更新。这一组合可以直接证明“计算发生但写 cache
未被捕获”。

日志：

```text
[dspark_diag][step=1][context_kv]
layers=[...5 layers...]
projected_k=[...]
projected_v=[...]
cached_k=[...]
cached_v=[...]
```

判读：

| eager/compile 对比 | 含义 |
| --- | --- |
| projected 已不同 | context hidden、position 或 fused KV/RoPE 图内刷新错误 |
| projected 一致，compile cached 未刷新/与 projected 不对应 | cache write 未捕获或 slot mapping 错误 |
| projected、cached 均一致 | context 预写基本排除，继续找第一层 FIA 分叉 |

注意：K cache 中保存的是 norm+RoPE 后的 K，所以同一层、同一 edge row 的
`projected_k` 和 `cached_k` 应一致；V 同理。BF16 日志允许极小的舍入差异，
不应出现量级或符号的大面积变化。

### 4.4 五层 decoder 的逐阶段输出

文件：`vllm_ascend/spec_decode/dspark_proposer.py`

对每个 draft layer 注册三处 hook，并保留 decoder tuple 的两个分量：

1. `fia`：`Attention` 输出，位于 `o_proj` 之前；
2. `self_attn`：完整 self-attention 输出，位于 `o_proj` 之后；
3. `hidden`：decoder layer 返回的 MLP 输出；
4. `residual`：decoder layer 返回的 residual。

日志：

```text
[dspark_diag][step=1][layer=0][stages]
name=model.layers.64.self_attn.attn
fia=[[...], ...]
self_attn=[[...], ...]
hidden=[[...], ...]
residual=[[...], ...]
```

五层均输出同样格式。按 eager/compile 顺序查找第一处明显分叉：

- embedding 一致、layer 0 FIA 分叉：context cache、query Q/K/V、RoPE 或 FIA；
- FIA 一致、`self_attn` 分叉：`o_proj`；
- self-attention 一致、layer hidden 分叉：post-attention norm 或 MLP；
- hidden 一致、residual 分叉：residual/RMSNorm 融合路径；
- 前一层输出一致、后一层 FIA 分叉：问题收敛到该层 attention。

## 5. 新增测试

文件：`tests/ut/spec_decode/test_dspark_proposer.py`

新增覆盖：

1. 通用 probe copy 只复制目标 buffer 能容纳的二维前缀；
2. projected K/V 探针正确选择第一个和最后一个 context row；
3. cache readback 根据实际 slot mapping 选择物理 cache row；
4. 两层 fake backbone 能安装预期数量的 hooks 和持久 buffers；
5. context probe buffer 正确挂载到 patched Qwen3 draft backbone。

本地桌面环境没有安装 vLLM/NPU Python 依赖、pytest 或 Ruff，无法运行 UT 和
真实图捕获。已完成三个修改文件的 AST 解析、行宽检查及 `git diff --check`。

## 6. 执行要求

需要用完全相同的 prompt 和启动参数分别执行一次 eager draft 与 compile
draft，建议保存为：

```text
/opt/zsy/vllm-ascend/dspark_analysis_log/dspark_eager_try_4.info
/opt/zsy/vllm-ascend/dspark_analysis_log/dspark_compile_try_4.info
```

日志使用 INFO 级别即可，不需要开启全局 DEBUG。compile 服务首先确认出现：

```text
[dspark_diag][probes] installed graph-safe stage probes ...
[dspark_diag][capture_context] ...
```

如果 capture 或 compile 因新增 cache readback 的 `index_select` 报不支持，请
保留完整错误日志；这意味着该只读探针需要替换为对应的 Ascend gather op，
不能据此判断原推理路径失败。

## 7. 下一轮最短判读顺序

只比较两份日志的 step 1，按以下顺序即可：

1. compile `capture_context.slot_mapping_is_none`；
2. eager/compile `query_embedding`；
3. eager/compile `projected_k/v`；
4. 每份日志内部 `projected_k/v` 对 `cached_k/v`；
5. eager/compile layer 0 `fia`；
6. 从 layer 0 到 layer 4 查找首个 stage 分叉；
7. `final_norm`、原有 sampled hidden、raw logits。

其中第 1 项如果为 `True`，已经足以进入下一次修复：在 DSpark `dummy_run`
捕图前按 `_layer_group_idx` 组装持久 context slot mapping 列表，并确保捕获五层
`do_kv_cache_update`；其余探针用于交叉验证该判断。
