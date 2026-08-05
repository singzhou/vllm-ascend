# DSpark ACLGraph 精度恢复尝试 3：拆分 8-token context 与 7-token query 图

## 1. Try 2 结论

严格对照中，主模型均保持 `FULL_DECODE_ONLY`：

- 仅草稿模型 eager 时，平均接受长度为 `8.00`，接受率为 `100%`；
- 草稿模型进入 ACLGraph 后，平均接受长度为 `2.80`，接受率为 `25.7%`。

五个 dense GQA full-attention 层全部位于 `gid=4`，第一次 decode 的
context position、context slot、query position、query slot 和 block table
在两组日志中一致。相同 seed 的第一组 Markov bias 也一致，但草稿
backbone hidden state 和 raw logits 已经分叉。因此错误位于草稿 backbone
的 FULL ACLGraph 路径，而不是主模型模式、context KV 寻址或 Markov head。

Try 2 暴露出的唯一结构差异是：

| 路径 | context hidden 数 | 草稿 query 数 | FIA query 布局 |
| --- | ---: | ---: | --- |
| eager | 8 | 7 | `[0, 7]` |
| 原 graph | 8 | 8 | `[0, 7, 8]` |

DSpark 在 `sample_from_anchor=True` 时，每个 request 的语义是：

- 主模型校验和传给草稿模型的 context hidden state 数：`N + 1`；
- 草稿 backbone 实际计算的 query 数：`N`。

原实现使用一个 `num_input_tokens` 同时表示这两个数量。为了复用主模型
8-token capture bucket，它把 7 个真实 query 扩展成 8 个 query，并增加一个
一 token 的虚拟 request。Try 1 已证明仅把虚拟 request 的 KV 长度从 0 改为
1 不能恢复精度，说明需要消除这层错误的 query 图等价假设。

## 2. 修改方案

核心方案是保留同一个主模型图调度键，但让 DSpark 草稿图使用自己的原生
query token 数：

```text
主模型 BatchDescriptor: num_tokens=8, num_reqs=1, uniform=True
                           │
                           ├── context KV 预写：8 tokens
                           │
                           └── DSpark query/FIA：1 * N = 7 tokens
```

`BatchDescriptor(8, 1)` 仍是 `ACLGraphWrapper` 的 capture/replay cache key，
所以不改变主模型 dispatcher，也不要求用户把
`cudagraph_capture_sizes:[8]` 改成 `[7]`。只有草稿模型内部计算形状和
attention graph params 的 key 从 8 改为 7。

### 2.1 为草稿模型增加图 token 数映射接口

文件：`vllm_ascend/spec_decode/llm_base_proposer.py`

基类默认保持现有行为：

```python
def get_graph_num_input_tokens(
    self,
    batch_descriptor: BatchDescriptor,
) -> int:
    return batch_descriptor.num_tokens
```

运行时两次 dispatcher 之后都通过该接口确定草稿图的输入 token 数，而不再
直接读取 `batch_descriptor.num_tokens`。因此 Eagle、DFlash、MTP 等方法行为
不变，只有显式 override 的 proposer 会使用不同图形状。

### 2.2 DSpark uniform decode 映射为原生 N-token 图

文件：`vllm_ascend/spec_decode/dspark_proposer.py`

```python
def get_graph_num_input_tokens(
    self,
    batch_descriptor: BatchDescriptor,
) -> int:
    if batch_descriptor.uniform and batch_descriptor.num_reqs is not None:
        return batch_descriptor.num_reqs * self.num_query_per_req
    return batch_descriptor.num_tokens
```

当前配置 `N=7`、`BatchDescriptor(8, 1, uniform=True)` 会得到草稿图大小 7。
如果启用 `dspark_bonus_anchor`，`num_query_per_req=N+1`，映射结果自然仍为 8。
非 uniform descriptor 或缺少 request 数时回退到原 token 数，避免改变 prefill
或 PIECEWISE 路径。

### 2.3 捕图时独立保留 context 和 query 两种宽度

文件：`vllm_ascend/spec_decode/dspark_proposer.py`

`dummy_run` 现在同时维护：

```python
graph_query_tokens = self.get_graph_num_input_tokens(batch_descriptor)
num_query_tokens = min(graph_query_tokens, self.max_query_tokens)
num_context_tokens = min(num_tokens, self.max_num_tokens)
```

随后：

- `num_context_tokens=8` 用于 `precompute_and_store_context_kv`；
- `num_input_tokens=7` 用于草稿 backbone、FIA metadata、position、logits 和
  Markov head；
- `_dflash_num_context` 在捕图时设置为 8，确保捕获的 context KV 预写算子与
  decode replay 的 8 个 target hidden states 一致；
- capture metadata 变为一个真实 request：`query_start_loc=[0, 7]`，不再为
  当前配置构造 `[0, 7, 8]` 虚拟 request。

如果未来使用更大的 batch bucket，仍保留 whole-request padding 的兜底逻辑；
该逻辑不再用于弥补每个 request 固有的 `N` 与 `N+1` 差值。

### 2.4 draft attention graph params 使用 7-token key

文件：`vllm_ascend/worker/model_runner_v1.py`

主模型和草稿模型的 attention graph params 不再强制共用同一组 token key：

```python
set_graph_params(capture_sizes)  # 主模型仍为 [8]

get_draft_graph_num_tokens = getattr(
    self.drafter,
    "get_graph_num_input_tokens",
    lambda desc: desc.num_tokens,
)
draft_capture_sizes = sorted({
    get_draft_graph_num_tokens(desc)
    for _, descs in capture_descs
    for desc in descs
})
set_draft_graph_params(draft_capture_sizes)  # DSpark 为 [7]
```

这一步必须与运行时和 `dummy_run` 同时修改。否则 FIA 捕图会访问不存在的
`draft_graph_params[7]`，或者运行时仍会错误更新 `draft_graph_params[8]`。
无 proposer 的 speculative method 继续沿用原始 capture sizes。

## 3. 回归测试

文件：`tests/ut/spec_decode/test_dspark_proposer.py`

新增或更新的断言覆盖：

1. uniform `BatchDescriptor(num_tokens=16, num_reqs=2)` 映射成 14-token
   DSpark 图；
2. bonus-anchor 配置仍保持 `N+1` 图宽；
3. 非 uniform descriptor 和缺少 `num_reqs` 时保持 16，不错误改变其他图；
4. 捕图输入为 target 16 tokens、DSpark 14 tokens 时：

    - capture query metadata 为 `[0, 7, 14]`；
    - 不增加虚拟 request；
    - `_runnable` 收到 `num_input_tokens=14`；
    - `_dflash_num_context=16`，context KV 预写宽度未被缩成 14。

本地环境没有安装可运行该 UT 所需的 `pytest`、vLLM/NPU Python 依赖，无法
代替 NPU 环境执行单测或服务。已完成修改文件的 AST 解析、Ruff check、
Ruff format check 和 `git diff --check`；最终精度仍需真实 NPU 服务验证。

## 4. Try 3 预期日志

使用与 `dspark_compile_try_2.info` 完全相同的启动参数和请求。第一次 decode
应出现：

```text
[dspark_diag][step=1][inputs]
context_tokens=8
num_query_total=7

[dspark_diag][step=1][metadata]
actual_tokens=7
graph_tokens=7
padded_reqs=1
actual_seq_q=[7]
actual_seq_kv=[40]
capture_layout={
  'capture_reqs': 1,
  'capture_padded_reqs': 1,
  'capture_actual_tokens': 7,
  'capture_qsl': [0, 7],
  ...
}

[dspark_diag][graph_update]
graph_tokens=7
captured_ops=5
```

以下任一情况都表示 try 3 没有走到预期路径：

- `graph_tokens` 仍为 8；
- `actual_seq_q` 仍为 `[7, 8]`；
- `capture_padded_reqs` 仍为 2；
- `context_tokens` 从 8 错误变成 7；
- `draft_graph_params[7]` 相关 KeyError 或未捕获 FIA op。

建议新日志保存为：

```text
/opt/zsy/vllm-ascend/dspark_analysis_log/dspark_compile_try_3.info
```

## 5. 验收标准

优先确认结构正确，再观察接受率：

1. 服务正常捕图和拉起；
2. 第一次 decode 同时满足 `context_tokens=8` 和 `graph_tokens=7`；
3. `actual_seq_q=[7]`、`padded_reqs=1`，没有单 token 虚拟 request；
4. 五个 FIA op 均在 `graph_tokens=7` 下更新；
5. hidden/raw-logits 不再呈现 try 2 的图内异常；
6. 接受率应向严格 eager 对照的 `100%` 恢复。

如果图形状已经全部符合上述预期但接受率仍低，下一步才需要在草稿模型的
embedding 输出和每一层 decoder 输出设置 graph-safe persistent probe，以定位
第一个发生数值分叉的具体算子；在执行 try 3 前不建议继续增加这类高开销打点。
