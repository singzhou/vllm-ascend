# DSpark ACLGraph 精度恢复尝试 1：修正虚拟请求的 KV 长度

## 1. 结论

本轮不需要继续增加日志。现有 eager/compile 对照已经把第一次有效分叉定位到 DSpark 草稿模型的 ACLGraph attention 输入，而不是采样、Markov head、拒绝步数更新或后续历史差异。

本次采用最小整改：当 anchor-first DSpark 的 `N` 个真实查询被放入 `N + 1` 图桶时，将补出的 FIA 虚拟请求的 `seq_lens` 从 `0` 改为合法的正数 `1`。真实请求的有效 KV 长度保持不变，DFlash 路径保持原行为。

## 2. 日志证据

输入日志：

- `dspark_eager.info`：全局 eager 对照。
- `dspark_compile.info`：草稿模型进入 ACLGraph 的实验组。

关键观测如下：

1. eager 接受率为 `100%`，`Accepted=14 / Drafted=14`；compile 接受率为 `25.7%`，`Accepted=9 / Drafted=35`。
2. step 0 尚未发生 ACLGraph replay。两组输出 token 完全相同：`[12305, 198, 727, 73111, 1393, 25, 514]`。这说明模型、权重、基本 DSpark 语义和非图路径可用。
3. step 1 是 compile 的第一次 ACLGraph replay。两组真实输入一致：seed 为 `8`，input token 为 `[8, mask x 6]`，position 为 `33..39`，slot 为 `7713..7719`，真实请求更新后 KV 长度为 `40`。
4. step 1 的 Markov bias 完全一致，说明随机输入及 Markov head 之前的随机分支没有漂移。
5. 第一次 replay 后，草稿 hidden state 的最大差异达到 `6.734375`，raw logits 最大差异达到 `6.421875`，输出 token 立即分叉。因此错误首先出现在草稿 backbone/attention 图路径。
6. compile step 1 的图桶为 8 个 token，但 DSpark 真实查询只有 7 个 token。运行时 metadata 为：

   ```text
   query_start_loc = [0, 7, 8]
   seq_lens       = [40, 0]
   num_reqs       = 2
   ```

   捕图时同一虚拟请求则使用了合法的正 KV 长度：

   ```text
   query_start_loc = [0, 7, 8]
   seq_lens       = [8, 1]
   num_reqs       = 2
   ```

这里第二个 request 不是业务请求，而是为补齐第 8 个图 token 人工添加的 FIA 虚拟请求。

## 3. 根因分析

DSpark 的 anchor-first 语义中：

- 投机查询数为 `SpecStep = N`；
- 校验/图桶按 `SpecStep + 1 = N + 1`；
- 因而在本例 `N=7` 时，运行时需要添加一个只覆盖第 8 个 padding token 的虚拟 request。

`llm_base_proposer.py` 原先通过通用 `_adjust_tensor` 扩展 `seq_lens`。该函数固定用 `0` 补齐，因此把真实 `[40]` 扩展成 `[40, 0]`。

但 Ascend FIA 的图 metadata 构建逻辑已经明确约束：虚拟 padding request 需要任意合法的正 KV 长度，并在捕图路径中使用 `1`。虚拟请求的 block table 为零行，padding token 的输出随后会被裁掉，KV cache 写入也按真实 token 数切片，所以 `1` 只用于满足 FIA metadata/tiling 的有效性，不会给真实请求引入额外上下文。

捕图时为 `[8, 1]`、回放时却更新为 `[40, 0]`，破坏了捕图与回放的参数不变量。它与“第一次 replay 立即在 backbone hidden state 分叉”的日志证据完全吻合。

## 4. 整改代码

文件：`vllm_ascend/spec_decode/llm_base_proposer.py`

原代码：

```python
if self.method in ("dflash", "dspark"):
    common_attn_metadata.seq_lens = self._adjust_tensor(
        common_attn_metadata.seq_lens, num_reqs_padded
    )
```

整改后：

```python
_GRAPH_PADDING_REQUEST_KV_LEN = 1

if self.method in ("dflash", "dspark"):
    common_attn_metadata.seq_lens = (
        self._adjust_parallel_draft_seq_lens_for_graph(
            common_attn_metadata.seq_lens, num_reqs_padded
        )
    )

def _adjust_parallel_draft_seq_lens_for_graph(self, seq_lens, desired_size):
    padding_value = (
        _GRAPH_PADDING_REQUEST_KV_LEN if self.method == "dspark" else 0
    )
    return self._adjust_tensor(
        seq_lens, desired_size, padding_value=padding_value
    )
```

同时给 `_adjust_tensor` 增加默认值为 `0` 的 `padding_value` 参数。所有既有调用保持原语义；只有 DSpark 图路径显式传入 `1`。

## 5. 影响范围

- DSpark anchor-first：存在 `N` 与 `N+1` 差值时，虚拟 request 从 `0` 改为 `1`。
- DSpark 真实 request：原始 `seq_lens` 原样保留，例如 `[40] -> [40, 1]`。
- DSpark bonus-anchor：查询数本身为 `N+1`、不需要虚拟 request 时，没有 padding，也没有数值变化。
- DFlash：仍使用 `0` 作为通用补齐值，避免扩大本次修复范围。
- eager：不进入该 FULL ACLGraph metadata 补齐分支，不受影响。
- GQA/dense/full-attention：修复发生在公共 FIA metadata 层，不修改模型层选择、KV head 数或 attention mask。

## 6. 回归保护

在 `tests/ut/spec_decode/test_dspark_proposer.py` 增加两条断言：

1. DSpark 将真实 `[40]` 补成 `[40, 1]`；
2. DFlash 仍将 `[40]` 补成 `[40, 0]`。

这两条测试分别保护本次修复和路径隔离。

## 7. NPU 实机验证标准

请使用与 `dspark_compile.info` 相同的 compile 启动参数重新执行相同请求。现有 `[dspark_diag]` 日志可继续复用，无需再加点。

首先确认第一次 replay 的 runtime metadata 已变为：

```text
query_start_loc = [0, 7, 8]
seq_lens       = [40, 1]
```

然后按以下优先级判断：

1. step 1 的 draft hidden probe、raw logits 和 output token 应显著靠近 eager；理想情况下 token 与 eager 对齐。
2. 接受率应从当前 `25.7%` 明显恢复，并接近同请求 eager 的 `100%`。
3. 后续 step 不再因 step 1 拒绝造成执行历史持续分叉。

若 `seq_lens=[40, 1]` 已生效但 step 1 仍在 backbone hidden state 大幅分叉，下一轮应保持主模型配置不变，仅在 speculative config 内令草稿模型 eager，先排除全局 `--enforce-eager` 同时改变主模型所带来的对照噪声；再继续核对图回放时的 block table、attention task/event 及 slot mapping。当前证据下，不建议在完成本次 A/B 前扩大修改范围。

## 8. 本地检查结果

- `python3 -m py_compile`：代码文件与新增测试通过语法编译。
- `git diff --check`：通过，未发现空白符错误。
- 当前主机 Python 环境未安装 `torch`、`pytest` 和 `ruff`，因此无法在该环境执行仓库单测或 lint；新增回归测试需要在项目开发/镜像环境中运行。
- 最终精度结论仍以 Ascend 实机的首次 replay 对齐结果和接受率为准。

## 9. Try 1 实机结果

新日志：`dspark_analysis_log/dspark_compile_try_1.info`。

整改已在运行时生效：step 1 的 `actual_seq_kv` 从 `[40, 0]` 变为 `[40, 1]`。但是 step 1 的 hidden probe、raw logits、draft token，以及最终 `Accepted=9 / Drafted=35`、`25.7%` 接受率均与整改前一致。

因此可以排除“虚拟请求 KV 长度为零是当前精度错误的直接原因”。正 KV 长度仍符合 FIA metadata 约束，可以保留，但下一轮定位转向：

1. 捕图时 5 个 FIA graph task/handle 的层顺序；
2. 运行时 per-layer metadata 的字典顺序；
3. 多 KV group 下各层 block table 与 context slot mapping 是否绑定到对应 handle；
4. 保持主模型图模式、仅让草稿模型 eager 的严格对照。
