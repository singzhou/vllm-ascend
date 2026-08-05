# DSpark ACLGraph 精度恢复尝试 2：FIA 层与 KV group 绑定诊断

## 1. Try 1 结论

`dspark_compile_try_1.info` 已证明代码整改生效：第一次 replay 的 `actual_seq_kv` 为 `[40, 1]`。但是 hidden probe、raw logits、draft token 和 `25.7%` 接受率与整改前完全一致。

因此，虚拟 request 的零 KV 长度不是当前精度问题的直接根因。下一步不继续修改 DSpark 算法语义，而是核对 ACLGraph 捕图 op 与运行时 attention metadata 的实际绑定关系。

## 2. 新增打点

### 2.1 Attention group 静态布局

日志：

```text
[dspark_diag][attn_groups]
```

内容：

- `model_layer_order`：proposer 记录的草稿 attention 层顺序；
- `groups`：每个 KV cache group 的 gid 和 layer names。

目的：确认五个 full-attention 草稿层是位于同一 KV group，还是分布在多个 group。如果分布在多个 group，graph handle 与 runtime key 的顺序错配会绑定错误的 block table/KV cache。

### 2.2 每步 context/KV group 输入

日志仍为：

```text
[dspark_diag][step=N][inputs]
```

新增字段：

- `context_positions`：预计算 context K/V 使用的 position；
- `context_layouts`：各 gid 的 layer names、context slot mapping 和 block table 前两行摘要。

目的：确认 eager 与 graph 第一次 replay 在写入草稿 context K/V 时使用相同的位置、slot 和物理 block。

### 2.3 Graph task 更新映射

日志：

```text
[dspark_diag][graph_update]
```

内容：

- `graph_tokens`：图桶大小；
- `captured_ops`：捕获的 FIA op/handle 数；
- `captured_layers`：每个捕获 op 对应的真实模型 layer name；
- `runtime_keys`：运行时按当前字典顺序展开的 `(draft_step, layer key)`。

目的：直接检查第 i 个捕获 FIA handle 是否与第 i 个运行时 layer metadata 对应。这里只记录 layer name，不改变 graph-task 的选择逻辑。

## 3. 需要执行的两组实验

### 实验 A：带新打点的原 compile 配置

保持 `dspark_compile_try_1.info` 的启动参数不变，重新执行同一请求。建议日志名：

```text
dspark_compile_try_2.info
```

### 实验 B：主模型 compile、仅草稿模型 eager

不要添加全局 `--enforce-eager`。只在 speculative config 中加入：

```json
{
  "num_speculative_tokens": 7,
  "method": "dspark",
  "model": "/opt/w00958190/dspark_gabbages/tige_training_output/0731_qwen3.6_27b_600k/step51480",
  "enforce_eager": true
}
```

其余参数，包括 `--compilation-config`，与实验 A 完全相同。建议日志名：

```text
dspark_draft_eager_target_compile.info
```

这组实验用于消除旧 eager 日志中“全局 `--enforce-eager` 同时改变主模型 hidden state”的对照噪声。

## 4. 判定方法

1. 如果 `captured_layers` 与 `runtime_keys` 顺序不同，且对应层位于不同 gid，则下一次整改应改为按捕获 layer name 选择 runtime metadata，不能依赖 set/dict 插入顺序。
2. 如果层映射一致，但 experiment A/B 的 `context_positions` 或相应 gid 的 `context_slots`、`block_table_head` 不同，则定位到 context K/V 预写路径。
3. 如果严格对照中草稿 eager 接受率恢复，而相同主模型 compile 下草稿 graph 仍错误，则可确认问题完全位于 draft ACLGraph，不再需要考虑主模型数值差异。
4. 如果严格对照的草稿 eager 接受率也低，则旧的全局 eager 结果不能作为草稿图精度基准，应先分析主模型 compile hidden state 与训练特征的一致性。
