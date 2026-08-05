# DSpark ACLGraph 精度恢复 try 5

## 结论

try4 日志已经把问题定位为一个确定的捕图生命周期缺陷，而不是 BF16 数值误差、
GQA 分支、SWA 分支、DSpark 查询宽度或 attention metadata 形状问题：

> DSpark 捕图发生在第一个真实请求之前。捕图时
> `_context_slot_mapping_buffers` 为 `None`，Qwen 的
> `precompute_and_store_context_kv` 因此在完成 K/V 投影后直接返回，5 层
> `do_kv_cache_update` 均未进入 ACLGraph。真实请求再创建 slot-mapping 列表，
> 无法把捕图时遗漏的算子补进已经生成的图。

本次修改让“按层 context slot-mapping 列表”从 attention backend 初始化完成起就
绑定到持久的 per-group tensor，并在 `dummy_run` 捕图入口进行防御性重绑定。
运行时 Triton kernel 仍然原位更新这些持久 tensor，所以捕获地址保持稳定，slot
内容可随请求变化。

## try4 日志证据

分析的日志：

- `dspark_analysis_log/dspark_compile_try_4.info`
- `dspark_analysis_log/dspark_eager_try_4.info`

### 1. 捕图时明确没有 context slot mapping

compile 日志第 408 行：

```text
[dspark_diag][capture_context] query_tokens=7 context_tokens=8
slot_mapping_is_none=True slot_mapping_layers=None slot_edges=None
```

这同时证明 try3 的宽度修复已经生效：DSpark query backbone 捕获 7 个 token，
target hidden context 仍保留验证宽度 8 个 token。当前错误不再是 7/8 的图宽问题。

### 2. eager 的 K/V 写入每步都正确

eager step 0 至 step 3 中，5 个 full-attention 层的：

```text
projected_k == cached_k
projected_v == cached_v
```

探针前后完全一致，最终接受率为 `14/14 = 100%`。因此投影、RoPE、层到 cache
的对应关系和真实请求 slot mapping 在 eager 路径上均正确。

### 3. compile 只在图外 step 0 写入，图重放不再写入

compile step 0 中投影值与缓存值一致；从 step 1 开始：

- projected K/V 随新的 target hidden states 正常变化；
- cached K/V 却始终逐值等于 compile step 0；
- step 1 至 step 5 的 cached-vs-step0 最大差均为 0；
- projected-vs-cached 的最大差已经达到约 `K=5.06~5.92`、
  `V=25.03~39.11`。

最终接受率为 `9/35 = 25.7%`。这说明 ACLGraph 中不是“写错值”，而是没有
context KV-cache 写操作。

### 4. 与代码控制流完全闭合

`vllm_ascend/patch/worker/patch_qwen3_dflash.py` 中存在：

```python
if context_slot_mapping is None:
    return
```

而 try4 捕图日志已经证明传入值为 `None`。该 return 位于 K/V 投影探针之后、
`do_kv_cache_update` 之前，恰好解释了 compile 中“projected 更新、cached 冻结”
的全部现象。

try4 中 Python module forward hook 的全零结果没有用于此结论。编译模型会绕过
这些 Python hook，因此它们不是有效的图内数值证据；直接嵌入
`precompute_and_store_context_kv` 的 projected/cached 探针才是有效证据。

## 代码修改

### 1. 新增统一绑定方法

文件：`vllm_ascend/spec_decode/dspark_proposer.py`

```python
def _bind_context_slot_mapping_buffers(self) -> None:
    """Bind each draft layer to its persistent context slot buffer."""
    self._context_slot_mapping_buffers = [
        self._per_group_context_slot_mapping_buffers[group_idx]
        for group_idx in self._layer_group_idx
    ]
```

作用：按照 draft layer 顺序，将每层映射到所属 KV-cache group 的持久 context
slot tensor。同一 group 的多层会共享同一个 slot tensor；这符合 slot layout 的
语义，同时支持未来 draft 层跨多个 KV group 的情况。

这里没有创建或复制新的设备 tensor，仅创建 Python 列表并保存原 tensor 引用，
所以不会引入 NPU 同步，也不会改变 graph 输入地址。

### 2. attention backend 初始化后立即绑定

在 `_per_group_context_slot_mapping_buffers` 分配完成后立即调用：

```python
self._bind_context_slot_mapping_buffers()
```

这是主要修复。ACLGraph 捕获早于第一个真实请求，但晚于 attention backend
初始化；因此 dummy capture 第一次执行 Qwen context-KV precompute 时已经能看到
非空的 per-layer slot 列表，5 层 cache-update 算子会被捕获。

初始 tensor 内容为 0 不影响捕图。捕图要求的是算子和 tensor 地址存在；真实重放
前，输入准备 kernel 会把这些 tensor 原位改为当前请求的真实 slot id。这与 DFlash
从构造阶段持有固定 context slot buffer 的机制一致。

### 3. 捕图入口防御性重绑定

在 `dummy_run` 确定 context 宽度后、调用 `_runnable` 前执行：

```python
self._bind_context_slot_mapping_buffers()
```

这保证即使后续重构改变初始化顺序，或某处清除了模型侧列表，捕图也不会再次以
`None` 进入 Qwen precompute。

### 4. 运行时不再临时清空列表

删除 `set_inputs_first_pass` 中：

```python
self._context_slot_mapping_buffers = None
```

Triton kernel 填充完成后仍调用统一绑定方法，作为 attention group 被重建时的
防御措施。底层 `_per_group_context_slot_mapping_buffers[gid]` 始终是同一 tensor，
所以运行时更新值不会破坏 ACLGraph 静态地址要求。

## 生命周期对比

修复前：

```text
分配 per-group tensor
  -> per-layer list 仍为 None
  -> dummy capture
  -> Qwen K/V projection
  -> context_slot_mapping is None，提前 return
  -> 图中没有 5 层 KV-cache update
  -> 首个真实请求才创建 list（已经太晚）
```

修复后：

```text
分配 per-group 持久 tensor
  -> 立即建立 per-layer list
  -> dummy capture 再次确认绑定
  -> Qwen K/V projection + 5 层 KV-cache update 全部入图
  -> 每个真实请求由 Triton kernel 原位刷新 slot tensor
  -> graph replay 使用当前请求 slot 写入当前 projected K/V
```

## 回归测试

修改 `tests/ut/spec_decode/test_dspark_proposer.py`：

1. `test_context_slot_mapping_binding_preserves_layer_group_order`
   - 构造 `[1, 0, 1]` 的 layer-to-group 映射；
   - 验证列表顺序正确；
   - 使用 `is` 验证列表引用的是持久 group tensor，不是 copy。
2. 扩展 `test_full_mode_separates_context_and_query_graph_widths`
   - 捕图前主动把模型侧列表恢复为旧故障状态 `None`；
   - 执行 FULL graph `dummy_run`；
   - 验证捕图入口已完成重绑定；
   - 同时继续验证 context=8、query=7 的 DSpark 图宽语义。

## 本地检查

- 两个修改过的 Python 文件通过 AST 解析。
- 当前本机不是 Ascend 服务运行环境，缺少项目所需的 `torch_npu`/vLLM 运行依赖，
  无法在本机完成 NPU graph capture 或完整 pytest。
- 当前 shell 也没有可用的 `ruff` 命令；已人工检查修改范围、类型和行宽。

## try5 运行后的预期判据

建议将新日志保存为：

```text
dspark_analysis_log/dspark_compile_try_5.info
```

必须首先满足：

```text
[dspark_diag][capture_context] query_tokens=7 context_tokens=8
slot_mapping_is_none=False slot_mapping_layers=5
```

捕图时 `slot_edges` 为初始 0 是允许的，因为重放前 tensor 内容会被原位刷新。

运行期的决定性判据是 compile 每个 step、每个层均满足：

```text
projected_k == cached_k
projected_v == cached_v
```

允许仅存在正常 BF16 精度范围内的表示差异，但 cached K/V 不能再逐步保持为 step 0。
接受率应恢复到与 eager 基线接近；当前短请求 eager 基线为 100%。

若 `slot_mapping_is_none=False` 且 projected/cached 已逐步一致，但接受率仍异常，才需
继续比较 query backbone/FIA 输出；在 try4 证据下，不需要在本次修复前再增加探针。

## try5 启动错误修正

首次 try5 执行日志：

```text
dspark_analysis_log/dspark_eager_try_5_error.info
```

服务在可用显存探测阶段失败，所有 TP worker 的首个有效异常一致：

```text
model_runner_v1.py:3363 -> self.drafter.dummy_run(...)
dspark_proposer.py:784 -> self._bind_context_slot_mapping_buffers()
dspark_proposer.py:536 -> for group_idx in self._layer_group_idx
AttributeError: 'AscendDSparkProposer' object has no attribute '_layer_group_idx'
```

这里的 `dummy_run` 是 `determine_available_memory -> profile_run`，发生在 KV cache
配置和 `initialize_attn_backend` 之前。因此此阶段尚不存在 `_layer_group_idx`，也没有
需要写入的 draft KV cache。try5 首版在所有 `dummy_run` 中无条件防御性绑定，错误地
把正式捕图阶段的前置条件应用到了 pre-KV memory profile。

修正后仅在 layer/group 映射已经建立时绑定：

```python
if getattr(self, "_layer_group_idx", None):
    self._bind_context_slot_mapping_buffers()
```

同时在构造阶段把 `_layer_group_idx` 显式初始化为空列表，表达“KV backend 尚未完成
初始化”的合法状态，避免生命周期状态依赖属性是否存在。

该条件区分两个合法生命周期：

```text
pre-KV memory profile
  -> 尚无 _layer_group_idx
  -> 跳过绑定，保持原来的 no-cache profile 行为

initialize_attn_backend
  -> 创建 _layer_group_idx 和持久 per-group slot tensor
  -> 立即完成第一次绑定

ACLGraph capture
  -> _layer_group_idx 已存在
  -> 防御性重绑定
  -> context KV-cache update 正常入图
```

新增回归测试
`test_profile_before_attn_backend_init_skips_context_binding`，显式设置空的
`_layer_group_idx` 后执行 `is_profile=True` 的 `dummy_run`，验证不会访问未初始化的
映射、context list 保持 `None`，且 profile model 调用仍正常发生。原有 FULL graph
测试继续覆盖正式捕图必须完成绑定，因而不会削弱 try5 的精度修复。
