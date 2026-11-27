# Ascend 910B3 · MegaChunkGdn 显存峰值

`MegaChunkGdn` 的单次临时显存随 token 数和本地 value head 数增长。TP4 单条 8192 token 的适配层新增活跃分配峰值为 **280.6 MiB**，16384 token 为 **514.6 MiB**。同条件下，14 组输入的峰值均低于当前 `030_naie_dev` FLA-NPU 基线，适配层降幅为 **20.5%–35.5%**。连续调用也复现了显存保留现象，释放输出后活跃分配回到调用前，缓存保留量在第 1 至第 40 次调用间保持不变。

## 实测

沿用[前次性能对比](report.md)的提交、模型头配置、BF16 输入和比较边界，新增 16384 token 和 32 条 64 token 的用例。本轮在同一容器的空闲物理 2 号卡上进行。每个用例和实现先预跑，构造输入后同步并清空闲置缓存，再重置统计测量单次调用，取 3 次中位数。数值为相对于已准备输入的新增活跃分配。单流执行，使用非零初始状态，适配层按模型配置传入和返回 FP32 状态。

| 本地 TP 头配置 | 输入 | 基线核心 MiB | Mega 核心 MiB | 基线适配层 MiB | Mega 适配层 MiB |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4 | 32 token | 74.5 | 48.4 | 75.8 | 48.9 |
| 4 | 2048 token | 126.0 | 101.5 | 137.1 | 105.9 |
| 4 | 32 条 × 64 token | 137.6 | 119.2 | 183.6 | 135.2 |
| 4 | 变长 5 条，共 4096 token | 184.5 | 157.4 | 210.5 | 167.4 |
| 4 | 8192 token | 324.8 | 264.2 | 365.9 | 280.6 |
| 4 | 16384 token | 609.1 | 482.3 | 690.3 | 514.6 |
| 1 | 2048 token | 326.4 | 265.9 | 371.4 | 283.9 |
| 8 | 2048 token | 99.4 | 74.2 | 105.4 | 76.4 |

TP4、8192 token 的 xLLM 适配层在输出仍被持有时新增活跃分配 **24.75 MiB**。释放输出后新增活跃分配为 **0 MiB**，分配器额外保留 **302 MiB**，执行 `empty_cache()` 后额外保留回到 **0 MiB**。整卡 HBM 读数还会计入模型权重、KV cache、CANN runtime 和其他进程。

## 峰值来源

以 TP4、单条 8192 token 为例，xLLM 核心的 **264.2 MiB** 来自一笔约 **46.9 MiB** 的 ACLNN workspace 和约 **217.3 MiB** 的绑定输出分配。分配历史与 MindStudio 的 `operator_memory.csv` 都记录了这两部分。workspace 在 14 组中保持恒定，与 20 个 block 的 tile/pad、16 MiB 对齐预算、8 MiB phase 预算和 CANN 系统 workspace 相符。[Tiling 代码](https://github.com/Sinle4Cat/xllm-ops/blob/43fc82b8f9f1d3fca063c4410635f78422bdabba/xllm_ops/mega_chunk_gdn/op_host/mega_chunk_gdn_tiling.cpp#L65-L73)

| xLLM 绑定分配 | 8192 token 下的大小 |
| --- | ---: |
| `a_inv_f32`（FP32） | 48 MiB |
| `a`、`a_inv`（BF16） | 48 MiB |
| `w`、`u`、`v_new`（BF16） | 72 MiB |
| 每个 chunk 的状态 `h`（BF16） | 24 MiB |
| `g_sum`、`g_t`、`beta_t` | 0.94 MiB |
| 注意力输出与 BF16 末状态 | 24.38 MiB |

原仓绑定每次调用都分配 12 个输出槽，其中 **10 个是本次推理适配层用完即释放的中间结果**，本例合计 **192.94 MiB**。上面的三块三角矩阵存储就占 96 MiB。矩阵按 `[1,T,Hv,128]` 分配，`h` 则按 `[Hv × Σceil(seq_len/128),128,128]` 分配，所以同样为 2048 token 时，32 条 64 token 使 `h` 从 6 MiB 增至 12 MiB。更多序列也会增加末状态和 FP32 cache 布局转换占用。[原仓绑定](https://github.com/Sinle4Cat/xllm-ops/blob/43fc82b8f9f1d3fca063c4410635f78422bdabba/test/python_test/RegisterOps.cpp#L525-L567)

FLA-NPU 基线在同一 8192 token 用例的核心峰值为 **324.8 MiB**，其中 workspace 约 **288 MiB**。xLLM 的公开输出分配更大，更小的 workspace 使其总峰值仍低于基线。

## 复现与证据

显存回放使用独立设备。保留了算子分配表和分配历史。

公开复测使用 [当前独立脚本](../../../mega_gdn_prefill.py) 和合成输入。

[14 组结果和分配历史](memory.json) · [8192 token 的算子分配表](memory_profile_8192.csv)
