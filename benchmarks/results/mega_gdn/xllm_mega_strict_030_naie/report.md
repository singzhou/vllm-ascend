# 910B3 · 严格 MegaGDN 整网复测

`030_naie_dev_xllm_mega` 已接入 `658e5fbdf2fcdccee5bd449ea324213919774647`，本地和容器内的分支引用一致。服务部署的文件逐个核对过 SHA-256。Mega worker 加载由 `singzhou/xllm-ops` 的 `44e774aca24cf474eaf99f9840e7ad9df42929c5` 编译的 910B vendor。vLLM Ascend 的 130 项 GDN 相关单测全部通过。[修复后的算子提交](https://github.com/singzhou/xllm-ops/commit/44e774aca24cf474eaf99f9840e7ad9df42929c5) · Mega 运行时库 · 单测日志

开关 `VLLM_ASCEND_ENABLE_XLLM_MEGA_GDN=1` 只执行 Mega prefill，去掉了逆矩阵幅度和有限性检查、设备标量回读与 FLA 重算。Mega 复用调度器的设备端序列边界，跳过原生 chunk metadata 构建，多个层共用同设备同 dtype 的掩码。状态 dtype 转换并入设备端缓存读写。开关为 `0` 时走原生 prefill。普通和推测 decode 继续执行各自的递推 GDN 路径。

## 已完成的同卡复测

在 910B3 物理卡 4–7 上依次运行原生和 Mega，两次采用 Qwen3.8-27B BF16、TP4、MRV2、DSpark-7 greedy、eager、关闭自适应校验和 Prefix Cache。最大模型长度 16384，调度上限每步 4096 token、32 条序列，`gpu_memory_utilization` 均为 `0.85`。显存以 `npu-smi` 整卡数据每 3 秒采样，峰值为原生 58,361 MiB、Mega 58,318 MiB，总容量为每卡 65,536 MiB。两个服务报告的 KV cache 内存均为 36.72 GiB，完整复测期间没有 OOM。通用启动参数 · [原生采样](baseline-results/hbm.jsonl) · [Mega 采样](mega-results/hbm.jsonl)

### 整网精度

| 用例 | 生成 token 完全一致 | 原生和 Mega 的答案命中 | 生成 token logprob 平均 / 最大绝对差 |
| --- | ---: | ---: | ---: |
| 22 个单请求 | 22/22 | 均为 20/22，长检索 10/10 | 0.00192 / 0.04844 |
| 40 个混合并发请求 | 40/40 | 均为 40/40，长检索 28/28 | 0.00066 / 0.05280 |

最长输入为 12,288 token，两个服务的生成 logprob 均有限。两组单请求都在相同两条短答上多生成了解释。另测四个请求的逐 prompt token logprob，Mega 对原生的平均绝对差为 3.17，最大为 34.25。当前原生对上次原生记录的同口径差异为 3.10 和 32.43，这项指标有较大的背景波动。[精度与性能汇总](summary.json) · [原生生成数据](baseline-results/precision.json) · [Mega 生成数据](mega-results/precision.json)

### 首字时延

沿用上次的 19 组组合。128、512 token 测并发 1、4、16、32，2048 测 1、4、8、16，4096 测 1、4、8，8192 测 1、4，混合输入测 4、16。混合输入等比例包含 128、512、2048、4096 token。每组预热一轮，再计时四轮。两组各 692 个有效流式请求。

Mega 有 16/19 组 p50 降低，逐组变化中位数为 −5.59%。p95 有 16/19 组降低，变化中位数为 −5.77%。128 token、并发 32 的 p50 从 810.3 ms 升至 1221.5 ms。2048 token、并发 1 的 p50 从 417.1 ms 升至 526.5 ms，该用例每组只有 4 个计时样本。详细 p50、p95 和逐请求数据见同卡报告。[19 组同卡对照](sequential_comparison.md)

## 八卡分组尝试

首次尝试分卡部署时，Mega 组的各卡仅有 29.18–32.54 GiB 可用。配置 `0.85` 需要 51.81 GiB，服务因此在压测前退出。后续完成的八卡分组结果见 [分卡复测](../xllm_mega_native_profile_20261127/report.md)。
