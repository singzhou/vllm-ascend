# MegaChunkGdn 在 Ascend 910B3 上的测试报告

当前实现位于 [`030_naie_dev_xllm_mega`](https://github.com/singzhou/vllm-ascend/tree/030_naie_dev_xllm_mega)。独立测量以同一组 post-conv Q/K/V、门控和状态输入对比原生 FLA-NPU 与 Mega，输出时延、输出误差和末状态误差。[独立基准脚本](../../mega_gdn_prefill.py) · [构建和使用](build_and_run.md)

| 阶段 | 报告 | 测量内容 |
| --- | --- | --- |
| 原始算子 | [单算子性能](mega_chunk_gdn_910b3/report.md) · [显存峰值](mega_chunk_gdn_910b3/memory_report.md) | 原始 `43fc82b8` 的核心 kernel、适配层时延和 workspace 分配 |
| 带在线校验的早期接入 | [服务 A/B](xllm_mega_serve_030_naie/report.md) · [时延归因](xllm_mega_serve_030_naie/attribution/report.md) · [求逆根因](xllm_mega_serve_030_naie/solver_root_cause.md) | 在线读回、失败重跑和 TP rank 之间的尾部等待 |
| Vendor 修复 | [三角求逆修复](mega_gdn_fix_910b3/report.md) · [末状态定位](mega_gdn_fix_910b3/state_diagnosis.md) · [末状态修复复测](mega_gdn_fix_910b3/state_fix_report.md) | CPU FP64、原生 NPU、强相关压力与真实模型输入，以及修复前后的时延 |
| 排他执行初测 | [严格模式复测](xllm_mega_strict_030_naie/report.md) · [同卡顺序对照](xllm_mega_strict_030_naie/sequential_comparison.md) | 移除在线校验和回退后的首字与整网精度，以及首次分卡时的显存预检 |
| TP4 分卡复测 | [19 组 TTFT、长上下文和原生 Profiling](xllm_mega_native_profile_20261127/report.md) · [采集索引](xllm_mega_native_profile_20261127/profile_index.md) | Qwen3.8-27B 的 TTFT、生成一致性、主核与整网耗时，并发波动和显存采样 |
| 最新接入 | [图模式与数据搬运](latest_integration.md) | 旧、新 Mega 适配层单层对照，原生基准复现入口，以及图模式验证 |

报告及配套数值文件已整理为可公开阅读的副本。历史结果保留原来的测试配置和统计边界。请求内容、运行日志与原始 trace 归档留在测试环境，公开目录提供时序统计和归档校验和。[归档 SHA-256](xllm_mega_native_profile_20261127/SHA256SUMS)
