# 2026-11-16 DSpark概率采样Review修复与上机验证

本次核对 `030_naie_dev_greedy_fix` 的 `eb34c66be69ec9ea518b6fd0aac402dbd7914491`。
原始开发基线仍为 `singzhou/030_naie_dev@7f93f07236def39e26b491798c7c461a552c2b23`。

## Review 结论与修复

| 问题 | 核对结果与处理 |
| --- | --- |
| P1 DP dummy 复用真实温度 | 成立。真实调用显式传入 metadata，dummy/profile 的 `None` 不再从 runner 补回。dummy 只走映射后的 argmax，且不导出 q。 |
| P1 同步语法裁剪造成分布偏差 | 成立。保留首个非法候选及其 q；仅在生成 grammar mask 的副本中使用占位，令 p(x)=0 触发正常拒绝与残差恢复。 |
| 动态 K 的 query 长度与缓冲区 | 成立。按当前 K 更新 query 长度、裁剪返回列和动态 confidence，保护 kernel 写入；K=0 返回前清缓存。 |
| 缺失请求或步数导致整批丢 q | 成立，主要影响接受率。保留已有行，缺失行和实际 -1 占位行使用零 q，强制拒绝并从 p 恢复。 |
| 无映射但草稿头比 target 宽 | 成立。额外列在 greedy 和概率采样前均按真实 target 词表裁掉。 |
| mixed batch 的 greedy 行 q | 与上游约定一致，不是当前验证器的分布错误。仅 random 行的 q 表示真实提议分布，greedy 行由独立 mask 验证，不读取该 q。 |
| 逐步 full-vocab 分配与 stack 峰值 | 分配风险成立，尚无此前代码实际 OOM 的证据。草稿在小词表上采样，再把 q 写入最终预分配缓存，减少 full-vocab logits/noise 和 list/stack 重复缓存。 |
| 上机额外发现的 GDN 短验证 NaN | 上轮接受 8 个 token、本轮 query 缩至 5 时，旧 FLA kernel 提前返回而未写输出。缩短的真实 spec query 改走已有 DCUT，保留完整 8 槽历史及原接受数。 |

语法反例中 `q=(0.4,0.1,0.5)`、约束后的 `p=(0.5,0.5,0)`：旧截断流程输出 `(0.65,0.35,0)`；保留首拒位置后，经 `(p-q)+` 恢复到 p。
语法保留补丁仅对 V1 DSpark、`draft_sample_method=probabilistic`、`rejection_sample_method=standard` 启用。

## 关键代码

- `vllm_ascend/spec_decode/llm_base_proposer.py`：metadata 契约、当前 K、缓存清理、小词表采样及最终概率缓存。
- `vllm_ascend/spec_decode/dspark_proposer.py`、`vllm_ascend/spec_decode/utils.py`、`vllm_ascend/ops/triton/spec_decode/utils.py`：dummy、动态 query/confidence 与 kernel 边界。
- `vllm_ascend/worker/model_runner_v1.py::_get_spec_decode_draft_probs`：按请求和步数对齐缓存，缺失与占位行强制拒绝。
- `vllm_ascend/patch/platform/patch_dspark_structured_output.py`：保留首个非法候选，只修改 mask 输入副本，保持 grammar 回滚。
- `tests/ut/spec_decode/test_dspark_probabilistic.py`、`tests/ut/spec_decode/test_dspark_review_regressions.py`、`tests/ut/spec_decode/test_dspark_probability_cache.py`、`tests/ut/patch/platform/test_dspark_grammar_rejection.py`：回归测试。
- `vllm_ascend/ops/gdn_attn_builder.py`：按真实 CPU query 长度选择 DCUT；`tests/ut/ops/test_gdn_shortened_verification.py` 验证 8→5、8→1 和完整历史状态读取。

验证阶段仍需保留完整 target 空间的 q；runner 按请求重排时也会分配对齐后的概率张量。
此次减少草稿采样的冗余分配，不保证任意批量和词表大小下均不会 OOM。

## 上机范围

用户本轮明确授权 `910b3` 的前四卡及停止其占用进程，覆盖此前仅使用后四卡的限制。
源码、配置与测试在 `wyd_va_030rc2_1028` 容器的独立目录内完成；后四卡原有服务继续保留。
配套 vLLM 为 `ced6857afa0ea7b2e3f0846a62e1394e90f15607`，使用 V1、eager、同步调度。
该版本在 DP>1 时会禁用按批量调整 K，因此分别验证 DP2×TP2 的静态 K=7 和单 DP×TP4 的动态 K。

## 验证结果

目标模型 `/opt/models/Qwen3.8-27B`，草稿 `/opt/models/Qwen3.8-27B-DSpark-Base`，BF16，草稿词表 64000、target 词表 248320。

| 验证 | 结果 |
| --- | --- |
| 本地五个回归文件 | 86 passed，3 个真实 NPU 测试因无设备跳过。 |
| 910b3 同一组测试 | **89 passed**；含真实 sampler 分布、strided q 写入、两种 anchor 的 kernel sentinel 和 GDN 历史状态。 |
| DCUT 实际 NPU 对照 | 9 项通过；覆盖 conv/recurrent、ragged、空行、历史 stride 8/16、BF16/FP32 和物理 state 布局。 |
| DP2×TP2、固定 K=7 | 两个 rank 的普通、混合温度、近上下文边界共 11 组通过；结构化测试在此阶段暴露 GDN NaN。 |
| GDN 修复后 TP2 | T=0/0.7/1 的 regex/JSON 共 6 组全通过；原失败 query=5 的 4 个验证步，每行 248320 个 finite logits、0 NaN。 |
| 最终 TP4 | K 按 B=1/2/3/4 配成 4/2/0/7；普通、混合、动态 K、上下文边界 16 组及 regex/JSON 6 组全通过。 |
| 活动批次 4→1 | 另 1 组通过；三条短请求先结束，长请求实际 draft 长度从 7 缩为 4，全部输出符合预期。 |
| 静态检查 | `bash format.sh ci`、`git diff --check` 通过。 |

最终调度日志包含 `(generation requests, generation input tokens)` 为 `(1,5)`、`(2,6)`、`(3,3)`、`(4,32)` 的稳定区间，分别对应 K=4/2/0/7。
`per_step_drafted` 是验证收到的实际草稿数；K=0 区间不追加记录，不能用恒为 7 的 `num_spec_tokens` 字段证明动态 K。

TP2 JSON 故障的临时 hook 确认：5 行 target logits 在应用 grammar mask 前均为 NaN，mask 非空，q 归一化；与安装 FLA 源码的 `accepted > seqLen` 早退一致。
修复后直接解决状态读取契约，没有对无效 p 使用 `nan_to_num`；最终服务已移除诊断 hook。

以上覆盖 V1、eager、同步调度；未进行完整图编译、异步调度或吞吐基准，不据此宣称接受率或性能提升。

## 部署与保留服务

- 概率采样仍使用原 `speculative_config` 中的 `draft_sample_method="probabilistic"`、`rejection_sample_method="standard"`，请求 T>0；T=0 保持 greedy。V1 设置为 `VLLM_USE_V2_MODEL_RUNNER=0`。
- GDN 短验证需要本分支已有的 `npu_dcut_causal_conv1d` 与 `npu_dcut_recurrent_gated_delta_rule` native op。旧扩展缺少接口时，仅替换 Python 或加启动参数不足，应使用包含这两个算子的匹配构建。
- 本次复用容器 `/opt/w00958190/vllm-ascend/vllm_ascend/` 的兼容产物（源码记录 `4a01608da7be1950eebc9902e3692d257f304759`），仅在独立源码目录链接。Python 3.12.16、torch 2.10.0、CXX11 ABI=True、CANN 9.2.1；原安装与后四卡服务未替换。
- 最终服务：`127.0.0.1:45111`，model=`dspark-review-validation`，设备 0–3、TP4、max length 2048、max sequences 4。上表动态 K 配置用于验证。
- 启动脚本：`/opt/w00958190/dspark_review_fix_20261116/serve_final_tp4.sh`，前台运行并 tee 到 `/opt/w00958190/log/dspark_review_final_tp4_front4_20261116_45111.log`。
- 结果：任务目录的 `final_summary.json`、`final_tp4_results/results.json`、`final_tp4_chat_results/results.json`、`final_active_shrink.json`；native 路径、备份和哈希见 `native_manifest.json`。
- 完成检查时，最终服务及后四卡原有 `45109`、`45110` 的 `/health` 均为 200。前四卡服务保留运行。

## 复核 prompt

```text
请 review fg11991/vllm-ascend 的 030_naie_dev_greedy_fix。
先比较 eb34c66be69ec9ea518b6fd0aac402dbd7914491..HEAD 核对本轮修复；原始基线是 7f93f07236def39e26b491798c7c461a552c2b23。
重点检查 DP dummy metadata 隔离、动态 K 与 K=0 缓存、q 的 target-ID 映射及缺失行、首个 grammar-invalid 候选与残差采样、以及 GDN previous accepted=8/current query=5 或1时完整历史状态的保留。
结合新增回归与 adaptation_docs/2026-11-16_DSpark概率采样Review修复与上机验证.md，只报告可定位的正确性/兼容性问题，给路径、行号、触发条件和最小复现；区分已验证 eager 同步路径与尚未跑的图/异步/性能组合。
```
