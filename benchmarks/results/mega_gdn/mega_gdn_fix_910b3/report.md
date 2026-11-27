# 910B3 MegaChunkGdn BF16 求逆修复报告

后续修复已解决下文压力用例的末状态超差，并在同一台 910B3 上复测了精度和单算子时延。[末状态修复与复测报告](state_fix_report.md)

**结论**　官方 `main` 在 910B3 上仍能复现原始 BF16 三角求逆发散。修复后的单算子消除了真实失败输入的发散和 NaN，在全部 8 个真实输入长度上同时通过 CPU FP64 和原生 FLA-NPU 对比。仓库的 74 项单算子测试全部通过。额外构造的一组极端强相关输入仍有末状态误差，见下文。

- 官方基线为 `a06a7c6e479bde41007a7d035cd648ba46b6723d`。当前 910B 路径的对角块求逆实现与原始复现提交 `43fc82b8` 相同。近期官方 MegaGDN 改动包含独立的 Ascend 950 路径。[官方基线](https://github.com/xLLM-AI/xllm-ops/commit/a06a7c6e479bde41007a7d035cd648ba46b6723d) · [官方 Ascend 950 改动](https://github.com/xLLM-AI/xllm-ops/commit/e24077d)
- 修复分支为 `codex/fix-mega-gdn-bf16-triangular-inverse`，提交 `614c9a13ab6749cd07a7e840c2a24d6626112add`。推送后，将待编译源码与该提交的文件逐字节核对，再强制重编 910B kernel、安装新包并重跑整套测试。[fork 修复分支](https://github.com/singzhou/xllm-ops/tree/codex/fix-mega-gdn-bf16-triangular-inverse) · [修复提交](https://github.com/singzhou/xllm-ops/commit/614c9a13ab6749cd07a7e840c2a24d6626112add)

## 修改

A2/A3 的 BF16、128-token 路径把 16×16 对角块求逆改成 FP32 三角前代。每行以单位矩阵为初值，用先前已解出的行做 FP32 更新，整个小块算完后才舍入为 BF16。向量核将结果放入 KKT 完成后已空闲的 `A_inv` 槽位，同步后由 Cube 读取并执行原有的 16→32→64→128 拼接。复用已有输出存储和同步机制，保持算子接口和 GM workspace 契约。该 BF16 路径改为在对角块全部就绪后启动 Cube 求逆，原先与 KKT 重叠的分组求逆流水在此路径关闭。

实现使用 Ascend C 的跨行 `DataCopy`、`Cast`、FP32 `Axpy` 和核间同步。新增的仓库回归用重复键、弱衰减和较大 beta 触发相消，覆盖多个序列和 127/129-token 边界。[Ascend C Axpy 接口](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/memory_vector_compute/composite_compute/Axpy.md) · [回归用例](https://github.com/singzhou/xllm-ops/commit/614c9a13ab6749cd07a7e840c2a24d6626112add)

## 单算子实测

在 `zsy-va-030-dev` 容器的 Ascend 910B3 物理 7 号卡、CANN 9.2.0 上运行。输入含服务中抓取且全部有限的 Qwen3.8-27B TP2 激活，截取 64、127、128、129、512、896、1024、2048 token，以及 5 组相关键、随机键、变长打包、非零状态和 FP16 用例。三条路径接收相同的已归一化 Q/K、V、gate、beta、序列边界和等价旧状态。

- CPU 参考为逐 token 的 FP64 GDN 递推。逆矩阵另与同一份 NPU KKT 矩阵的 FP64 三角解比较。
- NPU 参考调用当前 vLLM Ascend 原生 prefill 使用的 FLA-NPU `chunk_gated_delta_rule_fwd`，chunk 64，BNSD，应用相同的缩放。Mega 直接调用原始 API，输出按模型缩放后比较，过程中没有 Host 校验或 FLA 回退。
- 输出及末状态各自沿用官方测试容差。数值全部有限，且最大绝对误差 ≤ 0.01 或相对 RMSE < 5%，即为通过。

| 真实 2048-token 指标 | 官方 main | 修复后 |
| --- | ---: | ---: |
| 逆矩阵元素最大绝对值 | 2,056,192 | 1 |
| 逆矩阵对同一 KKT 的 FP64 解最大绝对误差 | 2,056,192 | 0.001930 |
| 输出非有限元素 | 132,736 | 0 |
| 末状态非有限元素 | 16,384 | 0 |
| 输出对 CPU FP64 的相对 RMSE | 输出含 NaN/Inf | 1.083% |
| 末状态对 CPU FP64 的相对 RMSE | 状态含 NaN/Inf | 0.404% |
| 输出对原生 FLA-NPU 的相对 RMSE | 输出含 NaN/Inf | 1.181% |
| 末状态对原生 FLA-NPU 的相对 RMSE | 状态含 NaN/Inf | 0.487% |

完整的 13 组中，逆矩阵、输出和末状态均为有限数，计算输出 13/13 通过两套参考的容差，末状态 12/13 通过。新增的相关键回归在官方二进制上失败，在修复二进制上通过。推送后的包执行同一测试文件得到 **74 passed**，其中包含新增回归和 73 项原有测试。[官方复现数据](upstream-validation.json) · [推送后双参考数据](published-validation.json) · 官方回归失败日志 · 修复版回归日志

## 极端输入的剩余误差

一组 2048-token 的合成压力用例将每个 key head 的向量固定为完全相同的方向，同时使用弱衰减及接近 1 的 beta。修复后逆矩阵的最大绝对误差为 0.000592，输出仍在容差内。末状态对 CPU FP64 和原生 FLA-NPU 的相对 RMSE 为 **5.63%** 和 **6.08%**，最大绝对误差为 **0.01298** 和 **0.01535**，超出仓库容差。直接拿这一输入实际产生的 NPU 逆矩阵，在 CPU 上用 FP64 完成后续计算，末状态相对 RMSE 为 2.01%。再换成设备的 BF16 W/U 后为 3.41%。这定位了剩余误差在后续 BF16 中间量和状态更新链路的贡献。[压力用例消融](stress-state.json)

独立运行原样本的前缀，确认第 3 个 128-token 块末首次超差。逐个重放 `chunk_h` 的舍入步骤后，确认最大的剩余误差来源是衰减后的 K 写成 BF16。CPU 重放中只让这一中间量保留 FP32，末状态相对 RMSE 从 5.63% 降到 3.73%。[复现与定位报告](state_diagnosis.md)

## 时延和构建记录

同卡分别预热，交错运行 8 轮，每轮 25 次，用 NPU Event 取每次调用的中位耗时。计入原始算子绑定创建输出和执行的开销，输入布局与元数据提前准备。测试边界为单算子。

| 输入 | 修复后 Mega 中位（µs） | 原生 FLA 中位（µs） |
| --- | ---: | ---: |
| 真实 128 token | 143.0 | 220.2 |
| 真实 512 token | 243.0 | 363.8 |
| 真实 2048 token | 506.0 | 931.2 |
| 变长强相关 23 + 129 + 387 token | 221.9 | 507.8 |

官方包位于 `upstream-install`，由发布提交复编的包位于 `pub`。单算子打包时，当前仓构建器会依赖两个没有内容的可选 INI。本次给构建副本补了空占位，并使构建用解析器跳过空文件。源码身份和 kernel 编译完成标记被显式更新，使本次构建产出新的 BF16 内核。推送后复编的 BF16 二进制 SHA-256 与已验证修复包一致，与官方二进制不同。FP16 二进制哈希保持一致。[内核哈希](kernel-sha256.txt) · [当前独立脚本](../../../mega_gdn_prefill.py)

从容器的上述目录运行原始捕获样本。

公开复测使用 [当前独立脚本](../../../mega_gdn_prefill.py) 和合成输入。
