# MegaChunkGdn 图模式与数据搬运复测

本轮使用 910B3、CANN 9.2.0 和 xLLM 修复提交 `44e774a`。接入改动包括允许图外的 PIECEWISE prefill、直接归一化 head-first Q/K，以及融合状态转置与 cache gather/scatter。[接入提交](https://github.com/singzhou/vllm-ascend/commit/28f871eff3afc1c7252606834db71c4e57530561)

## 旧 Mega 与新 Mega 单层对比

两路使用相同的 BF16 head-first Q/K/V、12 个 value heads、4 个 key heads 和 FP32 cache。测量从 cache gather 开始，到状态 scatter 结束，涵盖布局转换、Q/K 归一化、vendor 调用和输出缩放。预热后以 NPU Event 交替测量 6 组，每组 18 次，取每组单次平均耗时的中位数。输出和回写后 cache 在两路逐元素一致。

| 总 token | 序列数 | 旧 Mega（µs） | 新 Mega（µs） | 时延变化 |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1 | 1318.14 | 1055.66 | −19.91% |
| 4096 | 1 | 1330.90 | 1075.20 | −19.21% |
| 2048 | 16 | 1302.48 | 1047.61 | −19.57% |
| 4096 | 32 | 1332.69 | 1098.48 | −17.57% |

## 单项测量

以下是同卡微基准，计时用 ACLGraph 回放减少 Python 下发波动。只有被测的预处理和状态拷贝子图进入捕获。

| 微基准 | 输入 | 调整前（µs） | 调整后（µs） |
| --- | --- | ---: | ---: |
| BF16 单张量归一化加布局转换 | 128 token、4 heads | 23.47 | 15.09 |
| BF16 单张量归一化加布局转换 | 4096 token、4 heads | 40.97 | 18.54 |
| BF16 单张量归一化加布局转换 | 32768 token、16 heads | 533.35 | 338.20 |
| 状态 gather 加转置与 cast | 32 序列、8 heads | 47.88 | 23.31 |
| 状态 scatter 加转置与 cast | 32 序列、8 heads | 50.47 | 21.58 |

卷积和 Q/K/V 准备的联合对照中，`head_num` 为正值的 head-first 路径在 128、384、2048 和 4096 token 分别耗时 51.86、66.31、114.03 和 167.51 µs。对应的 `head_num=0` 路径为 56.40、68.58、117.31 和 171.45 µs。

## 精度与整网验证

- 384 和 2048 token 的 BF16 强相关压力用例、真实模型输入回放的 128、512 和 896 token 前缀，以及 BF16 和 FP16 随机变长用例中，新旧 Mega 的输出和末状态逐元素一致。输出和末状态均通过了 CPU FP64 和原生 NPU GDN 的精度口径。
- 将输出缩放移到 Q 的归一化时，2048 token 压力样本相对旧 Mega 的输出 RMSE 为 2.68%。该试验的状态逐元素一致。发布版本维持输出端缩放的计算顺序。
- 142 项单测和 27 项 NPU 测试通过。Qwen3.5-0.8B 服务在 `FULL_AND_PIECEWISE` 下完成两种图捕获，37、128、511 和 700 token 请求各成功生成 8 token。

Qwen3.8-27B 的 TTFT 结果来自上一阶段的排他执行接入，在最新热路径变更之前采集。[27B 服务与 Profiling 报告](xllm_mega_native_profile_20261127/report.md)

## 当前独立脚本对比原生 FLA-NPU

同一张 910B3 上以四个 key heads、十二个 value heads 和 FP32 cache 运行公开脚本。计入状态缓存读写、归一化、布局准备和 GDN 调用。CPU FP64 与原生 NPU 对照均通过。[复测命令](build_and_run.md)

| BF16 序列长度 | FLA-NPU（µs） | Mega（µs） | 加速 |
| --- | ---: | ---: | ---: |
| 128 | 1328.72 | 1015.35 | 1.31× |
| 384 | 1231.66 | 939.92 | 1.31× |
| 2048 | 1202.03 | 929.21 | 1.29× |
| 4096 | 1896.12 | 1013.25 | 1.87× |
| 63、129、257 | 1219.39 | 940.95 | 1.30× |

强相关 BF16 384、2048 token 用例分别测得 1.32×、1.30×，FP16 变长用例测得 1.32×。所有用例的输出与末状态均通过精度门限。[随机 BF16](standalone-default.json) · [强相关 BF16](standalone-stress.json) · [FP16](standalone-fp16.json)
