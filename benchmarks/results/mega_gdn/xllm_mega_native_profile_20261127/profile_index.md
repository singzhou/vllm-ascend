# vLLM 原生 Profiling 采集索引

基线在物理卡 4–7，Mega 在物理卡 0–3。每个用例和实现有四个 TP rank。归档内的 capture manifest 包含所有 rank 路径、每条请求的发出时间与首 token 时间。下表列出对应 rank 0。

| 路径 | 输入 token | 并发 | 解压后 rank 0 的目录 |
| --- | ---: | ---: | --- |
| baseline | 128 | 32 | `mega_gdn_native_20261127/baseline/rank0_6873_20261127015102132_ascend_pt` |
| baseline | 4096 | 1 | `mega_gdn_native_20261127/baseline/rank0_6873_20261127015106728_ascend_pt` |
| baseline | 12288 | 1 | `mega_gdn_native_20261127/baseline/rank0_6873_20261127015111380_ascend_pt` |
| mega | 128 | 32 | `mega_gdn_native_20261127/mega/rank0_6731_20261127014939813_ascend_pt` |
| mega | 4096 | 1 | `mega_gdn_native_20261127/mega/rank0_6731_20261127014943926_ascend_pt` |
| mega | 12288 | 1 | `mega_gdn_native_20261127/mega/rank0_6731_20261127014948424_ascend_pt` |

在 MindStudio Insight 或 Perfetto 打开该目录下的 `ASCEND_PROFILER_OUTPUT/trace_view.json`。同目录的 `kernel_details.csv` 是设备算子明细。公开统计汇总了这 24 份 trace，并将冗长的一个 FA 符号改用可读的显示名。[Profile 统计](profile-summary.json)

归档的 SHA-256 与文件名见 [SHA256SUMS](SHA256SUMS)。
