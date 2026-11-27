# 编译与运行 MegaChunkGdn

本流程使用 Ascend 910B、Linux、CANN 9.2.0，以及能运行本分支的 PyTorch、TorchNPU、Triton Ascend 和 FLA-NPU 环境。基准的原生对照来自 vLLM Ascend 的 A2 FLA-NPU prefill 路径，需要能够导入 `fla_npu.ops.ascendc.chunk_gated_delta_rule_fwd`。样本环境使用 Torch 2.10.0、TorchNPU 2.10.0.post7 和 FLA-NPU A2 26.9.1。编译使用 CANN Toolkit、GCC/G++、CMake 和 Ninja。

## 编译并安装隔离的厂商包

将 `CANN_HOME` 指到实际 Toolkit。克隆修复后的 fork，固定包含 BF16 三角求逆和末状态精度修复的版本。[xLLM fork 修复分支](https://github.com/singzhou/xllm-ops/tree/codex/fix-mega-gdn-bf16-triangular-inverse) · [已验证的提交](https://github.com/singzhou/xllm-ops/commit/44e774aca24cf474eaf99f9840e7ad9df42929c5)

```bash
export CANN_HOME=/usr/local/Ascend/cann-9.2.0
source "$CANN_HOME/set_env.sh"

git clone --branch codex/fix-mega-gdn-bf16-triangular-inverse \
  https://github.com/singzhou/xllm-ops.git
export XLLM_ROOT="$PWD/xllm-ops"
git -C "$XLLM_ROOT" checkout --detach 44e774aca24cf474eaf99f9840e7ad9df42929c5
git -C "$XLLM_ROOT" submodule update --init -- third_party/catlass third_party/pto-isa

export MEGA_BUILD_DIR="$XLLM_ROOT/build-mega-910b"
export MEGA_OPP_DIR="$XLLM_ROOT/opp-mega-910b"
(
  cd "$XLLM_ROOT/xllm_ops"
  bash build.sh --pkg --ops=mega_chunk_gdn --soc=ascend910b \
    --build-dir="$MEGA_BUILD_DIR" -j8
)
```

固定版本在 CANN 9.2.0 的干净单算子构建中可能报 `No rule to make target 'autogen/aic-ascend910b-ops-info.ini'`。Mega 的真实 INI 在 `autogen/exc`。确认它已生成，再给两份可选 INI 补空占位，使打包脚本跳过空文件，继续刚才的构建。

```bash
(
  set -e
  cd "$XLLM_ROOT"
  test -s "$MEGA_BUILD_DIR/autogen/exc/aic-ascend910b-ops-info.ini"
  mkdir -p "$MEGA_BUILD_DIR/autogen/inner"
  touch "$MEGA_BUILD_DIR/autogen/aic-ascend910b-ops-info.ini" \
    "$MEGA_BUILD_DIR/autogen/inner/aic-ascend910b-ops-info.ini"
  python - <<'PY'
from pathlib import Path

parser = Path("cmake/scripts/util/parse_ini_to_json.py")
source = parser.read_text()
old = "if os.path.exists(ini_file):"
assert source.count(old) == 1
parser.write_text(source.replace(old, "if os.path.isfile(ini_file) and os.path.getsize(ini_file) > 0:"))
PY
  cmake --build "$MEGA_BUILD_DIR" --target package -j8
)
```

打包成功后安装到隔离目录。

```bash
mkdir -p "$MEGA_OPP_DIR"
bash "$MEGA_BUILD_DIR"/cann-ops-xllm-custom*.run --quiet --install-path="$MEGA_OPP_DIR"
export MEGA_VENDOR="$MEGA_OPP_DIR/vendors/custom_xllm_math"
test -f "$MEGA_VENDOR/op_api/lib/libcust_opapi.so"
```

这里使用子目录打包入口并安装到工作目录。仓库根目录的构建入口会向 CANN 的 `opp` 树安装 vendor，隔离安装适合与其他算子包共存。

## 构建 Python 绑定

在运行基准的同一个 Python 环境中构建，保留与 `torch`、`torch_npu` 和 CANN 一致的 ABI。

```bash
export MEGA_BINDING_DIR="$XLLM_ROOT/python-mega-910b"
(
  cd "$XLLM_ROOT/test/python_test"
  USE_NINJA=1 MAX_JOBS=2 python setup.py build_ext \
    --build-lib "$MEGA_BINDING_DIR" --build-temp "$XLLM_ROOT/build-python-mega-910b"
)

export ASCEND_CUSTOM_OPP_PATH="$MEGA_VENDOR${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export LD_LIBRARY_PATH="$MEGA_VENDOR/op_api/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$MEGA_BINDING_DIR${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import torch, torch_npu, custom_ops_lib; print(custom_ops_lib.mega_chunk_gdn)'
```

路径要在导入 `custom_ops_lib` 前准备好。若环境还加载其他 OPP 包，所选 xLLM 包应排在旧 xLLM 包之前。最后一行检查 Python 入口，下一节的实际调用检查 CANN 算子也已正确加载。以上流程已从干净 checkout 重新打包并构建绑定，实际加载新包后通过了两组 BF16 压力用例。[重编后的精度与时延数据](rebuild-smoke.json)

## 单卡对比

在已安装本分支及其对应 vLLM 的环境中使用脚本，一张空闲卡即可。`--case` 可重复指定，每个值是一组打包序列的长度。默认包括 128、384、2048、4096 token 的单序列和一组变长序列。[单文件基准](../../mega_gdn_prefill.py)

```bash
git clone --branch 030_naie_dev_xllm_mega https://github.com/singzhou/vllm-ascend.git
cd vllm-ascend
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

ASCEND_RT_VISIBLE_DEVICES=0 python benchmarks/mega_gdn_prefill.py \
  --vendor "$MEGA_VENDOR" --key-heads 4 --value-heads 12 \
  --case 128 --case 4096 --case 63,129,257 --json mega-gdn.json

ASCEND_RT_VISIBLE_DEVICES=0 python benchmarks/mega_gdn_prefill.py \
  --vendor "$MEGA_VENDOR" --correlated-keys --case 384 --case 2048
```

时延从已准备好的 post-conv Q/K/V 和门控开始，覆盖布局、归一化、GDN 主算子和状态缓存读写。每条路径交替测 10 组，报告组内单次平均的中位数。CPU FP64 在计时之外运行。误差分别报告输出和末状态，并给出最大绝对误差、相对 RMSE 和非有限值计数。默认门限沿用算子回归口径，即最大绝对误差不超过 `0.01` 或相对 RMSE 小于 `5%`，且结果有限。精度失败时脚本返回非零退出码。

## 服务接入

服务加载相同的 vendor 和 Python 绑定后，设置 `VLLM_ASCEND_ENABLE_XLLM_MEGA_GDN=1`，用平常的 `vllm serve` 参数启动支持的 Qwen GDN 模型。默认值 `0` 选择原生 GDN。Mega prefill 支持 eager 以及 PIECEWISE 图间执行，符合条件的 decode 仍可使用 FULL ACLGraph。[分支的接入文档](../../../docs/source/user_guide/feature_guide/xllm_mega_gdn.md)
