# Environment Variables

vllm-ascend uses the following environment variables to configure the system:

**Note:** Some environment variables are being migrated to `--additional-config` options. These environment variables are still supported during the migration period, and it is recommended to use `--additional-config` for new deployments. See [Additional Configuration](additional_config.md) for details.

{{ include_code('vllm_ascend/envs.py', start_after='begin-env-vars-definition', end_before='end-env-vars-definition') }}

## MegaChunkGdn prefill on Atlas A2

The optional xLLM MegaChunkGdn backend handles eager Qwen GDN prefill with BF16
inputs, 128-wide Q/K and V heads, and prefill context parallel size 1. The
existing backend handles unsupported batches. MegaChunkGdn converts the
selected recurrent states to BF16 for computation and stores the final states
in the configured cache dtype.

Build the operator and Python binding in the same CANN and PyTorch NPU
environment used for serving. The following commands use the
[validated xLLM revision](https://github.com/Sinle4Cat/xllm-ops/commit/43fc82b8f9f1d3fca063c4410635f78422bdabba).

```bash
git clone https://github.com/Sinle4Cat/xllm-ops.git
cd xllm-ops
git checkout 43fc82b8f9f1d3fca063c4410635f78422bdabba
git submodule update --init
xllm_src="$PWD"
mega_gdn_build="$(mktemp -d)"
cd "$xllm_src/xllm_ops"
bash build.sh --pkg --ops=mega_chunk_gdn --soc=ascend910b \
    --build-dir="$mega_gdn_build/cann-build"
bash "$mega_gdn_build"/cann-build/cann-ops-xllm*.run --install-path="$mega_gdn_build/install"
cd "$xllm_src/test/python_test"
# Continue scanning custom vendor paths when an entry has no libcust_opapi.so.
sed -i '/if (cust_opapi_lib.empty()) {/{n;s/break;/continue;/;}' pytorch_npu_helper.hpp
python setup.py build_ext --build-lib "$mega_gdn_build/binding" --build-temp "$mega_gdn_build/binding-tmp"

vendor="$mega_gdn_build/install/vendors/custom_xllm_math"
export PYTHONPATH="$mega_gdn_build/binding${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_CUSTOM_OPP_PATH="$vendor${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export VLLM_ASCEND_ENABLE_MEGA_CHUNK_GDN=1
```
