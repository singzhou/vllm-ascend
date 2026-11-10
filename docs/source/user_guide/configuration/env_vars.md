# Environment Variables

vllm-ascend uses the following environment variables to configure the system:

**Note:** Some environment variables are being migrated to `--additional-config` options. These environment variables are still supported during the migration period, and it is recommended to use `--additional-config` for new deployments. See [Additional Configuration](additional_config.md) for details.

{{ include_code('vllm_ascend/envs.py', start_after='begin-env-vars-definition', end_before='end-env-vars-definition') }}

## MegaChunkGdn prefill on Atlas A2

The optional MegaChunkGdn backend handles eager Qwen GDN prefill with BF16
inputs, 128-wide Q/K and V heads, and prefill context parallel size 1. The
existing backend handles unsupported batches. MegaChunkGdn converts the
selected recurrent states to BF16 for computation and stores the final states
in the configured cache dtype.

Build or install vLLM Ascend with the Ascend910B custom operators. The source
build initializes the pinned PTO-ISA submodule. For direct builds from
`csrc`, initialize it first with
`git submodule update --init -- csrc/third_party/pto-isa` from the repository root.
Enable the backend with:

```bash
export VLLM_ASCEND_ENABLE_MEGA_CHUNK_GDN=1
```

The kernel and host operator are based on the
[xLLM MegaChunkGdn source at 43fc82b](https://github.com/Sinle4Cat/xllm-ops/tree/43fc82b8f9f1d3fca063c4410635f78422bdabba/xllm_ops/mega_chunk_gdn).
