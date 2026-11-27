# xLLM MegaChunkGdn prefill

Set `VLLM_ASCEND_ENABLE_XLLM_MEGA_GDN=1` to select xLLM `MegaChunkGdn`
exclusively for Gated DeltaNet prefill on Ascend 910B. The default `0` selects
the native prefill dispatch. Install the repaired `mega_chunk_gdn` CANN
vendor package and make its `custom_ops_lib` Python binding available to every
worker. The binding resolves the operator from `ASCEND_CUSTOM_OPP_PATH` when
it loads. Put the repaired vendor ahead of any older package for the same
operator.

Mega prefill requires eager execution on one prefill context-parallel rank,
FP16 or BF16 Q/K/V, and a 128-element head dimension. The scheduler compacts
empty segments and retains their cache slots. Unsupported configurations and
operator errors are reported to the caller. Each prefill uses one Mega call
and writes its state back through the device cache operators. Numerical
comparisons run offline during validation. Ordinary and speculative decode
use their existing recurrent GDN operators.

The binding still allocates intermediate tensors and zero-initializes five
of its outputs on each call. The adapter also normalizes Q and K and converts
the recurrent state between the cache and kernel layouts.

The tested vendor revision is `44e774aca24cf474eaf99f9840e7ad9df42929c5` in
`singzhou/xllm-ops`. It includes the BF16 triangular inverse and state update
precision fixes.
