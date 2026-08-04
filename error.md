Traceback (most recent call last):
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]   File "/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py", line 4857, in _replace_gpu_model_runner_function_wrapper
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]     yield
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]   File "/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py", line 4723, in capture_model
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]     cuda_graph_size = GPUModelRunner.capture_model(self)
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]   File "/vllm-workspace/vllm/vllm/tracing/otel.py", line 178, in sync_wrapper
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]     return func(*args, **kwargs)
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]   File "/vllm-workspace/vllm/vllm/v1/worker/gpu_model_runner.py", line 6811, in capture_model
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]     self._capture_cudagraphs(
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]   File "/vllm-workspace/vllm/vllm/v1/worker/gpu_model_runner.py", line 6938, in _capture_cudagraphs
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]     self._warmup_and_capture(
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]   File "/vllm-workspace/vllm/vllm/v1/worker/gpu_model_runner.py", line 6883, in _warmup_and_capture
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]     self._dummy_run(
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 124, in decorate_context
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]     return func(*args, **kwargs)
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]   File "/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py", line 3363, in _dummy_run
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]     self.drafter.dummy_run(
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 124, in decorate_context
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]     return func(*args, **kwargs)
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]   File "/vllm-workspace/vllm-ascend/vllm_ascend/spec_decode/dspark_proposer.py", line 468, in dummy_run
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]     self._runnable(
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]   File "/vllm-workspace/vllm-ascend/vllm_ascend/compilation/acl_graph.py", line 189, in __call__
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]     output = self.runnable(*args, **kwargs)
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]   File "/vllm-workspace/vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py", line 1065, in _run_merged_draft
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]     ret_hidden_states = self.model(**model_kwargs)
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]                         ^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]     return self._call_impl(*args, **kwargs)
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1787, in _call_impl
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]     return forward_call(*args, **kwargs)
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]   File "/vllm-workspace/vllm/vllm/model_executor/models/qwen3_dflash.py", line 705, in forward
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]     return self.model(input_ids, positions, inputs_embeds)
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]   File "/vllm-workspace/vllm/vllm/compilation/decorators.py", line 520, in __call__
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]     return self.aot_compiled_fn(self, *args, **kwargs)
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/torch/_dynamo/aot_compile.py", line 124, in __call__
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]     return self.fn(*args, **kwargs)
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]            ^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]   File "/vllm-workspace/vllm/vllm/model_executor/models/qwen3_dflash.py", line 614, in forward
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]     def forward(
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]   File "/vllm-workspace/vllm/vllm/compilation/caching.py", line 217, in __call__
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]     return self.optimized_call(*args, **kwargs)
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]   File "<string>", line 5, in execution_fn
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]   File "/vllm-workspace/vllm/vllm/compilation/piecewise_backend.py", line 380, in __call__
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]     return range_entry.runnable(*args)
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]   File "/vllm-workspace/vllm-ascend/vllm_ascend/compilation/compiler_interface.py", line 363, in compiled_fn
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]     result = _inner_fn(*args, **kwargs)
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]              ^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/dynamo/npugraph_ex/npu_fx_compiler.py", line 507, in __call__
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]     gm_result = self.run_kernel(*args, **kwargs)
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]   File "<string>", line 321, in kernel
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/dynamo/npugraph_ex/_acl_concrete_graph/acl_graph.py", line 823, in __call__
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]     return self.fx_run_eagerly(*args, **kwargs)
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/dynamo/npugraph_ex/_acl_concrete_graph/acl_graph.py", line 894, in fx_run_eagerly
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]     return self.fx_forward(*args, **kwargs)
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]   File "<string>", line 56, in forward
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]   File "/usr/local/python3.12.13/lib/python3.12/site-packages/torch/_ops.py", line 819, in __call__
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]     return self._op(*args, **kwargs)
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]            ^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]   File "/vllm-workspace/vllm/vllm/model_executor/layers/attention/kv_transfer_utils.py", line 40, in wrapper
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]     return func(*args, **kwargs)
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]   File "/vllm-workspace/vllm/vllm/model_executor/layers/attention/attention.py", line 835, in unified_attention_with_output
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]     self.impl.forward(
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]   File "/vllm-workspace/vllm-ascend/vllm_ascend/attention/attention_v1.py", line 1668, in forward
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]     attn_output = self.forward_impl(query, key, value, kv_cache, attn_metadata, output_padded)
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]   File "/vllm-workspace/vllm-ascend/vllm_ascend/attention/attention_v1.py", line 1604, in forward_impl
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]     output = self.forward_fused_infer_attention(query, key, value, attn_metadata, output, kv_cache)
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]   File "/vllm-workspace/vllm-ascend/vllm_ascend/attention/attention_v1.py", line 1290, in forward_fused_infer_attention
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]     attn_output, num_tokens = self.full_graph_fia(query, key, value, attn_metadata, output)
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]   File "/vllm-workspace/vllm-ascend/vllm_ascend/attention/attention_v1.py", line 960, in full_graph_fia
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]     graph_params.events[num_tokens].append(event)
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007]     ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^
(Worker_TP0 pid=135315) ERROR 08-04 09:36:46 [multiproc_executor.py:1007] KeyError: 14