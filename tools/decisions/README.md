# Qwen Decisions on Ascend

Install this vllm-ascend branch alongside stock vLLM 0.30.0. The platform
plugin registers `POST /v1/systemone` and initializes its handler for
`Qwen3ForCausalLM`, `Qwen3_5ForConditionalGeneration` and
`Qwen3_5MoeForConditionalGeneration` when serving a generation model.

```bash
VLLM_USE_V2_MODEL_RUNNER=0 vllm serve Qwen/Qwen3-0.6B \
  --served-model-name qwen-decisions --logprobs-mode raw_logprobs \
  --enable-prefix-caching --port 8010
```

Each named question reads one token's label probabilities. Supported question
types are `choice` (1–128 options), `noul` and `score`. The handler checks the
label tokens against the serving tokenizer. Disable thinking in the request.

```bash
curl http://localhost:8010/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "qwen-decisions",
  "state": "My package has not arrived.",
  "questions": {"team": {"type": "choice", "instructions": "Which team should reply?",
    "criteria": {"billing": null, "shipping": null}}},
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

Hybrid shared-prefix warmup is opt-in. Set
`VLLM_ASCEND_DECISIONS_WARM_SHARED_PREFIX=1` before starting the server to
evaluate it on eligible multi-question requests. The default is `0`.

## Verification

From the vllm-ascend checkout, point the CPU tools at a separate vLLM 0.30.0
checkout containing the source branch in its Git history. The offline suite uses a fake engine and tokenizer.

```bash
python -m pip install pytest pytest-asyncio numpy msgspec pydantic 'fastapi>=0.133,<0.137' httpx pillow regex
python tools/decisions/verify_upstream.py --vllm-root ../vllm
python tools/decisions/run_offline_tests.py --vllm-root ../vllm -q
python tools/decisions/probe_sampler_cpu.py --vllm-root ../vllm --output /tmp/sampler.json
```

The sampler probe additionally requires PyTorch. On an Ascend host with model
weights and the project test dependencies, run the real tokenizer and service tests.

```bash
pytest -q tests/ut/entrypoints/structured_decisions/test_next_token_tokenizer.py
pytest -q tests/e2e/pull_request/one_card/test_structured_decisions.py
python tools/decisions/smoke_qwen.py --base-url http://localhost:8010 \
  --model qwen-decisions --output /tmp/decisions-smoke.json
```

Use `benchmark_qwen.py` for warmup A/B, fixed-prompt gather or business request
measurements. Use `verify_qwen36_choice128.py` to regenerate the 128-label
tokenizer report for a local tokenizer. The [reference reports](../../docs/decisions/)
are copied from the [source branch at the ported revision](https://github.com/F00L42/vllm/tree/0e01034281f9e6fe94230a59e8535c48e20eb291/docs/design).
