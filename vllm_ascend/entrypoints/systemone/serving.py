# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""SystemOne HTTP endpoint backed by vLLM pooling requests."""

import logging
import math
import time

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from vllm import envs
from vllm.entrypoints.serve.engine.serving import BaseServing
from vllm.entrypoints.serve.utils.api_utils import load_aware_call, validate_json_request, with_cancellation
from vllm.exceptions import VLLMNotFoundError
from vllm.pooling_params import PoolingParams
from vllm.tracing import contains_trace_headers, extract_trace_headers, log_tracing_disabled_warning
from vllm.utils.async_utils import make_async, merge_async_iterators

from vllm_ascend.decision.config import KevConfig, is_kev_model, validate_runtime
from vllm_ascend.decision.planner import plan_request, validate_tokenizer

from .protocol import SystemOneRequest, output_tokens, to_answers

logger = logging.getLogger(__name__)


def probabilities(logits, temperature):
    if not logits or not all(math.isfinite(z) for z in logits):
        raise RuntimeError("Kev worker returned empty or non-finite logits")
    maximum = max(logits)
    values = [math.exp(z - maximum) for z in logits]
    total = sum(values)
    values = [v / total for v in values]
    if temperature != 1:
        # Match Kev's post-softmax clamp, rather than silently replacing it
        # with softmax(logits / T). Subtract max for very small temperatures.
        scaled = [math.log(max(v, 1e-9)) for v in values]
        largest = max(scaled)
        values = [math.exp((v - largest) / temperature) for v in scaled]
        total = sum(values)
        values = [v / total for v in values]
    return values


class DecisionService:
    def __init__(self, engine):
        validate_runtime(engine.vllm_config)
        self.engine = engine
        self.config = KevConfig.from_dict(engine.model_config.hf_config.kev_config)
        self.tokenizer = engine.renderer.get_tokenizer()
        validate_tokenizer(self.tokenizer, engine.model_config.hf_config.kev_delimiter_ids)
        self.max_model_len = engine.model_config.max_model_len
        names = engine.model_config.served_model_name
        self.names = {names} if isinstance(names, str) else set(names or [])
        self.cache_reads = engine.vllm_config.cache_config.enable_prefix_caching
        self.plan_async = make_async(plan_request, executor=engine.renderer._executor)
        self.response_async = make_async(self._build_response, executor=engine.renderer._executor)

    async def evaluate(self, request, raw_request):
        if not envs.VLLM_SKIP_MODEL_NAME_VALIDATION and request.model not in self.names:
            raise VLLMNotFoundError(f"Unknown served model: {request.model}")
        started = time.perf_counter()
        plan = await self.plan_async(request, self.tokenizer, self.config, self.max_model_len)
        distributions = await self._execute(plan, request, raw_request)
        return await self.response_async(request, plan, distributions, started)

    def _build_response(self, request, plan, distributions, started):
        # Like native pooling postprocessing, keep CPU formatting/tokenization
        # off the API event loop using the renderer's shared executor.
        answers = to_answers(distributions, plan.metadata)
        elapsed = round((time.perf_counter() - started) * 1000, 1)
        logger.info(
            "Kev complete questions=%d input_tokens=%d truncated=%s total_ms=%s",
            len(plan.rows),
            plan.input_tokens,
            plan.state_truncated,
            elapsed,
        )
        return {
            "model": request.model,
            "answers": answers,
            "usage": {
                "input_tokens": plan.input_tokens,
                "output_tokens": output_tokens(self.tokenizer, answers),
            },
            "latency_ms": elapsed,
        }

    async def _execute(self, plan, request, raw_request):
        parent = f"kev-{BaseServing._base_request_id(raw_request)}"
        trace_headers = None
        if await self.engine.is_tracing_enabled():
            trace_headers = extract_trace_headers(raw_request.headers)
        elif contains_trace_headers(raw_request.headers):
            log_tracing_disabled_warning()

        generators = []
        for index, row in enumerate(plan.rows):
            params = PoolingParams(
                task="plugin",
                skip_reading_prefix_cache=not self.cache_reads,
                extra_kwargs={"kev_readout": row.readout.as_dict()},
            )
            params.verify(self.engine.model_config)
            prompt = {"prompt_token_ids": row.token_ids}
            if request.cache_salt is not None:
                prompt["cache_salt"] = request.cache_salt
            generators.append(
                self.engine.encode(
                    prompt=prompt,
                    pooling_params=params,
                    request_id=f"{parent}-{index}",
                    priority=request.priority,
                    trace_headers=trace_headers,
                )
            )

        # Use the same batch merge and cancellation path as native pooling.
        # encode owns request IDs, DP routing, aborts and collector lifecycle.
        results = [None] * len(plan.rows)
        merged = merge_async_iterators(*generators)
        try:
            async for index, output in merged:
                if not output.finished:
                    continue
                tensor = output.outputs.data
                if tensor.ndim != 1 or tensor.shape[0] != len(plan.rows[index].readout.option_ends):
                    raise RuntimeError("Kev worker returned an invalid option distribution")
                results[index] = probabilities(tensor.tolist(), self.config.temperature)
        finally:
            await merged.aclose()
        if any(result is None for result in results):
            raise ValueError("Failed to generate results for all Kev questions")
        return results


class SystemOnePlugin:
    name = "ascend_systemone"
    required_tasks = ("plugin",)

    def attach_router(self, app: FastAPI):
        @app.post("/v1/systemone", dependencies=[Depends(validate_json_request)])
        @with_cancellation
        @load_aware_call
        async def systemone(request: SystemOneRequest, raw_request: Request):
            service = getattr(raw_request.app.state, "kev_service", None)
            if service is None:
                raise HTTPException(503, "Kev engine is not available")
            return JSONResponse(content=await service.evaluate(request, raw_request))

    async def init_state(self, engine_client, state, args):
        state.kev_service = None
        if engine_client is not None and is_kev_model(engine_client.model_config):
            state.kev_service = DecisionService(engine_client)
            logger.info(
                "Kev endpoint /v1/systemone initialized: effective max_context=%d "
                "(min of checkpoint limit and max_model_len), strict_length=%s, "
                "requests wait in the native vLLM scheduler queue",
                min(state.kev_service.config.max_context, state.kev_service.max_model_len),
                state.kev_service.config.strict_length,
            )
