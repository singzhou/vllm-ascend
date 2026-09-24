# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""SystemOne HTTP endpoint backed by vLLM pooling requests."""

import asyncio
import hashlib
import json
import logging
import math
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from vllm.pooling_params import PoolingParams

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


async def encode_on_rank(engine, token_ids, params, request_id, rank):
    """Use the native engine queue, with an optional stable DP replica hint.

    The current EngineClient.encode omits the routing hint exposed by
    AsyncLLM.add_request. This adapter owns collector consumption and aborts
    using its internal request ID, just like AsyncLLM.encode.
    """
    prompt = {"prompt_token_ids": token_ids}
    if rank is None:
        stream = engine.encode(prompt=prompt, pooling_params=params, request_id=request_id)
        try:
            async for output in stream:
                yield output
        finally:
            await stream.aclose()
        return
    collector = None
    finished = False
    try:
        collector = await engine.add_request(request_id, prompt, params, data_parallel_rank=rank)
        while not finished:
            output = collector.get_nowait() or await collector.get()
            finished = output.finished
            yield output
    finally:
        if collector is not None:
            try:
                if not finished:
                    await engine.abort(collector.request_id, internal=True)
            except Exception:
                logger.exception("Could not abort internally routed Kev request")
            finally:
                collector.close()


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
        parallel = engine.vllm_config.parallel_config
        self.dp_size = parallel.data_parallel_size
        self.route_internally = (
            self.config.dp_affinity
            and self.dp_size > 1
            and not parallel.data_parallel_external_lb
            and not parallel.data_parallel_hybrid_lb
        )
        if self.route_internally and not callable(getattr(engine, "add_request", None)):
            raise ValueError("Internal Kev DP affinity requires the native AsyncLLM engine")

    async def evaluate(self, request, raw_request):
        if request.model not in self.names:
            raise HTTPException(404, f"Unknown served model: {request.model}")
        # Submit children to the native engine queue. max_num_seqs limits
        # scheduler execution, not the number of HTTP requests allowed to wait.
        try:
            return await asyncio.wait_for(self._evaluate(request, raw_request), timeout=self.config.timeout_seconds)
        except asyncio.TimeoutError as exc:
            raise HTTPException(504, "Kev request deadline exceeded") from exc

    async def _evaluate(self, request, raw_request):
        started = time.perf_counter()
        try:
            plan = await asyncio.to_thread(
                plan_request,
                request,
                self.tokenizer,
                self.config,
                self.max_model_len,
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        inference = asyncio.create_task(self._execute(plan))
        disconnected = asyncio.create_task(self._wait_for_disconnect(raw_request))
        try:
            done, _ = await asyncio.wait((inference, disconnected), return_when=asyncio.FIRST_COMPLETED)
            if disconnected in done:
                raise HTTPException(499, "Client disconnected")
            distributions = await inference
        finally:
            for task in (inference, disconnected):
                if not task.done():
                    task.cancel()
            await asyncio.gather(inference, disconnected, return_exceptions=True)
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

    @staticmethod
    async def _wait_for_disconnect(request):
        while not await request.is_disconnected():
            await asyncio.sleep(0.1)

    async def _execute(self, plan):
        parent = uuid.uuid4().hex
        rows = iter(enumerate(plan.rows))
        results = [None] * len(plan.rows)
        active_ids = set()
        rank = None
        if self.route_internally:
            state_length = plan.rows[0].readout.state_length
            state = plan.rows[0].token_ids[:state_length]
            digest = hashlib.sha256(json.dumps(state, separators=(",", ":")).encode()).digest()
            rank = int.from_bytes(digest[:8], "big") % self.dp_size

        async def worker():
            for index, row in rows:
                child_id = f"kev-{parent}-{index}"
                params = PoolingParams(
                    task="plugin",
                    skip_reading_prefix_cache=not self.cache_reads,
                    extra_kwargs={"kev_readout": row.readout.as_dict()},
                )
                active_ids.add(child_id)
                stream = encode_on_rank(self.engine, row.token_ids, params, child_id, rank)
                try:
                    final = None
                    async for output in stream:
                        if output.finished:
                            final = output
                    if final is None:
                        raise RuntimeError("Kev child ended without a final result")
                    tensor = final.outputs.data
                    if tensor.ndim != 1 or tensor.shape[0] != len(row.readout.option_ends):
                        raise RuntimeError("Kev worker returned an invalid option distribution")
                    results[index] = probabilities(tensor.tolist(), self.config.temperature)
                    active_ids.discard(child_id)
                finally:
                    await stream.aclose()

        tasks = [asyncio.create_task(worker()) for _ in range(min(len(plan.rows), self.config.question_concurrency))]
        try:
            await asyncio.gather(*tasks)
            if any(result is None for result in results):
                raise RuntimeError("Incomplete Kev parent result")
            return results
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if active_ids:
                try:
                    await self.engine.abort(list(active_ids))
                except Exception:
                    logger.exception("Could not abort remaining Kev children")


class SystemOnePlugin:
    name = "ascend_systemone"
    required_tasks = ("plugin",)

    def attach_router(self, app: FastAPI):
        @app.post("/v1/systemone")
        async def systemone(request: SystemOneRequest, raw_request: Request):
            service = getattr(raw_request.app.state, "kev_service", None)
            if service is None:
                raise HTTPException(503, "Kev engine is not available")
            return await service.evaluate(request, raw_request)

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
