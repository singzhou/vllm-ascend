# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen backport contracts with a fake tokenizer and fake engine.

Unlike model E2E tests, these exercise service semantics and v0.30 renderer
interfaces without downloading a model or executing an accelerator kernel.
"""

import asyncio
import math
from argparse import Namespace
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.datastructures import State
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.exceptions import VLLMValidationError
from vllm.inputs import tokens_input
from vllm.logprobs import Logprob
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.sampling_params import SamplingParams

from vllm_ascend import envs
from vllm_ascend.entrypoints.structured_decisions.api_router import (
    init_structured_decisions_state,
    register_structured_decisions_api_router,
)
from vllm_ascend.entrypoints.structured_decisions.protocol import (
    ReadPromptRequest,
    StructuredDecisionRequest,
    StructuredDecisionResponse,
)
from vllm_ascend.entrypoints.structured_decisions.question_types import (
    CHOICE_LABELS_128,
    LABELS,
)
from vllm_ascend.entrypoints.structured_decisions.serving import (
    ServingStructuredDecisions,
)
from vllm_ascend.entrypoints.structured_decisions.strategies import (
    READ_STRATEGIES,
    NextTokenStrategy,
    ReadContext,
    common_prefix_len,
    reply_label_ids,
    select_read_strategy,
    type_label_ids,
)

if TYPE_CHECKING:
    from vllm.config import ModelConfig
    from vllm.engine.protocol import EngineClient
    from vllm.entrypoints.openai.models.serving import OpenAIServingModels
    from vllm.inputs import EngineInput
    from vllm.renderers import BaseRenderer
    from vllm.tokenizers import TokenizerLike

MODEL = "test-qwen"
ARCHITECTURES = (
    "Qwen3ForCausalLM",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
)
TRACE = {"traceparent": "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"}


class Tokenizer:
    """Deterministic stand-in; real Qwen tokenization belongs to model E2E."""

    max_chars_per_token = 4
    vocab_size = 12000
    all_special_ids = [10000]

    def __len__(self):
        return self.vocab_size

    @staticmethod
    def pair_ids():
        return {label: 11000 + i for i, label in enumerate(CHOICE_LABELS_128[26:])}

    def get_added_vocab(self):
        return {"assistant": 10000}

    def encode(self, text, **kwargs):
        if text[-2:] in self.pair_ids():
            return [*map(ord, text[:-2]), self.pair_ids()[text[-2:]]]
        if text.endswith("yes"):
            return [*map(ord, text[:-3]), 10001]
        if text.endswith("no"):
            return [*map(ord, text[:-2]), 10002]
        return list(map(ord, text))

    def decode(self, tokens):
        special = {
            10000: "<assistant>",
            10001: "yes",
            10002: "no",
            **{value: label for label, value in self.pair_ids().items()},
        }
        return "".join(special.get(t, chr(t)) for t in tokens)

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs.get("enable_thinking") is False
        text = "\n".join(m["role"] + ": " + m["content"] for m in messages)
        return self.encode(text) + [10000, 10, 10]


class Renderer:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.calls = []
        self.bad_tail = False

    def get_tokenizer(self):
        return self.tokenizer

    async def render_chat_async(self, messages, chat_params, tok_params, *, prompt_extras, skip_mm_cache):
        self.calls.append((messages, chat_params, tok_params, prompt_extras, skip_mm_cache))
        ids = self.tokenizer.apply_chat_template(messages[0], **chat_params.get_apply_chat_template_kwargs())
        tok_params._token_len_check(self.tokenizer, ids)
        if self.bad_tail:
            ids.append(0)
        return [messages[0]], [tokens_input(ids, cache_salt=prompt_extras.get("cache_salt"))]


class Engine:
    def __init__(self, architecture=ARCHITECTURES[0]):
        self.model_config = SimpleNamespace(
            architecture=architecture,
            logprobs_mode="raw_logprobs",
            model=MODEL,
            max_model_len=4096,
            is_encoder_decoder=False,
            multimodal_config=None,
            enable_prompt_embeds=False,
            is_hybrid=False,
            get_vocab_size=lambda: 12000,
        )
        self.vllm_config = SimpleNamespace(cache_config=SimpleNamespace(enable_prefix_caching=True))
        self.errored = False
        self.calls = []
        self.admissions = []
        self.failure = None
        self.warm_failure = None
        self.closed = []
        self.completed = []
        self.selected_token = 99999

    def check_admission(self, count):
        self.admissions.append(count)

    async def generate(self, prompt, params, request_id, **metadata):
        self.calls.append((prompt, params, request_id, metadata))
        failure = self.warm_failure if request_id.endswith("-prefix") else self.failure
        try:
            await asyncio.sleep(0)
            if failure == "raise":
                raise RuntimeError("producer execution failed")
            if failure == "empty_stream":
                return
            outputs = []
            if failure != "empty_outputs":
                label_ids = params.logprob_token_ids or []
                values = [0.04, 0.16, 0.10]
                logprobs = {
                    token: Logprob(math.log(values[index] if len(label_ids) <= 3 else 0.5 / len(label_ids)))
                    for index, token in enumerate(label_ids)
                }
                if failure == "missing_label":
                    logprobs.pop(label_ids[-1])
                logprobs.setdefault(self.selected_token, Logprob(math.log(0.3)))
                outputs = [
                    CompletionOutput(
                        index=0,
                        text="outside labels",
                        token_ids=[] if failure == "empty_tokens" else [self.selected_token],
                        cumulative_logprob=None,
                        logprobs=None if failure == "no_logprobs" else [logprobs],
                        finish_reason={
                            "engine_error": "error",
                            "abort": "abort",
                            "unfinished": None,
                        }.get(failure or "", "length"),
                    )
                ]
            if failure == "multiple_outputs":
                outputs = outputs * 2
            yield RequestOutput(
                request_id=request_id,
                prompt=None,
                prompt_token_ids=prompt["prompt_token_ids"],
                prompt_logprobs=None,
                outputs=outputs,
                finished=failure != "unfinished",
            )
            self.completed.append(request_id)
        finally:
            self.closed.append(request_id)


class Models:
    def __init__(self):
        self.lora_requests = {}

    def is_base_model(self, name):
        return name == MODEL

    def model_name(self, adapter):
        return MODEL if adapter is None else adapter.lora_name


@pytest.fixture(autouse=True)
def disable_warm_by_default(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_ASCEND_DECISIONS_WARM_SHARED_PREFIX", False, raising=False)


@pytest.fixture
def setup_service():
    engine = Engine()
    renderer = Renderer()
    online = OnlineRenderer.__new__(OnlineRenderer)
    online.model_config = cast("ModelConfig", engine.model_config)
    online.renderer = cast("BaseRenderer", renderer)
    strategy = NextTokenStrategy(ReadContext(cast("EngineClient", engine), online, None, "auto", {}))
    models = cast("OpenAIServingModels", Models())
    return ServingStructuredDecisions(models, strategy), engine, renderer


def request(**overrides):
    return StructuredDecisionRequest(
        **(
            {
                "model": MODEL,
                "state": {"message": "Français 日本語"},
                "questions": {
                    "team": {
                        "type": "choice",
                        "instructions": "Which team?",
                        "criteria": {"billing": "money", "shipping": None},
                    },
                },
                "request_id": "fixed",
            }
            | overrides
        )
    )


@pytest.mark.asyncio
async def test_choice_noul_score_use_candidate_probabilities_and_full_vocab_confidence(
    setup_service,
):
    service, engine, _ = setup_service
    body = request()
    body.questions.update(
        StructuredDecisionRequest(
            state="unused",
            questions={
                "urgent": {"type": "noul"},
                "tone": {"type": "score", "criteria": ["calm", "annoyed", "furious"]},
            },
        ).questions
    )
    response = await service.create_decision(body)
    assert isinstance(response, StructuredDecisionResponse)
    assert list(response.answers) == ["team", "urgent", "tone"]
    assert response.answers["team"]["choice"] == "shipping"
    assert response.answers["team"]["probabilities"] == pytest.approx({"billing": 0.2, "shipping": 0.8})
    assert response.answers["urgent"]["noul"] == pytest.approx(0.2)
    tone = response.answers["tone"]
    assert tone["legend"] == {"0": "calm", "1": "annoyed", "2": "furious"}
    assert tone["score"] == pytest.approx(1.2)
    assert all(answer["confidence"] == pytest.approx(0.16) for answer in response.answers.values())
    assert response.diagnostics["team"].label_mass == pytest.approx(0.2)
    assert all(not diag.argmax_is_label for diag in response.diagnostics.values())
    assert engine.admissions == [3]
    assert response.usage.output_tokens == 3
    assert response.usage.input_tokens == sum(len(c[0]["prompt_token_ids"]) for c in engine.calls)
    for _, params, _, _ in engine.calls:
        assert isinstance(params, SamplingParams)
        assert params.max_tokens == 1 and params.temperature == 0.0
        assert params.allowed_token_ids is None


@pytest.mark.asyncio
async def test_one_option_confidence_does_not_become_one(setup_service):
    service, _, _ = setup_service
    response = await service.create_decision(
        request(
            questions={
                "only": {"type": "choice", "criteria": {"one": None}},
            }
        )
    )
    assert response.answers["only"]["probabilities"] == {"one": 1.0}
    assert response.answers["only"]["confidence"] == pytest.approx(0.04)


@pytest.mark.asyncio
async def test_metadata_and_unicode_reach_real_v030_renderer_interface(setup_service):
    service, engine, renderer = setup_service
    raw = Request(
        {
            "type": "http",
            "headers": [
                (b"x-request-id", b"from-header"),
                (b"authorization", b"secret"),
                *[(k.encode(), v.encode()) for k, v in TRACE.items()],
            ],
        }
    )
    response = await service.create_decision(
        request(
            instructions="Be accurate.",
            cache_salt="tenant-a",
            priority=7,
            chat_template_kwargs={"custom": 42},
        ),
        raw,
    )
    assert response.id == "decision-from-header"
    prompt, params, rid, metadata = engine.calls[0]
    assert prompt["cache_salt"] == "tenant-a"
    assert rid == "decision-from-header-0"
    assert metadata == {"lora_request": None, "trace_headers": TRACE, "priority": 7}
    messages, chat, tok, extras, skip_mm = renderer.calls[0]
    assert messages[0][0] == {"role": "system", "content": "Be accurate."}
    assert '"message": "Français 日本語"' in messages[0][1]["content"]
    assert chat.chat_template_kwargs["enable_thinking"] is False
    assert chat.chat_template_kwargs["custom"] == 42
    assert chat.chat_template_kwargs["add_generation_prompt"] is True
    assert tok.max_output_tokens == 1 and tok.max_total_tokens == 4096
    assert tok.add_special_tokens is False
    assert extras == {"cache_salt": "tenant-a"} and skip_mm


@pytest.mark.asyncio
async def test_true_label_argmax_is_reported(setup_service):
    service, engine, _ = setup_service
    engine.selected_token = ord("A")
    response = await service.create_decision(request())
    assert response.diagnostics["team"].argmax_is_label is True


@pytest.mark.asyncio
async def test_lora_model_selection_reaches_each_engine_request(setup_service):
    service, engine, _ = setup_service
    adapter = SimpleNamespace(lora_name="qwen-adapter")
    service.models.lora_requests[adapter.lora_name] = adapter
    response = await service.create_decision(request(model=adapter.lora_name))
    assert response.model == adapter.lora_name
    assert engine.calls[0][3]["lora_request"] is adapter


@pytest.mark.asyncio
async def test_render_failure_does_not_leave_partially_submitted_decisions(warm_service, monkeypatch):
    service, engine, renderer = warm_service
    original = renderer.render_chat_async

    async def render(*args, **kwargs):
        if renderer.calls:
            raise VLLMValidationError("second prompt is too long")
        return await original(*args, **kwargs)

    monkeypatch.setattr(renderer, "render_chat_async", render)
    body = request(state="a" * 300, questions={"one": {"type": "noul"}, "two": {"type": "noul"}})
    with pytest.raises(VLLMValidationError, match="second prompt"):
        await service.create_decision(body)
    assert not engine.calls


@pytest.mark.asyncio
async def test_questions_run_concurrently_and_results_keep_request_order(setup_service, monkeypatch):
    service, engine, _ = setup_service
    original = engine.generate
    ready = asyncio.Event()
    started = []

    async def generate(prompt, params, request_id, **metadata):
        started.append(request_id)
        if len(started) == 2:
            ready.set()
        await asyncio.wait_for(ready.wait(), timeout=1)
        if request_id.endswith("-0"):
            await asyncio.sleep(0)
        async for result in original(prompt, params, request_id, **metadata):
            probability = 0.1 if request_id.endswith("-0") else 0.2
            result.outputs[0].logprobs = [{token: Logprob(math.log(probability)) for token in params.logprob_token_ids}]
            yield result

    monkeypatch.setattr(engine, "generate", generate)
    body = request(questions={"one": {"type": "noul"}, "two": {"type": "noul"}})
    response = await service.create_decision(body)
    assert list(response.answers) == ["one", "two"]
    assert response.answers["one"]["confidence"] == pytest.approx(0.1)
    assert response.answers["two"]["confidence"] == pytest.approx(0.2)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"chat_template_kwargs": {"enable_thinking": True}}, "thinking must be off"),
        ({"images": ["data:image/png;base64,AAAA"]}, "does not take image inputs"),
        ({"questions": {}}, "at least one question"),
        (
            {
                "questions": {
                    "q": {
                        "type": "choice",
                        "criteria": {"a": None},
                        "depends_on": ["x"],
                    }
                }
            },
            "unknown field",
        ),
        (
            {"questions": {"q": {"type": "score", "criteria": ["one"]}}},
            "ordered list of levels",
        ),
        (
            {
                "questions": {
                    "q": {
                        "type": "choice",
                        "criteria": dict.fromkeys(map(str, range(129))),
                    }
                }
            },
            "at most 128",
        ),
        ({"questions": {f"q{i}": {"type": "noul"} for i in range(65)}}, "at most 64"),
        ({"model": "missing"}, "does not exist"),
    ],
)
async def test_invalid_requests_fail_before_engine_submission(setup_service, overrides, match):
    service, engine, _ = setup_service
    response = await service.create_decision(request(**overrides))
    assert isinstance(response, ErrorResponse)
    assert match in response.error.message
    assert response.error.code == (404 if overrides.get("model") else 400)
    assert not engine.calls


@pytest.mark.asyncio
async def test_changed_generation_suffix_fails_before_engine_submission(setup_service):
    service, engine, renderer = setup_service
    renderer.bad_tail = True
    response = await service.create_decision(request())
    assert isinstance(response, ErrorResponse)
    assert "end the prompt differently" in response.error.message
    assert not engine.calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure,match",
    [
        ("empty_stream", "Failed to generate result"),
        ("empty_outputs", "No output generated"),
        ("engine_error", "Generation error"),
        ("no_logprobs", "No logprobs available"),
        ("missing_label", "not found in logprobs"),
    ],
)
async def test_incomplete_engine_outputs_are_errors(setup_service, failure, match):
    service, engine, _ = setup_service
    engine.failure = failure
    with pytest.raises(ValueError, match=match):
        await service.create_decision(request())


@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize("mode", ["raw_logprobs", "processed_logprobs"])
def test_official_qwen_architectures_select_next_token(architecture, mode):
    config = SimpleNamespace(architecture=architecture, logprobs_mode=mode)
    assert select_read_strategy(cast("ModelConfig", config), ("generate",)) is NextTokenStrategy


@pytest.mark.parametrize(
    "architecture,mode,tasks,match",
    [
        (ARCHITECTURES[0], "raw_logits", ("generate",), "not raw_logits"),
        (ARCHITECTURES[0], "processed_logits", ("generate",), "not processed_logits"),
        (ARCHITECTURES[0], "raw_logprobs", ("embed",), "need the generate task"),
        ("Qwen2ForCausalLM", "raw_logprobs", ("generate",), "does not support"),
        ("LlamaForCausalLM", "raw_logprobs", ("generate",), "does not support"),
    ],
)
def test_unsupported_architectures_tasks_and_modes_are_not_enabled(architecture, mode, tasks, match):
    with pytest.raises(ValueError, match=match):
        select_read_strategy(
            cast(
                "ModelConfig",
                SimpleNamespace(architecture=architecture, logprobs_mode=mode),
            ),
            tasks,
        )


def test_backport_registry_is_limited_to_official_qwen_scope():
    assert set(READ_STRATEGIES) == set(ARCHITECTURES)


@pytest.mark.parametrize("salt", ["", "a/b", "a\\b", "a@b", "a\x00b", "x" * 129, 7])
def test_v030_cache_salt_validation_is_preserved(salt):
    with pytest.raises(VLLMValidationError, match="cache_salt"):
        request(cache_salt=salt)


def test_read_prompt_forces_generation_and_reserves_one_output_token():
    read = ReadPromptRequest(chat_template_kwargs={"continue_final_message": True})
    chat = read.build_chat_params(None, "auto").with_defaults({"enable_thinking": True})
    assert chat.chat_template_kwargs["enable_thinking"] is False
    assert chat.chat_template_kwargs["continue_final_message"] is False
    assert chat.chat_template_kwargs["add_generation_prompt"] is True
    tok = read.build_tok_params(cast("ModelConfig", SimpleNamespace(max_model_len=200)))
    assert tok.max_input_tokens == 199


def test_distinct_single_token_labels_are_required():
    tokenizer = cast("TokenizerLike", Tokenizer())
    _, ids = reply_label_ids(tokenizer, [10000, 10, 10])
    assert ids == list(map(ord, LABELS))
    with pytest.raises(ValueError, match="not one distinct token"):
        reply_label_ids(tokenizer, [10000, 10, 10], ("AAA", "B"))


def test_router_serves_json_and_preserves_error_status(setup_service):
    service, _, _ = setup_service
    app = FastAPI()
    app.state.serving_structured_decisions = service
    register_structured_decisions_api_router(app)
    with TestClient(app) as client:
        response = client.post("/v1/systemone", json=request().model_dump())
        assert response.status_code == 200
        assert response.json()["answers"]["team"]["choice"] == "shipping"
        response = client.post("/v1/systemone", json=request(model="absent").model_dump())
        assert response.status_code == 404
        assert response.json()["error"]["type"] == "NotFoundError"


def test_service_initialization_enables_supported_qwen_and_disables_unsupported(
    setup_service,
):
    service, engine, _ = setup_service
    state = State(
        {
            "online_renderer": service.strategy.context.online_renderer,
            "openai_serving_models": Models(),
        }
    )
    args = Namespace(chat_template=None, chat_template_content_format="auto")
    init_structured_decisions_state(engine, state, args, None, ("generate",), {})
    assert isinstance(state.serving_structured_decisions, ServingStructuredDecisions)
    engine.model_config.architecture = "LlamaForCausalLM"
    init_structured_decisions_state(engine, state, args, None, ("generate",), {})
    assert state.serving_structured_decisions is None


@pytest.fixture
def warm_service(setup_service, monkeypatch):
    service, engine, renderer = setup_service
    monkeypatch.setattr(envs, "VLLM_ASCEND_DECISIONS_WARM_SHARED_PREFIX", True)
    engine.model_config.is_hybrid = True
    # Graph and asynchronous modes must not become new API-level exclusions.
    engine.model_config.enforce_eager = False
    engine.vllm_config.scheduler_config = SimpleNamespace(async_scheduling=True)
    return service, engine, renderer


def render_with_prefix(service, monkeypatch, prefix_length):
    rendered: list[EngineInput] = []

    async def render(read_request, messages):
        ids = [7] * prefix_length + [31 + len(rendered), 10000, 10, 10]
        prompt = tokens_input(ids, cache_salt=read_request.cache_salt)
        rendered.append(prompt)
        return prompt, ids

    monkeypatch.setattr(service.strategy, "_render", render)
    return rendered


def two_questions(**overrides):
    return request(**({"questions": {"one": {"type": "noul"}, "two": {"type": "noul"}}} | overrides))


@pytest.mark.parametrize(
    "prompts,expected",
    [
        ([], 0),
        ([[]], 0),
        ([[1, 2]], 2),
        ([[1, 2], [1, 2]], 2),
        ([[1, 2], [1]], 1),
        ([[1], [2]], 0),
        ([[1, 2], []], 0),
        ([[1, 2, 3], [1, 2, 4], [1, 5]], 1),
    ],
)
def test_common_prefix_uses_complete_token_sequences(prompts, expected):
    assert common_prefix_len(prompts) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "enabled,hybrid,apc,count,prefix,warms",
    [
        (False, True, True, 2, 128, False),
        (True, False, True, 2, 128, False),
        (True, True, False, 2, 128, False),
        (True, True, True, 1, 128, False),
        (True, True, True, 2, 127, False),
        (True, True, True, 2, 128, True),
    ],
)
async def test_warm_gates_and_exact_threshold(warm_service, monkeypatch, enabled, hybrid, apc, count, prefix, warms):
    service, engine, _ = warm_service
    monkeypatch.setattr(envs, "VLLM_ASCEND_DECISIONS_WARM_SHARED_PREFIX", enabled)
    engine.model_config.is_hybrid = hybrid
    engine.vllm_config.cache_config.enable_prefix_caching = apc
    rendered = render_with_prefix(service, monkeypatch, prefix)
    body = request(questions={f"q{i}": {"type": "noul"} for i in range(count)})
    response = await service.create_decision(body)
    assert isinstance(response, StructuredDecisionResponse)
    assert len(engine.calls) == count + int(warms)
    assert sum(call[2].endswith("-prefix") for call in engine.calls) == int(warms)
    if warms:
        assert engine.calls[0][0]["prompt_token_ids"] == [7] * prefix
    branches = [call for call in engine.calls if not call[2].endswith("-prefix")]
    assert [call[0] for call in branches] == rendered


@pytest.mark.asyncio
async def test_warm_finishes_and_closes_before_original_branches(warm_service, monkeypatch):
    service, engine, _ = warm_service
    rendered = render_with_prefix(service, monkeypatch, 160)
    adapter = SimpleNamespace(lora_name="qwen-adapter")
    service.models.lora_requests[adapter.lora_name] = adapter
    original = engine.generate
    started, release = asyncio.Event(), asyncio.Event()
    seen = []

    async def generate(prompt, params, request_id, **metadata):
        seen.append(request_id)
        if request_id.endswith("-prefix"):
            started.set()
            await release.wait()
        else:
            assert "decision-fixed-prefix" in engine.completed
            assert "decision-fixed-prefix" in engine.closed
        async for result in original(prompt, params, request_id, **metadata):
            yield result

    monkeypatch.setattr(engine, "generate", generate)
    raw = Request(
        {
            "type": "http",
            "headers": [(k.encode(), v.encode()) for k, v in TRACE.items()],
        }
    )
    body = two_questions(model=adapter.lora_name, cache_salt="tenant-a", priority=7)
    task = asyncio.create_task(service.create_decision(body, raw))
    await asyncio.wait_for(started.wait(), timeout=1)
    assert seen == ["decision-fixed-prefix"]
    release.set()
    response = await asyncio.wait_for(task, timeout=1)
    assert len(engine.calls) == 3
    assert engine.calls[0][1].logprob_token_ids is None
    assert engine.calls[0][1].max_tokens == 1
    assert engine.calls[0][1].temperature == 0
    for prompt, _, _, metadata in engine.calls:
        assert prompt["cache_salt"] == "tenant-a"
        assert metadata == {
            "lora_request": adapter,
            "priority": 7,
            "trace_headers": TRACE,
        }
    assert [call[0] for call in engine.calls[1:]] == rendered
    assert response.usage.input_tokens == sum(len(p["prompt_token_ids"]) for p in rendered)
    assert response.usage.output_tokens == 2
    warm_answers = response.answers
    warm_inputs = [dict(call[0]) for call in engine.calls[1:]]
    rendered.clear()
    engine.calls.clear()
    monkeypatch.setattr(envs, "VLLM_ASCEND_DECISIONS_WARM_SHARED_PREFIX", False)
    response = await service.create_decision(body, raw)
    assert [call[0] for call in engine.calls] == warm_inputs
    assert response.answers == warm_answers


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        "empty_stream",
        "empty_outputs",
        "empty_tokens",
        "multiple_outputs",
        "engine_error",
        "abort",
        "unfinished",
    ],
)
async def test_warm_invalid_completion_fails_without_submitting_branches(warm_service, monkeypatch, failure):
    service, engine, _ = warm_service
    render_with_prefix(service, monkeypatch, 128)
    engine.warm_failure = failure
    with pytest.raises(ValueError, match="[Ww]arm"):
        await service.create_decision(two_questions())
    assert [call[2] for call in engine.calls] == ["decision-fixed-prefix"]
    assert engine.closed == ["decision-fixed-prefix"]


@pytest.mark.asyncio
async def test_warm_engine_exception_is_not_swallowed(warm_service, monkeypatch):
    service, engine, _ = warm_service
    render_with_prefix(service, monkeypatch, 128)
    engine.warm_failure = "raise"
    with pytest.raises(RuntimeError, match="producer execution failed"):
        await service.create_decision(two_questions())
    assert [call[2] for call in engine.calls] == ["decision-fixed-prefix"]
    assert engine.closed == ["decision-fixed-prefix"]


@pytest.mark.asyncio
async def test_cancelling_warm_closes_producer_and_submits_no_branches(warm_service, monkeypatch):
    service, engine, _ = warm_service
    render_with_prefix(service, monkeypatch, 128)
    started, hold = asyncio.Event(), asyncio.Event()
    seen, closed = [], []

    async def generate(prompt, params, request_id, **metadata):
        seen.append(request_id)
        try:
            started.set()
            await hold.wait()
            raise AssertionError("Cancelled producer must never yield")
            yield
        finally:
            closed.append(request_id)

    monkeypatch.setattr(engine, "generate", generate)
    task = asyncio.create_task(service.create_decision(two_questions()))
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    response = await asyncio.wait_for(task, timeout=1)
    assert isinstance(response, ErrorResponse)
    assert "Client disconnected" in response.error.message
    assert seen == closed == ["decision-fixed-prefix"]


@pytest.mark.asyncio
async def test_parent_requests_keep_producer_ids_and_cache_salts_separate(warm_service, monkeypatch):
    service, engine, _ = warm_service
    render_with_prefix(service, monkeypatch, 128)
    responses = await asyncio.gather(
        service.create_decision(two_questions(request_id="left", cache_salt="left")),
        service.create_decision(two_questions(request_id="right", cache_salt="right")),
    )
    assert all(isinstance(response, StructuredDecisionResponse) for response in responses)
    ids = [call[2] for call in engine.calls]
    assert len(ids) == len(set(ids)) == 6
    for prompt, _, rid, _ in engine.calls:
        assert prompt["cache_salt"] == ("left" if rid.startswith("decision-left") else "right")


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [1, 2, 26, 27, 62, 64, 127, 128])
async def test_choice_capacity_remains_one_engine_request_per_question(warm_service, count):
    service, engine, _ = warm_service
    response = await service.create_decision(
        request(
            questions={
                "q": {
                    "type": "choice",
                    "instructions": "Pick one.",
                    "criteria": dict.fromkeys(f"option-{i}" for i in range(count)),
                }
            }
        )
    )
    assert isinstance(response, StructuredDecisionResponse)
    assert len(engine.calls) == 1
    _, params, _, _ = engine.calls[0]
    assert len(params.logprob_token_ids) == count
    assert len(set(params.logprob_token_ids)) == count
    assert params.max_tokens == 1 and params.logprobs is None
    assert params.allowed_token_ids is None
    assert len(response.answers["q"]["probabilities"]) == count
    assert sum(response.answers["q"]["probabilities"].values()) == pytest.approx(1)
    assert response.usage.output_tokens == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("sampled_inside", [False, True])
async def test_all_128_candidates_are_read_without_sampled_token_contamination(setup_service, sampled_inside):
    service, engine, _ = setup_service
    if sampled_inside:
        engine.selected_token = ord("A")
    response = await service.create_decision(
        request(questions={"q": {"type": "choice", "criteria": dict.fromkeys(map(str, range(128)))}})
    )
    probabilities = response.answers["q"]["probabilities"]
    assert len(probabilities) == 128
    assert set(probabilities) == set(map(str, range(128)))
    assert response.diagnostics["q"].label_mass == pytest.approx(0.5)
    assert response.answers["q"]["confidence"] == pytest.approx(0.5 / 128)
    assert response.diagnostics["q"].argmax_is_label is sampled_inside


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [1, 2, 26])
async def test_small_choice_complete_prompt_ids_preserve_original_contract(setup_service, count):
    service, _, renderer = setup_service
    criteria = {f"option-{i}": f"description-{i}" for i in range(count)}
    await service.create_decision(
        request(
            state="Original state",
            questions={
                "q": {
                    "type": "choice",
                    "instructions": "Choose a team",
                    "criteria": criteria,
                }
            },
        )
    )
    expected = "Original state\n\nQuestion: Choose a team\n"
    expected += "\n".join(
        f"{label}: {name} - {description}" for label, (name, description) in zip(LABELS, criteria.items())
    )
    expected += "\nAnswer with the letter of one option only."
    prompt = renderer.calls[0][0][0]
    assert prompt == [{"role": "user", "content": expected}]
    expected_ids = renderer.tokenizer.apply_chat_template(
        [{"role": "user", "content": expected}], enable_thinking=False
    )
    assert service.strategy.tail == [10, 10]
    assert renderer.tokenizer.apply_chat_template(prompt, enable_thinking=False) == expected_ids


@pytest.mark.parametrize("failure", ["split", "duplicate", "special", "added", "negative", "range", "empty"])
def test_extended_labels_reject_invalid_single_token_mapping(failure, monkeypatch):
    tokenizer = Tokenizer()
    original = tokenizer.encode
    target = {
        "duplicate": ord("A"),
        "special": 10000,
        "added": 11000,
        "negative": -1,
        "range": len(tokenizer),
    }

    def encode(text, **kwargs):
        if text.endswith("AA"):
            if failure == "split":
                return [*map(ord, text[:-2]), ord("A"), ord("A")]
            if failure == "empty":
                return []
            return [*map(ord, text[:-2]), target[failure]]
        return original(text, **kwargs)

    monkeypatch.setattr(tokenizer, "encode", encode)
    if failure == "added":
        monkeypatch.setattr(tokenizer, "get_added_vocab", lambda: {"assistant": 10000, "AA": 11000})
    with pytest.raises(ValueError, match="AA"):
        reply_label_ids(cast("TokenizerLike", tokenizer), [10000, 10, 10], ("A", "AA"))


def test_choice_codebook_initialization_failure_is_visible_and_disables_service(setup_service, monkeypatch, caplog):
    service, engine, renderer = setup_service
    original = renderer.tokenizer.encode

    def split_label(text, **kwargs):
        if text.endswith("AA"):
            return [*map(ord, text[:-2]), ord("A"), ord("A")]
        return original(text, **kwargs)

    monkeypatch.setattr(renderer.tokenizer, "encode", split_label)
    state = State(
        {
            "online_renderer": service.strategy.context.online_renderer,
            "openai_serving_models": Models(),
        }
    )
    args = Namespace(chat_template=None, chat_template_content_format="auto")
    with caplog.at_level("ERROR"):
        init_structured_decisions_state(engine, state, args, None, ("generate",), {})
    assert state.serving_structured_decisions is None
    assert any(record.levelname == "ERROR" and "initialization failed" in record.message for record in caplog.records)
    assert "AA" in caplog.text and MODEL in caplog.text
    assert ARCHITECTURES[0] in caplog.text and "chat_template_sha256" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scores",
    [(float("nan"), -1.0), (float("inf"), -1.0), (-float("inf"), -float("inf"))],
)
async def test_invalid_candidate_scores_fail_before_nan_response(setup_service, monkeypatch, scores):
    service, engine, _ = setup_service
    original = engine.generate

    async def generate(prompt, params, request_id, **metadata):
        async for result in original(prompt, params, request_id, **metadata):
            values = result.outputs[0].logprobs[0]
            for token, score in zip(params.logprob_token_ids, scores):
                values[token] = Logprob(score)
            yield result

    monkeypatch.setattr(engine, "generate", generate)
    with pytest.raises(ValueError, match="[Ll]ogprob"):
        await service.create_decision(request())


@pytest.mark.asyncio
async def test_negative_infinite_candidate_has_zero_probability(setup_service, monkeypatch):
    service, engine, _ = setup_service
    original = engine.generate

    async def generate(prompt, params, request_id, **metadata):
        async for result in original(prompt, params, request_id, **metadata):
            values = result.outputs[0].logprobs[0]
            values[params.logprob_token_ids[0]] = Logprob(-math.inf)
            values[params.logprob_token_ids[1]] = Logprob(math.log(0.2))
            yield result

    monkeypatch.setattr(engine, "generate", generate)
    response = await service.create_decision(request())
    assert response.answers["team"]["probabilities"] == {
        "billing": 0.0,
        "shipping": 1.0,
    }
    assert response.answers["team"]["confidence"] == pytest.approx(0.2)
    assert response.diagnostics["team"].label_mass == pytest.approx(0.2)


def test_union_label_initialization_is_not_limited_to_per_request_capacity(
    setup_service,
):
    service, _, _ = setup_service
    assert len(service.strategy.label_ids) == 140
    assert len(set(service.strategy.label_ids.values())) == 140


def test_model_vocabulary_bound_is_checked_besides_tokenizer_vocabulary():
    tokenizer = cast("TokenizerLike", Tokenizer())
    with pytest.raises(ValueError, match="AA"):
        reply_label_ids(tokenizer, [10000, 10, 10], ("AA",), vocab_size=1000)


@pytest.mark.asyncio
@pytest.mark.parametrize("unsupported", ["data_parallel", "external_cache"])
async def test_warm_falls_back_before_submission_for_unsupported_cache_routing(warm_service, monkeypatch, unsupported):
    service, engine, _ = warm_service
    render_with_prefix(service, monkeypatch, 128)
    if unsupported == "data_parallel":
        engine.vllm_config.parallel_config = SimpleNamespace(data_parallel_size=2)
    else:
        engine.vllm_config.kv_transfer_config = object()
    response = await service.create_decision(two_questions())
    assert isinstance(response, StructuredDecisionResponse)
    assert [call[2] for call in engine.calls] == [
        "decision-fixed-0",
        "decision-fixed-1",
    ]


def test_different_question_types_cannot_share_a_label_token(monkeypatch):
    tokenizer = Tokenizer()
    original = tokenizer.encode

    def encode(text, **kwargs):
        if text.endswith("yes"):
            return [*map(ord, text[:-3]), ord("A")]
        return original(text, **kwargs)

    monkeypatch.setattr(tokenizer, "encode", encode)
    with pytest.raises(ValueError, match="yes.*shares token ID"):
        type_label_ids(cast("TokenizerLike", tokenizer), [10000, 10, 10])


@pytest.mark.asyncio
async def test_branch_failure_closes_other_pending_engine_generators(setup_service, monkeypatch):
    service, engine, _ = setup_service
    pending = asyncio.Event()
    hold = asyncio.Event()
    closed = []
    all_closed = asyncio.Event()

    async def generate(prompt, params, request_id, **metadata):
        try:
            if request_id.endswith("-0"):
                await pending.wait()
                raise RuntimeError("branch failed")
            pending.set()
            await hold.wait()
            yield
        finally:
            closed.append(request_id)
            if len(closed) == 2:
                all_closed.set()

    monkeypatch.setattr(engine, "generate", generate)
    with pytest.raises(RuntimeError, match="branch failed"):
        await asyncio.wait_for(service.create_decision(two_questions()), timeout=1)
    await asyncio.wait_for(all_closed.wait(), timeout=1)
    assert sorted(closed) == ["decision-fixed-0", "decision-fixed-1"]
