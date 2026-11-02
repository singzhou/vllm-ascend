# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read strategies: how a model is asked for label probabilities.

Only models in READ_STRATEGIES are currently supported.
"""

import asyncio
import json
import logging
import math
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any, ClassVar

import vllm.envs as envs
from PIL import Image
from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import ChatTemplateContentFormatOption
from vllm.inputs import EngineInput, tokens_input
from vllm.logger import logger
from vllm.lora.request import LoRARequest
from vllm.multimodal.media import (
    MEDIA_CONNECTOR_REGISTRY,
    MediaConnector,
)
from vllm.renderers.inputs.preprocess import extract_prompt_components
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.sampling_params import SamplingParams
from vllm.tasks import SupportedTask
from vllm.tokenizers import TokenizerLike

from vllm_ascend import envs as ascend_envs
from vllm_ascend.entrypoints.structured_decisions.label_reads import next_token_label_reads

from .protocol import ReadPromptRequest
from .question_types import (
    CHOICE_LABELS_128,
    LABELS,
    QUESTION_TYPES,
    Question,
    StructuredDecisionError,
    label_softmax,
)

# Fewer shared tokens are not worth an extra round trip before the reads.
MIN_WARM_PREFIX_TOKENS = 128


def common_prefix_len(prompts: Sequence[Sequence[int]]) -> int:
    """The number of leading token ids that every prompt starts with."""
    if not prompts:
        return 0
    shortest = min(prompts, key=len)
    for i, token_id in enumerate(shortest):
        if any(p[i] != token_id for p in prompts):
            return i
    return len(shortest)


@dataclass
class ReadContext:
    engine_client: EngineClient
    online_renderer: OnlineRenderer
    chat_template: str | None
    chat_template_content_format: ChatTemplateContentFormatOption
    default_chat_template_kwargs: dict[str, Any]


@dataclass(frozen=True)
class DecisionLimits:
    max_questions: int
    max_options: int


@dataclass
class QuestionRead:
    probs: list[float]
    label_mass: float
    argmax_is_label: bool
    input_tokens: int
    output_tokens: int


def state_text(state: Any) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


class ReadStrategy(ABC):
    #: The engine task reads run as.
    task: ClassVar[SupportedTask] = "generate"
    #: The modalities a request may attach media of.
    modalities: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, context: ReadContext):
        self.context = context

    @cached_property
    def _connector(self) -> MediaConnector:
        model_config = self.context.engine_client.model_config
        return MEDIA_CONNECTOR_REGISTRY.load(
            envs.VLLM_MEDIA_CONNECTOR,
            media_io_kwargs=model_config.get_multimodal_config().media_io_kwargs,
            allowed_local_media_path=model_config.allowed_local_media_path,
            allowed_media_domains=model_config.allowed_media_domains,
        )

    @abstractmethod
    def limits(self) -> DecisionLimits: ...

    def render_state(self, state: Any) -> str:
        """The request's state as the model reads it."""
        return state_text(state)

    def num_requests(self, questions: list[Question]) -> int:
        """The engine requests a decision takes."""
        return len(questions)

    @abstractmethod
    async def read(
        self,
        questions: list[Question],
        instructions: str | None,
        state: str,
        *,
        request_id: str,
        chat_template_kwargs: dict[str, Any] | None,
        lora_request: LoRARequest | None,
        priority: int,
        cache_salt: str | None = None,
        trace_headers: Mapping[str, str] | None = None,
        media: Mapping[str, Sequence[str]] | None = None,
    ) -> list[QuestionRead]:
        """One read per question, in order."""


def prompt_tail(tokenizer: TokenizerLike, prompt_ids: Sequence[int]) -> list[int]:
    """The prompt's ids after its last added token."""
    added = set(tokenizer.get_added_vocab().values())
    start = max((i + 1 for i, t in enumerate(prompt_ids) if t in added), default=0)
    return list(prompt_ids[start:])


def reply_label_ids(
    tokenizer: TokenizerLike,
    prompt_ids: Sequence[int],
    labels: Sequence[str] = LABELS,
    *,
    vocab_size: int | None = None,
) -> tuple[list[int], list[int]]:
    """The prompt's ids after its last added token, and the token each of
    ``labels`` adds after them as the first token of the reply. Raises
    ValueError if a label is not one distinct token there."""
    tail = prompt_tail(tokenizer, prompt_ids)
    text = tokenizer.decode(tail)
    ids: list[int] = []
    forbidden = set(tokenizer.all_special_ids) | set(tokenizer.get_added_vocab().values())
    upper_bound = len(tokenizer)
    if vocab_size is not None:
        upper_bound = min(upper_bound, vocab_size)
    for label in labels:
        extended = tokenizer.encode(text + label, add_special_tokens=False)
        if len(extended) != len(tail) + 1 or extended[:-1] != tail:
            raise ValueError(f"label {label!r} is not one distinct token after this model's chat prompt")
        token_id = extended[-1]
        if token_id in ids or token_id in forbidden or not 0 <= token_id < upper_bound:
            raise ValueError(
                f"label {label!r} has duplicate, special, added or out-of-range "
                f"token ID {token_id} for this model's chat prompt"
            )
        ids.append(token_id)
    return tail, ids


def type_label_ids(
    tokenizer: TokenizerLike,
    prompt_ids: Sequence[int],
    *,
    vocab_size: int | None = None,
) -> dict[str, int]:
    """The reply's first token for each label of every registered question
    type. Raises ValueError as reply_label_ids does."""
    label_ids: dict[str, int] = {}
    for qtype in QUESTION_TYPES.values():
        _, ids = reply_label_ids(tokenizer, prompt_ids, qtype.label_set, vocab_size=vocab_size)
        for label, token_id in zip(qtype.label_set, ids):
            if any(old_label != label and old_id == token_id for old_label, old_id in label_ids.items()):
                raise ValueError(f"label {label!r} shares token ID {token_id} with another label")
            label_ids[label] = token_id
    return label_ids


def cap_pixels(image: Image.Image, max_pixels: int) -> Image.Image:
    """Shrink ``image`` to at most ``max_pixels`` pixels (bicubic)."""
    w, h = image.size
    if w * h <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / (w * h))
    size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return image.resize(size, Image.Resampling.BICUBIC)


class NextTokenStrategy(ReadStrategy):
    """Autoregressive models. Each question is one request: the state, then the
    question with its labeled options, and the logprobs of the label tokens as
    the reply's first token. The requests share the state. With prefix caching,
    it is prefilled once."""

    #: Images are shrunk to at most this many pixels before the processor.
    max_image_pixels: ClassVar[int | None] = None

    def __init__(self, context: ReadContext):
        super().__init__(context)
        # Every prompt ends with the same generation prompt, so a label's token
        # is the same for every question.
        prompt = self._generation_prompt()
        self.tail = prompt_tail(self._tokenizer(), prompt)
        self.label_ids = type_label_ids(
            self._tokenizer(),
            prompt,
            vocab_size=context.engine_client.model_config.get_vocab_size(),
        )

    def _tokenizer(self) -> TokenizerLike:
        return self.context.online_renderer.renderer.get_tokenizer()

    def _generation_prompt(self) -> list[int]:
        """A chat prompt's ids up to the generation prompt, with thinking off."""
        context, tokenizer = self.context, self._tokenizer()
        probe = tokenizer.apply_chat_template(
            [{"role": "user", "content": "x"}],
            **{
                **context.default_chat_template_kwargs,
                "chat_template": context.chat_template,
                "add_generation_prompt": True,
                "tokenize": True,
                "return_dict": False,
                "enable_thinking": False,
            },
        )
        if isinstance(probe, str):
            probe = tokenizer.encode(probe, add_special_tokens=False)
        return list(probe)

    def _sampling_params(self, label_ids: list[int], prompt_ids: list[int]) -> SamplingParams:
        return SamplingParams(max_tokens=1, temperature=0.0, logprob_token_ids=label_ids)

    def _read_input(self, engine_input: EngineInput, prompt_ids: list[int]) -> EngineInput:
        """The engine input for a read of the rendered prompt."""
        return engine_input

    def _label_groups(self, question: Question) -> list[list[int]]:
        """The tokens that count for each label. A label scores its best one."""
        return [[self.label_ids[label]] for label in question.labels]

    def _content(self, state: str, question: Question) -> str:
        """The user turn: the state, then the question."""
        return f"{state}\n\n{question.type.prompt(question)}"

    async def _image_parts(self, urls: Sequence[str]) -> list[dict[str, Any]]:
        """The images as chat content parts, fetched once so that every
        question reads the same pictures."""
        if not urls:
            return []
        try:
            images = await asyncio.gather(*map(self._connector.fetch_image_async, urls))
        except ValueError as e:
            raise StructuredDecisionError(f"media: {e}") from e
        if self.max_image_pixels is not None:
            images = [cap_pixels(image, self.max_image_pixels) for image in images]
        return [{"type": "image_pil", "image_pil": image} for image in images]

    def limits(self) -> DecisionLimits:
        return DecisionLimits(max_questions=64, max_options=len(CHOICE_LABELS_128))

    def _read_request(
        self, chat_template_kwargs: dict[str, Any] | None, cache_salt: str | None = None
    ) -> ReadPromptRequest:
        if (chat_template_kwargs or {}).get("enable_thinking"):
            raise StructuredDecisionError("a read is the reply's first token, so thinking must be off")
        return ReadPromptRequest(chat_template_kwargs=chat_template_kwargs, cache_salt=cache_salt)

    async def _render(
        self, read_request: ReadPromptRequest, messages: list[dict[str, Any]]
    ) -> tuple[EngineInput, list[int]]:
        ctx = self.context
        _, (engine_input,) = await ctx.online_renderer.preprocess_chat(
            read_request,
            messages,
            default_template=ctx.chat_template,
            default_template_content_format=ctx.chat_template_content_format,
            default_template_kwargs=ctx.default_chat_template_kwargs,
            # A read that fails before it reaches the engine must not leave the
            # sender cache believing the engine has the images.
            skip_mm_cache=True,
        )
        prompt_ids = extract_prompt_components(ctx.engine_client.model_config, engine_input).token_ids
        return engine_input, list(prompt_ids or [])

    async def _warm_shared_prefix(
        self,
        prompts: Sequence[Sequence[int]],
        request_id: str,
        *,
        lora_request: LoRARequest | None,
        priority: int,
        cache_salt: str | None,
        trace_headers: Mapping[str, str] | None,
    ) -> None:
        """Complete a shared-prefix producer before submitting hybrid reads."""
        engine_client = self.context.engine_client
        reason = None
        if not ascend_envs.VLLM_ASCEND_DECISIONS_WARM_SHARED_PREFIX:
            reason = "disabled"
        elif len(prompts) < 2:
            reason = "single_question"
        elif not engine_client.model_config.is_hybrid:
            reason = "not_hybrid"
        elif not engine_client.vllm_config.cache_config.enable_prefix_caching:
            reason = "prefix_caching_disabled"
        elif (
            getattr(
                getattr(engine_client.vllm_config, "parallel_config", None),
                "data_parallel_size",
                1,
            )
            > 1
        ):
            reason = "data_parallel_unsupported"
        elif getattr(engine_client.vllm_config, "kv_transfer_config", None) is not None:
            reason = "kv_transfer_unsupported"
        if reason is not None:
            logger.debug(
                "Decisions prefix warm: request_id=%s status=skipped reason=%s",
                request_id,
                reason,
            )
            return
        shared = common_prefix_len(prompts)
        if shared < MIN_WARM_PREFIX_TOKENS:
            logger.debug(
                "Decisions prefix warm: request_id=%s status=skipped reason=short_prefix prefix_tokens=%d",
                request_id,
                shared,
            )
            return

        generator = engine_client.generate(
            tokens_input(list(prompts[0][:shared]), cache_salt=cache_salt),
            SamplingParams(max_tokens=1, temperature=0.0),
            f"{request_id}-prefix",
            lora_request=lora_request,
            trace_headers=trace_headers,
            priority=priority,
        )
        started = time.perf_counter()
        result = None
        try:
            async for result in generator:
                pass
        finally:
            # AsyncLLM.generate owns abort-on-cancel/GeneratorExit. Closing the
            # iterator invokes that contract without a second explicit abort.
            await generator.aclose()
        if (
            result is None
            or not result.finished
            or len(result.outputs) != 1
            or not result.outputs[0].token_ids
            or result.outputs[0].finish_reason not in ("stop", "length")
        ):
            raise ValueError("Shared-prefix warm request did not complete successfully")
        logger.debug(
            "Decisions prefix warm: request_id=%s status=completed prefix_tokens=%d cached_tokens=%d elapsed_ms=%.3f",
            request_id,
            shared,
            result.num_cached_tokens,
            (time.perf_counter() - started) * 1000,
        )

    async def read(
        self,
        questions: list[Question],
        instructions: str | None,
        state: str,
        *,
        request_id: str,
        chat_template_kwargs: dict[str, Any] | None,
        lora_request: LoRARequest | None,
        priority: int,
        cache_salt: str | None = None,
        trace_headers: Mapping[str, str] | None = None,
        media: Mapping[str, Sequence[str]] | None = None,
    ) -> list[QuestionRead]:
        ctx = self.context
        read_request = self._read_request(chat_template_kwargs, cache_salt)
        image_parts = await self._image_parts((media or {}).get("image", ()))

        slots, engine_inputs, params, prompts = [], [], [], []
        for q in questions:
            content: Any = self._content(state, q)
            if image_parts:
                content = [*image_parts, {"type": "text", "text": content}]
            messages = [{"role": "user", "content": content}]
            if instructions:
                messages.insert(0, {"role": "system", "content": instructions})
            engine_input, prompt_ids = await self._render(read_request, messages)
            if prompt_ids[len(prompt_ids) - len(self.tail) :] != self.tail:
                raise StructuredDecisionError(
                    "these chat options end the prompt differently, so the labels' tokens are unknown"
                )
            prompts.append(prompt_ids)
            slots.append(self._label_groups(q))
            engine_inputs.append(self._read_input(engine_input, prompt_ids))
            params.append(self._sampling_params([i for ids in slots[-1] for i in ids], prompt_ids))

        if media:
            logger.debug(
                "Decisions prefix warm: request_id=%s status=skipped reason=unsupported_media",
                request_id,
            )
        else:
            await self._warm_shared_prefix(
                prompts,
                request_id,
                lora_request=lora_request,
                priority=priority,
                cache_salt=cache_salt,
                trace_headers=trace_headers,
            )
        label_reads = await next_token_label_reads(
            ctx.engine_client,
            engine_inputs,
            params,
            request_id,
            lora_request=lora_request,
            trace_headers=trace_headers,
            priority=priority,
        )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Decisions reads: request_id=%s questions=%d input_tokens=%d branch_cached_tokens=%s",
                request_id,
                len(label_reads),
                sum(len(read.result.prompt_token_ids or ()) for read in label_reads),
                [read.result.num_cached_tokens for read in label_reads],
            )
        reads = []
        for groups, label_read in zip(slots, label_reads):
            output = label_read.result.outputs[0]
            logprobs = iter(label_read.logprobs)
            reads.append(
                QuestionRead(
                    probs=label_softmax([max(next(logprobs) for _ in g) for g in groups]),
                    label_mass=sum(math.exp(lp) for lp in label_read.logprobs),
                    argmax_is_label=bool(output.token_ids) and any(output.token_ids[0] in ids for ids in groups),
                    input_tokens=len(label_read.result.prompt_token_ids or ()),
                    output_tokens=len(output.token_ids),
                )
            )
        return reads


READ_STRATEGIES: dict[str, type[ReadStrategy]] = {
    "Qwen3ForCausalLM": NextTokenStrategy,
    "Qwen3_5ForConditionalGeneration": NextTokenStrategy,
    "Qwen3_5MoeForConditionalGeneration": NextTokenStrategy,
}


# label_mass sums the labels' full-vocabulary probabilities. The logprobs modes
# return those probabilities. The logits modes return raw logits instead.
LOGPROBS_MODES = frozenset({"raw_logprobs", "processed_logprobs"})


def select_read_strategy(model_config: ModelConfig, supported_tasks: Sequence[SupportedTask]) -> type[ReadStrategy]:
    """Raise ValueError when the model cannot serve structured decisions."""
    strategy = READ_STRATEGIES.get(model_config.architecture)
    if strategy is None:
        raise ValueError(
            "The structured decisions API does not support "
            f"{model_config.architecture}. Supported architectures: "
            f"{sorted(READ_STRATEGIES)}"
        )
    if strategy.task not in supported_tasks:
        raise ValueError(f"Structured decisions on {model_config.architecture} need the {strategy.task} task")
    if strategy.task == "generate" and model_config.logprobs_mode not in LOGPROBS_MODES:
        raise ValueError(
            "Structured decisions need --logprobs-mode raw_logprobs or "
            f"processed_logprobs, not {model_config.logprobs_mode}"
        )
    return strategy
