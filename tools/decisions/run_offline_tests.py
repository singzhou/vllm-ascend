# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only contract tests without installing the Linux accelerator runtime.

Run with a small Python environment containing pytest, pytest-asyncio, numpy,
msgspec, pydantic, FastAPI, httpx and regex. This does not load model weights.

The changed Decisions modules are imported in full. Original v0.30 protocol,
sampling, input, async, BaseServing and renderer-parameter implementations run
unchanged. Import-only heavyweight dependencies are replaced. The real v0.30
OnlineRenderer.preprocess_chat method is extracted with AST, so its call to the
fake low-level renderer exercises the actual v0.30 interface. Engine execution,
tokenization, media handling, exception registration and server initialization
are outside this runner's evidence; use the normal suites and remote smoke test.
"""

import argparse
import ast
import importlib.util
import logging
import sys
import types
import uuid
from pathlib import Path
from typing import Any

ASCEND_ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--vllm-root", type=Path, required=True)
options, pytest_args = parser.parse_known_args()
ROOT = options.vllm_root.resolve()


def module(name: str, **members: Any) -> types.ModuleType:
    if name in sys.modules:
        result = sys.modules[name]
    else:
        if "." in name:
            parent, child = name.rsplit(".", 1)
            package = module(parent)
        result = types.ModuleType(name)
        result.__path__ = [str(ROOT.joinpath(*name.split(".")))]
        sys.modules[name] = result
        if "." in name:
            setattr(package, child, result)
    result.__dict__.update(members)
    return result


def load(name: str, path: str | None = None) -> types.ModuleType:
    base = ASCEND_ROOT if name.startswith("vllm_ascend") else ROOT
    target = base / (path or (name.replace(".", "/") + ".py"))
    result = module(name)
    spec = importlib.util.spec_from_file_location(name, target)
    assert spec is not None and spec.loader is not None
    result.__file__ = str(target)
    result.__package__ = name.rpartition(".")[0]
    spec.loader.exec_module(result)
    return result


def definitions(name: str, path: str, names: set[str], **scope: Any):
    """Execute original selected definitions, with explicit import boundaries."""
    result = module(name, **scope)
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    selected = [
        n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names
    ]
    assert {n.name for n in selected} == names
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    code = ast.fix_missing_locations(ast.Module([future, *selected], type_ignores=[]))
    exec(compile(code, str(ROOT / path), "exec"), result.__dict__)
    return result


def bootstrap():
    import asyncio
    import functools
    from collections.abc import Mapping
    from typing import ClassVar, get_type_hints

    import msgspec
    import pydantic
    from fastapi import Request
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic_core import core_schema
    from starlette.background import BackgroundTask, BackgroundTasks

    logger: Any = logging.getLogger("decisions.offline")
    logger.warning_once = logger.warning
    logger.debug_once = logger.debug
    module("vllm.logger", init_logger=lambda _: logger, logger=logger)
    envs = module(
        "vllm.envs",
        VLLM_MAX_N_SEQUENCES=16384,
        VLLM_ALLOW_RUNTIME_LORA_UPDATING=False,
        VLLM_SKIP_MODEL_NAME_VALIDATION=False,
    )
    module(
        "vllm.config",
        ModelConfig=object,
        SpeculativeConfig=object,
        StructuredOutputsConfig=object,
    )
    module("vllm.engine.protocol", EngineClient=object)
    module("vllm.tokenizers", TokenizerLike=object)
    module("vllm.lora.request", LoRARequest=type("LoRARequest", (), {}))
    module("vllm.entrypoints.serve.utils.request_logger", RequestLogger=object)
    module(
        "vllm.entrypoints.openai.models.serving",
        OpenAIServingModels=object,
        OpenAIModelRegistry=object,
    )
    module("vllm.entrypoints.pooling.typing", AnyPoolingRequest=object)
    module("vllm.entrypoints.serve.engine.typing", AnyRequest=object)
    module(
        "vllm.entrypoints.chat_utils",
        ChatTemplateContentFormatOption=str,
        load_chat_template=lambda template: template,
    )
    module(
        "vllm.utils",
        random_uuid=lambda: uuid.uuid4().hex,
        length_from_prompt_token_ids_or_embeds=lambda ids, _: len(ids),
    )
    module("vllm.utils.mistral", is_mistral_tokenizer=lambda _: False)
    module(
        "vllm.multimodal.media",
        MEDIA_CONNECTOR_REGISTRY=object(),
        MediaConnector=object,
    )
    module("vllm.utils.import_utils", LazyLoader=lambda *args: object())
    module(
        "vllm.multimodal.media.connector",
        merge_media_io_kwargs=lambda defaults, overrides: (defaults or {}) | (overrides or {}),
    )
    module("vllm_ascend").__path__ = [str(ASCEND_ROOT / "vllm_ascend")]
    load("vllm_ascend.envs")
    load("vllm.exceptions")
    load("vllm.logprobs")
    engine_inputs = load("vllm.inputs.engine")
    inputs = module(
        "vllm.inputs",
        **{n: v for n, v in vars(engine_inputs).items() if not n.startswith("_")},
    )
    for name in (
        "TokensPrompt",
        "EmbedsPrompt",
        "TextPrompt",
        "ExplicitEncoderDecoderPrompt",
        "PromptType",
        "SingletonPrompt",
    ):
        setattr(inputs, name, dict)
    module("vllm", PromptType=dict, envs=envs)
    module(
        "vllm.utils.collection_utils",
        is_list_of=lambda value, cls: isinstance(value, list) and all(isinstance(item, cls) for item in value),
    )
    load("vllm.renderers.inputs.preprocess")
    params = load("vllm.renderers.params")
    module(
        "vllm.renderers",
        ChatParams=params.ChatParams,
        TokenizeParams=params.TokenizeParams,
        merge_kwargs=params.merge_kwargs,
    )
    definitions(
        "vllm.v1.serial_utils",
        "vllm/v1/serial_utils.py",
        {"PydanticMsgspecMixin"},
        ClassVar=ClassVar,
        Any=Any,
        msgspec=msgspec,
        get_type_hints=get_type_hints,
        core_schema=core_schema,
        GetCoreSchemaHandler=pydantic.GetCoreSchemaHandler,
    )
    sampling = load("vllm.sampling_params")
    module("vllm", SamplingParams=sampling.SamplingParams)
    # Outputs only use torch for pooling annotations; no tensor operations run.
    module("torch", Tensor=type("Tensor", (), {}))
    module(
        "vllm.v1.metrics.stats",
        RequestSpecDecodeMetrics=object,
        RequestStateStats=object,
    )
    load("vllm.outputs")
    load("vllm.entrypoints.serve.engine.protocol")
    exceptions = sys.modules["vllm.exceptions"]
    definitions(
        "vllm.entrypoints.generate.base.protocol",
        "vllm/entrypoints/generate/base/protocol.py",
        {"validate_cache_salt"},
        _CACHE_SALT_FORBIDDEN_CHARS=frozenset("@/\\\x00"),
        _MAX_CACHE_SALT_LENGTH=128,
        VLLMValidationError=exceptions.VLLMValidationError,
    )
    error_utils = load("vllm.entrypoints.serve.exception_handling.utils")
    error_response = load("vllm.entrypoints.serve.exception_handling.error_response")
    module(
        "vllm.entrypoints.serve",
        create_error_response=error_response.create_error_response,
    )
    assert error_utils.sanitize_message("example") == "example"
    load("vllm.entrypoints.serve.engine.serving")
    load("vllm.utils.async_utils")
    definitions(
        "vllm.tracing",
        "vllm/tracing/utils.py",
        {"contains_trace_headers", "extract_trace_headers"},
        Mapping=Mapping,
        TRACE_HEADERS=["traceparent", "tracestate"],
    )
    online = definitions(
        "vllm.renderers.online_renderer",
        "vllm/renderers/online_renderer.py",
        {"_reused_prompt_token_ids"},
    )
    tree = ast.parse((ROOT / "vllm/renderers/online_renderer.py").read_text(encoding="utf-8"))
    renderer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OnlineRenderer")
    renderer.body = [n for n in renderer.body if getattr(n, "name", None) == "preprocess_chat"]
    online.__dict__.update(
        merge_kwargs=params.merge_kwargs,
        is_mistral_tokenizer=lambda _: False,
        tokens_input=inputs.tokens_input,
    )
    ast_module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            renderer,
        ],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(ast_module), "online_renderer.py", "exec"),
        online.__dict__,
    )
    definitions(
        "vllm.entrypoints.serve.utils.api_utils",
        "vllm/entrypoints/serve/utils/api_utils.py",
        {
            "listen_for_disconnect",
            "with_cancellation",
            "decrement_server_load",
            "load_aware_call",
            "validate_json_request",
        },
        asyncio=asyncio,
        functools=functools,
        Request=Request,
        RequestValidationError=RequestValidationError,
        JSONResponse=JSONResponse,
        StreamingResponse=StreamingResponse,
        BackgroundTask=BackgroundTask,
        BackgroundTasks=BackgroundTasks,
    )
    module("vllm.tasks", SupportedTask=str)
    print(
        "Offline boundaries: fake tokenizer and engine; no weights, NPU, APC, "
        "DG, server startup or performance validation.",
        flush=True,
    )


if __name__ == "__main__":
    bootstrap()
    import pytest

    suite = ASCEND_ROOT / "tests/ut/entrypoints/structured_decisions"
    raise SystemExit(
        pytest.main(
            [
                "--confcutdir=" + str(suite),
                "--basetemp=" + str(ASCEND_ROOT / ".pytest-decisions-offline"),
                str(suite / "test_question_types.py"),
                str(suite / "test_next_token_strategy.py"),
                str(suite / "test_platform_patch.py"),
                *pytest_args,
            ]
        )
    )
