# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import runpy
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from starlette.datastructures import State
from vllm.entrypoints import serve
from vllm.entrypoints.generate import api_router

from vllm_ascend.entrypoints.structured_decisions import api_router as decisions


@pytest.mark.asyncio
async def test_platform_hooks_register_once_and_initialize_after_generate(monkeypatch):
    app = FastAPI()
    state = State()
    register = Mock(spec=[])
    events = []
    initialize = AsyncMock(side_effect=lambda *args: events.append("generate"))
    init_decisions = Mock(side_effect=lambda *args: events.append("decisions"))
    monkeypatch.setattr(serve, "register_vllm_serve_api_routers", register, raising=False)
    monkeypatch.setattr(api_router, "init_generate_state", initialize)
    monkeypatch.setattr(decisions, "init_structured_decisions_state", init_decisions)

    patch = Path(__file__).resolve().parents[4] / "vllm_ascend/patch/platform/patch_structured_decisions.py"
    runpy.run_path(str(patch))
    installed = serve.register_vllm_serve_api_routers
    runpy.run_path(str(patch))
    assert serve.register_vllm_serve_api_routers is installed
    serve.register_vllm_serve_api_routers(app)
    register.assert_called_once_with(app)
    assert "post" in app.openapi()["paths"]["/v1/systemone"]

    args = (Mock(), state, Mock(), None, ("generate",), {})
    await api_router.init_generate_state(*args)
    initialize.assert_awaited_once_with(*args)
    init_decisions.assert_called_once_with(*args)
    assert events == ["generate", "decisions"]
