# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attach the Decisions backport to the unmodified vLLM 0.30 server.

Only the HTTP router and generate-state hooks are extended. Remove this patch
and the local serving package when the minimum vLLM version supplies Decisions
with the same choice128 and optional prefix-warmup contracts.
"""

from functools import wraps

from vllm.entrypoints import serve
from vllm.entrypoints.generate import api_router


def _patch_structured_decisions() -> None:
    original_register = serve.register_vllm_serve_api_routers
    if getattr(original_register, "_ascend_structured_decisions", False):
        return
    original_init = api_router.init_generate_state

    @wraps(original_register)
    def register_with_decisions(app):
        # Lazy imports keep serving dependencies out of engine/worker startup.
        from vllm_ascend.entrypoints.structured_decisions.api_router import register_structured_decisions_api_router

        original_register(app)
        register_structured_decisions_api_router(app)

    @wraps(original_init)
    async def init_with_decisions(
        engine_client, state, args, request_logger, supported_tasks, default_chat_template_kwargs
    ):
        from vllm_ascend.entrypoints.structured_decisions.api_router import init_structured_decisions_state

        await original_init(engine_client, state, args, request_logger, supported_tasks, default_chat_template_kwargs)
        init_structured_decisions_state(
            engine_client, state, args, request_logger, supported_tasks, default_chat_template_kwargs
        )

    register_with_decisions._ascend_structured_decisions = True
    serve.register_vllm_serve_api_routers = register_with_decisions
    api_router.init_generate_state = init_with_decisions


_patch_structured_decisions()
