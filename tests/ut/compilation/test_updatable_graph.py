# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import vllm_ascend.compilation.updatable_graph as graph_module
from vllm_ascend.compilation.updatable_graph import ContextSource, GraphUpdateTask, SharedSource, UpdatableGraph


@pytest.mark.parametrize("shared", [False, True])
def test_resolved_tasks_remain_independent_across_replays(shared):
    """Runtime parameters must not mutate capture kwargs or older resolutions."""
    provider = MagicMock()
    provider.layer_name = "layer"
    operation, handle, event = MagicMock(), object(), MagicMock()
    captured = {"query": object(), "block_table": object(), "actual_seq_lengths": [1]}
    task = GraphUpdateTask(operation, captured, provider, 0, handle, event)
    graph = SimpleNamespace(provider_sizes={provider: 1}, tasks=[task])
    tables = [object(), object()]
    resolutions = []
    for length, table in enumerate(tables, 2):
        params = {"block_table": table, "actual_seq_lengths": [length]}
        if shared:
            source = SharedSource([{**params, "layer_name": "layer"}])
        else:
            provider.resolve.return_value = params
            source = ContextSource(object())
        resolutions.append(UpdatableGraph.resolve_tasks(graph, source)[0])
    assert task.kwargs is captured
    assert captured["actual_seq_lengths"] == [1]
    for length, (resolved, table) in enumerate(zip(resolutions, tables), 2):
        assert resolved.kwargs["block_table"] is table
        assert resolved.kwargs["actual_seq_lengths"] == [length]
        assert resolved.kwargs["query"] is captured["query"]
        assert resolved.operation is operation
        assert resolved.provider is provider
        assert resolved.provider_index == 0
        assert resolved.handle is handle
        assert resolved.event is event
        assert "layer_name" not in resolved.kwargs


@pytest.mark.parametrize("raises", [False, True])
def test_update_restores_known_replay_stream(raises):
    graph = SimpleNamespace(_update_tasks=MagicMock())
    if raises:
        graph._update_tasks.side_effect = RuntimeError("task failed")
    replay_stream, update_stream = MagicMock(), MagicMock()
    replay_stream.device = update_stream.device = "npu:0"
    tasks = ()
    with patch.object(graph_module.torch.npu, "set_stream") as set_stream:
        if raises:
            with pytest.raises(RuntimeError, match="task failed"):
                UpdatableGraph.update(graph, update_stream, tasks, replay_stream=replay_stream)
        else:
            UpdatableGraph.update(graph, update_stream, tasks, replay_stream=replay_stream)
    assert [call.args for call in set_stream.call_args_list] == [(update_stream,), (replay_stream,)]
    graph._update_tasks.assert_called_once_with(update_stream, tasks)


@pytest.mark.parametrize("replay_device", [None, "npu:1"])
def test_update_keeps_stream_context_for_other_callers(replay_device):
    graph = SimpleNamespace(_update_tasks=MagicMock())
    update_stream = SimpleNamespace(device="npu:0")
    replay_stream = SimpleNamespace(device=replay_device) if replay_device else None
    with patch.object(graph_module.torch.npu, "stream", return_value=nullcontext()) as stream:
        UpdatableGraph.update(graph, update_stream, (), replay_stream=replay_stream)
    stream.assert_called_once_with(update_stream)
    graph._update_tasks.assert_called_once_with(update_stream, ())
