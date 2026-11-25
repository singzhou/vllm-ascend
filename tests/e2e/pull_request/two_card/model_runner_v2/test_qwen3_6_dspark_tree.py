# SPDX-License-Identifier: Apache-2.0

from tests.e2e.pull_request.two_card.model_runner_v2.test_qwen3_6_dspark import (
    QWEN36_DENSE_MODEL,
    QWEN36_DSPARK_DRAFT_MODEL,
)


def test_qwen_dspark_tree_matches_greedy_chain(vllm_runner, monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    prompts = [
        "Write an ordered list of eight different vegetables.",
        "Explain why the sky appears blue in two short paragraphs.",
        "Write a short Python function that sorts a list of strings by length.",
    ]
    runner_kwargs = dict(
        tensor_parallel_size=2,
        max_model_len=2048,
        max_num_batched_tokens=256,
        dtype="bfloat16",
        enforce_eager=True,
        async_scheduling=False,
        enable_prefix_caching=False,
        mamba_cache_mode="none",
        generation_config="vllm",
        speculative_config={
            "method": "dspark",
            "model": QWEN36_DSPARK_DRAFT_MODEL,
            "num_speculative_tokens": 7,
            "draft_sample_method": "greedy",
        },
    )
    with vllm_runner(QWEN36_DENSE_MODEL, **runner_kwargs) as runner:
        baseline = runner.generate_greedy(prompts, max_tokens=64)
    with vllm_runner(
        QWEN36_DENSE_MODEL,
        **runner_kwargs,
        additional_config={"tree_spec_config": {"enabled": True, "topk": 3}},
    ) as runner:
        tree = runner.generate_greedy(prompts, max_tokens=64)
    assert [ids for ids, _ in tree] == [ids for ids, _ in baseline]
