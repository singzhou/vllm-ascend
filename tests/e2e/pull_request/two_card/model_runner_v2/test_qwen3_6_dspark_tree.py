# SPDX-License-Identifier: Apache-2.0

from vllm import SamplingParams

from tests.e2e.pull_request.two_card.model_runner_v2.test_qwen3_6_dspark import (
    QWEN36_DENSE_MODEL,
    QWEN36_DSPARK_DRAFT_MODEL,
)


def _install_tree_verification_probe(worker):
    runner = worker.model_runner
    sample = runner.sample
    worker.verified_tree_rows = 0

    def observe_tree_verification(hidden_states, input_batch, grammar_output):
        result = sample(hidden_states, input_batch, grammar_output)
        tree = getattr(input_batch, "tree_batch", None)
        if tree is not None:
            worker.verified_tree_rows += len(tree.verified)
        return result

    runner.sample = observe_tree_verification


def _read_verified_tree_rows(worker):
    return worker.verified_tree_rows


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
    sampling_params = SamplingParams(temperature=0, max_tokens=64, ignore_eos=True)
    with vllm_runner(QWEN36_DENSE_MODEL, **runner_kwargs) as runner:
        baseline = runner.model.generate(prompts, sampling_params, use_tqdm=False)
    with vllm_runner(
        QWEN36_DENSE_MODEL,
        **runner_kwargs,
        additional_config={"tree_spec_config": {"enabled": True, "topk": 3}},
    ) as runner:
        runner.model.collective_rpc(_install_tree_verification_probe)
        tree = runner.model.generate(prompts, sampling_params, use_tqdm=False)
        verified_rows = runner.model.collective_rpc(_read_verified_tree_rows)
    assert len(verified_rows) == runner_kwargs["tensor_parallel_size"]
    assert all(count > 0 for count in verified_rows), "DSpark did not verify a tree on every TP worker"
    assert [output.outputs[0].token_ids for output in tree] == [output.outputs[0].token_ids for output in baseline]
