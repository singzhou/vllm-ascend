# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check a running Qwen Decisions server; optionally compare with v0.32.0."""

import argparse
import copy
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def request_json(base_url, path, body, timeout, api_key=None):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(
        base_url.rstrip("/") + path,
        data=None if body is None else json.dumps(body).encode(),
        headers=headers,
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.load(error)


def decision_body(model):
    return {
        "model": model,
        "state": {"ticket": "My card was charged twice. Please refund the extra charge."},
        "questions": {
            "team": {
                "type": "choice",
                "instructions": "Which team handles this ticket?",
                "criteria": {"billing": "Payments", "shipping": "Delivery"},
            },
            "money": {"type": "noul", "instructions": "Does this concern money?"},
            "urgency": {
                "type": "score",
                "instructions": "How urgent is the ticket?",
                "criteria": ["low", "medium", "high"],
            },
        },
        "chat_template_kwargs": {"enable_thinking": False},
    }


def check_answer(body, question_ids):
    assert body["object"] == "structured_decision", body
    assert list(body["answers"]) == list(question_ids), body
    assert body["usage"]["input_tokens"] > 0, body
    assert body["usage"]["output_tokens"] == len(question_ids), body
    for qid, answer in body["answers"].items():
        probabilities = answer["probabilities"]
        assert all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities.values())
        assert math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-5)
        diagnostic = body["diagnostics"][qid]
        assert 0 <= diagnostic["label_mass"] <= 1 + 1e-5, diagnostic
        assert isinstance(diagnostic["argmax_is_label"], bool), diagnostic
        assert math.isclose(
            answer["confidence"],
            max(probabilities.values()) * diagnostic["label_mass"],
            abs_tol=1e-5,
        ), answer
        if answer["type"] == "choice":
            assert answer["choice"] == max(probabilities, key=probabilities.get)
        elif answer["type"] == "noul":
            assert set(probabilities) == {"yes", "no"}
            assert answer["noul"] == probabilities["yes"]
        elif answer["type"] == "score":
            assert set(answer["legend"]) == set(probabilities)
            assert math.isclose(
                answer["score"],
                sum(int(i) * p for i, p in probabilities.items()),
                abs_tol=1e-5,
            )
        else:
            raise AssertionError(answer)


def compare(actual, expected, atol, path=""):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), (path, actual, expected)
        for key, value in expected.items():
            compare(actual[key], value, atol, f"{path}.{key}")
    elif isinstance(expected, float):
        assert math.isclose(actual, expected, abs_tol=atol, rel_tol=0), (
            path,
            actual,
            expected,
        )
    else:
        assert actual == expected, (path, actual, expected)


def run(args, report):
    """Exercise the live Decisions endpoint and optionally compare its answers with a reference server."""

    def call(path, body=None, base_url=None):
        return request_json(base_url or args.base_url, path, body, args.timeout, args.api_key)

    status, models = call("/v1/models")
    assert status == 200, (status, models)
    model = args.model or models["data"][0]["id"]
    report.update(model=model, models=models)
    report["version"] = call("/version")
    payload = decision_body(model)
    status, result = call("/v1/systemone", payload)
    report["decision"] = {"status": status, "body": result}
    assert status == 200, (status, result)
    check_answer(result, payload["questions"])

    one = copy.deepcopy(payload)
    one["questions"] = {"only": {"type": "choice", "criteria": {"billing": None}}}
    status, result_one = call("/v1/systemone", one)
    report["one_option"] = {"status": status, "body": result_one}
    assert status == 200, result_one
    check_answer(result_one, one["questions"])
    assert result_one["answers"]["only"]["probabilities"] == {"billing": 1.0}

    limits = payload | {
        "questions": {
            "all_letters": {
                "type": "choice",
                "instructions": "Pick one option.",
                "criteria": dict.fromkeys(map(str, range(26))),
            },
            "all_digits": {
                "type": "score",
                "instructions": "Rate urgency.",
                "criteria": [f"level {i}" for i in range(10)],
            },
        }
    }
    status, result_limits = call("/v1/systemone", limits)
    report["supported_limits"] = {"status": status, "body": result_limits}
    assert status == 200, result_limits
    check_answer(result_limits, limits["questions"])
    assert len(result_limits["answers"]["all_letters"]["probabilities"]) == 26
    assert len(result_limits["answers"]["all_digits"]["probabilities"]) == 10

    extended = payload | {
        "questions": {
            "choice128": {
                "type": "choice",
                "instructions": "Pick one option.",
                "criteria": dict.fromkeys(map(str, range(128))),
            }
        }
    }
    status, result_extended = call("/v1/systemone", extended)
    report["choice128"] = {"status": status, "body": result_extended}
    assert status == 200, result_extended
    check_answer(result_extended, extended["questions"])
    assert len(result_extended["answers"]["choice128"]["probabilities"]) == 128

    failures = {
        "empty_questions": {"questions": {}},
        "too_many_questions": {"questions": {f"q{i}": {"type": "noul"} for i in range(65)}},
        "too_many_options": {"questions": {"q": {"type": "choice", "criteria": dict.fromkeys(map(str, range(129)))}}},
        "thinking": {"chat_template_kwargs": {"enable_thinking": True}},
        "unknown_question_field": {"questions": {"q": {"type": "noul", "depends_on": ["team"]}}},
        "text_only": {"images": ["https://example.invalid/image.png"]},
        "invalid_score": {"questions": {"q": {"type": "score", "criteria": ["low"]}}},
        "unknown_model": {"model": "__nonexistent_decisions_model__"},
    }
    report["validation"] = {}
    for name, changes in failures.items():
        status, body = call("/v1/systemone", payload | changes)
        report["validation"][name] = {"status": status, "body": body}
        assert status == (404 if name == "unknown_model" else 400), (name, status, body)

    generation = {
        "model": model,
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 8,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [
            pool.submit(call, path, body)
            for path, body in [
                ("/v1/systemone", payload),
                ("/v1/chat/completions", generation),
            ]
            * 2
        ]
        report["mixed_requests"] = [job.result() for job in jobs]
    for index, (status, body) in enumerate(report["mixed_requests"]):
        assert status == 200, (index, status, body)
        if index % 2 == 0:
            check_answer(body, payload["questions"])
        else:
            assert body["choices"], body

    if args.reference_url:
        reference_payload = payload | {"model": args.reference_model or model}
        status, reference = call("/v1/systemone", reference_payload, base_url=args.reference_url)
        report["reference"] = {"status": status, "body": reference}
        assert status == 200, reference
        for field in ("answers", "diagnostics", "usage"):
            compare(result[field], reference[field], args.atol, field)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--model")
    parser.add_argument("--api-key")
    parser.add_argument("--reference-url")
    parser.add_argument("--reference-model")
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output", type=Path, default=Path("decisions-smoke.json"))
    args = parser.parse_args()
    report = {
        "started_unix": time.time(),
        "base_url": args.base_url,
        "reference_url": args.reference_url,
        "atol": args.atol,
        "status": "failed",
    }
    try:
        run(args, report)
        report["status"] = "passed"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{report['status']}: {args.output.resolve()}")


if __name__ == "__main__":
    main()
