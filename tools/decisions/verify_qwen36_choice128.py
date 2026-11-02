# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify the fixed Qwen3.6-27B choice128 codebook without loading model weights.

Example:
  python verify_qwen36_choice128.py --tokenizer /models/Qwen3.6-27B \
    --output qwen36_choice128_verified.json
Requires transformers, tokenizers, jinja2. Reference run: 5.5.4 / 0.22.2 / 3.1.6.
"""

import argparse
import hashlib
import importlib.metadata
import json
import string
from collections import Counter
from pathlib import Path

REFERENCE_MODEL = "Qwen/Qwen3.6-27B"
REFERENCE_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
PAIR_LABELS = [
    "AA",
    "AB",
    "AC",
    "AD",
    "AE",
    "AF",
    "AG",
    "AH",
    "AI",
    "AJ",
    "AK",
    "AL",
    "AM",
    "AN",
    "AO",
    "AP",
    "AQ",
    "AR",
    "AS",
    "AT",
    "AU",
    "AV",
    "AW",
    "AX",
    "AY",
    "AZ",
    "BA",
    "BB",
    "BC",
    "BD",
    "BE",
    "BF",
    "BG",
    "BH",
    "BI",
    "BJ",
    "BK",
    "BL",
    "BM",
    "BN",
    "BO",
    "BP",
    "BR",
    "BS",
    "BT",
    "BU",
    "BV",
    "BW",
    "BX",
    "BY",
    "CA",
    "CB",
    "CC",
    "CD",
    "CE",
    "CF",
    "CG",
    "CH",
    "CI",
    "CK",
    "CL",
    "CM",
    "CN",
    "CO",
    "CP",
    "CR",
    "CS",
    "CT",
    "CU",
    "CV",
    "CW",
    "CX",
    "CY",
    "DA",
    "DB",
    "DC",
    "DD",
    "DE",
    "DF",
    "DG",
    "DH",
    "DI",
    "DJ",
    "DK",
    "DL",
    "DM",
    "DN",
    "DO",
    "DP",
    "DR",
    "DS",
    "DT",
    "DU",
    "DV",
    "DW",
    "DX",
    "DY",
    "EA",
    "EB",
    "EC",
    "ED",
    "EE",
]
CHOICE_LABELS_128 = tuple(string.ascii_uppercase) + tuple(PAIR_LABELS)
EXPECTED_IDS = [
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    5840,
    1803,
    1646,
    1745,
    13276,
    8018,
    1825,
    28946,
    15015,
    29595,
    11568,
    939,
    1354,
    1058,
    18183,
    2456,
    88898,
    905,
    1846,
    802,
    33869,
    7839,
    14006,
    2860,
    2926,
    22828,
    6844,
    9798,
    4738,
    9265,
    11261,
    19278,
    36513,
    93801,
    8335,
    14544,
    85266,
    9110,
    28000,
    15137,
    4525,
    25261,
    12717,
    7116,
    17078,
    14497,
    57339,
    74909,
    52072,
    19305,
    4887,
    12607,
    3580,
    6281,
    2036,
    9362,
    8533,
    2080,
    10911,
    2925,
    3040,
    9690,
    27731,
    8023,
    6901,
    8702,
    6211,
    1123,
    16307,
    18990,
    64045,
    63037,
    33380,
    6151,
    3392,
    5449,
    3967,
    1113,
    5095,
    51923,
    49600,
    17099,
    51483,
    17756,
    16037,
    8135,
    30237,
    5683,
    9992,
    7444,
    5751,
    10284,
    20887,
    59884,
    52396,
    16103,
    67547,
    18535,
    8006,
    7263,
    1425,
    6878,
]
REFERENCE_HASHES = {
    "tokenizer.json": ("5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"),
    "tokenizer_config.json": ("5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b"),
    "chat_template.jinja": ("e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259"),
    "config.json": ("69db4eb7196bc8190813231b3018ca05d8c2e3abc7b1af19d55c157af44a9d9c"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def run(folder, chat_template_path=None):
    """Verify the label codebook at the rendered reply boundary and record tokenizer fingerprints."""
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(folder), local_files_only=True, trust_remote_code=False)
    if chat_template_path:
        tokenizer.chat_template = chat_template_path.read_text(encoding="utf-8")
    raw = Tokenizer.from_file(str(folder / "tokenizer.json"))
    labels = CHOICE_LABELS_128
    require(len(labels) == len(EXPECTED_IDS) == 128, "Codebook size is not 128")
    require(len(set(labels)) == len(set(EXPECTED_IDS)) == 128, "Duplicate label or ID")
    added_ids = set(tokenizer.get_added_vocab().values())
    special_ids = set(tokenizer.all_special_ids)
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    text_config = config.get("text_config", config)
    vocab_size = text_config["vocab_size"]
    for label, expected in zip(labels, EXPECTED_IDS):
        require(
            tokenizer.encode(label, add_special_tokens=False) == [expected],
            f"Isolated encoding/ID differs: {label!r}",
        )
        require(
            raw.encode(label, add_special_tokens=False).ids == [expected],
            f"Raw tokenizer differs: {label!r}",
        )
        require(
            expected not in (added_ids | special_ids) and 0 <= expected < vocab_size,
            f"Added/special/out-of-range label: {label!r}",
        )
        require(
            tokenizer.decode([expected], skip_special_tokens=False) == label,
            f"Decode does not round-trip: {label!r}",
        )

    states = [
        "x",
        "设备无法接入网络，请选择处理团队。",
        "The customer was charged twice. Please identify the responsible team.",
        "历史记录：链路正常，用户认证失败。\n" * 64,
    ]
    contexts = []
    for state_index, state in enumerate(states):
        for with_system in (False, True):
            for k in (26, 27, 128):
                instruction = (
                    "Answer with the letter of one option only."
                    if k <= 26
                    else "Answer with the exact label of one option only."
                )
                lines = ["Question: Which option matches the state?"]
                lines.extend(f"{label}: option_{i}" for i, label in enumerate(labels[:k]))
                content = state + "\n\n" + "\n".join(lines + [instruction])
                messages = [{"role": "system", "content": "Read the evidence and choose."}] if with_system else []
                messages.append({"role": "user", "content": content})
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                base = tokenizer.encode(prompt, add_special_tokens=False)
                rendered_ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                require(list(rendered_ids) == base, "Rendered token IDs differ from encode")
                start = max(
                    (i + 1 for i, token in enumerate(base) if token in added_ids),
                    default=0,
                )
                tail = base[start:]
                tail_text = tokenizer.decode(tail, skip_special_tokens=False)
                require(
                    tail == [271] and tail_text == "\n\n",
                    "Reply boundary differs from the validated official template",
                )
                for label, expected in zip(labels, EXPECTED_IDS):
                    require(
                        tokenizer.encode(tail_text + label, add_special_tokens=False) == tail + [expected],
                        f"Reply-tail append failed: {label!r}",
                    )
                    require(
                        tokenizer.encode(prompt + label, add_special_tokens=False) == base + [expected],
                        f"Full-prompt append failed: {label!r}",
                    )
                    require(
                        raw.encode(prompt + label, add_special_tokens=False).ids == base + [expected],
                        f"Raw full-prompt append failed: {label!r}",
                    )
                contexts.append(
                    {
                        "state_index": state_index,
                        "system": with_system,
                        "options": k,
                        "prompt_tokens": len(base),
                        "reply_tail_ids": tail,
                    }
                )
    hashes = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in REFERENCE_HASHES}
    original_numeric_labels = list(string.ascii_uppercase) + [str(i) for i in range(102)]
    numeric_failures = [s for s in original_numeric_labels if len(tokenizer.encode(s, add_special_tokens=False)) != 1]
    return {
        "status": "passed",
        "model": REFERENCE_MODEL,
        "reference_revision": REFERENCE_REVISION,
        "versions": {name: importlib.metadata.version(name) for name in ("transformers", "tokenizers", "jinja2")},
        "tokenizer_class": type(tokenizer).__name__,
        "architectures": config.get("architectures"),
        "text_layer_counts": dict(Counter(text_config.get("layer_types", []))),
        "model_vocab_size": vocab_size,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "local_files_sha256": hashes,
        "matches_reference_files": hashes == REFERENCE_HASHES,
        "effective_chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode("utf-8")).hexdigest(),
        "label_count": len(labels),
        "labels": list(labels),
        "token_ids": EXPECTED_IDS,
        "label_to_token_id": dict(zip(labels, EXPECTED_IDS)),
        "checks": {
            "isolated_single_token": True,
            "unique_ids": True,
            "not_special_or_added": True,
            "reply_tail_append": True,
            "full_prompt_append": True,
            "raw_tokenizer_agrees": True,
            "contexts": len(contexts),
            "context_label_pairs": len(contexts) * len(labels),
        },
        "tested_contexts": contexts,
        "negative_examples": {
            s: tokenizer.encode(s, add_special_tokens=False) for s in ("10", "101", "BQ", "BZ", "CJ")
        },
        "previous_numeric_candidate_failed_count": len(numeric_failures),
        "scope": ("Tokenizer and reply boundary verified; no model inference, accuracy or NPU performance claim."),
        "sources": {
            name: f"https://huggingface.co/{REFERENCE_MODEL}/resolve/{REFERENCE_REVISION}/{name}"
            for name in REFERENCE_HASHES
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True, type=Path, help="Local model/tokenizer directory")
    parser.add_argument("--chat-template", type=Path, help="Optional deployed template to validate")
    parser.add_argument("--output", type=Path, help="Write the complete verification JSON")
    args = parser.parse_args()
    report = run(args.tokenizer, args.chat_template)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "reference_revision",
                    "versions",
                    "tokenizer_class",
                    "text_layer_counts",
                    "label_count",
                    "checks",
                    "negative_examples",
                    "previous_numeric_candidate_failed_count",
                    "matches_reference_files",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
