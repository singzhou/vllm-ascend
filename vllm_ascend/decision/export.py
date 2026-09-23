# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Export a dense Kev adapter as native vLLM backbone and pointer weights."""

import argparse
import hashlib
import json
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

from .config import KevConfig

ARCHITECTURES = {
    "qwen2": "AscendKevQwen2ForDecision",
    "qwen3": "AscendKevQwen3ForDecision",
    "qwen3_5_text": "AscendKevQwen35ForDecision",
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_checkpoint(args):
    import torch
    from huggingface_hub import save_torch_state_dict, snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from vllm_ascend.decision.encoding import SPECIAL

    output = Path(args.out).expanduser().resolve()
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    run = Path(args.run).expanduser()
    if not run.is_dir():
        repo, _, revision = args.run.partition("@")
        run = Path(
            snapshot_download(
                repo,
                revision=revision or None,
                allow_patterns=["*.json", "*.safetensors", "head.pt"],
            )
        )
    meta = torch.load(run / "head.pt", map_location="cpu", weights_only=True)
    adapter = json.loads((run / "adapter_config.json").read_text())
    if meta.get("option_isolation"):
        raise ValueError("Option-isolated checkpoints require a different execution model")
    if meta.get("weights_dtype") == "bf16":
        raise ValueError("BF16-trained unmerged checkpoints need a separate export path")
    if adapter.get("trainable_token_indices") or adapter.get("modules_to_save"):
        raise ValueError("Trainable embeddings/modules_to_save are not yet supported")
    base = args.base or meta["base"]
    revision = args.base_revision
    if revision is None and base == meta["base"]:
        revision = meta.get("base_revision")
    if Path(base).is_dir():
        revision = None
    max_context = args.max_context if args.max_context is not None else KevConfig().max_context
    settings = KevConfig.from_dict(
        {
            "head_dim": meta.get("head_dim", 256),
            "max_context": max_context,
            "temperature": args.temperature,
            "strict_length": args.strict_length,
            "date_facts": args.date_facts,
            "dp_affinity": not args.no_dp_affinity,
        }
    )
    tokenizer = AutoTokenizer.from_pretrained(base, revision=revision)
    delimiter_ids = [tokenizer.convert_tokens_to_ids(t) for t in SPECIAL]
    if any(t is None or t == tokenizer.unk_token_id for t in delimiter_ids):
        raise ValueError("Base tokenizer is missing Kev delimiters")
    if len(set(delimiter_ids)) != len(SPECIAL):
        raise ValueError("Kev delimiter token IDs must be distinct")

    source = AutoModelForCausalLM.from_pretrained(
        base, revision=revision, dtype=torch.float32, attn_implementation="eager"
    )
    backbone = source.model
    config = backbone.config.to_dict()
    model_type = config["model_type"]
    if model_type not in ARCHITECTURES:
        raise ValueError(f"Unsupported Kev backbone: {model_type}")
    backbone = PeftModel.from_pretrained(backbone, str(run))
    backbone = backbone.merge_and_unload(safe_merge=True)
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    backbone = backbone.to(dtype=dtype)
    weights = {f"model.{name}": tensor.detach().cpu().contiguous() for name, tensor in backbone.state_dict().items()}
    head = meta["head"]
    expected = {"q.weight", "q.bias", "k.weight", "k.bias"}
    if set(head) != expected:
        raise ValueError(f"Unexpected Kev head keys: {set(head)}")
    for name, value in head.items():
        shape = (settings.head_dim, config["hidden_size"]) if name.endswith("weight") else (settings.head_dim,)
        if tuple(value.shape) != shape or not torch.isfinite(value).all():
            raise ValueError(f"Invalid Kev head weight {name}")
        weights[f"pooler.{name}"] = value.detach().float().cpu().contiguous()
    config.update(
        architectures=[ARCHITECTURES[model_type]],
        kev_config=asdict(settings),
        kev_delimiter_ids=delimiter_ids,
        is_causal=True,
        dtype="bfloat16" if args.dtype == "bf16" else "float32",
        torch_dtype="bfloat16" if args.dtype == "bf16" else "float32",
    )
    config.pop("auto_map", None)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        save_torch_state_dict(
            weights,
            staging,
            max_shard_size=args.max_shard_size,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(staging)
        (staging / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        manifest = {
            "schema_version": 2,
            "base": base,
            "base_revision": revision,
            "resolved_base_revision": getattr(source.config, "_commit_hash", None),
            "source_run": str(run),
            "merge_dtype": "float32",
            "lora_scale": 1.0,
            "backbone_dtype": args.dtype,
            "head_dtype": "float32",
            "architecture": ARCHITECTURES[model_type],
            "kev_config": asdict(settings),
            "max_context_provenance": {
                "value": settings.max_context,
                "source": "cli" if args.max_context is not None else "exporter_default",
                "verified_against_training_run": False,
            },
            "source_hashes": {
                p.name: sha256(p)
                for p in sorted(run.iterdir())
                if p.is_file() and (p.suffix == ".safetensors" or p.name in ("head.pt", "adapter_config.json"))
            },
            "files": {p.name: sha256(p) for p in sorted(staging.iterdir()) if p.is_file()},
        }
        (staging / "kev_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Kev run directory or Hub ID[@revision]")
    parser.add_argument("--out", required=True, help="New export directory")
    parser.add_argument("--base")
    parser.add_argument("--base-revision")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--max-context",
        type=int,
        help=(
            "Serving token limit for state plus each question branch "
            f"(default: {KevConfig().max_context}); not inferred from training. "
            "State may be truncated; overlong branches are rejected."
        ),
    )
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--strict-length", action="store_true")
    parser.add_argument("--date-facts", action="store_true")
    parser.add_argument("--no-dp-affinity", action="store_true")
    export_checkpoint(parser.parse_args())


if __name__ == "__main__":
    main()
