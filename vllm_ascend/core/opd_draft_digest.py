# SPDX-License-Identifier: Apache-2.0
"""Content identity of a local DSpark draft, independent of HF shard layout."""

import hashlib
import json
from pathlib import Path


def tensor_digest(tensor):
    import torch

    tensor = tensor.detach().cpu().contiguous()
    raw = tensor.reshape(-1).view(torch.uint8).numpy()
    return [str(tensor.dtype), list(tensor.shape), hashlib.sha256(raw).hexdigest()]


def weights_digest(entries):
    payload = json.dumps(entries, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def safetensors_digest(root):
    from safetensors import safe_open

    root = Path(root)
    index = root / "model.safetensors.index.json"
    single = root / "model.safetensors"
    if index.is_file() == single.is_file():
        raise ValueError(f"expected one local HF safetensors draft or shard index in {root}")
    weight_map = json.loads(index.read_text())["weight_map"] if index.is_file() else None
    shards = sorted(set(weight_map.values())) if weight_map is not None else [single.name]
    if set(shards) != {file.name for file in root.glob("*.safetensors")}:
        raise ValueError(f"draft safetensors files disagree with the selected weights in {root}")
    entries = {}
    for shard in shards:
        with safe_open(root / shard, framework="pt", device="cpu") as source:
            for name in source.keys():  # noqa: SIM118 - safe_open exposes keys(), not __iter__
                if weight_map is not None and weight_map.get(name) != shard:
                    raise ValueError(f"draft shard/index mismatch for {name}")
                entries[name] = tensor_digest(source.get_tensor(name))
    if not entries or (weight_map is not None and entries.keys() != weight_map.keys()):
        raise ValueError("incomplete draft safetensors index")
    return weights_digest(entries)
