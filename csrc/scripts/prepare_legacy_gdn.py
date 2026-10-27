# SPDX-License-Identifier: Apache-2.0
"""Give the unchanged historical GDN sources an isolated build-time namespace.

CANN resolves kernels by operator type process-wide. Registering the historical
RecurrentGatedDeltaRule would shadow FLA's operator of the same name, even when
its ACLNN entry point comes from FLA's own library. Only build inputs are renamed;
the archived sources, arithmetic, tiling and data layouts remain unchanged.
"""

import sys
from pathlib import Path

ALIASES = (
    (b"recurrent_gated_delta_rule", b"legacy_recurrent_gated_delta_rule"),
    (b"RecurrentGatedDeltaRule", b"LegacyRecurrentGatedDeltaRule"),
    (b"RECURRENT_GATED_DELTA_RULE", b"LEGACY_RECURRENT_GATED_DELTA_RULE"),
)


def prepare(csrc: Path) -> None:
    source = csrc / "attention/recurrent_gated_delta_rule"
    destination = csrc / "attention/legacy_recurrent_gated_delta_rule"
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        relative = path.relative_to(source).as_posix().encode()
        content = path.read_bytes()
        for original, alias in ALIASES:
            relative = relative.replace(original, alias)
            content = content.replace(original, alias)
        target = destination / relative.decode()
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.read_bytes() != content:
            target.write_bytes(content)


if __name__ == "__main__":
    prepare(Path(sys.argv[1]))
