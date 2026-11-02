# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare the port's declarations with the final, pinned Decisions backport."""

import argparse
import ast
import subprocess
from pathlib import Path

SOURCE = "0e01034281f9e6fe94230a59e8535c48e20eb291"
PACKAGE = Path(__file__).resolve().parents[2] / "vllm_ascend/entrypoints/structured_decisions"


def declarations(source):
    # The plugin relocates imports and uses the Ascend environment registry and
    # the shared vLLM logger. Compare every other executable declaration.
    tree = ast.parse(source.replace("ascend_envs.VLLM_ASCEND_DECISIONS_", "envs.VLLM_DECISIONS_"))
    return [
        ast.dump(node)
        for node in tree.body
        if not isinstance(node, (ast.Import, ast.ImportFrom)) and ast.unparse(node) != "logger = init_logger(__name__)"
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-root", type=Path, required=True)
    args = parser.parse_args()
    directory = "vllm/entrypoints/generate"
    files = subprocess.check_output(
        [
            "git",
            "-C",
            args.vllm_root,
            "ls-tree",
            "-r",
            "--name-only",
            SOURCE,
            f"{directory}/structured_decisions",
            f"{directory}/label_reads.py",
        ],
        text=True,
    ).splitlines()
    assert {Path(path).name for path in files} == {path.name for path in PACKAGE.glob("*.py")}
    for path in files:
        source = subprocess.check_output(["git", "-C", args.vllm_root, "show", f"{SOURCE}:{path}"], text=True)
        assert declarations((PACKAGE / Path(path).name).read_text()) == declarations(source), path
    print(f"Verified {len(files)} Decisions modules against {SOURCE}")


if __name__ == "__main__":
    main()
