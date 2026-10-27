# SPDX-License-Identifier: Apache-2.0
import runpy
from pathlib import Path


def test_build_alias_preserves_archived_sources(tmp_path):
    script = Path(__file__).resolve().parents[3] / "csrc/scripts/prepare_legacy_gdn.py"
    prepare = runpy.run_path(str(script))["prepare"]
    source = tmp_path / "attention/recurrent_gated_delta_rule/op_kernel"
    source.mkdir(parents=True)
    original = source / "recurrent_gated_delta_rule.cpp"
    content = b"RecurrentGatedDeltaRule\r\nrecurrent_gated_delta_rule\r\nRECURRENT_GATED_DELTA_RULE\r\nx += y;\r\n"
    original.write_bytes(content)
    prepare(tmp_path)
    target = tmp_path / "attention/legacy_recurrent_gated_delta_rule/op_kernel/legacy_recurrent_gated_delta_rule.cpp"
    expected = (
        b"LegacyRecurrentGatedDeltaRule\r\nlegacy_recurrent_gated_delta_rule\r\n"
        b"LEGACY_RECURRENT_GATED_DELTA_RULE\r\nx += y;\r\n"
    )
    assert original.read_bytes() == content
    assert target.read_bytes() == expected
    modified = target.stat().st_mtime_ns
    prepare(tmp_path)
    assert target.stat().st_mtime_ns == modified
