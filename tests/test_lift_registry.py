"""Lifted-code drift gate (CDR-3 / impl-plan Section 4.1, T0.1).

Runs the registry's ``check`` (re-extract every lifted symbol from the vendored
MuJoCo tree and diff against the stored snapshot + hash) and ``verify-headers``
(every lifted file is attributed and registered, and vice versa) as part of the
normal suite, so lifted-code drift also fires on every MuJoCo bump.

Authored to drive tools/lift_registry.py through its module surface; the
vendored-tree-dependent ``check`` skips cleanly when the tree is absent, while
``verify-headers`` (repo-only) always runs.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "lift_registry", ROOT / "tools" / "lift_registry.py"
)
lr = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lr)


class _Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_registry_nonempty_and_wellformed():
    reg = lr.load_registry()
    assert reg["entries"], "registry has no entries"
    for e in reg["entries"]:
        for key in ("id", "upstream_rel_path", "symbol", "content_hash",
                    "snapshot", "local_path"):
            assert e.get(key), f"entry {e.get('id')} missing {key}"
        # the mjuu unit + the mj_makeModel allocator must both be present.
    ids = {e["id"] for e in reg["entries"]}
    assert {"user_util_cc", "make_model"} <= ids


def test_verify_headers():
    assert lr.cmd_verify_headers(_Args()) == 0, "verify-headers failed"


def test_snapshots_present_and_match_registry_hash():
    reg = lr.load_registry()
    for e in reg["entries"]:
        snap = ROOT / e["snapshot"]
        assert snap.is_file(), f"snapshot missing: {e['snapshot']}"
        text = lr.lf(snap.read_text(encoding="utf-8"))
        assert lr.sha256(text) == e["content_hash"], (
            f"snapshot hash != registry hash for {e['id']}")


@pytest.mark.skipif(
    lr.find_mujoco_root() is None,
    reason="vendored MuJoCo tree not found (set PROTOSPEC_MUJOCO_ROOT)",
)
def test_check_against_vendored_tree():
    assert lr.cmd_check(_Args(mujoco_root=None)) == 0, (
        "lifted code diverged from the vendored MuJoCo tree")
