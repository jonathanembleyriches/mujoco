"""Semantic equivalence gate for the generated native-reader keyword maps.

``protospec_gen.emit_native`` regenerates MuJoCo's schema-backed
``const mjMap NAME[...]`` keyword tables (previously hand-written in
``src/xml/xml_native_reader.cc``) into ``src/xml/xml_native_keywords.inc`` from
``schema/mujoco.spec``: the keyword spellings and order come from the IDL enum
members, the value column (C constants / integers) and array-size tokens are the
MuJoCo-side mapping the emitter owns (``KEYWORD_MAPS``).

This test proves the generated tables are identical to the ones they replaced:

  * the original hand-written maps are recovered from git
    (``main:src/xml/xml_native_reader.cc``);
  * the generated ``.inc`` is read from the working tree;

both are parsed with the extractor's own ``parse_keyword_maps`` (the same reader
the drift snapshot uses) and compared per map as an *ordered* list of
``{key, value}`` pairs -- ``FindKey``/``MapValue`` consume the array in order, so
order and values are the contract.

The seven maps deliberately left hand-written (not schema-backed 1:1) are checked
to still live in the reader and to be *absent* from the generated include:
``bool_map``/``enable_map`` (generic on/off, no enum), ``TFAuto_map`` (generic
tri-state; two enums -- ``TriState``/``InertiaFromGeom`` -- share its spelling),
``equality_map`` (equality is a union of child elements), ``jkind_map``/
``shape_map`` (composite keyword sets), ``meshtype_map`` (bool-spelled inertia
constants).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "bootstrap"))

import extract_mjcf_schema as ex  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
KEYWORDS_INC = REPO / "src" / "xml" / "xml_native_keywords.inc"
READER = REPO / "src" / "xml" / "xml_native_reader.cc"
HAND_REF = "main:src/xml/xml_native_reader.cc"

# Maps still hand-written in the reader (fit none of the generated paths: enum,
# union, or primitive) -- mirrors the header in xml_native_reader.cc.
HAND_ONLY = {"equality_map", "jkind_map", "shape_map"}


def _git_show(ref: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "show", ref],
            capture_output=True, text=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return out.stdout


def _pairs(entries: list[dict[str, str]]) -> list[tuple[str, str]]:
    return [(e["key"], e["value"]) for e in entries]


@pytest.fixture(scope="module")
def hand_maps():
    source = _git_show(HAND_REF)
    if source is None:
        pytest.skip(f"cannot recover {HAND_REF} (git or branch unavailable)")
    return ex.parse_keyword_maps(source)


@pytest.fixture(scope="module")
def generated_maps():
    if not KEYWORDS_INC.exists():
        pytest.skip(f"{KEYWORDS_INC} not generated; run emit_native --write")
    return ex.parse_keyword_maps(KEYWORDS_INC.read_text(encoding="utf-8"))


def test_generated_include_carries_expected_map_count(generated_maps):
    # 48 schema-region maps total, 3 left hand-written -> 45 generated
    # (41 enum-backed + 4 primitive-backed: bool/enable/TFAuto/meshtype).
    assert len(generated_maps) == 45


def test_no_hand_only_map_leaked_into_generated(generated_maps):
    leaked = HAND_ONLY & set(generated_maps)
    assert not leaked, f"hand-only maps must not be generated: {sorted(leaked)}"


def test_generated_maps_equal_hand_maps(generated_maps, hand_maps):
    missing = [m for m in generated_maps if m not in hand_maps]
    assert not missing, f"generated maps absent from hand source: {missing}"
    diffs = []
    for name, entries in generated_maps.items():
        want = _pairs(hand_maps[name])
        got = _pairs(entries)
        if got != want:
            diffs.append(f"{name}: generated={got} hand={want}")
    assert not diffs, "generated keyword maps diverge from hand maps:\n  " + "\n  ".join(diffs)


def test_hand_only_maps_still_in_reader_and_match_hand(hand_maps):
    reader_maps = ex.parse_keyword_maps(READER.read_text(encoding="utf-8"))
    for name in HAND_ONLY:
        assert name in reader_maps, f"{name} must stay hand-written in the reader"
        assert _pairs(reader_maps[name]) == _pairs(hand_maps[name]), (
            f"{name} drifted from its hand definition"
        )


def test_reader_plus_include_covers_every_hand_map(generated_maps, hand_maps):
    reader_maps = ex.parse_keyword_maps(READER.read_text(encoding="utf-8"))
    union = set(reader_maps) | set(generated_maps)
    assert union == set(hand_maps), (
        f"coverage gap: reader+include={sorted(union)} hand={sorted(hand_maps)}"
    )
