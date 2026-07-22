"""Semantic equivalence gate for the generated native-reader schema table.

``protospec_gen.emit_native`` regenerates MuJoCo's hand-maintained MJCF grammar
table (``std::vector<const char*> MJCF[]`` in ``src/xml/xml_native_reader.cc``)
into ``src/xml/xml_native_schema.inc`` from ``schema/mujoco.spec``. This test
proves the generated table is *semantically identical* to the table it replaced:

  * the original hand-written table is recovered from git
    (``protospec-upstream:src/xml/xml_native_reader.cc``);
  * the generated ``.inc`` is read from the working tree;

both are parsed with the extractor's own row/nesting logic (the same reconstruction
``mjXSchema`` performs) and compared as element trees: every element's tag,
occurrence code, and attribute *set* (order-insensitive), recursively over the
children (also order-insensitive). Attribute and child order do not matter to
``mjXSchema``; the *set* and the tree shape are the contract.

Upstream main already carried the SO3 ``orientation`` actuator (072e963f), the
geom/pair ``adhesion`` attribute (a264d0bc), the general ``input`` attribute and
the ``so3`` gain/bias keywords, so the schema catch-up that added them to
``schema/mujoco.spec`` makes the two tables coincide exactly -- the comparison is
strict equality with no waived deltas.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "bootstrap"))

import extract_mjcf_schema as ex  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
INC = REPO / "src" / "xml" / "xml_native_schema.inc"
UPSTREAM_REF = "protospec-upstream:src/xml/xml_native_reader.cc"


def _git_show(ref: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "show", ref],
            capture_output=True, text=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return out.stdout


def _tree(source: str):
    return ex.parse_schema_tree(source)


def _diff(a, b, path: str = "") -> list[str]:
    """Order-insensitive structural diff of two extractor Element trees."""
    from collections import defaultdict

    p = f"{path}/{a.name}"
    diffs: list[str] = []
    if a.name != b.name:
        return [f"{path}: element tag {a.name!r} vs {b.name!r}"]
    if a.typecode != b.typecode:
        diffs.append(f"{p}: occurrence {a.typecode!r} (generated) vs "
                     f"{b.typecode!r} (hand)")
    sa, sb = set(a.attributes), set(b.attributes)
    if sa != sb:
        diffs.append(f"{p}: attrs generated-only={sorted(sa - sb)} "
                     f"hand-only={sorted(sb - sa)}")
    ga: dict = defaultdict(list)
    gb: dict = defaultdict(list)
    for c in a.children:
        ga[c.name].append(c)
    for c in b.children:
        gb[c.name].append(c)
    for name in sorted(set(ga) | set(gb)):
        if name not in ga:
            diffs.append(f"{p}: missing child {name!r} (hand has {len(gb[name])})")
            continue
        if name not in gb:
            diffs.append(f"{p}: extra child {name!r} (generated has {len(ga[name])})")
            continue
        if len(ga[name]) != len(gb[name]):
            diffs.append(f"{p}: child {name!r} count generated={len(ga[name])} "
                         f"hand={len(gb[name])}")
        for x, y in zip(ga[name], gb[name]):
            diffs.extend(_diff(x, y, p))
    return diffs


@pytest.fixture(scope="module")
def hand_tree():
    source = _git_show(UPSTREAM_REF)
    if source is None:
        pytest.skip(f"cannot recover {UPSTREAM_REF} (git or branch unavailable)")
    return _tree(source)


@pytest.fixture(scope="module")
def generated_tree():
    if not INC.exists():
        pytest.skip(f"{INC} not generated; run emit_native --write")
    return _tree(INC.read_text(encoding="utf-8"))


def test_inc_parses_and_is_rooted_at_mujoco(generated_tree):
    assert generated_tree.name == "mujoco"
    assert generated_tree.typecode == "!"


def test_row_count_matches_hand_table(generated_tree):
    hand = _git_show(UPSTREAM_REF)
    if hand is None:
        pytest.skip("upstream reader unavailable")
    hand_block, base = ex._extract_block(
        hand, r"std::vector<const char\*>\s+MJCF\s*\[\s*nMJCF\s*\]\s*="
    )
    hand_rows = len(ex._parse_rows(hand_block, base))
    gen_block, gbase = ex._extract_block(
        INC.read_text(encoding="utf-8"),
        r"std::vector<const char\*>\s+MJCF\s*\[\s*nMJCF_GENERATED\s*\]\s*=",
    )
    gen_rows = len(ex._parse_rows(gen_block, gbase))
    assert gen_rows == hand_rows


def test_generated_table_semantically_equals_hand_table(generated_tree, hand_tree):
    diffs = _diff(generated_tree, hand_tree)
    assert not diffs, "generated table diverges from hand table:\n  " + "\n  ".join(diffs)
