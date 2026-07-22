"""Two-way join-completeness gate for the native reader's keyword-map value column.

``protospec_gen.emit_native`` owns the value column of the generated keyword maps
(``KEYWORD_MAPS`` / ``PRIMITIVE_MAPS``) -- an otherwise unguarded join between the
schema keyword side and MuJoCo's C ``mjt*`` enums. ``tools/bootstrap/
extract_c_enums.py`` extracts those enums (``snapshots/mjt_enums.json``); this test
pins the join in both directions so that a wrong-but-compiling constant, or an
upstream-added enumerator with no keyword, becomes a loud failure:

  (a) every C constant a generated map references exists in the header extraction,
      and all constants of a map belong to one enum;
  (b) every member of each mapped C enum is either covered by a map row, a size
      sentinel (``mjN*``), or a documented exclusion (internal-only members);
  plus: the integer maps are pure ordinals; the prefix+UPPER(keyword) derivability
  classification has exactly the known exceptions; and the duplicated member-joins
  in ``KEYWORD_MAPS`` (emit_native) and ``ENUM_MJT`` (emit_mjs) agree.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "bootstrap"))

from protospec_gen import emit_native  # noqa: E402
from protospec_gen import emit_mjs  # noqa: E402
from protospec_gen.idl import parse_spec  # noqa: E402
import extract_c_enums as cx  # noqa: E402

SNAP = REPO / "snapshots" / "mjt_enums.json"

# Members of a mapped enum deliberately absent from its map, with a reason.
# (Size sentinels ``mjN*`` are excluded generically.)
MEMBER_EXCLUSIONS = {
    "mjtGeom": {  # visualization-only geom types, not authorable MJCF shapes
        "mjGEOM_ARROW", "mjGEOM_ARROW1", "mjGEOM_ARROW2", "mjGEOM_LINE",
        "mjGEOM_LINEBOX", "mjGEOM_FLEX", "mjGEOM_SKIN", "mjGEOM_LABEL",
        "mjGEOM_TRIANGLE", "mjGEOM_NONE",
    },
    "mjtSleepPolicy": {  # composite auto-policies with no distinct MJCF spelling
        "mjSLEEP_AUTO_NEVER", "mjSLEEP_AUTO_ALLOWED",
    },
    "mjtTextureRole": {"mjTEXROLE_USER"},  # beyond texrole_sz == mjNTEXROLE-1
}

# Generated maps whose constants belong to an INTERNAL composite/flexcomp enum
# (defined in src/user/, not a public mjt* header) -- excluded from the enum join.
INTERNAL_MAPS = {"comp_map", "fcomp_map", "fdof_map"}

# The prefix+UPPER(keyword) derivability exceptions, with why each is irregular.
DERIVABILITY_EXCEPTIONS = {
    "mjCAMOUT_DIST": "abbreviation of keyword 'distance'",
    "mjCAMOUT_SEG": "abbreviation of keyword 'segmentation'",
    "mjINERTIA_VOLUME": "meshtype_map is a bool spelling (false) -> inertia constant",
    "mjINERTIA_SHELL": "meshtype_map is a bool spelling (true) -> inertia constant",
}


@pytest.fixture(scope="module")
def snap():
    return json.loads(SNAP.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def const2enum(snap):
    return {m["name"]: name for name, mem in snap["enums"].items() for m in mem}


def _cmaps():
    """(map_name, [(keyword, value), ...]) for every generated map."""
    doc = parse_spec(emit_native.SCHEMA).to_json()
    enums = {e["name"]: e for e in doc["enums"]}
    out = []
    for b in emit_native.KEYWORD_MAPS:
        keys = [m["value"] for m in enums[b.enum]["members"]]
        out.append((b.map, list(zip(keys, b.values))))
    for pb in emit_native.PRIMITIVE_MAPS:
        out.append((pb.map, list(pb.pairs)))
    return out


def _c_const_maps(const2enum):
    """Only maps whose value column is C constants (not pure ordinals), tagged with
    their resolved enum; internal-enum maps flagged with enum=None."""
    out = []
    for mp, pairs in _cmaps():
        cvals = [v for _, v in pairs if not v.lstrip("-").isdigit()]
        if not cvals:
            continue
        enums = {const2enum.get(v) for v in cvals}
        out.append((mp, pairs, enums.pop() if len(enums) == 1 else None))
    return out


def test_snapshot_matches_headers():
    """The checked-in snapshot equals a fresh extraction (byte-gate for the join)."""
    fresh = cx.build_snapshot(REPO.parent)
    current = json.loads(SNAP.read_text(encoding="utf-8"))
    assert fresh == current, ("snapshots/mjt_enums.json is stale; re-run "
                              "tools/bootstrap/extract_c_enums.py --mujoco-src ..")


def test_every_mapped_constant_exists_in_one_enum(const2enum):
    for mp, pairs, enum in _c_const_maps(const2enum):
        cvals = [v for _, v in pairs if not v.lstrip("-").isdigit()]
        if mp in INTERNAL_MAPS:
            # constants live in an internal src/user/ enum, not a public header
            assert enum is None, f"{mp}: expected internal enum, resolved {enum}"
            continue
        missing = [v for v in cvals if v not in const2enum]
        assert not missing, f"{mp}: constants not found in any C enum: {missing}"
        assert enum is not None, f"{mp}: constants span multiple enums {cvals}"


def test_mapped_enum_members_covered_or_excluded(const2enum, snap):
    for mp, pairs, enum in _c_const_maps(const2enum):
        if mp in INTERNAL_MAPS or enum is None:
            continue
        mapped = {v for _, v in pairs if not v.lstrip("-").isdigit()}
        excl = MEMBER_EXCLUSIONS.get(enum, set())
        uncovered = []
        for m in snap["enums"][enum]:
            n = m["name"]
            if n in mapped or n in excl or n.startswith("mjN"):
                continue  # covered / documented exclusion / size sentinel
            uncovered.append(n)
        assert not uncovered, (
            f"{mp} ({enum}): enum members with no keyword and no exclusion: "
            f"{uncovered} -- add a keyword to the schema or document the exclusion")


def test_integer_maps_are_pure_ordinals():
    for mp, pairs in _cmaps():
        vals = [v for _, v in pairs]
        if any(not v.lstrip("-").isdigit() for v in vals):
            continue
        assert [int(v) for v in vals] == list(range(len(vals))), (
            f"{mp}: integer map is not a 0..n-1 ordinal sequence: {vals}")


def test_derivability_exceptions_are_documented(snap):
    got = set(snap["derivability"]["exceptions"])
    assert got == set(DERIVABILITY_EXCEPTIONS), (
        f"derivability exceptions changed: {sorted(got)} vs "
        f"{sorted(DERIVABILITY_EXCEPTIONS)}")


def test_keyword_maps_agree_with_enum_mjt():
    """The ~member-joins duplicated in KEYWORD_MAPS (emit_native) and ENUM_MJT
    (emit_mjs) agree today by discipline only -- pin them until ENUM_MJT is retired."""
    doc = parse_spec(emit_native.SCHEMA).to_json()
    schema_enums = {e["name"]: [m["name"] for m in e["members"]]
                    for e in doc["enums"]}
    kw_by_enum = {b.enum: b for b in emit_native.KEYWORD_MAPS}

    checked = 0
    for enum, b in kw_by_enum.items():
        mjt = emit_mjs.ENUM_MJT.get(enum)
        if mjt is None:
            continue
        members = schema_enums[enum]
        kw_val = dict(zip(members, b.values))  # schema member name -> keyword-map value
        for member, const in mjt[1].items():
            assert member in kw_val, f"{enum}.{member} in ENUM_MJT but not the schema"
            assert kw_val[member] == const, (
                f"{enum}.{member}: KEYWORD_MAPS says {kw_val[member]!r} but "
                f"ENUM_MJT says {const!r}")
            checked += 1
    assert checked > 20, f"expected many pinned joins, only checked {checked}"
