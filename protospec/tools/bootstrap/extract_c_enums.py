"""Extract MuJoCo's C ``mjt*`` enum definitions into a JSON snapshot.

The native reader's keyword maps join two worlds: the *keyword* column comes from
``schema/mujoco.spec`` (an IDL enum's member spellings) while the *value* column is
a MuJoCo C constant (``mjGAIN_FIXED``) or a bare ordinal. ``protospec_gen.emit_native``
owns that value column in ``KEYWORD_MAPS`` / ``PRIMITIVE_MAPS`` -- an unguarded join
against ``include/mujoco/*.h``. This module extracts every ``typedef enum NAME_ { ... }
NAME;`` block (members with resolved integer values) so a test can pin the join in
both directions: every mapped constant exists with the right enum membership, and
every member of a mapped enum is either covered by a map row or explicitly excluded.

It also records a *derivability* classification per enum: the common ``mjPREFIX_``
of its members, so a future schema-annotation migration can tell which
constant == prefix + upper(keyword) rows are regular and which are genuine
exceptions (only ``mjCAMOUT_DIST`` / ``mjCAMOUT_SEG`` across the whole join).

Run:  uv run python tools/bootstrap/extract_c_enums.py --mujoco-src ..
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Headers that carry the mjt* enums the reader's keyword maps reference.
HEADERS = ("mjtype.h", "mjspec.h", "mjmodel.h")

# The tag before the brace is optional and may or may not carry a trailing
# underscore (``typedef enum mjtGain {...} mjtGain;`` and ``typedef enum mjtFoo_
# {...} mjtFoo;`` both occur); the canonical name is the identifier after ``}``.
_ENUM_RE = re.compile(
    r"typedef\s+enum\s+(?:mj\w+\s+)?\{(?P<body>.*?)\}\s*(?P<name>mj\w+)\s*;",
    re.DOTALL,
)
_MEMBER_RE = re.compile(r"^\s*(mj[A-Z0-9_]+)\s*(?:=\s*([^,]+?))?\s*,?\s*(?://.*)?$")


class CEnumError(Exception):
    pass


def _strip_block_comments(text: str) -> str:
    return re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)


def _resolve(expr: str, values: dict[str, int]) -> int | None:
    """Resolve a member initializer to an int, if possible (literal, prior member,
    or a simple ``A | B`` / ``1 << N`` / ``A + k`` form). Returns None if opaque."""
    expr = expr.strip()
    try:
        if expr in values:
            return values[expr]
        # bare integer (dec or hex)
        return int(expr, 0)
    except ValueError:
        pass
    # 1 << N
    m = re.fullmatch(r"1\s*<<\s*(\w+)", expr)
    if m:
        n = values.get(m.group(1))
        if n is None:
            try:
                n = int(m.group(1), 0)
            except ValueError:
                return None
        return 1 << n
    # NAME + k  /  NAME - k
    m = re.fullmatch(r"(\w+)\s*([+-])\s*(\w+)", expr)
    if m and m.group(1) in values:
        try:
            k = int(m.group(3), 0)
        except ValueError:
            return None
        return values[m.group(1)] + (k if m.group(2) == "+" else -k)
    return None


def parse_enums(source: str) -> dict[str, list[dict]]:
    """Parse every ``typedef enum NAME_ { ... } NAME;`` into name -> [{name,value}]."""
    out: dict[str, list[dict]] = {}
    for m in _ENUM_RE.finditer(source):
        name = m.group("name")
        members: list[dict] = []
        values: dict[str, int] = {}
        counter = 0
        for raw in _strip_block_comments(m.group("body")).splitlines():
            line = raw.strip()
            if not line or line.startswith("//"):
                continue
            mm = _MEMBER_RE.match(line)
            if not mm:
                continue
            mname, expr = mm.group(1), mm.group(2)
            if expr is not None:
                val = _resolve(expr, values)
                if val is not None:
                    counter = val
            val = counter
            values[mname] = val
            members.append({"name": mname, "value": val})
            counter += 1
        if members:
            out.setdefault(name, members)
    return out


def _common_prefix(members: list[dict]) -> str:
    names = [m["name"] for m in members]
    if not names:
        return ""
    pref = os.path.commonprefix(names)
    # trim back to the last underscore so the prefix is mjXXX_
    if "_" in pref:
        return pref[: pref.rindex("_") + 1]
    return pref


def _derivability(enums: dict, prefixes: dict) -> dict:
    """Classify each generated keyword-map row against the prefix+UPPER(keyword)
    naming convention, so a future schema-annotation migration can consume which
    constants are regular vs genuine exceptions. Joins the header enums against
    ``emit_native``'s KEYWORD_MAPS / PRIMITIVE_MAPS (a data join, no codegen)."""
    # Deferred import: the tool lives beside the package on disk.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from protospec_gen import emit_native as en

    const2enum = {m["name"]: name for name, mem in enums.items() for m in mem}

    rows = []  # {map, enum, keyword, constant, regular}
    ordinal_maps = []
    for mp, pairs in ([(b.map, list(zip(_keywords(en, b), b.values)))
                       for b in en.KEYWORD_MAPS]
                      + [(pb.map, list(pb.pairs)) for pb in en.PRIMITIVE_MAPS]):
        cvals = [v for _, v in pairs if not v.lstrip("-").isdigit()]
        if not cvals:  # pure-ordinal integer map
            ordinal_maps.append(mp)
            continue
        enum_names = {const2enum.get(v) for v in cvals}
        enum = enum_names.pop() if len(enum_names) == 1 else None
        if enum is None:  # constants not in a public header (internal enum)
            rows.append({"map": mp, "enum": None, "regular": None})
            continue
        # Prefix from the map's OWN constants (the enum-wide prefix is polluted by
        # sentinel members like mjNGEOMTYPES).
        pref = os.path.commonprefix(cvals)
        if "_" in pref:
            pref = pref[: pref.rindex("_") + 1]
        for kw, const in pairs:
            if const.lstrip("-").isdigit():
                continue
            expected = pref + re.sub(r"[^A-Za-z0-9]", "", kw).upper()
            rows.append({"map": mp, "enum": enum, "keyword": kw,
                         "constant": const, "regular": const == expected})
    exceptions = sorted(r["constant"] for r in rows if r.get("regular") is False)
    return {
        "convention": "constant == prefix + UPPER(alnum(keyword))",
        "ordinal_maps": ordinal_maps,
        "exceptions": exceptions,
        "rows": rows,
    }


def _keywords(en, bind):
    """Return the keyword column of a KEYWORD_MAPS bind (its enum members' values)."""
    from protospec_gen.idl import parse_spec
    doc = parse_spec(en.SCHEMA).to_json()
    enums = {e["name"]: e for e in doc["enums"]}
    return [m["value"] for m in enums[bind.enum]["members"]]


def build_snapshot(mujoco_src: Path) -> dict:
    enums: dict[str, list[dict]] = {}
    for header in HEADERS:
        path = mujoco_src / "include" / "mujoco" / header
        if not path.exists():
            continue
        for name, members in parse_enums(path.read_text(encoding="utf-8")).items():
            enums.setdefault(name, members)
    prefixes = {name: _common_prefix(members) for name, members in enums.items()}
    return {
        "source": {"files": list(HEADERS)},
        "enums": enums,
        "prefixes": prefixes,
        "derivability": _derivability(enums, prefixes),
    }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mujoco-src", type=Path, required=True)
    ap.add_argument("--out", type=Path,
                    default=_repo_root() / "snapshots" / "mjt_enums.json")
    args = ap.parse_args(argv)

    snap = build_snapshot(args.mujoco_src)
    if not snap["enums"]:
        print("error: no enums parsed", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(snap, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {args.out} ({len(snap['enums'])} enums)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
