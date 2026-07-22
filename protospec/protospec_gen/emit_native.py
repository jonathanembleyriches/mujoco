"""Generate MuJoCo's native XML-reader schema table from ProtoSpec's IDL.

MuJoCo's hand-maintained MJCF grammar lives in ``src/xml/xml_native_reader.cc``
as ``std::vector<const char*> MJCF[nMJCF]`` -- rows of
``{element, occurrence, attr...}`` interleaved with ``{"<"}`` / ``{">"}`` nesting
markers, consumed by ``mjXSchema`` (``src/xml/xml_util.cc``). This module emits
that exact table from ``schema/mujoco.spec`` into
``src/xml/xml_native_schema.inc`` so the count and every row are derived rather
than hand-curated.

Run from the ``protospec/`` directory:

    uv run python -m protospec_gen.emit_native --write   # regenerate the .inc
    uv run python -m protospec_gen.emit_native --check   # byte-gate (CI)

Structural mapping (the schema is the source of truth for content; this emitter
owns the shape decisions the reader table encodes but the IR models differently):

  * occurrence codes -- ``one`` -> ``!``, ``zero_or_one`` -> ``?``,
    ``zero_or_more`` -> ``*``; a self-recursive element (``body``, ``default``)
    is emitted once with ``R`` and never re-listed at its recursion point.
  * unions -- an ordered heterogeneous child list expands to one row per member
    element (actuator/sensor/tendon/equality/body-child spellings).
  * body aliases -- ``frame``/``replicate`` are not their own rows; mjXSchema
    validates them against the ``body`` row (xml_util.cc NameMatch), so they are
    dropped from the ``body`` child list here.
  * variants -- an aliasing group contributes its arms' attributes (a struct
    arm's field name, or the arm tag for an enum/scalar arm).
  * input aliases -- a field's read-only ``aliases`` (euler/axisangle/... for a
    quat, ``diameter`` for cylinder area) and the element-level material
    ``texture`` alias are accepted attributes and so appear in the row.
  * ``<default>`` templates -- carry the *defaultable* subset of each element:
    identity (``name``/``class``) and per-family binding attributes (transmission
    targets, geom1/geom2, camera/light targets, the bulk of ``mesh``) are
    dropped, and children are suppressed -- except ``<material>``, whose
    ``<layer>`` list is part of the defaultable material spec. This projection
    mirrors MuJoCo's hand-written default subtree (what ``mjsDefault`` carries);
    it is table data here because the IR reuses the full element type in both
    the authored and the default context.
"""

from __future__ import annotations

import argparse
import os
import sys

from .idl import parse_spec

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCHEMA = os.path.join(ROOT, "schema", "mujoco.spec")
# src/xml/ lives at the vendored-MuJoCo checkout root (protospec/'s parent).
MUJOCO_ROOT = os.path.dirname(ROOT)
OUT_PATH = os.path.join(MUJOCO_ROOT, "src", "xml", "xml_native_schema.inc")

HEADER = (
    "// Generated from protospec/schema/mujoco.spec by protospec_gen.emit_native"
    " -- do not edit.\n"
    "//\n"
    "// The MJCF grammar table (rows of {element, occurrence, attr...} plus\n"
    "// {\"<\"}/{\">\"} nesting markers) and its row count, consumed by mjXSchema.\n"
    "// Regenerate with: uv run python -m protospec_gen.emit_native --write\n"
)

_CARD_CODE = {"zero_or_more": "*", "zero_or_one": "?", "one": "!"}

# frame/replicate are body aliases: mjXSchema::NameMatch validates them against
# the body row, so they are never their own schema rows.
_BODY_ALIAS_SKIP = frozenset({"Frame", "Replicate"})

# The transmission-target / length-range attributes an actuator carries that a
# <default> template cannot set (they name a specific joint/site/tendon/body).
_ACT_STRIP = (
    "lengthrange", "joint", "jointinparent", "tendon", "slidersite",
    "cranksite", "site", "refsite", "body",
)

# Attributes dropped from an element's <default> row, beyond the universal
# name/class. Keyed by IR element name. (Mesh is handled by _DEFAULT_KEEP.)
_DEFAULT_STRIP = {
    "ActuatorGeneral": _ACT_STRIP,
    "Motor": _ACT_STRIP,
    "Position": _ACT_STRIP,
    "Velocity": _ACT_STRIP,
    "IntVelocity": _ACT_STRIP,
    "Damper": _ACT_STRIP,
    "Cylinder": _ACT_STRIP,
    "DcMotor": _ACT_STRIP,
    "Muscle": _ACT_STRIP + ("tausmooth",),
    "Adhesion": ("body",),
    "Camera": ("target",),
    "Light": ("target", "texture"),
    "Pair": ("geom1", "geom2"),
}

# Elements whose <default> row keeps only an explicit attribute allow-list
# (everything else, name/class included, is dropped).
_DEFAULT_KEEP = {"Mesh": ("scale", "maxhullvert", "inertia")}

# Elements whose child lists survive into the <default> template. Only
# <material> qualifies: its <layer> list is part of the defaultable spec.
_DEFAULT_KEEP_CHILDREN = frozenset({"Material"})

# Element-level MJCF input aliases: an attribute accepted on input and folded
# into a child list rather than a field (material texture -> a <layer> entry).
# Mirrors emit.ELEMENT_INPUT_ALIASES (kept in step; the two encode the same fold).
_ELEMENT_INPUT_ALIASES = {"Material": ["texture"]}


class Schema:
    def __init__(self, doc: dict):
        self.elements = {e["name"]: e for e in doc["elements"]}
        self.structs = {s["name"]: s for s in doc["structs"]}
        self.variants = {v["name"]: v for v in doc["variants"]}
        self.unions = {u["name"]: u for u in doc["unions"]}
        self.recursive = self._recursive()

    @staticmethod
    def xml_of(e: dict) -> str:
        return e.get("annotations", {}).get("xml", e["name"].lower())

    @staticmethod
    def field_xml(f: dict) -> str:
        return f.get("annotations", {}).get("xml", f["name"])

    def _child_members(self, elem: dict) -> set:
        out = set()
        for c in elem["children"]:
            if c.get("union") is not None:
                out.update(self.unions[c["union"]]["members"])
            else:
                out.add(c["element"])
        return out

    def _recursive(self) -> set:
        """Element names reachable from themselves through child lists."""
        rec = set()
        for name in self.elements:
            seen, stack = set(), list(self._child_members(self.elements[name]))
            while stack:
                n = stack.pop()
                if n == name:
                    rec.add(name)
                    break
                if n in seen:
                    continue
                seen.add(n)
                stack.extend(self._child_members(self.elements[n]))
        return rec

    def _field_aliases(self, f: dict) -> list:
        ann = f.get("annotations", {})
        if ann.get("resolver") is None:
            return []
        raw = ann.get("aliases")
        return raw.split() if raw else []

    def attrs(self, elem: dict) -> list:
        """Ordered, de-duplicated MJCF attribute names an element accepts."""
        out: list[str] = []
        for f in elem["fields"]:
            t = f["type"]
            if t["kind"] == "variant":
                for arm in self.variants[t["target"]]["arms"]:
                    at = arm["type"]
                    if at["kind"] == "named" and at.get("category") == "struct":
                        for sf in self.structs[at["name"]]["fields"]:
                            out.append(self.field_xml(sf))
                    else:
                        out.append(arm["tag"])
            else:
                out.append(self.field_xml(f))
                out.extend(self._field_aliases(f))
        out.extend(_ELEMENT_INPUT_ALIASES.get(elem["name"], ()))
        seen, res = set(), []
        for a in out:
            if a not in seen:
                seen.add(a)
                res.append(a)
        return res


class Row:
    """An emitted schema row (an element header, or a "<"/">" nesting marker)."""

    __slots__ = ("tokens", "depth", "kind")

    def __init__(self, tokens, depth, kind):
        self.tokens = tokens  # list[str]
        self.depth = depth
        self.kind = kind  # "elem" | "open" | "close"


def _default_attrs(s: Schema, name: str, attrs: list) -> list:
    if name in _DEFAULT_KEEP:
        keep = set(_DEFAULT_KEEP[name])
        return [a for a in attrs if a in keep]
    strip = {"name", "class"} | set(_DEFAULT_STRIP.get(name, ()))
    return [a for a in attrs if a not in strip]


def build_rows(s: Schema) -> list:
    rows: list[Row] = []

    def emit(name: str, card: str, ancestors: frozenset, default_ctx: bool,
             depth: int) -> None:
        elem = s.elements[name]
        code = "R" if name in s.recursive else _CARD_CODE[card]
        attrs = s.attrs(elem)
        if default_ctx:
            attrs = _default_attrs(s, name, attrs)
        rows.append(Row([s.xml_of(elem), code] + attrs, depth, "elem"))

        emit_children = (not default_ctx) or (name in _DEFAULT_KEEP_CHILDREN)
        if not emit_children:
            return
        anc2 = ancestors | {name}
        child_default = default_ctx or (name == "Default")
        kids: list = []
        for c in elem["children"]:
            ccard = c["cardinality"]
            if c.get("union") is not None:
                members = s.unions[c["union"]]["members"]
            else:
                members = [c["element"]]
            for m in members:
                if m in _BODY_ALIAS_SKIP or m in anc2:
                    continue  # body alias, or recursion point (R): not re-listed
                kids.append((m, ccard))
        if not kids:
            return
        rows.append(Row(["<"], depth, "open"))
        for m, ccard in kids:
            emit(m, ccard, anc2, child_default, depth + 1)
        rows.append(Row([">"], depth, "close"))

    emit("Model", "one", frozenset(), False, 0)
    return rows


def _render_row(tokens: list, depth: int) -> str:
    """Render one brace-group, wrapping long attribute lists for readability."""
    indent = "    " * (depth + 1)
    lits = [f'"{t}"' for t in tokens]
    one_line = indent + "{" + ", ".join(lits) + "},"
    if len(one_line) <= 96 or len(lits) <= 2:
        return one_line
    # Wrap: keep {tag, code, on the first line, then attrs wrapped.
    cont = indent + "    "
    lines = [indent + "{" + lits[0] + ", " + lits[1] + ","]
    cur = cont
    for lit in lits[2:]:
        piece = lit + ", "
        if len(cur) + len(piece) > 96 and cur != cont:
            lines.append(cur.rstrip())
            cur = cont
        cur += piece
    lines.append(cur.rstrip().rstrip(",") + "},")
    return "\n".join(lines)


def generate(schema_path: str = SCHEMA) -> str:
    s = Schema(parse_spec(schema_path).to_json())
    rows = build_rows(s)
    n = len(rows)

    out: list[str] = [HEADER.rstrip("\n"), ""]
    out.append(f"#define nMJCF_GENERATED {n}")
    out.append("")
    out.append("// clang-format off")
    out.append("std::vector<const char*> MJCF[nMJCF_GENERATED] = {")
    for r in rows:
        if r.kind == "open":
            out.append("    " * (r.depth + 1) + '{"<"},')
        elif r.kind == "close":
            out.append("    " * (r.depth + 1) + '{">"},')
        else:
            out.append(_render_row(r.tokens, r.depth))
    out.append("};")
    out.append("// clang-format on")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true",
                   help="verify the checked-in .inc is up to date; do not write")
    g.add_argument("--write", action="store_true",
                   help="regenerate the .inc (default)")
    ap.add_argument("--out", default=OUT_PATH, help="destination .inc path")
    args = ap.parse_args(argv)

    content = generate()

    if args.check:
        if not os.path.exists(args.out):
            sys.stderr.write(f"missing generated file: {args.out}\n")
            return 1
        with open(args.out, "r", encoding="utf-8", newline="") as fh:
            current = fh.read().replace("\r\n", "\n")
        if current != content:
            sys.stderr.write(
                "xml_native_schema.inc is out of date; re-run "
                "`python -m protospec_gen.emit_native --write`\n"
            )
            return 1
        n = content.split("#define nMJCF_GENERATED ", 1)[1].split("\n", 1)[0]
        print(f"{args.out} is up to date ({n} rows)")
        return 0

    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(content)
    n = content.split("#define nMJCF_GENERATED ", 1)[1].split("\n", 1)[0]
    print(f"wrote {args.out} ({n} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
