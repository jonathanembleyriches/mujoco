"""One-shot migration: write ProtoSpec binding facts into the schema as annotations.

Historically the ProtoSpec->MuJoCo binding facts lived as Python dict literals in
``protospec_gen/emit_native.py`` (KEYWORD_MAPS value columns, ENUM_PRIM_MAP,
LENIENT_EXACT_FALSE) and ``protospec_gen/emit_mjs.py`` (ELEMENT_STRUCT, ELEM_ALIAS,
GLOBAL_ALIAS, ENUM_MJT). This script consumed those tables -- frozen in
``tools/bootstrap/legacy_binding_tables.json`` -- and mechanically wrote the
equivalent EXPLICIT annotations into ``schema/mujoco.spec``:

  * enum   ``(ctype=mjtGain)``          <- ENUM_MJT key column
  * enum   ``(cmap=TFAuto_map)``        <- ENUM_PRIM_MAP
  * member ``(c=mjGAIN_FIXED)``         <- KEYWORD_MAPS value column
  * member ``(mjs_c=mjLIMITED_FALSE)``  <- ENUM_MJT value column that diverges from
                                           (or has no) reader ``c`` value
  * member ``(no_mjs)``                 <- a reader-only member of a ctype'd enum,
                                           excluded from the mjs ToMjt switch
  * element ``(mjs=mjsBody)``           <- ELEMENT_STRUCT, non-identity only
                                           (mjs<Name> is identity, unannotated)
  * field  ``(mjs=ipos)``               <- ELEM_ALIAS / GLOBAL_ALIAS
  * field  ``(lenient)``                <- LENIENT_EXACT_FALSE

It is an APPEND-mode one-shot, NOT re-run in the pipeline; it is kept as the
provenance of the annotations now living in the schema. Rerunning the writer
against a freshly un-annotated schema reproduces the committed annotations. It
never touches the migration-scaffolding tables (ATTR_ELEMENTS, waivers,
include-lists), which stay in Python.

``--check`` does NOT re-write; it re-parses the (already annotated) schema and
asserts its binding annotations still reconstruct every legacy table byte-for-byte
-- the standing provenance guard that the schema and this frozen snapshot agree.

Run from ``protospec/``:  python -m tools.write_binding_annotations [--check]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from protospec_gen.idl import parse_spec  # noqa: E402

SCHEMA = os.path.join(ROOT, "schema", "mujoco.spec")
FIELDS_JSON = os.path.join(ROOT, "snapshots", "mjspec_fields.json")
LEGACY = os.path.join(HERE, "bootstrap", "legacy_binding_tables.json")

# Render order of the binding annotations we add, per context.
_ORDER = {
    "enum": ["ctype", "cmap"],
    "member": ["c", "mjs_c", "no_mjs"],
    "element": ["mjs"],
    "field": ["mjs", "lenient"],
}


def _load_legacy():
    with open(LEGACY, encoding="utf-8") as fh:
        d = json.load(fh)
    d["ELEM_ALIAS"] = {tuple(k.split(".", 1)): v for k, v in d["ELEM_ALIAS"].items()}
    d["LENIENT_EXACT_FALSE"] = {tuple(x) for x in d["LENIENT_EXACT_FALSE"]}
    return d


def compute_annotations(schema_json, mjs_json, legacy):
    """Return {line_no: (context, {key: value})} edits to apply to the schema.

    Every edit is keyed by the 1-based source line of the entity it annotates;
    a mixin field shared by many elements resolves to one line and one edit.
    """
    KEYWORD_MAPS = legacy["KEYWORD_MAPS"]  # [ [map, enum, size, sz, [values...]], ...]
    kw_values = {row[1]: row[4] for row in KEYWORD_MAPS}
    ENUM_PRIM_MAP = legacy["ENUM_PRIM_MAP"]
    ENUM_MJT = legacy["ENUM_MJT"]
    ELEMENT_STRUCT = legacy["ELEMENT_STRUCT"]
    ELEM_ALIAS = legacy["ELEM_ALIAS"]
    GLOBAL_ALIAS = legacy["GLOBAL_ALIAS"]
    LENIENT = legacy["LENIENT_EXACT_FALSE"]

    edits: dict[int, tuple[str, dict]] = {}

    def add(line, context, key, value):
        ctx, d = edits.setdefault(line, (context, {}))
        assert ctx == context, (line, ctx, context)
        if key in d and d[key] != value:
            raise SystemExit(
                f"conflicting {key!r} at line {line}: {d[key]!r} vs {value!r}"
            )
        d[key] = value

    # ---- enums + members --------------------------------------------------- #
    for enum in schema_json["enums"]:
        ename = enum["name"]
        mjt = ENUM_MJT.get(ename)
        if mjt is not None:
            add(enum["line"], "enum", "ctype", mjt["ctype"])
        if ename in ENUM_PRIM_MAP:
            add(enum["line"], "enum", "cmap", ENUM_PRIM_MAP[ename][0])
        values = kw_values.get(ename)
        mjt_members = mjt["members"] if mjt else {}
        for idx, m in enumerate(enum["members"]):
            mname = m["name"]
            cval = values[idx] if values is not None else None
            if cval is not None:
                add(m["line"], "member", "c", cval)
            mjtc = mjt_members.get(mname)
            if mjtc is not None:
                if cval is not None and cval == mjtc:
                    pass  # reader c value doubles as the ToMjt constant
                else:
                    add(m["line"], "member", "mjs_c", mjtc)
            elif mjt is not None and cval is not None:
                # ctype'd enum, reader-only member: exclude from the ToMjt switch.
                add(m["line"], "member", "no_mjs", True)

    # ---- elements (non-identity struct) ------------------------------------ #
    elems = {e["name"]: e for e in schema_json["elements"]}
    for ename, struct in ELEMENT_STRUCT.items():
        if struct == "mjs" + ename:
            continue  # identity, no annotation
        add(elems[ename]["line"], "element", "mjs", struct)

    # ---- fields (alias + lenient), derived from the legacy resolution ------ #
    struct_by = {s["name"]: {f["name"]: f for f in s["fields"]}
                 for s in mjs_json["structs"]}

    def effective_mjs(elem_name, fname, struct):
        sf = struct_by[struct]
        n = ELEM_ALIAS.get((elem_name, fname))
        if n is not None:
            return n
        if fname in sf:
            return fname  # exact
        return GLOBAL_ALIAS.get(fname, fname)

    def field_xml(f):
        return f.get("annotations", {}).get("xml", f["name"])

    for ename, struct in ELEMENT_STRUCT.items():
        for f in elems[ename]["fields"]:
            fname = f["name"]
            eff = effective_mjs(ename, fname, struct)
            if eff != fname and eff in struct_by[struct]:
                add(f["line"], "field", "mjs", eff)
            if (ename, field_xml(f)) in LENIENT:
                add(f["line"], "field", "lenient", True)

    return edits


# --------------------------------------------------------------------------- #
# Line rewriting.                                                              #
# --------------------------------------------------------------------------- #
def _split_comment(line: str) -> tuple[str, str]:
    """Split a source line into (code, comment) at the first '#' outside a string."""
    in_str = False
    for i, ch in enumerate(line):
        if ch == '"':
            in_str = not in_str
        elif ch == "#" and not in_str:
            return line[:i].rstrip(), line[i:]
    return line, ""


def _render_clause(context: str, annots: dict) -> str:
    parts = []
    for key in _ORDER[context]:
        if key not in annots:
            continue
        v = annots[key]
        parts.append(key if v is True else f"{key}={v}")
    return ", ".join(parts)


def _apply(line: str, context: str, annots: dict) -> str:
    code, comment = _split_comment(line)
    trailing = ""
    if comment:
        # preserve the original gap between code and comment
        raw_code = line[: len(line) - len(line.lstrip()) + 0]  # unused
        gap = line[len(code):]
        gap = gap[: len(gap) - len(gap.lstrip())]
        trailing = gap + comment
    clause = _render_clause(context, annots)

    if "(" in code:  # merge into an existing annotation clause
        rp = code.rindex(")")
        new_code = code[:rp] + ", " + clause + code[rp:]
        return new_code + trailing

    if context in ("enum", "element"):
        assert code.rstrip().endswith("{")
        head = code.rstrip()[:-1].rstrip()
        return f"{head} ({clause}) {{" + trailing
    if context == "member":
        return code.rstrip() + f" ({clause})" + trailing
    # field: insert after the type, before an optional default.
    eq = code.find("=")
    if eq != -1:
        head, tail = code[:eq].rstrip(), code[eq:]
        return f"{head} ({clause}) {tail}" + trailing
    return code.rstrip() + f" ({clause})" + trailing


def rewrite(text: str, edits: dict) -> str:
    lines = text.split("\n")
    for line_no, (context, annots) in edits.items():
        i = line_no - 1
        lines[i] = _apply(lines[i], context, annots)
    return "\n".join(lines)


def reconstruct_check(schema_json, mjs_json, legacy) -> None:
    """Assert the schema's binding annotations still encode every legacy table."""
    import protospec_gen.binding as B  # noqa: N814

    enums = {e["name"]: e for e in schema_json["enums"]}
    elems = {e["name"]: e for e in schema_json["elements"]}
    structs = {s["name"]: {f["name"]: f for f in s["fields"]}
               for s in mjs_json["structs"]}

    for row in legacy["KEYWORD_MAPS"]:
        enum, values = row[1], row[4]
        got = B.keyword_values(enums[enum])
        assert got == values, f"KEYWORD_MAPS {enum}: {got} != {values}"

    ev = B.load_enum_values()
    for enum, info in legacy["ENUM_MJT"].items():
        assert B.enum_ctype(enums[enum]) == info["ctype"], enum
        recon = dict(B.tomjt_members(enums[enum]))
        assert recon == info["members"], f"ENUM_MJT {enum}: {recon} != {info['members']}"

    for enum, (mp, _sz) in legacy["ENUM_PRIM_MAP"].items():
        assert B.enum_cmap(enums[enum]) == mp, enum

    for ename, struct in legacy["ELEMENT_STRUCT"].items():
        assert B.element_struct(elems[ename]) == struct, ename

    def old_eff(en, fn, st):
        n = legacy["ELEM_ALIAS"].get((en, fn))
        if n:
            return n
        if fn in structs[st]:
            return fn
        return legacy["GLOBAL_ALIAS"].get(fn, fn)

    for ename, struct in legacy["ELEMENT_STRUCT"].items():
        for f in elems[ename]["fields"]:
            fn = f["name"]
            eff = old_eff(ename, fn, struct)
            got = B.field_mjs(f) or fn
            if eff in structs[struct] and eff != fn:
                assert got == eff, f"{ename}.{fn}: mjs={got} != {eff}"

    for en, xml in legacy["LENIENT_EXACT_FALSE"]:
        f = next(x for x in elems[en]["fields"]
                 if x.get("annotations", {}).get("xml", x["name"]) == xml)
        assert B.field_lenient(f), f"{en}.{xml} not lenient"
    print("OK: schema annotations reconstruct every legacy binding table")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="re-parse the schema and assert its annotations still "
                         "reconstruct the frozen legacy tables; do not write")
    args = ap.parse_args(argv)

    legacy = _load_legacy()
    with open(FIELDS_JSON, encoding="utf-8") as fh:
        mjs_json = json.load(fh)
    schema_json = parse_spec(SCHEMA).to_json()

    if args.check:
        reconstruct_check(schema_json, mjs_json, legacy)
        return 0

    edits = compute_annotations(schema_json, mjs_json, legacy)
    with open(SCHEMA, encoding="utf-8") as fh:
        text = fh.read()
    new_text = rewrite(text, edits)

    census = {"enum": 0, "member": 0, "element": 0, "field": 0}
    keycount: dict = {}
    for _, (context, annots) in edits.items():
        for k in annots:
            keycount[k] = keycount.get(k, 0) + 1
        census[context] += 1
    print("edited lines by context:", census)
    print("annotation count by key:", dict(sorted(keycount.items())))

    with open(SCHEMA, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(new_text)
    print(f"wrote {SCHEMA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
