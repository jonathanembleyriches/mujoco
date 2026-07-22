"""ProtoSpec binding facts: read the schema's binding annotations + cross-validate.

The ProtoSpec->MuJoCo binding facts (which C constant a keyword maps to, which
mjsStruct an element binds, which mjs field an attribute renames to) live as
EXPLICIT annotations in ``schema/mujoco.spec`` -- ``(ctype=)``/``(c=)`` on enums and
members, ``(mjs=)`` on elements and fields, ``(lenient)``, ``(cmap=)``. This module
is the single reader of those facts (the accessors ``emit_native``/``emit_mjs``
consume) and their cross-validator against the two ground-truth snapshots:

  * ``snapshots/mjmodel_enums.json`` -- every MuJoCo C enum constant + owner, so a
    ``(c=mjGAIN_FOO)``/``(mjs_c=)`` naming a constant that does not exist, or that
    does not belong to its enum's declared ``ctype``, fails loudly;
  * ``snapshots/mjspec_fields.json`` -- the mjsStruct field inventory, so an
    ``(mjs=)`` naming a missing or kind-incompatible mjs field fails loudly.

Every diagnostic is a :class:`~protospec_gen.idl.SchemaError` located at the schema
file:line of the offending annotation -- the maintenance contract that replaced the
deleted Python dict literals. ``validate`` also enforces completeness: a ctype'd
enum must annotate EVERY member (no partial ToMjt switch).
"""

from __future__ import annotations

import json
import os
import re

from .idl import SchemaError, _SourceCtx

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCHEMA = os.path.join(ROOT, "schema", "mujoco.spec")
MJMODEL_ENUMS_JSON = os.path.join(ROOT, "snapshots", "mjmodel_enums.json")
MJSPEC_FIELDS_JSON = os.path.join(ROOT, "snapshots", "mjspec_fields.json")

_INT_RE = re.compile(r"-?\d+")
# A declared ctype that is a primitive, not an enum -- membership of `(c=)`/
# `(mjs_c=)` constants is not checkable (they are raw integers into a byte/int).
_PRIMITIVE_CTYPES = frozenset({"int", "mjtByte"})


# --------------------------------------------------------------------------- #
# Snapshot loading.                                                            #
# --------------------------------------------------------------------------- #
def load_enum_index(path: str = MJMODEL_ENUMS_JSON) -> dict[str, str]:
    """Return ``{C-constant -> owning mjt enum name}`` from the mjmodel snapshot."""
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    index: dict[str, str] = {}
    for enum in doc["enums"]:
        for en in enum["enumerators"]:
            index[en["name"]] = enum["name"]
    return index


def load_enum_values(path: str = MJMODEL_ENUMS_JSON) -> dict[str, int]:
    """Return ``{C-constant -> integer value}`` from the mjmodel snapshot.

    Used to order a ToMjt switch's cases by their underlying enum value (the
    switch is emitted in enum-declaration order, which can differ from the schema
    member order -- the reader keyword order -- e.g. mjtMeshInertia)."""
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    out: dict[str, int] = {}
    for enum in doc["enums"]:
        for en in enum["enumerators"]:
            if en["value"] is not None:
                out[en["name"]] = en["value"]
    return out


def load_mjs_structs(path: str = MJSPEC_FIELDS_JSON) -> dict[str, dict]:
    """Return ``{mjsStruct -> {field-name -> field-dict}}`` from the fields snapshot."""
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    return {s["name"]: {f["name"]: f for f in s["fields"]} for s in doc["structs"]}


# --------------------------------------------------------------------------- #
# Accessors over the schema's binding annotations.                            #
# --------------------------------------------------------------------------- #
def is_int_literal(value) -> bool:
    return bool(_INT_RE.fullmatch(str(value)))


def enum_ctype(enum: dict):
    """The mjt cast type an enum is authored to (``ctype=``), or None."""
    return enum.get("annotations", {}).get("ctype")


def enum_cmap(enum: dict):
    """The shared primitive keyword map an enum's attributes bind through, or None."""
    return enum.get("annotations", {}).get("cmap")


def ctype_enum_names(schema_json: dict) -> set:
    """Names of enums authored to mjSpec (those carrying ``ctype=``)."""
    return {e["name"] for e in schema_json["enums"] if enum_ctype(e) is not None}


def keyword_values(enum: dict) -> list:
    """The reader keyword-map value column: each member's ``(c=)`` value, in order."""
    return [m.get("annotations", {}).get("c") for m in enum["members"]]


def tomjt_members(enum: dict, enum_values: dict | None = None) -> list[tuple[str, str]]:
    """Ordered ``(member-name, C-constant)`` for the enum's ToMjt switch.

    The constant is the member's ``mjs_c`` override, else its reader ``c`` value.
    Members flagged ``no_mjs`` (reader-only) are excluded. When ``enum_values``
    (``{C-constant -> int}``) is given, cases are ordered by the underlying enum
    value -- the switch is emitted in enum-declaration order, which can diverge
    from the schema member order (the reader keyword order); pass None to keep
    schema order."""
    out = []
    for m in enum["members"]:
        a = m.get("annotations", {})
        if a.get("no_mjs"):
            continue
        const = a.get("mjs_c", a.get("c"))
        if const is None:
            continue
        out.append((m["name"], const))
    if enum_values is not None:
        def key(pair):
            const = pair[1]
            if is_int_literal(const):
                return int(const)
            return enum_values.get(const, 1 << 30)
        out.sort(key=key)
    return out


def element_struct(elem: dict) -> str:
    """The mjsStruct an element binds: explicit ``mjs=`` or the ``mjs<Name>`` identity."""
    return elem.get("annotations", {}).get("mjs") or "mjs" + elem["name"]


def field_mjs(field: dict):
    """The renamed destination mjs field (``mjs=``), or None (identity: same name)."""
    return field.get("annotations", {}).get("mjs")


def field_lenient(field: dict) -> bool:
    """Whether the attribute reads a short prefix of a fixed/scalar (``lenient``)."""
    return bool(field.get("annotations", {}).get("lenient"))


# --------------------------------------------------------------------------- #
# Kind classification + compatibility (shared by the emitter + validator).     #
# --------------------------------------------------------------------------- #
def ps_shape(t: dict) -> str:
    """The ProtoSpec storage shape of a field type."""
    k = t["kind"]
    if k == "variant":
        return "variant"
    if k == "ref":
        return "ref_list" if t.get("arity") is not None else "ref"
    if k == "named":  # enum
        return "enum_set" if t.get("arity") is not None else "enum_scalar"
    arity = t.get("arity")
    if t["prim"] == "string" and arity is None:
        return "string"
    if arity is None:
        return "scalar"
    return {"fixed": "fixed", "range": "range", "unbounded": "vec"}[arity["kind"]]


_VEC_KIND = {"double": "double_vec", "float": "float_vec", "int32": "int_vec",
             "uint64": "int_vec"}


def compatible(t: dict, m: dict, ctype_enums: set) -> bool:
    """True if ProtoSpec field type ``t`` may be written into mjs field ``m``.

    ``ctype_enums`` is the set of enum names authored to mjSpec (an enum-scalar is
    only writable when its enum has a ToMjt switch, i.e. a ``ctype=`` annotation)."""
    shape = ps_shape(t)
    mk = m["kind"]
    if shape == "scalar":
        return mk == "scalar"
    if shape == "enum_scalar":
        return mk == "scalar" and t["name"] in ctype_enums
    if shape == "string":
        return mk == "string"
    if shape == "ref":
        return mk == "string"
    if shape == "ref_list":
        return mk == "string_vec"
    if shape == "fixed":
        return mk == "fixed_array" and m["array"]["value"] == t["arity"]["size"]
    if shape == "range":
        return mk == "fixed_array" and t["arity"]["max"] <= m["array"]["value"]
    if shape == "vec":
        if mk == _VEC_KIND.get(t["prim"]):
            return True
        return t["prim"] == "double" and mk == "float_vec"
    return False  # enum_set, variant


# --------------------------------------------------------------------------- #
# Cross-validation.                                                            #
# --------------------------------------------------------------------------- #
def _ctx(filename: str) -> _SourceCtx:
    lines = []
    if os.path.exists(filename):
        with open(filename, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    return _SourceCtx(filename=filename, lines=lines)


def validate_enum_bindings(schema_json: dict, enum_index: dict, ctx: _SourceCtx) -> None:
    """(a) every ``(c=)``/``(mjs_c=)`` constant exists and belongs to the declared
    ctype where applicable; (b) a ctype'd enum annotates EVERY member."""
    for enum in schema_json["enums"]:
        ctype = enum_ctype(enum)
        for m in enum["members"]:
            a = m.get("annotations", {})
            if ctype is not None and not (
                "c" in a or "mjs_c" in a or a.get("no_mjs")
            ):
                raise ctx.error(
                    f"enum {enum['name']!r} is authored to mjSpec (ctype={ctype}) "
                    f"but member {m['name']!r} carries no c=/mjs_c=/no_mjs binding "
                    "(a ctype'd enum must annotate every member)",
                    m["line"], m["col"],
                )
            for key in ("c", "mjs_c"):
                if key not in a:
                    continue
                val = a[key]
                if is_int_literal(val):
                    continue  # a raw ordinal, not a named constant
                owner = enum_index.get(val)
                if owner is None:
                    raise ctx.error(
                        f"{key}={val}: no such MuJoCo constant "
                        "(not in snapshots/mjmodel_enums.json)",
                        m["line"], m["col"],
                    )
                if ctype is not None and ctype not in _PRIMITIVE_CTYPES and owner != ctype:
                    raise ctx.error(
                        f"{key}={val} belongs to enum {owner} but "
                        f"{enum['name']!r} declares ctype={ctype}",
                        m["line"], m["col"],
                    )


def validate_mjs_bindings(
    schema_json: dict, structs: dict, waivers, ctx: _SourceCtx
) -> None:
    """(c) every element/field ``(mjs=)`` resolves against the mjsStruct inventory
    with kind compatibility. ``waivers`` names elements with no per-instance struct
    (skipped)."""
    ctype_enums = ctype_enum_names(schema_json)
    for elem in schema_json["elements"]:
        name = elem["name"]
        if name in waivers:
            continue
        struct = element_struct(elem)
        if struct not in structs:
            raise ctx.error(
                f"element {name!r} binds mjs struct {struct!r}, which does not "
                "exist in snapshots/mjspec_fields.json",
                elem["line"], elem["col"],
            )
        sf = structs[struct]
        for f in elem["fields"]:
            mjs = field_mjs(f)
            if mjs is None:
                continue
            m = sf.get(mjs)
            if m is None:
                raise ctx.error(
                    f"{name}.{f['name']}: mjs={mjs} is not a field of {struct}",
                    f["line"], f["col"],
                )
            if not compatible(f["type"], m, ctype_enums):
                raise ctx.error(
                    f"{name}.{f['name']}: mjs={mjs} is kind-incompatible with "
                    f"{struct}.{mjs}",
                    f["line"], f["col"],
                )


def validate(schema_json: dict, *, waivers=frozenset(), filename: str = SCHEMA,
             enum_index: dict | None = None, structs: dict | None = None) -> None:
    """Run every binding cross-check; raise the first located error found."""
    ctx = _ctx(filename)
    if enum_index is None:
        enum_index = load_enum_index()
    if structs is None:
        structs = load_mjs_structs()
    validate_enum_bindings(schema_json, enum_index, ctx)
    validate_mjs_bindings(schema_json, structs, waivers, ctx)


__all__ = [
    "SchemaError", "load_enum_index", "load_mjs_structs", "is_int_literal",
    "enum_ctype", "enum_cmap", "ctype_enum_names", "keyword_values",
    "tomjt_members", "element_struct", "field_mjs", "field_lenient",
    "ps_shape", "compatible", "validate", "validate_enum_bindings",
    "validate_mjs_bindings",
]
