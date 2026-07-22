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
from collections import namedtuple

from .idl import parse_spec

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCHEMA = os.path.join(ROOT, "schema", "mujoco.spec")
# src/xml/ lives at the vendored-MuJoCo checkout root (protospec/'s parent).
MUJOCO_ROOT = os.path.dirname(ROOT)
OUT_PATH = os.path.join(MUJOCO_ROOT, "src", "xml", "xml_native_schema.inc")
KEYWORD_OUT_PATH = os.path.join(MUJOCO_ROOT, "src", "xml", "xml_native_keywords.inc")
ATTRBIND_OUT_PATH = os.path.join(MUJOCO_ROOT, "src", "xml", "xml_native_attrbind.inc")

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


# ---------------------------------------------------------------------------
# Keyword-map generation (xml_native_keywords.inc).
#
# The reader carries ~48 ``const mjMap NAME[...] = {{"kw", VALUE}, ...}`` keyword
# tables. The keyword *spellings* and their *order* are schema facts (they are the
# ``value`` of an IDL enum's members); the VALUE column is a MuJoCo-side C constant
# (``mjGAIN_FIXED``) or a bare integer that the IDL does not model. Following the
# prior wave's rule -- "the schema owns content, the emitter owns the MuJoCo-side
# mapping" -- the C value column and the C array-size tokens live here, in an
# explicit table, extracted once from the hand-written reader. Each row is:
#
#   MapBind(map_name, enum_name, size_token, sz_const_expr_or_None, [value, ...])
#
# where ``size_token`` is the ``[...]`` array bound the reader uses (a literal, an
# ``mjN*`` macro, or a ``NAME_sz`` symbol), ``sz_const_expr`` is the value of the
# ``const int NAME_sz = <expr>;`` line the reader emits before the map (None when
# the reader used a literal/macro bound directly, so no such line is emitted), and
# the value list is positionally aligned with the enum's members.
#
# generate_keywords() derives every keyword from the IDL enum and asserts, per map,
# that the enum's member-value list has the same length as this value column; the
# equivalence test (tests/test_native_keyword_maps.py) then proves the generated
# {keyword, value} pairs equal the git-recovered hand tables. A schema enum that
# gains/loses a member, or is reordered, therefore fails loudly here at emit time.
#
# Some reader maps have no backing IDL enum because their keywords are type-system
# spellings, not enum members: bool_map / enable_map (boolean on/off with two
# spellings), TFAuto_map (the tri-state limited/auto primitive, reused across many
# attributes -- two enums, TriState and InertiaFromGeom, share its spelling, so it
# is NOT an enum but the primitive itself), meshtype_map (a boolean shellinertia
# flag whose two spellings map to inertia constants). These are emitted from
# PRIMITIVE_MAPS below -- the emitter owns both the keyword spellings and the value
# column, since the schema models them as primitives, not enums.
#
# Maps still left hand-written (do not fit either path), enumerated in the test and
# report: equality_map (the EqualityAny union supplies 7 of its 8 arms but the 8th,
# distance->mjEQ_DISTANCE, has no union member, so it is not cleanly union-derived),
# jkind_map / shape_map (single-entry / expression-keyword composite maps; composite
# is quarantined and promoting them to schema enums would ripple through the other
# generators for no behavioral gain).

MapBind = namedtuple("MapBind", "map enum size sz_expr values")

# Primitive-backed maps: no IDL enum, the (keyword, value) column is emitter-owned
# because the schema models these as boolean / tri-state primitives.
PrimMapBind = namedtuple("PrimMapBind", "map size sz_expr pairs")

PRIMITIVE_MAPS = [
    PrimMapBind('bool_map', '2', None, [('false', '0'), ('true', '1')]),
    PrimMapBind('enable_map', '2', None, [('disable', '0'), ('enable', '1')]),
    PrimMapBind('TFAuto_map', '3', None,
                [('false', '0'), ('true', '1'), ('auto', '2')]),
    PrimMapBind('meshtype_map', '2', None,
                [('false', 'mjINERTIA_VOLUME'), ('true', 'mjINERTIA_SHELL')]),
]

KEYWORD_MAPS = [
    MapBind('coordinate_map', 'Coordinate', '2', None, ['0', '1']),
    MapBind('angle_map', 'AngleUnit', '2', None, ['0', '1']),
    MapBind('fluid_map', 'FluidShape', '2', None, ['0', '1']),
    MapBind('FAuto_map', 'SimpleMode', '2', None, ['0', '1']),
    MapBind('bodysleep_map', 'BodySleep', 'bodysleep_sz', '4', ['mjSLEEP_AUTO', 'mjSLEEP_NEVER', 'mjSLEEP_ALLOWED', 'mjSLEEP_INIT']),
    MapBind('joint_map', 'JointType', 'joint_sz', '4', ['mjJNT_FREE', 'mjJNT_BALL', 'mjJNT_SLIDE', 'mjJNT_HINGE']),
    MapBind('geom_map', 'GeomType', 'mjNGEOMTYPES', None, ['mjGEOM_PLANE', 'mjGEOM_HFIELD', 'mjGEOM_SPHERE', 'mjGEOM_CAPSULE', 'mjGEOM_ELLIPSOID', 'mjGEOM_CYLINDER', 'mjGEOM_BOX', 'mjGEOM_MESH', 'mjGEOM_SDF']),
    MapBind('projection_map', 'CameraProjection', 'projection_sz', '2', ['mjPROJ_PERSPECTIVE', 'mjPROJ_ORTHOGRAPHIC']),
    MapBind('camlight_map', 'CamLightMode', 'camlight_sz', '5', ['mjCAMLIGHT_FIXED', 'mjCAMLIGHT_TRACK', 'mjCAMLIGHT_TRACKCOM', 'mjCAMLIGHT_TARGETBODY', 'mjCAMLIGHT_TARGETBODYCOM']),
    MapBind('lighttype_map', 'LightType', 'lighttype_sz', '4', ['mjLIGHT_SPOT', 'mjLIGHT_DIRECTIONAL', 'mjLIGHT_POINT', 'mjLIGHT_IMAGE']),
    MapBind('texrole_map', 'TexRole', 'texrole_sz', 'mjNTEXROLE - 1', ['mjTEXROLE_RGB', 'mjTEXROLE_OCCLUSION', 'mjTEXROLE_ROUGHNESS', 'mjTEXROLE_METALLIC', 'mjTEXROLE_NORMAL', 'mjTEXROLE_OPACITY', 'mjTEXROLE_EMISSIVE', 'mjTEXROLE_RGBA', 'mjTEXROLE_ORM']),
    MapBind('integrator_map', 'Integrator', 'integrator_sz', '4', ['mjINT_EULER', 'mjINT_RK4', 'mjINT_IMPLICIT', 'mjINT_IMPLICITFAST']),
    MapBind('cone_map', 'Cone', 'cone_sz', '2', ['mjCONE_PYRAMIDAL', 'mjCONE_ELLIPTIC']),
    MapBind('jac_map', 'JacobianType', 'jac_sz', '3', ['mjJAC_DENSE', 'mjJAC_SPARSE', 'mjJAC_AUTO']),
    MapBind('solver_map', 'SolverType', 'solver_sz', '3', ['mjSOL_PGS', 'mjSOL_CG', 'mjSOL_NEWTON']),
    MapBind('texture_map', 'TextureType', 'texture_sz', '3', ['mjTEXTURE_2D', 'mjTEXTURE_CUBE', 'mjTEXTURE_SKYBOX']),
    MapBind('colorspace_map', 'ColorSpace', 'colorspace_sz', '3', ['mjCOLORSPACE_AUTO', 'mjCOLORSPACE_LINEAR', 'mjCOLORSPACE_SRGB']),
    MapBind('builtin_map', 'TextureBuiltin', 'builtin_sz', '4', ['mjBUILTIN_NONE', 'mjBUILTIN_GRADIENT', 'mjBUILTIN_CHECKER', 'mjBUILTIN_FLAT']),
    MapBind('mark_map', 'TextureMark', 'mark_sz', '4', ['mjMARK_NONE', 'mjMARK_EDGE', 'mjMARK_CROSS', 'mjMARK_RANDOM']),
    MapBind('dyn_map', 'DynType', 'dyn_sz', '7', ['mjDYN_NONE', 'mjDYN_INTEGRATOR', 'mjDYN_FILTER', 'mjDYN_FILTEREXACT', 'mjDYN_MUSCLE', 'mjDYN_DCMOTOR', 'mjDYN_USER']),
    MapBind('dcmotorinput_map', 'DcMotorInput', 'dcmotorinput_sz', '3', ['0', '1', '2']),
    MapBind('gain_map', 'GainType', 'gain_sz', '6', ['mjGAIN_FIXED', 'mjGAIN_AFFINE', 'mjGAIN_MUSCLE', 'mjGAIN_DCMOTOR', 'mjGAIN_SO3', 'mjGAIN_USER']),
    MapBind('input_map', 'SO3Input', 'input_sz', '2', ['mjCHART_EXPMAP', 'mjCHART_QUAT']),
    MapBind('bias_map', 'BiasType', 'bias_sz', '6', ['mjBIAS_NONE', 'mjBIAS_AFFINE', 'mjBIAS_MUSCLE', 'mjBIAS_DCMOTOR', 'mjBIAS_SO3', 'mjBIAS_USER']),
    MapBind('interp_map', 'InterpType', 'interp_sz', '3', ['0', '1', '2']),
    MapBind('stage_map', 'NeedStage', 'stage_sz', '4', ['mjSTAGE_NONE', 'mjSTAGE_POS', 'mjSTAGE_VEL', 'mjSTAGE_ACC']),
    MapBind('datatype_map', 'DataType', 'datatype_sz', '4', ['mjDATATYPE_REAL', 'mjDATATYPE_POSITIVE', 'mjDATATYPE_AXIS', 'mjDATATYPE_QUATERNION']),
    MapBind('condata_map', 'ContactData', 'mjNCONDATA', None, ['mjCONDATA_FOUND', 'mjCONDATA_FORCE', 'mjCONDATA_TORQUE', 'mjCONDATA_DIST', 'mjCONDATA_POS', 'mjCONDATA_NORMAL', 'mjCONDATA_TANGENT']),
    MapBind('raydata_map', 'RayData', 'mjNRAYDATA', None, ['mjRAYDATA_DIST', 'mjRAYDATA_DIR', 'mjRAYDATA_ORIGIN', 'mjRAYDATA_POINT', 'mjRAYDATA_NORMAL', 'mjRAYDATA_DEPTH']),
    MapBind('camout_map', 'CameraOutput', 'mjNCAMOUT', 'mjNCAMOUT', ['mjCAMOUT_RGB', 'mjCAMOUT_DEPTH', 'mjCAMOUT_DIST', 'mjCAMOUT_NORMAL', 'mjCAMOUT_SEG']),
    MapBind('reduce_map', 'ContactReduce', 'reduce_sz', '4', ['0', '1', '2', '3']),
    MapBind('conflict_map', 'Conflict', 'conflict_sz', '3', ['mjCONFLICT_WARNING', 'mjCONFLICT_MERGE', 'mjCONFLICT_ERROR']),
    MapBind('lrmode_map', 'LRMode', 'lrmode_sz', '4', ['mjLRMODE_NONE', 'mjLRMODE_MUSCLE', 'mjLRMODE_MUSCLEUSER', 'mjLRMODE_ALL']),
    MapBind('comp_map', 'CompositeType', 'mjNCOMPTYPES', None, ['mjCOMPTYPE_PARTICLE', 'mjCOMPTYPE_GRID', 'mjCOMPTYPE_ROPE', 'mjCOMPTYPE_LOOP', 'mjCOMPTYPE_CABLE', 'mjCOMPTYPE_CLOTH']),
    MapBind('meshinertia_map', 'MeshInertia', '4', None, ['mjMESH_INERTIA_CONVEX', 'mjMESH_INERTIA_LEGACY', 'mjMESH_INERTIA_EXACT', 'mjMESH_INERTIA_SHELL']),
    MapBind('meshbuiltin_map', 'MeshBuiltin', 'meshbuiltin_sz', '8', ['mjMESH_BUILTIN_NONE', 'mjMESH_BUILTIN_SPHERE', 'mjMESH_BUILTIN_HEMISPHERE', 'mjMESH_BUILTIN_CONE', 'mjMESH_BUILTIN_SUPERTORUS', 'mjMESH_BUILTIN_SUPERSPHERE', 'mjMESH_BUILTIN_WEDGE', 'mjMESH_BUILTIN_PLATE']),
    MapBind('fcomp_map', 'FlexcompType', 'mjNFCOMPTYPES', None, ['mjFCOMPTYPE_GRID', 'mjFCOMPTYPE_BOX', 'mjFCOMPTYPE_CYLINDER', 'mjFCOMPTYPE_ELLIPSOID', 'mjFCOMPTYPE_SQUARE', 'mjFCOMPTYPE_DISC', 'mjFCOMPTYPE_CIRCLE', 'mjFCOMPTYPE_MESH', 'mjFCOMPTYPE_GMSH', 'mjFCOMPTYPE_DIRECT']),
    MapBind('fdof_map', 'FlexDof', 'mjNFCOMPDOFS', None, ['mjFCOMPDOF_FULL', 'mjFCOMPDOF_RADIAL', 'mjFCOMPDOF_TRILINEAR', 'mjFCOMPDOF_QUADRATIC', 'mjFCOMPDOF_2D']),
    MapBind('flexself_map', 'FlexSelfCollide', '5', None, ['mjFLEXSELF_NONE', 'mjFLEXSELF_NARROW', 'mjFLEXSELF_BVH', 'mjFLEXSELF_SAP', 'mjFLEXSELF_AUTO']),
    MapBind('elastic2d_map', 'Elastic2D', '5', None, ['0', '1', '2', '3']),
    MapBind('flexeq_map', 'FlexEquality', '4', None, ['0', '1', '2', '3']),
]

KEYWORD_HEADER = (
    "// Generated from protospec/schema/mujoco.spec by protospec_gen.emit_native"
    " -- do not edit.\n"
    "//\n"
    "// The reader's schema-backed `const mjMap NAME[]` keyword tables. Each map's\n"
    "// keyword spellings and order are an IDL enum's member values; the value\n"
    "// column (C constants / integers) and array-size tokens are the MuJoCo-side\n"
    "// mapping the emitter owns (KEYWORD_MAPS in emit_native.py). The trailing\n"
    "// primitive-backed maps (bool/enable/TFAuto/meshtype) come from PRIMITIVE_MAPS.\n"
    "// Maps still hand-written in xml_native_reader.cc (equality_map, jkind_map,\n"
    "// shape_map) are not cleanly enum/union/primitive-derivable -- see the header\n"
    "// there. Regenerate with: uv run python -m protospec_gen.emit_native --write\n"
)


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


# ---------------------------------------------------------------------------
# Attribute-binding generation (xml_native_attrbind.inc, Stage B).
#
# Each mjXReader::One<Elem> parser is a long run of mechanical
# ``ReadAttr(elem, "attr", arity, &field, text[, req, exact])`` /
# ``ReadAttrInt`` / ``MapValue`` / ``ReadAttrTxt``+``mjs_setString`` calls -- a
# straight attr->mjs-field copy. This emits, per converted element, a table of
# ``AttrBind`` rows ``{xml_name, kind, len, offsetof(mjsStruct, field), exact,
# map, mapsz}`` that a single generic ``ReadAttrBinds`` loop in the reader drives.
#
# Everything is derived from ground truth: the attr name + arity + enum come from
# the IDL element field (schema/mujoco.spec); the destination mjs field name comes
# from emit_mjs's alias tables (ELEM_ALIAS / GLOBAL_ALIAS, else same name); the C
# type (double vs float vs int) and array extent come from snapshots/
# mjspec_fields.json; the enum keyword map + size come from KEYWORD_MAPS above.
# ``exact`` is derived from the schema arity kind -- a fixed/scalar field reads
# exactly N values (exact=true), a range field reads a filled prefix (exact=false)
# -- which reproduces the reader's per-attr exact flag.
#
# A field is left hand-written (NOT emitted here) when it needs semantics beyond a
# copy: the ``name`` identity (mjs_setName + error), the ``class`` default ref
# (read by the caller, not the parser), orientation (quat/alt -> ReadQuat/
# ReadAlternative), unbounded vectors (user/output -> ReadVector/MapValues), bool
# flags (bool_map + ``(n==1)`` into an mjtBool field), variant "shape" groups
# (fromto/type inference), and the per-element interplay quirks named in
# ATTR_ELEMENTS. The generator asserts every emitted field resolves to a real mjs
# field of a compatible kind; the differential suite proves per-element identity.

# `include`, when non-empty, restricts generation to exactly those xml attrs (an
# allow-list, used when only a contiguous mechanical prefix of a heavily
# special-cased parser is converted, e.g. the actuator common attributes shared
# by every transmission/gain shortcut).
AttrElem = namedtuple("AttrElem", "elem struct quirks include")
AttrElem.__new__.__defaults__ = ((),)  # include defaults to empty

# Elements whose mechanical attribute reads are generated. `quirks` names fields
# kept hand-written beyond the automatic skips (name/class/orientation/unbounded/
# bool/variant), with the reason:
ATTR_ELEMENTS = [
    AttrElem("Site", "mjsSite", ()),               # fromto/type-inference ride the `shape` variant (auto-skipped)
    AttrElem("Camera", "mjsCamera", ("sensorsize", "fovy")),  # fovy/sensorsize are mutually exclusive (throw)
    AttrElem("Light", "mjsLight", ("type",)),      # `type` interplays with the `directional` bool alias (throw)
    AttrElem("Material", "mjsMaterial", ()),       # `texture`/`layer` fold into textures[] via mjs_setInStringVec
    AttrElem("Pair", "mjsPair", ("geom1", "geom2")),  # geom names are read only when !readingdefaults
    AttrElem("Geom", "mjsGeom", ("shellinertia", "fluidshape")),  # shellinertia->typeinertia via meshtype_map; fluidshape folds to (n==1)
    AttrElem("Joint", "mjsJoint", ("actuatorgravcomp",)),  # actgravcomp folds to (n==1)
    # Only the actuator common prefix (shared by every gain/transmission
    # shortcut). Transmission targets, slidercrank/refsite, the type dispatch and
    # every mjs_setToXxx shortcut stay hand-written.
    AttrElem("ActuatorGeneral", "mjsActuator", (), (
        "group", "nsample", "interp", "delay", "ctrllimited", "forcelimited",
        "actlimited", "ctrlrange", "forcerange", "actrange", "lengthrange",
        "gear", "damping", "armature",
    )),
    # Tendon: OneTendon is shared by spatial and fixed; Spatial is the superset.
    AttrElem("Spatial", "mjsTendon", ("springlength",)),  # springlength: 1-or-2 with copy-to-second logic
    # Equality: OneEquality is dominated by objtype/objname dispatch into data[];
    # only the common solref/solimp tail is a clean copy (active folds to (n==1)).
    AttrElem("Connect", "mjsEquality", (), ("solref", "solimp")),
    # Mesh: refpos/refquat/scale/inertia/material are plain copies. content_type
    # (direct std::string assign), file (vfs), smoothnormal (bool), maxhullvert
    # (validated), builtin (drives mjs_makeMesh, not a field) stay hand-written.
    AttrElem("Mesh", "mjsMesh",
             ("content_type", "file", "smoothnormal", "maxhullvert", "builtin")),
    # Sensor common prefix (identical across the whole sensor family; derived from
    # Touch as a representative). The per-type objtype/objname/reftype dispatch and
    # dataspec/intprm bitmask folding stay hand-written.
    AttrElem("Touch", "mjsSensor", (),
             ("cutoff", "noise", "nsample", "interp", "delay", "interval")),
    # Texture: plain copies incl. content_type (string field). gridlayout (memcpy
    # + length check) and the six cube-face files (fold into cubefiles[] via
    # mjs_setInStringVec) stay hand-written; builtin/file ride the `source` variant
    # and hflip/vflip are bool folds (all auto-skipped).
    AttrElem("Texture", "mjsTexture", (
        "gridlayout", "fileright", "fileleft", "fileup", "filedown",
        "filefront", "fileback",
    )),
    # Hfield: content_type/nrow/ncol are plain copies. size is a required read and
    # file/elevation need vfs / reverse-row handling, so they stay hand-written.
    AttrElem("Hfield", "mjsHField", ("file", "size")),
    # Body: only the unambiguous plain copies (pos, gravcomp, sleep). name, quat/
    # alt (orientation), mocap (bool), simple (byte-width enum), childclass (also
    # drives the childdef lookup) and user stay hand-written in the recursive Body
    # parser.
    AttrElem("Body", "mjsBody", (), ("pos", "gravcomp", "sleep")),
]

# Schema enums whose reader keyword map is a shared primitive/hand map, not their
# own KEYWORD_MAPS entry (so the enum path can still emit a kBindEnum row).
ENUM_PRIM_MAP = {
    "TriState": ("TFAuto_map", "3"),  # limited/actuatorfrclimited/align tri-state
}

# (element, xml) fields the reader reads leniently (exact=false) against the
# fixed/scalar -> exact-true default. The handful of fixed/scalar reads the reader
# accepts a short prefix of. (armature is scalar, so exact is behaviorally moot,
# but listed to keep the generated flag faithful to the hand read.)
LENIENT_EXACT_FALSE = {
    ("Geom", "surfacevel"),
    ("ActuatorGeneral", "armature"),
}

_FIELDS_JSON = os.path.join(ROOT, "snapshots", "mjspec_fields.json")

ATTRBIND_HEADER = (
    "// Generated from protospec/schema/mujoco.spec + snapshots/mjspec_fields.json"
    " by\n"
    "// protospec_gen.emit_native -- do not edit.\n"
    "//\n"
    "// Per-element AttrBind tables: the mechanical attr->mjs-field reads of each\n"
    "// One<Elem> parser, driven by mjXReader::ReadAttrBinds. Fields needing more\n"
    "// than a copy (name, class, orientation, vectors, bool flags, variant shape,\n"
    "// per-element interplay quirks) stay hand-written in xml_native_reader.cc.\n"
    "// Regenerate with: uv run python -m protospec_gen.emit_native --write\n"
)


def _attr_binds():
    """Resolve every ATTR_ELEMENTS element into (elem, struct, [bind-row, ...]).

    A bind-row is (xml_name, kind, length, field, exact, map, mapsz). Raises on any
    field that cannot be cleanly disposed (unknown mjs field, kind mismatch, missing
    enum map) -- that failure is the maintenance contract.
    """
    import json as _json

    doc = parse_spec(SCHEMA).to_json()
    elems = {e["name"]: e for e in doc["elements"]}
    enum_to_map = {b.enum: b for b in KEYWORD_MAPS}

    with open(_FIELDS_JSON, encoding="utf-8") as fh:
        fields_doc = _json.load(fh)
    structs = {s["name"]: {f["name"]: f for f in s["fields"]}
               for s in fields_doc["structs"]}

    # Imported lazily so emit_native stays importable without the mjSpec shim on
    # the path; both modules live in this package.
    from . import emit_mjs

    def field_xml(f):
        return f.get("annotations", {}).get("xml", f["name"])

    out = []
    for ae in ATTR_ELEMENTS:
        elem = elems[ae.elem]
        mjs = structs[ae.struct]
        rows = []
        for f in elem["fields"]:
            xml = field_xml(f)
            t = f["type"]
            kind = t["kind"]
            # --- allow-list (when set, only these attrs are generated) ---
            if ae.include and xml not in ae.include:
                continue
            # --- automatic skips (hand-written) ---
            if xml == "name":
                continue
            if kind == "ref" and t.get("target") == "Default":  # class
                continue
            if kind == "variant":
                continue
            if xml in ae.quirks:
                continue
            arity = t.get("arity")
            if arity is not None and arity.get("kind") == "unbounded":
                continue
            # orientation quat (resolver alias) is ReadQuat, hand-written
            if f.get("annotations", {}).get("resolver") == "orientation":
                continue

            # --- resolve destination mjs field ---
            # Element-specific alias wins; then an exact same-name mjs field (a
            # GLOBAL_ALIAS like solreffriction->solref_friction is joint/tendon
            # specific and must not shadow mjsPair.solreffriction); then the
            # global alias.
            mjs_name = emit_mjs.ELEM_ALIAS.get((ae.elem, f["name"]))
            if mjs_name is None:
                if f["name"] in mjs:
                    mjs_name = f["name"]
                else:
                    mjs_name = emit_mjs.GLOBAL_ALIAS.get(f["name"], f["name"])
            if mjs_name not in mjs:
                raise KeyError(
                    f"{ae.elem}.{xml}: mjs field {mjs_name!r} not in {ae.struct}")
            mf = mjs[mjs_name]
            ctype = mf["ctype"]

            # bool flags are read via bool_map + (n==1); leave hand-written
            if ctype == "mjtBool":
                continue

            # --- classify kind / length / exact ---
            if arity is None:
                length = 1
                exact = (ae.elem, xml) not in LENIENT_EXACT_FALSE
            elif arity["kind"] == "fixed":
                length = arity["size"]
                exact = (ae.elem, xml) not in LENIENT_EXACT_FALSE
            elif arity["kind"] == "range":
                length, exact = arity["max"], False
            else:
                raise ValueError(f"{ae.elem}.{xml}: unexpected arity {arity}")

            bmap, bmapsz = "nullptr", "0"
            if kind == "named":  # enum
                enum = t["name"]
                b = enum_to_map.get(enum)
                if b is not None:
                    bmap, bmapsz = b.map, b.size
                elif enum in ENUM_PRIM_MAP:
                    bmap, bmapsz = ENUM_PRIM_MAP[enum]
                else:
                    raise KeyError(
                        f"{ae.elem}.{xml}: enum {enum!r} has no keyword map")
                bkind, length, exact = "kBindEnum", 0, True
            elif ctype == "mjString":  # ref name, or a direct string field
                bkind, length, exact = "kBindString", 0, True
            elif ctype in ("double", "mjtNum"):
                bkind = "kBindDouble"
            elif ctype == "float":
                bkind = "kBindFloat"
            elif ctype == "int":
                bkind = "kBindInt" if arity is None else "kBindIntArr"
            else:
                raise TypeError(f"{ae.elem}.{xml}: unhandled ctype {ctype!r}")

            rows.append((xml, bkind, length, ae.struct, mjs_name, exact,
                         bmap, bmapsz))
        out.append((ae.elem, ae.struct, rows))
    return out


def generate_attrbinds(schema_path: str = SCHEMA) -> str:
    binds = _attr_binds()
    out = [ATTRBIND_HEADER.rstrip("\n"), "", "// clang-format off"]
    for elem, struct, rows in binds:
        var = f"k{elem}Binds"
        out.append("")
        out.append(f"const AttrBind {var}[] = {{")
        # column widths for readability
        wname = max(len(f'"{r[0]}",') for r in rows)
        wkind = max(len(r[1] + ",") for r in rows)
        for xml, bkind, length, st, field, exact, bmap, bmapsz in rows:
            off = f"offsetof({st}, {field})"
            name_col = ('"' + xml + '",').ljust(wname)
            kind_col = (bkind + ",").ljust(wkind)
            exact_col = "true" if exact else "false"
            row = ("  {" + name_col + " " + kind_col + " "
                   + f"{length}, {off}, {exact_col}, {bmap}, {bmapsz}" + "},")
            out.append(row)
        out.append("};")
        out.append(f"const int {var}N = {len(rows)};")
    out.append("// clang-format on")
    return "\n".join(out) + "\n"


def generate_keywords(schema_path: str = SCHEMA) -> str:
    """Emit xml_native_keywords.inc: the schema-backed mjMap keyword tables."""
    doc = parse_spec(schema_path).to_json()
    enums = {e["name"]: e for e in doc["enums"]}

    out: list[str] = [KEYWORD_HEADER.rstrip("\n"), "", "// clang-format off"]
    for b in KEYWORD_MAPS:
        enum = enums.get(b.enum)
        if enum is None:
            raise KeyError(f"{b.map}: schema enum {b.enum!r} not found")
        keys = [m["value"] for m in enum["members"]]
        if len(keys) != len(b.values):
            raise ValueError(
                f"{b.map}: schema enum {b.enum!r} has {len(keys)} members but the "
                f"value column has {len(b.values)} -- schema drift; update "
                f"KEYWORD_MAPS in emit_native.py"
            )
        out.append("")
        out.append(f"// {b.enum} -> {b.map}")
        if b.sz_expr is not None:
            # The reader's size-constant symbol follows the map name: NAME_map -> NAME_sz.
            sz_name = b.map[:-len("_map")] + "_sz" if b.map.endswith("_map") else b.map + "_sz"
            out.append(f"const int {sz_name} = {b.sz_expr};")
        keylits = [f'"{k}"' for k in keys]
        width = max(len(k) for k in keylits)
        out.append(f"const mjMap {b.map}[{b.size}] = {{")
        for k, v in zip(keylits, b.values):
            out.append(f"  {{{(k + ',').ljust(width + 1)} {v}}},")
        out.append("};")

    out.append("")
    out.append("// Primitive-backed maps (boolean / tri-state spellings; no IDL"
               " enum). The")
    out.append("// keyword spellings and value column are emitter-owned"
               " (PRIMITIVE_MAPS).")
    for pb in PRIMITIVE_MAPS:
        out.append("")
        out.append(f"// primitive -> {pb.map}")
        keylits = [f'"{k}"' for k, _ in pb.pairs]
        width = max(len(k) for k in keylits)
        out.append(f"const mjMap {pb.map}[{pb.size}] = {{")
        for (k, v), klit in zip(pb.pairs, keylits):
            out.append(f"  {{{(klit + ',').ljust(width + 1)} {v}}},")
        out.append("};")
    out.append("// clang-format on")
    return "\n".join(out) + "\n"


# (path, generator, out-of-date message, summary-line builder) for every emitted
# include -- --write regenerates all, --check byte-verifies all.
def _targets():
    return [
        (OUT_PATH, generate,
         "xml_native_schema.inc is out of date",
         lambda c: f"{c.split('#define nMJCF_GENERATED ', 1)[1].splitlines()[0]} rows"),
        (KEYWORD_OUT_PATH, generate_keywords,
         "xml_native_keywords.inc is out of date",
         lambda c: f"{len(KEYWORD_MAPS) + len(PRIMITIVE_MAPS)} maps"),
        (ATTRBIND_OUT_PATH, generate_attrbinds,
         "xml_native_attrbind.inc is out of date",
         lambda c: f"{c.count('const AttrBind ')} elements"),
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true",
                   help="verify the checked-in .inc files are up to date; do not write")
    g.add_argument("--write", action="store_true",
                   help="regenerate the .inc files (default)")
    args = ap.parse_args(argv)

    rc = 0
    for path, gen, stale_msg, summary in _targets():
        content = gen()
        if args.check:
            if not os.path.exists(path):
                sys.stderr.write(f"missing generated file: {path}\n")
                rc = 1
                continue
            with open(path, "r", encoding="utf-8", newline="") as fh:
                current = fh.read().replace("\r\n", "\n")
            if current != content:
                sys.stderr.write(
                    f"{stale_msg}; re-run "
                    "`python -m protospec_gen.emit_native --write`\n"
                )
                rc = 1
                continue
            print(f"{path} is up to date ({summary(content)})")
        else:
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(content)
            print(f"wrote {path} ({summary(content)})")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
