"""ProtoSpec Python-binding emitter: schema AST -> generated pybind11 glue.

Consumes the same canonical JSON AST the C++ emitters use
(:func:`protospec_gen.idl.parse_spec`) and emits the *generated* half of the
Python bindings into ``cpp/python/generated/``. The hand-written half lives in
``cpp/python/`` (``ps_bind.h`` casters + field helpers, ``module.cc`` the module
definition + sim surface + builders).

Generated files:

    py_enums.h        ``ps::py_gen::is_enum<T>`` trait specialised for every IDL
                      enum -- drives the one pybind ``type_caster`` that maps an
                      enum field to/from its MJCF keyword string (so a Python
                      user writes ``geom.type = "sphere"``, never an enum object).
    py_bind_gen.h     declarations of the chunked register functions.
    py_structs.cc     ``RegisterStructs`` -- the POD variant-arm structs
                      (Quat/Euler/FromTo/...) as pybind classes with read/write
                      fields, so ``body.orient = ps.Quat(); ...`` round-trips a
                      tagged variant.
    py_elements_<i>.cc  ``RegisterElements<i>`` -- every element class with its
                      fields exposed as Python properties (``opt<T>`` -> a
                      None-able property; required fields plain; the ``name``
                      field routes to ``NameField`` so an in-place rename warns),
                      and every child list as a sequence (union child lists
                      iterate yielding the typed member). Chunked for compile
                      time.
    py_builders.cc    ``RegisterModelBuilders`` + ``WrapElement`` -- the
                      schema-driven Model authoring verbs (``add_actuator`` /
                      ``add_sensor`` / ``add_tendon`` / ``add_equality`` union
                      families dispatched by MJCF keyword, plus ``add_material``
                      / ``add_mesh`` / ``add_pair`` / ``add_key`` / ... owned
                      families) and the ElementType -> typed-handle map the
                      ``duplicate`` verb returns a clone through. New union
                      members / owned families flow in at the next emit.

Every generated class registration ends with a call to
``ps::py::Augment(cls)`` -- a no-op by default, overloaded in ``module.cc`` for
the handful of elements (Body/Frame/Replicate/Model) that gain ergonomic
builders (``add_body``/``add_geom``/...). The generator never needs to know
about those; it just leaves the hook.

Entry point: imported and driven by :mod:`protospec_gen.emit` (so a single
``python -m protospec_gen.emit`` / ``--check`` covers both the C++ and the
Python generated trees). ``run(check=...)`` writes or verifies.
"""

from __future__ import annotations

import os
import sys

from .emit import BANNER, Schema, ident, stored_type, value_type
from .idl import parse_spec

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCHEMA = os.path.join(ROOT, "schema", "mujoco.spec")
OUT_DIR = os.path.join(ROOT, "lib", "python", "generated")

# How many py_elements_<i>.cc chunks to split the element registrations across
# (compile-time knob only; behaviour is identical for any value >= 1).
NUM_CHUNKS = 6

# Child lists the generic sequence accessor deliberately skips; the hand-written
# Augment hook exposes a friendlier surface instead. Keyed (element, child).
# Cross-ref: emit.py `_CHILD_XML_OVERRIDE` encodes the same (Model, worldbody) edge
# for the C++ XML tag -- keep the two in step.
_CHILD_SKIP = {("Model", "worldbody")}

# --------------------------------------------------------------------------- #
# Model authoring families (drive the generated builders in py_builders.cc)     #
# --------------------------------------------------------------------------- #
# These name WHICH families are authorable; the CONTENT of each builder (every
# member, every owned family, and -- via FinishChild -- every field) is derived
# from the schema, so a new upstream union member or asset family flows into the
# Python surface at the next `emit` with no hand edit. Adding a whole new family
# is the only thing that touches these lists (the same shape as _CHILD_SKIP).

# Union sections: Model gains `add_<verb>(kind, **kw)` dispatching over the
# union's members by their MJCF keyword. Each entry:
#   (python method, union type, SDK builder template, default kind or None).
# The SDK exposes AddActuator<T>/AddSensor<T>/AddTendon<T>/AddEquality<T>, each
# Ensure*-ing its section; the generated dispatch just picks the member type.
_UNION_FAMILIES = [
    ("add_actuator", "ActuatorAny", "AddActuator", "motor"),
    ("add_sensor", "SensorAny", "AddSensor", None),
    ("add_tendon", "TendonAny", "AddTendon", None),
    ("add_equality", "EqualityAny", "AddEquality", None),
]

# Owned-list sections: for each (Model child list, section element), Model gains
# an `add_<child>` for every owned child list the section carries. The builder
# calls AddOwnedChild(m.<list>, &<Section>::<member>) (ps_bind.h), so a family
# added to the section upstream appears automatically.
_OWNED_FAMILIES = [
    ("assets", "Asset"),
    ("contacts", "Contact"),
    ("keyframes", "Keyframe"),
    ("customs", "Custom"),
]


def _snake(camel: str) -> str:
    """CamelCase element name -> snake_case builder suffix (Material->material,
    ModelAsset->model_asset)."""
    out: list[str] = []
    for i, ch in enumerate(camel):
        if ch.isupper() and i and not camel[i - 1].isupper():
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


def _chunks(seq: list, n: int) -> list[list]:
    """Split `seq` into `n` roughly equal contiguous chunks."""
    n = max(1, min(n, len(seq)))
    size, rem = divmod(len(seq), n)
    out, start = [], 0
    for i in range(n):
        stop = start + size + (1 if i < rem else 0)
        out.append(seq[start:stop])
        start = stop
    return out


# --------------------------------------------------------------------------- #
# Field / child registration lines                                             #
# --------------------------------------------------------------------------- #
def _is_opt_string(f: dict) -> bool:
    t = f["type"]
    return f["optional"] and t.get("kind") == "prim" and t.get("prim") == "string"


def _field_line(elem_name: str, f: dict) -> str:
    """One property registration for a field of `elem_name`."""
    member = f"&{elem_name}::{ident(f['name'])}"
    prop = f['name']
    # The element name field routes to NameField, which warns on an in-place
    # rename (referrers are not rewritten by a bare `.name =`; Model.rename is).
    if prop == "name" and _is_opt_string(f):
        return f'  NameField(c, {member});'
    if f["optional"]:
        return f'  OptField(c, "{prop}", {member});'
    return f'  PlainField(c, "{prop}", {member});'


def _child_line(elem_name: str, c: dict) -> str | None:
    """One sequence accessor for a child list, or None if skipped."""
    if (elem_name, c["name"]) in _CHILD_SKIP:
        return None
    member = f"&{elem_name}::{ident(c['name'])}"
    if c.get("union") is not None:
        return f'  UnionList(c, "{c["name"]}", {member});'
    return f'  ChildList(c, "{c["name"]}", {member});'


# --------------------------------------------------------------------------- #
# Emitters                                                                     #
# --------------------------------------------------------------------------- #
def emit_py_enums_h(s: Schema) -> str:
    o: list[str] = []
    w = o.append
    w(BANNER)
    w("//")
    w("// Enum-identity trait: `is_enum<T>` is true exactly for the IDL enums,")
    w("// so a single pybind type_caster (ps_bind.h) can map every enum field")
    w("// to/from its MJCF keyword string. Nothing else references these types.")
    w("#ifndef PROTOSPEC_PYTHON_GENERATED_PY_ENUMS_H")
    w("#define PROTOSPEC_PYTHON_GENERATED_PY_ENUMS_H")
    w("")
    w("#include <type_traits>")
    w("")
    w('#include "types.h"')
    w("")
    w("namespace ps::py_gen {")
    w("")
    w("template <class T>")
    w("struct is_enum : std::false_type {};")
    w("")
    for e in s.enums:
        w(f"template <> struct is_enum<ps::mjcf::{e['name']}> : std::true_type {{}};")
    w("")
    w("template <class T>")
    w("inline constexpr bool is_enum_v = is_enum<T>::value;")
    w("")
    w("}  // namespace ps::py_gen")
    w("")
    w("#endif  // PROTOSPEC_PYTHON_GENERATED_PY_ENUMS_H")
    return "\n".join(o) + "\n"


def emit_py_nocopy_h(s: Schema) -> str:
    o: list[str] = []
    w = o.append
    w(BANNER)
    w("//")
    w("// Element types are entities (stable identity via `serial`, DR-10); they")
    w("// are never copied by value -- Python always holds them by reference. But")
    w("// an element that owns a `vector<unique_ptr<Child>>` still reports as")
    w("// copy-constructible to `std::is_copy_constructible` (std::vector declares")
    w("// a copy ctor that is only ill-formed on instantiation), and pybind11's")
    w("// container-recursion workaround does not see through a struct member, so")
    w("// pybind would try to synthesize a copy ctor and fail to compile. Marking")
    w("// every element non-copyable to pybind stops that; struct (variant-arm)")
    w("// value types stay copyable.")
    w("#ifndef PROTOSPEC_PYTHON_GENERATED_PY_NOCOPY_H")
    w("#define PROTOSPEC_PYTHON_GENERATED_PY_NOCOPY_H")
    w("")
    w("#include <type_traits>")
    w("")
    w("#include <pybind11/detail/type_caster_base.h>")
    w("")
    w('#include "types.h"')
    w("")
    w("namespace pybind11 {")
    w("namespace detail {")
    w("")
    for e in s.elements:
        t = f"ps::mjcf::{e['name']}"
        w(f"template <> struct is_copy_constructible<{t}> : std::false_type {{}};")
        w(f"template <> struct is_copy_assignable<{t}> : std::false_type {{}};")
    w("")
    w("}  // namespace detail")
    w("}  // namespace pybind11")
    w("")
    w("#endif  // PROTOSPEC_PYTHON_GENERATED_PY_NOCOPY_H")
    return "\n".join(o) + "\n"


def emit_py_bind_gen_h(s: Schema, chunk_count: int) -> str:
    o: list[str] = []
    w = o.append
    w(BANNER)
    w("//")
    w("// Declarations of the chunked register functions the module init calls.")
    w("#ifndef PROTOSPEC_PYTHON_GENERATED_PY_BIND_GEN_H")
    w("#define PROTOSPEC_PYTHON_GENERATED_PY_BIND_GEN_H")
    w("")
    w("#include <pybind11/pybind11.h>")
    w("")
    w("namespace ps::py {")
    w("")
    w("void RegisterStructs(pybind11::module_& m);")
    for i in range(chunk_count):
        w(f"void RegisterElements{i}(pybind11::module_& m);")
    w("")
    w("// Registers the POD structs then every element class (all chunks).")
    w("inline void RegisterGenerated(pybind11::module_& m) {")
    w("  RegisterStructs(m);")
    for i in range(chunk_count):
        w(f"  RegisterElements{i}(m);")
    w("}")
    w("")
    w("}  // namespace ps::py")
    w("")
    w("#endif  // PROTOSPEC_PYTHON_GENERATED_PY_BIND_GEN_H")
    return "\n".join(o) + "\n"


def emit_py_structs_cc(s: Schema) -> str:
    o: list[str] = []
    w = o.append
    w(BANNER)
    w('#include "ps_bind.h"')
    w("")
    w("namespace ps::py {")
    w("")
    w("void RegisterStructs(pybind11::module_& m) {")
    for st in s.structs:
        name = st["name"]
        w(f'  {{ pyb::class_<{name}> c(m, "{name}");')
        w("    c.def(pyb::init<>());")
        for f in st["fields"]:
            # Struct (POD) fields are plain values; a read/write property whose
            # getter/setter go through the field casters (array/InlineVec/enum).
            w(f'    StructField(c, "{f["name"]}", &{name}::{ident(f["name"])});')
        w("  }")
    w("}")
    w("")
    w("}  // namespace ps::py")
    return "\n".join(o) + "\n"


def emit_py_elements_cc(s: Schema, chunk_index: int, elements: list[dict]) -> str:
    o: list[str] = []
    w = o.append
    w(BANNER)
    w('#include "ps_bind.h"')
    w("")
    w("namespace ps::py {")
    w("")
    w(f"void RegisterElements{chunk_index}(pybind11::module_& m) {{")
    for e in elements:
        name = e["name"]
        w(f'  {{ pyb::class_<{name}> c(m, "{name}");')
        w("    c.def(pyb::init<>());")
        w("    InitKwargs(c);  // ps.Elem(name=..., field=...) keyword ctor")
        w("    ElementBase(c);")
        for f in e["fields"]:
            w("  " + _field_line(name, f))
        for c in e["children"]:
            line = _child_line(name, c)
            if line is not None:
                w("  " + line)
        w("    Augment(c);")
        w("  }")
    w("}")
    w("")
    w("}  // namespace ps::py")
    return "\n".join(o) + "\n"


def _union_dispatcher(s: Schema, method: str, union: str, builder: str) -> list[str]:
    """The `Add<Family>ByKind` helper: MJCF keyword -> typed SDK builder call."""
    o: list[str] = []
    w = o.append
    members = s.union_by_name[union]["members"]
    fn = "".join(part.capitalize() for part in method.split("_")) + "ByKind"
    w(f"pyb::object {fn}(pyb::object self, const std::string& kind,")
    w("                 const pyb::kwargs& kw) {")
    w("  Model& m = self.cast<Model&>();")
    kinds = []
    for member in members:
        tag = s.elem_xml(s.element_by_name[member])
        kinds.append(tag)
        w(f'  if (kind == "{tag}")')
        w(f"    return FinishChild(self, ps::sdk::{builder}<{member}>(m), kw);")
    listing = "/".join(kinds)
    w(f'  throw pyb::value_error("unknown {method[4:]} kind \'" + kind +')
    w(f'                         "\' ({listing})");')
    w("}")
    w("")
    return o


def emit_py_builders_cc(s: Schema) -> str:
    """RegisterModelBuilders + WrapElement -- the schema-driven Model authoring
    surface (union-family + owned-list add_* builders) and the ElementType ->
    typed-handle wrapper the duplicate() verb returns through."""
    o: list[str] = []
    w = o.append
    w(BANNER)
    w("//")
    w("// The schema-driven Model builders. Union families (actuator/sensor/")
    w("// tendon/equality) dispatch by MJCF keyword to the typed SDK builder;")
    w("// owned families (asset/contact/keyframe/custom) get one add_* per owned")
    w("// child list. WrapElement re-types a duplicate() clone. Everything here")
    w("// is derived from the schema, so new members/families need no hand edit.")
    w('#include "ps_bind.h"')
    w("")
    w('#include "protospec/sdk.h"')
    w("")
    w("namespace ps::py {")
    w("")

    # --- WrapElement: runtime ElementType -> typed handle ------------------- #
    w("pyb::object WrapElement(int etype, void* p, pyb::object parent) {")
    w("  switch (static_cast<ElementType>(etype)) {")
    for e in s.elements:
        name = e["name"]
        w(f"    case ElementType::{name}:")
        w(f"      return pyb::cast(static_cast<{name}*>(p),")
        w("                       pyb::return_value_policy::reference_internal,"
          " parent);")
    w("    default:")
    w("      return pyb::none();")
    w("  }")
    w("}")
    w("")

    # --- Union-family dispatchers ------------------------------------------- #
    for method, union, builder, _default in _UNION_FAMILIES:
        o.extend(_union_dispatcher(s, method, union, builder))

    # --- RegisterModelBuilders --------------------------------------------- #
    w("void RegisterModelBuilders(pyb::class_<Model>& c) {")
    for method, union, _builder, default in _UNION_FAMILIES:
        fn = "".join(part.capitalize() for part in method.split("_")) + "ByKind"
        family = method[4:]
        w(f'  c.def(')
        w(f'      "{method}",')
        w("      [](pyb::object self, const std::string& kind, pyb::kwargs kw) {")
        w(f"        return {fn}(self, kind, kw);")
        w("      },")
        if default is not None:
            w(f'      pyb::arg("kind") = "{default}",')
        else:
            w('      pyb::arg("kind"),')
        w(f'      "Append a {family} of the given MJCF keyword; fields as '
          'keywords.");')
    # Owned-list families.
    for model_list, section in _OWNED_FAMILIES:
        sec = s.element_by_name[section]
        for child in sec["children"]:
            child_elem = child.get("element")
            if child_elem is None:  # skip any union child on a section
                continue
            suffix = _snake(child_elem)
            member = ident(child["name"])
            w(f'  c.def(')
            w(f'      "add_{suffix}",')
            w("      [](pyb::object self, pyb::kwargs kw) {")
            w("        Model& m = self.cast<Model&>();")
            w(f"        return FinishChild(")
            w(f"            self, AddOwnedChild(m.{ident(model_list)}, "
              f"&{section}::{member}), kw);")
            w("      },")
            w(f'      "Append a <{s.elem_xml(s.element_by_name[child_elem])}> '
              f'to the model {section.lower()} section.");')
    w("}")
    w("")
    w("}  // namespace ps::py")
    return "\n".join(o) + "\n"


# --------------------------------------------------------------------------- #
# Driver                                                                        #
# --------------------------------------------------------------------------- #
def generate_files(schema_path: str = SCHEMA) -> dict[str, str]:
    """Return {filename: content} for every generated Python-binding file."""
    s = Schema(parse_spec(schema_path).to_json())
    chunks = _chunks(s.elements, NUM_CHUNKS)
    files = {
        "py_enums.h": emit_py_enums_h(s),
        "py_nocopy.h": emit_py_nocopy_h(s),
        "py_bind_gen.h": emit_py_bind_gen_h(s, len(chunks)),
        "py_structs.cc": emit_py_structs_cc(s),
        "py_builders.cc": emit_py_builders_cc(s),
    }
    for i, elems in enumerate(chunks):
        files[f"py_elements_{i}.cc"] = emit_py_elements_cc(s, i, elems)
    return files


def run(check: bool = False, out_dir: str = OUT_DIR) -> list[str]:
    """Write (or, when `check`, verify) the generated files.

    Returns the list of stale/missing filenames (empty on success). Never exits;
    the caller (emit.main) aggregates and decides the process status.
    """
    files = generate_files()
    if check:
        stale = []
        for name, content in files.items():
            path = os.path.join(out_dir, name)
            if not os.path.exists(path):
                stale.append(f"{name} (missing)")
                continue
            with open(path, "r", encoding="utf-8", newline="") as fh:
                current = fh.read()
            if current.replace("\r\n", "\n") != content:
                stale.append(name)
        return stale

    os.makedirs(out_dir, exist_ok=True)
    for name, content in files.items():
        path = os.path.join(out_dir, name)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(content)
    return []


def main() -> None:
    check = "--check" in sys.argv[1:]
    stale = run(check=check)
    if check:
        if stale:
            sys.stderr.write(
                "cpp/python/generated/ is out of date; re-run "
                "`python -m protospec_gen.emit`:\n  " + "\n  ".join(stale) + "\n"
            )
            sys.exit(1)
        print("cpp/python/generated/ is up to date")
    else:
        print(f"wrote {len(generate_files())} files to {OUT_DIR}")


if __name__ == "__main__":
    main()
