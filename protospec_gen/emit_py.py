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
                      None-able property; required fields plain), and every child
                      list as a sequence (union child lists iterate yielding the
                      typed member). Chunked purely for compile time.

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
OUT_DIR = os.path.join(ROOT, "cpp", "python", "generated")

# How many py_elements_<i>.cc chunks to split the element registrations across
# (compile-time knob only; behaviour is identical for any value >= 1).
NUM_CHUNKS = 6

# Child lists the generic sequence accessor deliberately skips; the hand-written
# Augment hook exposes a friendlier surface instead. Keyed (element, child).
# Cross-ref: emit.py `_CHILD_XML_OVERRIDE` encodes the same (Model, worldbody) edge
# for the C++ XML tag -- keep the two in step.
_CHILD_SKIP = {("Model", "worldbody")}


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
def _field_line(elem_name: str, f: dict) -> str:
    """One property registration for a field of `elem_name`."""
    member = f"&{elem_name}::{ident(f['name'])}"
    prop = f['name']
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
