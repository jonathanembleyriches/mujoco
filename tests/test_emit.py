"""Milestone-2 emitter tests: --check parity, file manifest, content spot-checks.

These guard the Python->C++ codegen contract without a compiler: the checked-in
``cpp/generated/`` must be byte-identical to a fresh emit, the expected file set
must be present, and a few load-bearing shapes (the Geom struct's fields, the
ActuatorAny union child-list type, presence wrapping, the field-count constant)
must appear in the output. The C++ side (compile, run, property tests) is driven
by cpp/CMakeLists.txt and cpp/test/test_model.cc.
"""

from __future__ import annotations

import os

from protospec_gen import emit, parse_spec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN_DIR = os.path.join(ROOT, "cpp", "generated")

EXPECTED_FILES = {
    "types.h",
    "types.cc",
    "visit.h",
    "reflect.h",
    "reflect.cc",
    "keywords.h",
    "keywords.cc",
    "defaults.h",
    "defaults.cc",
}


def _generated() -> dict[str, str]:
    return emit.generate()


def test_generate_produces_expected_manifest():
    files = _generated()
    assert set(files) == EXPECTED_FILES


def test_checked_in_matches_fresh_emit():
    """The checked-in cpp/generated/ is byte-identical to a fresh emit (the
    same invariant `python -m protospec_gen.emit --check` enforces)."""
    files = _generated()
    for name, content in files.items():
        path = os.path.join(GEN_DIR, name)
        assert os.path.exists(path), f"{name} is not checked in"
        with open(path, "r", encoding="utf-8", newline="") as fh:
            current = fh.read().replace("\r\n", "\n")
        assert current == content, (
            f"{name} is stale; re-run `python -m protospec_gen.emit`"
        )


def test_every_file_carries_the_banner():
    for content in _generated().values():
        assert content.splitlines()[0] == emit.BANNER


def test_geom_struct_contains_expected_fields():
    types_h = _generated()["types.h"]
    # Isolate the Geom struct definition.
    start = types_h.index("struct Geom {")
    body = types_h[start : types_h.index("\n};", start)]
    # Presence-tracked scalar, enum, ref, variant, arity, and child-list shapes.
    assert "ps::opt<GeomType> type" in body
    assert "ps::opt<ps::Ref<Material>> material" in body
    assert "ps::opt<ps::InlineVec<double, 3>> friction" in body  # range arity
    assert "ps::opt<GeomShape> shape" in body  # variant field
    assert "std::vector<std::unique_ptr<PluginRef>> plugin" in body  # child list
    assert "std::uint64_t serial" in body  # creation serial
    assert "ps::SourceLoc loc" in body  # provenance


def test_actuator_union_child_list_type_exists():
    types_h = _generated()["types.h"]
    assert "struct ActuatorAny {" in types_h
    # The container element carries the one ordered heterogeneous list.
    start = types_h.index("struct Actuator {")
    body = types_h[start : types_h.index("\n};", start)]
    assert "std::vector<ActuatorAny> actuators" in body


def test_field_count_constant_matches_ast():
    """The generated kFieldCount_<Element> constants equal the AST field counts;
    this is the constant the C++ reflection self-check pins against."""
    schema = parse_spec(os.path.join(ROOT, "schema", "mujoco.spec")).to_json()
    reflect_h = _generated()["reflect.h"]
    for e in schema["elements"]:
        n = len(e["fields"])
        assert f"kFieldCount_{e['name']} = {n};" in reflect_h


def test_keyword_table_sanitizes_reserved_enumerators():
    keywords_cc = _generated()["keywords.cc"]
    # `auto` is a C++ keyword: enumerator sanitized, MJCF string preserved.
    assert 'case TriState::auto_: return "auto";' in keywords_cc


def test_defaults_only_where_idl_has_them():
    defaults_cc = _generated()["defaults.cc"]
    start = defaults_cc.index("void ApplyDefault(Geom& e)")
    body = defaults_cc[start : defaults_cc.index("\n}", start)]
    assert "e.type = GeomType::sphere;" in body
    assert "e.contype = 1;" in body
    assert "e.mass" not in body  # no IDL default -> not applied
