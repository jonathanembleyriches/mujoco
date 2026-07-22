"""Tests for the MJCF schema bootstrap extractor.

Unit tests run against small embedded fixtures and never touch the vendored MuJoCo
tree. One integration test exercises the real vendored source and is skipped when the
checkout is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from _mujoco_src import MUJOCO_SRC, SKIP_REASON

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "bootstrap"))

import extract_mjcf_schema as ex  # noqa: E402

# A realistic excerpt of the MJCF[] table: an element with a simple child, a complex
# child carrying its own bracketed block, and a doubly-nested block (plugin > config).
SCHEMA_FIXTURE = """
std::vector<const char*> MJCF[nMJCF] = {
{"mujoco", "!", "model"},
{"<"},
    {"size", "*", "memory", "njmax", "nkey"},

    {"default", "R", "class"},
    {"<"},
        {"mesh", "?", "scale", "inertia"},
    {">"},

    {"body", "R", "name", "pos", "quat"},
    {"<"},
        {"inertial", "?", "pos", "mass"},
        {"geom", "*", "name", "type", "size"},
        {"plugin", "*", "plugin", "instance"},
        {"<"},
          {"config", "*", "key", "value"},
        {">"},
    {">"},
{">"}
};
"""

MAP_FIXTURE = """
// geom type
const mjMap geom_map[mjNGEOMTYPES] = {
  {"plane",         mjGEOM_PLANE},
  {"sphere",        mjGEOM_SPHERE},
  {"mesh",          mjGEOM_MESH},
};


// interpolation type
const int interp_sz = 3;
const mjMap interp_map[interp_sz] = {
  {"zoh",           0},
  {"linear",        1},
  {"cubic",         2}
};

// this is a function parameter, not a table, and must be ignored:
// bool op(const char* attr, const mjMap* map, int mapsz) { ... }
"""


def test_schema_tree_shape():
    tree = ex.parse_schema_tree(SCHEMA_FIXTURE)
    assert tree.name == "mujoco"
    assert tree.typecode == "!"
    assert tree.attributes == ["model"]
    top = {c.name: c for c in tree.children}
    assert set(top) == {"size", "default", "body"}

    assert top["size"].typecode == "*"
    assert top["size"].attributes == ["memory", "njmax", "nkey"]
    assert top["size"].children == []

    assert top["default"].typecode == "R"
    assert [c.name for c in top["default"].children] == ["mesh"]


def test_schema_nested_block_reconstruction():
    tree = ex.parse_schema_tree(SCHEMA_FIXTURE)
    body = next(c for c in tree.children if c.name == "body")
    assert body.typecode == "R"
    assert [c.name for c in body.children] == ["inertial", "geom", "plugin"]

    plugin = next(c for c in body.children if c.name == "plugin")
    assert [c.name for c in plugin.children] == ["config"]
    assert plugin.children[0].attributes == ["key", "value"]


def test_keyword_maps_parse():
    maps = ex.parse_keyword_maps(MAP_FIXTURE)
    assert set(maps) == {"geom_map", "interp_map"}
    assert maps["geom_map"] == [
        {"key": "plane", "value": "mjGEOM_PLANE"},
        {"key": "sphere", "value": "mjGEOM_SPHERE"},
        {"key": "mesh", "value": "mjGEOM_MESH"},
    ]
    assert maps["interp_map"] == [
        {"key": "zoh", "value": "0"},
        {"key": "linear", "value": "1"},
        {"key": "cubic", "value": "2"},
    ]


def test_element_to_dict_roundtrip():
    tree = ex.parse_schema_tree(SCHEMA_FIXTURE)
    d = tree.to_dict()
    assert d["name"] == "mujoco"
    assert d["children"][0]["attributes"] == ["memory", "njmax", "nkey"]


def test_unbalanced_bracket_raises_with_line_number():
    # 'body' opens a child block whose inner '<' (line 8) is never closed before the
    # enclosing element ends.
    malformed = """
std::vector<const char*> MJCF[nMJCF] = {
{"mujoco", "!", "model"},
{"<"},
    {"body", "R", "name"},
    {"<"},
        {"geom", "*", "size"},
        {"<"},
            {"config", "*", "key"},
{">"}
};
"""
    with pytest.raises(ex.SchemaParseError) as excinfo:
        ex.parse_schema_tree(malformed)
    message = str(excinfo.value)
    assert "line 5" in message
    assert "body" in message


def test_unknown_typecode_raises_with_line_number():
    malformed = """
std::vector<const char*> MJCF[nMJCF] = {
{"mujoco", "!", "model"},
{"<"},
    {"body", "Z", "name"},
{">"}
};
"""
    with pytest.raises(ex.SchemaParseError) as excinfo:
        ex.parse_schema_tree(malformed)
    assert "line 5" in str(excinfo.value)
    assert "'Z'" in str(excinfo.value)


def test_empty_row_raises():
    malformed = """
std::vector<const char*> MJCF[nMJCF] = {
{"mujoco", "!", "model"},
{},
};
"""
    with pytest.raises(ex.SchemaParseError):
        ex.parse_schema_tree(malformed)


@pytest.mark.skipif(MUJOCO_SRC is None, reason=SKIP_REASON)
def test_integration_real_source():
    snapshot = ex.build_snapshot(MUJOCO_SRC)

    tree = ex._dict_to_element(snapshot["elements"])
    ex.run_sanity_checks(tree, snapshot["keyword_maps"])

    assert snapshot["source"]["file"] == "src/xml/xml_native_reader.cc"
    assert snapshot["source"]["mujoco_version"].count(".") == 2

    assert tree.name == "mujoco"
    assert ex._count_elements(tree) > 100
    for required in ("angle_map", "geom_map"):
        assert required in snapshot["keyword_maps"]

    assert "body_row_aliases" in snapshot["special_cases"]
    assert set(snapshot["special_cases"]["body_row_aliases"]["aliases"]) == {
        "worldbody",
        "frame",
        "replicate",
    }
