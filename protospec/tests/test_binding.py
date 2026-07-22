"""Tests for protospec_gen.binding: annotation accessors + cross-validation.

The binding facts moved from Python dict literals into schema annotations; this
module proves the accessors reconstruct the switches/columns and that the
cross-validator fires loudly (with a schema file:line) on every failure kind.
"""

import copy

import pytest

from protospec_gen import binding, emit_mjs
from protospec_gen.idl import SchemaError, parse_spec


def _schema():
    return parse_spec(binding.SCHEMA).to_json()


@pytest.fixture(scope="module")
def schema():
    return _schema()


@pytest.fixture(scope="module")
def waivers():
    return frozenset(emit_mjs.ELEMENT_WAIVERS)


# --------------------------------------------------------------------------- #
# Accessors.                                                                    #
# --------------------------------------------------------------------------- #
def test_tomjt_excludes_no_mjs_and_uses_mjs_c(schema):
    enums = {e["name"]: e for e in schema["enums"]}
    # GainType: so3 is (no_mjs) -> excluded from the switch; the rest reuse (c=).
    gain = dict(binding.tomjt_members(enums["GainType"]))
    assert "so3" not in gain
    assert gain["fixed"] == "mjGAIN_FIXED"
    # TriState backs no reader map; every member's constant is its (mjs_c=).
    tri = dict(binding.tomjt_members(enums["TriState"]))
    assert tri == {"false_": "mjLIMITED_FALSE", "true_": "mjLIMITED_TRUE",
                   "auto": "mjLIMITED_AUTO"}


def test_tomjt_orders_by_enum_value(schema):
    enums = {e["name"]: e for e in schema["enums"]}
    ev = binding.load_enum_values()
    # mjtMeshInertia's declaration order (convex, exact, legacy, shell) differs
    # from the schema/reader order (convex, legacy, exact, shell); the switch
    # follows enum-declaration order.
    ordered = [m for m, _ in binding.tomjt_members(enums["MeshInertia"], ev)]
    assert ordered == ["convex", "exact", "legacy", "shell"]


def test_keyword_values_are_c_annotations(schema):
    enums = {e["name"]: e for e in schema["enums"]}
    assert binding.keyword_values(enums["GainType"]) == [
        "mjGAIN_FIXED", "mjGAIN_AFFINE", "mjGAIN_MUSCLE", "mjGAIN_DCMOTOR",
        "mjGAIN_SO3", "mjGAIN_USER"]
    assert binding.keyword_values(enums["InterpType"]) == ["0", "1", "2"]


def test_element_struct_identity_and_explicit(schema):
    elems = {e["name"]: e for e in schema["elements"]}
    assert binding.element_struct(elems["Body"]) == "mjsBody"      # identity
    assert binding.element_struct(elems["Inertial"]) == "mjsBody"  # explicit
    assert binding.element_struct(elems["Hfield"]) == "mjsHField"  # explicit


# --------------------------------------------------------------------------- #
# Validation: passes on the real schema.                                        #
# --------------------------------------------------------------------------- #
def test_validate_passes(schema, waivers):
    binding.validate(schema, waivers=waivers)  # no raise


# --------------------------------------------------------------------------- #
# Validation: fires loudly, located at the schema line.                         #
# --------------------------------------------------------------------------- #
def _mutate(schema, fn):
    j = copy.deepcopy(schema)
    fn(j)
    return j


def test_unknown_c_constant_fires(schema, waivers):
    def m(j):
        e = next(x for x in j["enums"] if x["name"] == "GainType")
        e["members"][0]["annotations"]["c"] = "mjGAIN_NOPE"
    with pytest.raises(SchemaError, match=r"no such MuJoCo constant"):
        binding.validate(_mutate(schema, m), waivers=waivers)


def test_wrong_ctype_membership_fires(schema, waivers):
    def m(j):
        e = next(x for x in j["enums"] if x["name"] == "GainType")
        e["members"][0]["annotations"]["c"] = "mjBIAS_NONE"  # wrong enum
    with pytest.raises(SchemaError, match=r"belongs to enum mjtBias"):
        binding.validate(_mutate(schema, m), waivers=waivers)


def test_partial_ctyped_enum_fires(schema, waivers):
    def m(j):
        e = next(x for x in j["enums"] if x["name"] == "GainType")
        e["members"][1]["annotations"] = {}  # strip a member's binding
    with pytest.raises(SchemaError, match=r"must annotate every member"):
        binding.validate(_mutate(schema, m), waivers=waivers)


def test_bad_element_struct_fires(schema, waivers):
    def m(j):
        e = next(x for x in j["elements"] if x["name"] == "Body")
        e["annotations"] = {"mjs": "mjsNope"}
    with pytest.raises(SchemaError, match=r"does not exist in"):
        binding.validate(_mutate(schema, m), waivers=waivers)


def test_bad_field_mjs_fires(schema, waivers):
    def m(j):
        e = next(x for x in j["elements"] if x["name"] == "Inertial")
        f = next(x for x in e["fields"] if x["name"] == "pos")
        f["annotations"]["mjs"] = "ghost"
    with pytest.raises(SchemaError, match=r"not a field of mjsBody"):
        binding.validate(_mutate(schema, m), waivers=waivers)


# --------------------------------------------------------------------------- #
# Standing provenance guard: schema still reconstructs the frozen legacy tables. #
# --------------------------------------------------------------------------- #
def test_schema_reconstructs_legacy_tables():
    from tools import write_binding_annotations as W

    legacy = W._load_legacy()
    import json
    with open(W.FIELDS_JSON, encoding="utf-8") as fh:
        mjs_json = json.load(fh)
    W.reconstruct_check(_schema(), mjs_json, legacy)  # asserts internally
