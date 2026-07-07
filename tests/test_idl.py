"""Tests for the ProtoSpec IDL parser (protospec_gen.idl)."""

import json
from pathlib import Path

import pytest

from protospec_gen import SchemaError, parse_spec
from protospec_gen.idl import Schema

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> Schema:
    return parse_spec(FIXTURES / f"{name}.spec")


# --------------------------------------------------------------------------- #
# Golden: full AST of the plan Section 5 example                               #
# --------------------------------------------------------------------------- #
def test_section5_golden_json():
    schema = _load("section5")
    expected = json.loads((FIXTURES / "section5_expected.json").read_text("utf-8"))
    assert schema.to_json() == expected


def test_section5_shape_spotchecks():
    j = _load("section5").to_json()
    assert j["mujoco_version"] == "3.10.0"
    # Top-level lists are sorted by name (order is not semantic).
    assert [e["name"] for e in j["elements"]] == [
        "Body",
        "Default",
        "Frame",
        "Geom",
        "Joint",
        "Material",
        "Mesh",
    ]
    geom = next(e for e in j["elements"] if e["name"] == "Geom")
    # Mixin fields come first (source order preserved), then local fields.
    assert [f["name"] for f in geom["fields"][:3]] == ["pos", "orient", "name"]
    assert geom["fields"][0]["source_mixin"] == "Posed"
    assert geom["uses"] == ["Posed"]
    # `dclass` carries an xml-divergence annotation; default tags are not stored.
    dclass = next(f for f in geom["fields"] if f["name"] == "dclass")
    assert dclass["annotations"] == {"xml": "class"}
    assert dclass["type"] == {
        "kind": "ref",
        "target": "Default",
        "line": dclass["type"]["line"],
        "col": dclass["type"]["col"],
    }


# --------------------------------------------------------------------------- #
# Round-trip                                                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "name", ["section5", "annotations", "cardinality", "arity", "unions"]
)
def test_json_roundtrip_fixpoint(name):
    schema = _load(name)
    first = schema.to_json()
    second = Schema.from_json(first).to_json()
    assert first == second
    # A second cycle is still identical.
    third = Schema.from_json(second).to_json()
    assert third == first


def test_json_str_is_deterministic():
    schema = _load("section5")
    assert schema.to_json_str() == schema.to_json_str()


def test_parse_from_text_and_path_agree():
    text = (FIXTURES / "section5.spec").read_text("utf-8")
    from_text = parse_spec(text).to_json()
    from_path = parse_spec(FIXTURES / "section5.spec").to_json()
    # line/col are identical; only the reported filename differs.
    assert from_text == from_path


# --------------------------------------------------------------------------- #
# Mixin flattening                                                             #
# --------------------------------------------------------------------------- #
def test_mixin_flattening_order_and_provenance():
    schema = parse_spec(
        """
        mujoco_version "3.10.0"
        mixin A { a1 : int32  a2 : int32 }
        mixin B { b1 : int32 }
        element E {
          use A
          use B
          local : int32
        }
        """
    )
    elem = next(e for e in schema.elements if e.name == "E")
    assert [f.name for f in elem.fields] == ["a1", "a2", "b1", "local"]
    assert [f.source_mixin for f in elem.fields] == ["A", "A", "B", None]


def test_use_of_undefined_mixin():
    exc = _expect_error(
        """
        mujoco_version "3.10.0"
        element E { use Nope  x : int32 }
        """
    )
    assert "undefined mixin 'Nope'" in exc.message


def test_use_of_non_mixin():
    exc = _expect_error(
        """
        mujoco_version "3.10.0"
        struct S { x : int32 }
        element E { use S }
        """
    )
    assert "not a mixin" in exc.message


# --------------------------------------------------------------------------- #
# Arity forms                                                                  #
# --------------------------------------------------------------------------- #
def test_arity_forms():
    j = _load("arity").to_json()
    fields = {f["name"]: f for f in j["elements"][0]["fields"]}
    assert fields["fixed3"]["type"]["arity"] == {"kind": "fixed", "size": 3}
    assert fields["range03"]["type"]["arity"] == {"kind": "range", "min": 0, "max": 3}
    assert fields["range13"]["type"]["arity"] == {"kind": "range", "min": 1, "max": 3}
    assert fields["unbounded"]["type"]["arity"] == {"kind": "unbounded"}
    assert "arity" not in fields["scalar_d"]["type"]


def test_defaults_kinds():
    j = _load("arity").to_json()
    fields = {f["name"]: f for f in j["elements"][0]["fields"]}
    assert fields["color"]["default"] == {
        "kind": "array",
        "values": [0.5, 0.5, 0.5, 1],
        "line": fields["color"]["default"]["line"],
        "col": fields["color"]["default"]["col"],
    }
    assert fields["neg"]["default"]["value"] == -1.5
    assert fields["count"]["default"]["value"] == 7
    assert fields["on"]["default"]["value"] is True
    assert fields["mode"]["default"]["kind"] == "enum"
    assert fields["mode"]["default"]["member"] == "a"


# --------------------------------------------------------------------------- #
# Annotations                                                                  #
# --------------------------------------------------------------------------- #
def test_all_annotation_kinds():
    j = _load("annotations").to_json()
    body2 = next(e for e in j["elements"] if e["name"] == "Body2")
    assert body2["annotations"] == {"xml": "body"}
    thing = next(e for e in j["elements"] if e["name"] == "Thing")
    fields = {f["name"]: f for f in thing["fields"]}
    assert fields["name"]["optional"] is False  # required -> optional False
    assert "annotations" not in fields["name"]  # required is folded into optional
    assert fields["angle"]["annotations"] == {"unit": "angle"}
    assert fields["tag"]["annotations"] == {"xml": "class"}
    assert fields["orient"]["annotations"] == {
        "variant_group": "orient",
        "variant_tag": "quat",
    }
    assert fields["text"]["annotations"] == {"element_text": True}


def test_optional_default_true():
    schema = parse_spec(
        'mujoco_version "3.10.0"\nelement E { x : int32  y : int32 (required) }'
    )
    fields = {f.name: f for f in schema.elements[0].fields}
    assert fields["x"].optional is True
    assert fields["y"].optional is False


# --------------------------------------------------------------------------- #
# Cardinality                                                                  #
# --------------------------------------------------------------------------- #
def test_cardinality_forms():
    j = _load("cardinality").to_json()
    root = next(e for e in j["elements"] if e["name"] == "Root")
    cards = {c["name"]: c["cardinality"] for c in root["children"]}
    assert cards == {
        "many": "zero_or_more",
        "optional": "zero_or_one",
        "exactly": "one",
    }


# --------------------------------------------------------------------------- #
# Comments as docs                                                             #
# --------------------------------------------------------------------------- #
def test_trailing_comment_becomes_doc():
    schema = parse_spec(
        'mujoco_version "3.10.0"\nelement E {\n  x : int32  # the x value\n}'
    )
    assert schema.elements[0].fields[0].doc == "the x value"


def test_full_line_comment_is_not_a_doc():
    schema = parse_spec(
        'mujoco_version "3.10.0"\nelement E {\n  # standalone\n  x : int32\n}'
    )
    assert schema.elements[0].fields[0].doc is None


def test_hash_inside_string_is_not_a_comment():
    schema = parse_spec('mujoco_version "3.10.0"\nelement E (xml="a#b") { x : int32 }')
    assert schema.elements[0].annotations == {"xml": "a#b"}


# --------------------------------------------------------------------------- #
# Variant definition form                                                      #
# --------------------------------------------------------------------------- #
def test_variant_definition():
    schema = _load("section5")
    orient = next(v for v in schema.variants if v.name == "Orientation")
    assert [a.tag for a in orient.arms] == ["quat", "euler", "axisangle"]
    assert orient.arms[0].type.name == "Quat"
    assert orient.arms[0].type.category == "struct"
    assert orient.arms[0].doc == "unit quaternion (canonical form)"


# --------------------------------------------------------------------------- #
# Union definition form                                                        #
# --------------------------------------------------------------------------- #
def test_union_definition_and_positions():
    j = _load("unions").to_json()
    # Members are element names in source order.
    u = next(u for u in j["unions"] if u["name"] == "ActuatorAny")
    assert u["members"] == ["Motor", "Position", "Plugin"]
    # A union child list carries `union` (heterogeneous, ordered), not `element`.
    actuator = next(e for e in j["elements"] if e["name"] == "Actuator")
    child = actuator["children"][0]
    assert child["union"] == "ActuatorAny"
    assert "element" not in child
    assert child["cardinality"] == "zero_or_more"
    # ref<UnionName> resolves to a plain ref whose target is the union name.
    trans = next(e for e in j["elements"] if e["name"] == "Transmission")
    target = next(f for f in trans["fields"] if f["name"] == "target")
    assert target["type"]["kind"] == "ref"
    assert target["type"]["target"] == "ActuatorAny"


def test_union_members_serialize_as_bare_names():
    # from_json restores a union with the same members (fixpoint already covers
    # this; this pins the flat member-name shape explicitly).
    u = next(u for u in _load("unions").unions if u.name == "ActuatorAny")
    assert [m.name for m in u.members] == ["Motor", "Position", "Plugin"]


def test_union_child_list_and_element_child_list_coexist():
    schema = parse_spec(
        'mujoco_version "3.10.0"\n'
        "element A { name : string }\n"
        "element B { name : string }\n"
        "union U = A | B\n"
        "element E {\n"
        "  children plain : A *\n"
        "  children mixed : U *\n"
        "}"
    )
    e = next(el for el in schema.elements if el.name == "E")
    plain, mixed = e.children
    assert plain.element == "A" and plain.union is None
    assert mixed.union == "U" and mixed.element is None


# --------------------------------------------------------------------------- #
# Negative suite (each asserts message + line/col)                             #
# --------------------------------------------------------------------------- #
def _expect_error(text: str, *, filename=None) -> SchemaError:
    with pytest.raises(SchemaError) as ei:
        parse_spec(text, filename=filename)
    return ei.value


def test_error_missing_header():
    exc = _expect_error('enum E { a = "a" }')
    assert "mujoco_version" in exc.message
    assert (exc.line, exc.col) == (1, 1)


def test_error_empty_schema():
    exc = _expect_error("\n\n")
    assert "empty schema" in exc.message


def test_error_unbalanced_brace():
    # Element opened on line 2, never closed.
    exc = _expect_error('mujoco_version "3.10.0"\nelement E {\n  x : int32\n')
    assert "unbalanced" in exc.message and "}" in exc.message
    assert exc.line == 2
    assert exc.col == 11  # the '{'


def test_error_unknown_type():
    exc = _expect_error(
        'mujoco_version "3.10.0"\nelement E {\n  f : Nope\n}'
    )
    assert exc.message == "unknown type 'Nope'"
    assert (exc.line, exc.col) == (3, 7)


def test_error_duplicate_field_via_mixin():
    exc = _expect_error(
        'mujoco_version "3.10.0"\n'
        "mixin M { dup : int32 }\n"
        "element E {\n"
        "  use M\n"
        "  dup : int32\n"
        "}"
    )
    assert "duplicate field 'dup'" in exc.message
    assert "mixin 'M'" in exc.message
    assert exc.line == 5  # the offending local field
    assert exc.col == 3


def test_error_duplicate_field_between_mixins():
    exc = _expect_error(
        'mujoco_version "3.10.0"\n'
        "mixin A { dup : int32 }\n"
        "mixin B { dup : int32 }\n"
        "element E {\n"
        "  use A\n"
        "  use B\n"
        "}"
    )
    assert "duplicate field 'dup'" in exc.message
    assert exc.line == 6  # the 'use B' that re-injects it


def test_error_bad_annotation_key():
    exc = _expect_error(
        'mujoco_version "3.10.0"\nelement E {\n  f : int32 (bogus=1)\n}'
    )
    assert "unknown annotation key 'bogus'" in exc.message
    assert exc.line == 3
    assert exc.col == 14


def test_error_ref_to_non_element():
    exc = _expect_error(
        'mujoco_version "3.10.0"\n'
        "struct S { x : int32 }\n"
        "element E {\n"
        "  r : ref<S>\n"
        "}"
    )
    assert "is a struct, not an element" in exc.message
    assert exc.line == 4


def test_error_ref_to_unknown():
    exc = _expect_error(
        'mujoco_version "3.10.0"\nelement E {\n  r : ref<Nope>\n}'
    )
    assert "unknown element 'Nope'" in exc.message


def test_error_variant_referencing_undefined_type():
    exc = _expect_error(
        'mujoco_version "3.10.0"\n'
        "variant V {\n"
        "  arm : Missing\n"
        "}\n"
        "element E { v : variant V }"
    )
    assert exc.message == "unknown type 'Missing'"
    assert exc.line == 3
    assert exc.col == 9


def test_error_variant_field_referencing_undefined():
    exc = _expect_error(
        'mujoco_version "3.10.0"\nelement E {\n  v : variant Nope\n}'
    )
    assert "unknown variant 'Nope'" in exc.message


def test_error_duplicate_top_level_name():
    exc = _expect_error(
        'mujoco_version "3.10.0"\n'
        "struct Dup { x : int32 }\n"
        "element Dup { y : int32 }"
    )
    assert "duplicate definition of 'Dup'" in exc.message
    assert exc.line == 3


def test_error_arity_on_named_type():
    exc = _expect_error(
        'mujoco_version "3.10.0"\n'
        'enum M { a = "a" }\n'
        "element E {\n  f : M[3]\n}"
    )
    assert "only valid on primitive types" in exc.message


def test_error_enum_default_not_a_member():
    exc = _expect_error(
        'mujoco_version "3.10.0"\n'
        'enum M { a = "a"  b = "b" }\n'
        "element E {\n  f : M = c\n}"
    )
    assert "not a member of enum 'M'" in exc.message
    assert exc.line == 4


def test_error_array_default_wrong_length():
    exc = _expect_error(
        'mujoco_version "3.10.0"\nelement E {\n  f : double[3] = {1, 2}\n}'
    )
    assert "2 values but the field arity is [3]" in exc.message


def test_error_unit_only_angle():
    exc = _expect_error(
        'mujoco_version "3.10.0"\nelement E {\n  f : double (unit=meter)\n}'
    )
    assert "only accepts 'angle'" in exc.message


def test_error_element_annotation_rejects_unit():
    exc = _expect_error(
        'mujoco_version "3.10.0"\nelement E (unit=angle) {\n  f : int32\n}'
    )
    assert "not valid on element" in exc.message


def test_error_unterminated_string():
    exc = _expect_error('mujoco_version "3.10.0')
    assert exc.line == 1


def test_error_message_renders_caret():
    exc = _expect_error(
        'mujoco_version "3.10.0"\nelement E {\n  f : Nope\n}', filename="x.spec"
    )
    text = str(exc)
    assert "x.spec:3:7: error: unknown type 'Nope'" in text
    assert "f : Nope" in text
    assert "^" in text


def test_error_duplicate_child_and_field_collision():
    exc = _expect_error(
        'mujoco_version "3.10.0"\n'
        "element Leaf { name : string }\n"
        "element E {\n"
        "  dup : int32\n"
        "  children dup : Leaf *\n"
        "}"
    )
    assert "duplicate member 'dup'" in exc.message


def test_error_union_unknown_member():
    exc = _expect_error(
        'mujoco_version "3.10.0"\n'
        "element A { name : string }\n"
        "union U = A | Nope\n"
    )
    assert "union 'U' references unknown element 'Nope'" in exc.message
    assert exc.line == 3


def test_error_union_duplicate_member():
    exc = _expect_error(
        'mujoco_version "3.10.0"\n'
        "element A { name : string }\n"
        "union U = A | A\n"
    )
    assert "duplicate member 'A' in union 'U'" in exc.message
    assert exc.line == 3


def test_error_union_contains_union():
    exc = _expect_error(
        'mujoco_version "3.10.0"\n'
        "element A { name : string }\n"
        "union Inner = A\n"
        "union Outer = A | Inner\n"
    )
    assert "cannot contain another union" in exc.message
    assert "Inner" in exc.message
    assert exc.line == 4


def test_error_union_member_not_element():
    exc = _expect_error(
        'mujoco_version "3.10.0"\n'
        "struct S { x : int32 }\n"
        "union U = S\n"
    )
    assert "union member 'S' is a struct, not an element" in exc.message


def test_error_union_as_field_type():
    exc = _expect_error(
        'mujoco_version "3.10.0"\n'
        "element A { name : string }\n"
        "union U = A\n"
        "element E {\n"
        "  f : U\n"
        "}"
    )
    assert "'U' is a union and cannot be used as a field type" in exc.message
    assert "ref<U>" in exc.message  # message points at the two legal positions
    assert (exc.line, exc.col) == (5, 7)  # located at the type token, not the name


def test_error_union_empty_members():
    # `union U =` with no member is a parse error (at least one element required).
    exc = _expect_error('mujoco_version "3.10.0"\nunion U =\n')
    assert "union member element name" in exc.message
