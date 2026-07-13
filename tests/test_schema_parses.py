"""schema/mujoco.spec parses clean, plus targeted structural assertions.

These checks pin the human-judgment shape of the curated schema (plan Sections
5-7): the Posed/Orientation treatment of the spatial elements, the per-spelling
typed actuator and equality surfaces, the layered Default element, and the
keyword enums matching the snapshot's mjMap tables.
"""

import json
from pathlib import Path

import pytest

from protospec_gen import parse_spec

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = ROOT / "schema" / "mujoco.spec"
MJCF = ROOT / "snapshots" / "mjcf_schema.json"


@pytest.fixture(scope="module")
def schema():
    return parse_spec(SCHEMA).to_json()


@pytest.fixture(scope="module")
def index(schema):
    return {
        "elements": {e["name"]: e for e in schema["elements"]},
        "enums": {e["name"]: e for e in schema["enums"]},
        "variants": {v["name"]: v for v in schema["variants"]},
        "structs": {s["name"]: s for s in schema["structs"]},
        "mixins": {m["name"]: m for m in schema["mixins"]},
    }


def _field(elem, name):
    for f in elem["fields"]:
        if f["name"] == name:
            return f
    return None


def test_parses_clean():
    # A raising parse_spec would fail the module import fixtures; assert version.
    assert parse_spec(SCHEMA).mujoco_version == "3.10.0"


# --------------------------------------------------------------------------- #
# Posed + Orientation on the spatial elements                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["Body", "Geom", "Site", "Camera", "Frame"])
def test_posed_elements_use_posed_and_canonical_quat(name, index):
    # Q-ORIENT canonicalization (docs/plan_canonicalization.md Wave A): orientation
    # is a single canonical `quat` field; euler/axisangle/xyaxes/zaxis resolve into
    # it at read. The five-arm Orientation variant no longer exists.
    elem = index["elements"][name]
    assert "Posed" in elem["uses"], f"{name} should use the Posed mixin"
    quat = _field(elem, "quat")
    assert quat is not None, f"{name} has no quat field"
    assert quat["type"] == {
        "kind": "prim", "prim": "double",
        "arity": {"kind": "fixed", "size": 4},
        "line": quat["type"]["line"], "col": quat["type"]["col"],
    }
    assert quat["source_mixin"] == "Posed"
    assert quat["annotations"]["resolver"] == "orientation"
    assert quat["annotations"]["aliases"].split() == [
        "euler", "axisangle", "xyaxes", "zaxis"
    ]


def test_orientation_variant_is_canonicalized_away(index):
    # The Orientation variant and its arm structs are gone; the four non-canonical
    # spellings survive only as read-only input aliases on the canonical quat.
    assert "Orientation" not in index["variants"]
    for struct in ("Quat", "AxisAngle", "XYAxes", "ZAxis", "Euler"):
        assert struct not in index["structs"], struct
    posed_quat = next(
        f for f in index["mixins"]["Posed"]["fields"] if f["name"] == "quat"
    )
    assert posed_quat["annotations"]["resolver"] == "orientation"


def test_posed_mixin_has_pos(index):
    posed = index["mixins"]["Posed"]
    pos = next(f for f in posed["fields"] if f["name"] == "pos")
    assert pos["type"] == {
        "kind": "prim", "prim": "double",
        "arity": {"kind": "fixed", "size": 3},
        "line": pos["type"]["line"], "col": pos["type"]["col"],
    }


# --------------------------------------------------------------------------- #
# Typed actuator + equality spellings (plan Section 6, Q-ACT / Q-EQ)           #
# --------------------------------------------------------------------------- #
ACTUATOR_SPELLINGS = {
    "ActuatorGeneral": "general", "Motor": "motor", "Position": "position",
    "Velocity": "velocity", "IntVelocity": "intvelocity", "Damper": "damper",
    "Cylinder": "cylinder", "Muscle": "muscle", "Adhesion": "adhesion",
    "DcMotor": "dcmotor",
}


@pytest.mark.parametrize("name,tag", ACTUATOR_SPELLINGS.items())
def test_actuator_spelling_is_typed_element(name, tag, index):
    elem = index["elements"].get(name)
    assert elem is not None, f"missing actuator element {name}"
    xml = (elem.get("annotations") or {}).get("xml", name.lower())
    assert xml == tag


def test_position_carries_named_shortcut_params(index):
    # A loaded <position> is a typed element you mutate in kp/kv terms, not prm
    # arrays (DR-3 / Q-ACT).
    pos = index["elements"]["Position"]
    names = {f["name"] for f in pos["fields"]}
    assert {"kp", "kv"} <= names
    assert "gainprm" not in names and "biasprm" not in names


def test_general_actuator_exposes_lowering_params(index):
    gen = index["elements"]["ActuatorGeneral"]
    names = {f["name"] for f in gen["fields"]}
    assert {"gaintype", "biastype", "dyntype", "gainprm", "biasprm", "dynprm"} <= names


EQUALITY_SPELLINGS = {
    "Connect": "connect", "Weld": "weld", "EqualityJoint": "joint",
    "EqualityTendon": "tendon", "EqualityFlex": "flex", "Flexvert": "flexvert",
    "Flexstrain": "flexstrain",
}


@pytest.mark.parametrize("name,tag", EQUALITY_SPELLINGS.items())
def test_equality_spelling_is_typed_element(name, tag, index):
    elem = index["elements"].get(name)
    assert elem is not None, f"missing equality element {name}"
    xml = (elem.get("annotations") or {}).get("xml", name.lower())
    assert xml == tag


def test_equality_spellings_have_named_params(index):
    # No opaque data[11] vector; each spelling names its own parameters (Q-EQ).
    weld = index["elements"]["Weld"]
    names = {f["name"] for f in weld["fields"]}
    assert {"anchor", "relpose", "torquescale"} <= names


# --------------------------------------------------------------------------- #
# Sensors stay a per-tag typed surface (not one generic sensor element)        #
# --------------------------------------------------------------------------- #
def test_sensors_are_per_tag_elements(index):
    # spot-check a few concrete sensor elements exist with their own slots
    fp = index["elements"]["Framepos"]
    names = {f["name"] for f in fp["fields"]}
    assert {"objtype", "objname", "reftype", "refname"} <= names
    touch = index["elements"]["Touch"]
    assert _field(touch, "site")["type"]["kind"] == "ref"


# --------------------------------------------------------------------------- #
# Default: layered classes with per-family optional sub-structures (Section 7) #
# --------------------------------------------------------------------------- #
def test_default_is_recursive_with_family_children(index):
    default = index["elements"]["Default"]
    child_targets = {c["element"] for c in default["children"]}
    # recursion + a representative spread of defaultable families
    assert "Default" in child_targets
    for fam in ("Geom", "Joint", "Material", "Position", "EqualityDefault",
                "TendonDefault"):
        assert fam in child_targets, f"Default missing {fam} sub-structure"
    # the class name is the one genuine rename (class -> dclass)
    dclass = _field(default, "dclass")
    assert dclass is not None and dclass["annotations"]["xml"] == "class"


# --------------------------------------------------------------------------- #
# Replicate is a first-class pass-through element (plan DR-7 / Q-MACRO)         #
# --------------------------------------------------------------------------- #
def test_replicate_control_attributes(index):
    # <replicate> owns no MJCF[] row (the reader validates it against the body
    # row, xml_util.cc:492); its control attributes come from the reader's
    # replicate branch (xml_native_reader.cc:3806-3874): count (3812, required),
    # offset (3813), euler (3814, accumulated rotation -> unit=angle), sep (3815)
    # and childclass (3826). prefix is corpus-written (namespaces the clones) and
    # modelled for pass-through fidelity though this vendored reader hardcodes it.
    rep = index["elements"]["Replicate"]
    fields = {f["name"]: f for f in rep["fields"]}
    assert set(fields) == {"count", "offset", "euler", "sep", "prefix", "childclass"}
    # count int32 and required -- the reader reads it with exact/required = true
    assert fields["count"]["type"]["kind"] == "prim"
    assert fields["count"]["type"]["prim"] == "int32"
    assert fields["count"]["optional"] is False
    # offset/euler are double[3]; euler carries unit=angle
    for arm in ("offset", "euler"):
        t = fields[arm]["type"]
        assert t["prim"] == "double"
        assert t["arity"] == {"kind": "fixed", "size": 3}
    assert fields["euler"]["annotations"]["unit"] == "angle"
    assert fields["sep"]["type"]["prim"] == "string"
    assert fields["prefix"]["type"]["prim"] == "string"
    assert fields["childclass"]["type"]["kind"] == "ref"
    assert fields["childclass"]["type"]["target"] == "Default"


def _child_key(c):
    return c.get("element") or c.get("union")


def test_replicate_mirrors_body_children(index, schema):
    # replicate/frame/body all wrap body-context content, so they share the same
    # children shape: a unique optional <inertial> plus one ordered BodyChildAny
    # union (geom/joint/site/camera/light/plugin + body/frame and the macros),
    # whose interleave preserves MuJoCo's document-order id assignment.
    rep = index["elements"]["Replicate"]
    body = index["elements"]["Body"]
    frame = index["elements"]["Frame"]
    body_shape = {_child_key(c) for c in body["children"]}
    assert body_shape == {"Inertial", "BodyChildAny"}
    assert {_child_key(c) for c in rep["children"]} == body_shape
    assert {_child_key(c) for c in frame["children"]} == body_shape

    unions = {u["name"]: u for u in schema["unions"]}
    members = set(unions["BodyChildAny"]["members"])
    for fam in ("Geom", "Joint", "Site", "Camera", "Light", "PluginRef",
                "Body", "Frame", "Composite", "Flexcomp", "Replicate", "Attach"):
        assert fam in members, f"BodyChildAny missing {fam} member"


# --------------------------------------------------------------------------- #
# Keyword enums match the snapshot mjMap tables                                #
# --------------------------------------------------------------------------- #
def _map(name):
    km = json.loads(MJCF.read_text("utf-8"))["keyword_maps"]
    return [e["key"] for e in km[name]]


@pytest.mark.parametrize("enum_name,map_name", [
    ("GeomType", "geom_map"),
    ("Integrator", "integrator_map"),
    ("BiasType", "bias_map"),
    ("GainType", "gain_map"),
    ("DynType", "dyn_map"),
    ("JointType", "joint_map"),
])
def test_enum_values_match_snapshot(enum_name, map_name, index):
    values = [m["value"] for m in index["enums"][enum_name]["members"]]
    assert values == _map(map_name)


@pytest.mark.parametrize("elem,attr", [
    ("Rangefinder", "data"),
    ("SensorContact", "data"),
    ("Camera", "output"),
])
def test_mapvalues_attrs_are_enum_keyword_lists(elem, attr, index):
    # MuJoCo reads these attributes with MapValues -- space-separated keyword
    # LISTS OR'd into a bitmask, not scalar enums. The schema types them as
    # `EnumName[]` (an unbounded-arity enum reference).
    field = _field(index["elements"][elem], attr)
    assert field is not None, f"{elem}.{attr} missing"
    t = field["type"]
    assert t["kind"] == "named", f"{elem}.{attr} is not an enum reference"
    assert t.get("category") == "enum"
    assert t.get("arity") == {"kind": "unbounded"}, (
        f"{elem}.{attr} should be a keyword list (EnumName[])"
    )
    # No scalar default survives the list retype (the bitmask default is a
    # read-time concern).
    assert "default" not in field


def test_contactdata_and_raydata_enum_values(index):
    # The keyword sets match MuJoCo's condata_map / raydata_map surface.
    ray = [m["value"] for m in index["enums"]["RayData"]["members"]]
    assert ray == ["dist", "dir", "origin", "point", "normal", "depth"]
    con = [m["value"] for m in index["enums"]["ContactData"]["members"]]
    assert con == ["found", "force", "torque", "dist", "pos", "normal", "tangent"]


def test_tristate_enum(index):
    # limited-style tri-states are an enum {false,true,auto}, not a variant.
    values = [m["value"] for m in index["enums"]["TriState"]["members"]]
    assert values == ["false", "true", "auto"]
    joint = index["elements"]["Joint"]
    assert _field(joint, "limited")["type"] == {
        "kind": "named", "name": "TriState", "category": "enum",
        "line": _field(joint, "limited")["type"]["line"],
        "col": _field(joint, "limited")["type"]["col"],
    }
