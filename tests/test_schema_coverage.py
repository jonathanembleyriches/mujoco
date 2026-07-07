"""Drift gate v0: schema/mujoco.spec covers the whole MJCF surface.

Walks the MJCF element tree (snapshots/mjcf_schema.json) and the parsed IDL
schema (schema/mujoco.spec) in parallel from the model root, asserting that
every MJCF element and every one of its XML attributes is covered by the schema
-- or listed in the explicit WAIVERS table with a reason.

Coverage mapping rules (plan Section 5 / task brief):

* an IDL element covers the MJCF element reached by matching XML tag along the
  child links (the tree walk disambiguates same-tag elements in different
  positions -- e.g. body/joint vs equality/joint vs tendon/fixed/joint);
* an IDL field covers the MJCF attribute named by its ``xml=`` annotation, or,
  by default, its own name;
* a ``variant`` field covers every XML attribute named by its variant
  definition's arm tags (Orientation's quat/axisangle/xyaxes/zaxis/euler);
* a plain child list covers its one child element;
* a ``union`` child list covers every member element's XML tag (the ordered
  heterogeneous sections: actuator/sensor/equality/tendon and the spatial path).

worldbody/frame/replicate own no MJCF[] row; MuJoCo validates them against the
body row (``special_cases.body_row_aliases``). Each still resolves to a covering
schema element -- worldbody -> Body, frame -> Frame, replicate -> Replicate (a
first-class pass-through element, plan DR-7) -- asserted by
``test_body_row_aliases_resolve_to_elements``.
"""

import json
from pathlib import Path

import pytest

from protospec_gen import parse_spec

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = ROOT / "schema" / "mujoco.spec"
MJCF = ROOT / "snapshots" / "mjcf_schema.json"


# Elements deliberately absent from the schema, each with a reason. These are
# MJCF nodes that expand at read time (they never reach the object model) or
# fold into another element (plan Section 6, DR-7, Q-MACRO).
WAIVERS_ELEMENTS = {
    "attach": "read-time expansion: deep-copy with prefix namespacing (DR-7); "
              "never persists in the object model",
}

# XML attributes with no covering field, each with a reason. Empty: the schema
# models every attribute of every non-waived element.
WAIVERS_ATTRIBUTES = {}


def _schema_index():
    schema = parse_spec(SCHEMA).to_json()
    elements = {e["name"]: e for e in schema["elements"]}
    variants = {v["name"]: v for v in schema["variants"]}
    unions = {u["name"]: u for u in schema["unions"]}

    def xml_tag(elem):
        return (elem.get("annotations") or {}).get("xml", elem["name"].lower())

    tag_of = {name: xml_tag(e) for name, e in elements.items()}

    def covered_attrs(elem):
        attrs = set()
        for f in elem["fields"]:
            t = f["type"]
            if t["kind"] == "variant":
                for arm in variants[t["target"]]["arms"]:
                    attrs.add(arm["tag"])
            else:
                attrs.add((f.get("annotations") or {}).get("xml", f["name"]))
        return attrs

    def child_targets(elem):
        # child MJCF tag -> covering IDL element. A plain child list contributes
        # its one element; a union child list contributes every member element.
        targets = {}
        for c in elem["children"]:
            if "union" in c:
                members = unions[c["union"]]["members"]
            else:
                members = [c["element"]]
            for member in members:
                targets[tag_of[member]] = member
        return targets

    return elements, tag_of, covered_attrs, child_targets


def _mjcf_root():
    return json.loads(MJCF.read_text("utf-8"))["elements"]


def test_schema_covers_mjcf_tree():
    elements, tag_of, covered_attrs, child_targets = _schema_index()
    # tag -> idl element name, restricted to the children reachable from a node
    root = _mjcf_root()

    missing = []  # (context, kind, name)

    def visit(mjcf_node, idl_name, path):
        idl = elements[idl_name]
        have = covered_attrs(idl)
        for attr in mjcf_node["attributes"]:
            if attr not in have and attr not in WAIVERS_ATTRIBUTES:
                missing.append((path, "attribute", attr))
        # child tag -> idl element target, from this element's child links
        targets = child_targets(idl)
        for child in mjcf_node["children"]:
            tag = child["name"]
            if tag in WAIVERS_ELEMENTS:
                continue
            target = targets.get(tag)
            if target is None:
                missing.append((path, "element", tag))
                continue
            visit(child, target, f"{path}/{tag}")

    visit(root, "Model", "mujoco")

    if missing:
        lines = [f"  {path}: uncovered {kind} '{name}'" for path, kind, name in missing]
        pytest.fail(
            f"{len(missing)} uncovered MJCF item(s) (add a field/child or a "
            "WAIVERS entry):\n" + "\n".join(lines)
        )


def test_body_row_aliases_resolve_to_elements():
    # worldbody/frame/replicate have no MJCF[] row of their own; MuJoCo validates
    # them against the body row (special_cases.body_row_aliases). Each must still
    # resolve to a covering schema element so the alias usages are modelled:
    # worldbody -> Body, frame -> Frame, replicate -> Replicate (first-class, DR-7).
    elements, tag_of, _, _ = _schema_index()
    name_by_tag = {tag: name for name, tag in tag_of.items()}
    aliases = json.loads(MJCF.read_text("utf-8"))["special_cases"][
        "body_row_aliases"
    ]["aliases"]
    expected = {"worldbody": "Body", "frame": "Frame", "replicate": "Replicate"}
    for alias in aliases:
        assert alias in expected, f"new body-row alias {alias!r} needs a schema mapping"
        assert expected[alias] in elements, f"{alias} -> {expected[alias]} missing"
    # replicate is now a real element (its own tag), not a bare body-row passthrough
    assert name_by_tag.get("replicate") == "Replicate"


def test_waivers_are_justified():
    # Every waiver must carry a non-empty reason string.
    for tag, reason in WAIVERS_ELEMENTS.items():
        assert reason and isinstance(reason, str), tag
    for attr, reason in WAIVERS_ATTRIBUTES.items():
        assert reason and isinstance(reason, str), attr


def test_waivers_are_used():
    # A waiver that no longer corresponds to an uncovered MJCF item is dead
    # weight -- it should be removed so the list stays honest.
    elements, tag_of, covered_attrs, child_targets = _schema_index()
    root = _mjcf_root()

    present_tags = set()

    def collect(node):
        for c in node["children"]:
            present_tags.add(c["name"])
            collect(c)

    collect(root)
    for tag in WAIVERS_ELEMENTS:
        assert tag in present_tags, f"waiver for '{tag}' matches no MJCF element"
