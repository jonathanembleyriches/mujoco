"""Coverage/drift gate tests for the ProtoSpec->mjSpec field-mapping emitter.

Asserts generation succeeds today, and that the named-field gate fires on a
synthetic unmapped field and on orphaned alias/waiver table entries."""

import copy

import pytest

from protospec_gen import emit_mjs
from protospec_gen.idl import SchemaError


def _data():
    """Fresh (schema, mjs) snapshots so tests can mutate without cross-talk."""
    return emit_mjs._load()


def test_generation_succeeds():
    files = emit_mjs.generate_files()
    assert set(files) == {"mjs_binding.h", "mjs_binding.cc"}
    assert "void ApplyMjs(const Geom& e, mjsGeom* out)" in files["mjs_binding.cc"]
    # canonical-quat orientation maps as a plain fixed array (no mjsOrientation)
    assert "out->quat[i]" in files["mjs_binding.cc"]


def test_every_element_is_mapped_or_waived():
    schema, mjs = _data()
    names = {e["name"] for e in schema["elements"]}
    # Mapped = complement of the retained waiver partition; the element->struct
    # join itself lives in the schema (binding.element_struct).
    mapped = set(emit_mjs._mapped_structs(schema))
    waived = set(emit_mjs.ELEMENT_WAIVERS)
    assert mapped.isdisjoint(waived)
    assert names == mapped | waived, names ^ (mapped | waived)
    # Every mapped element resolves to a real mjs struct.
    struct_names = {s["name"] for s in mjs["structs"]}
    for name, struct in emit_mjs._mapped_structs(schema).items():
        assert struct in struct_names, (name, struct)


def test_stats_shape():
    s = emit_mjs.stats()
    assert s["mapped_elements"] > 0 and s["waived_elements"] > 0
    # exact matches dominate; every representative kind is exercised somewhere.
    tot = {k: 0 for k in ("exact", "alias", "waive", "name", "variant")}
    for c in s["per_element"].values():
        for k in tot:
            tot[k] += c[k]
    assert tot["exact"] > 100
    assert tot["variant"] == 3  # GeomShape on Geom/Site + TextureSource on Texture


def test_unmapped_field_fires_gate():
    schema, mjs = _data()
    geom = next(e for e in schema["elements"] if e["name"] == "Geom")
    geom["fields"].append({
        "name": "bogus_unmapped",
        "type": {"kind": "prim", "prim": "double"},
        "optional": True,
    })
    with pytest.raises(emit_mjs.MjsCoverageError, match=r"Geom\.bogus_unmapped"):
        emit_mjs.generate_files(schema=schema, mjs=mjs)


def test_bad_mjs_annotation_target_fires_gate():
    # A field's `mjs=` binding annotation naming a non-existent mjs field fails
    # loudly (was ELEM_ALIAS/GLOBAL_ALIAS orphan detection; now the inline
    # annotation is validated against the mjsStruct inventory with a schema line).
    schema, mjs = _data()
    geom = next(e for e in schema["elements"] if e["name"] == "Geom")
    f = next(x for x in geom["fields"] if x["name"] == "hfield")
    f["annotations"]["mjs"] = "no_such_mjs_field"
    with pytest.raises(SchemaError, match=r"not a field of mjsGeom"):
        emit_mjs.generate_files(schema=schema, mjs=mjs)


def test_kind_incompatible_mjs_annotation_fires_gate():
    # A `mjs=` binding to a real but kind-incompatible mjs field fails loudly.
    schema, mjs = _data()
    geom = next(e for e in schema["elements"] if e["name"] == "Geom")
    f = next(x for x in geom["fields"] if x["name"] == "pos")  # double[3]
    f.setdefault("annotations", {})["mjs"] = "mass"  # scalar double -> incompatible
    with pytest.raises(SchemaError, match=r"kind-incompatible"):
        emit_mjs.generate_files(schema=schema, mjs=mjs)


def test_orphan_waiver_fires_gate(monkeypatch):
    monkeypatch.setitem(emit_mjs.FIELD_WAIVERS, ("Geom", "ghostfield"), "reason")
    with pytest.raises(emit_mjs.MjsCoverageError, match=r"dead"):
        emit_mjs.generate_files()
