"""Coverage/drift gate tests for the ProtoSpec->mjSpec field-mapping emitter.

Asserts generation succeeds today, and that the named-field gate fires on a
synthetic unmapped field and on orphaned alias/waiver table entries."""

import copy

import pytest

from protospec_gen import emit_mjs


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
    schema, _ = _data()
    names = {e["name"] for e in schema["elements"]}
    covered = set(emit_mjs.ELEMENT_STRUCT) | set(emit_mjs.ELEMENT_WAIVERS)
    assert names == covered, names ^ covered


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


def test_orphan_alias_fires_gate(monkeypatch):
    # An alias keyed on a schema field that does not exist is never consumed.
    monkeypatch.setitem(emit_mjs.ELEM_ALIAS, ("Geom", "ghostfield"), "pos")
    with pytest.raises(emit_mjs.MjsCoverageError, match=r"dead"):
        emit_mjs.generate_files()


def test_orphan_alias_bad_target_fires_gate(monkeypatch):
    # An alias whose mjs target field does not exist on the struct.
    monkeypatch.setitem(emit_mjs.ELEM_ALIAS, ("Geom", "hfield"), "no_such_mjs_field")
    with pytest.raises(emit_mjs.MjsCoverageError, match=r"no such mjs field"):
        emit_mjs.generate_files()


def test_orphan_waiver_fires_gate(monkeypatch):
    monkeypatch.setitem(emit_mjs.FIELD_WAIVERS, ("Geom", "ghostfield"), "reason")
    with pytest.raises(emit_mjs.MjsCoverageError, match=r"dead"):
        emit_mjs.generate_files()
