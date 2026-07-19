"""Orphan-key drift guard for the emitter's hand-maintained tables.

emit.py and emit_py.py each carry small hand-written tables keyed by schema names
(element names, and (element, child-list) pairs). Nothing regenerates these from the
schema, so a schema rename could silently orphan an entry -- the table would keep a key
that no longer names anything, and the special-case it encodes would quietly stop firing.

These tests assert every key of those tables still resolves against the *live* parsed
schema, so a rename that orphans a hand-table entry fails here instead of shipping.
"""

from __future__ import annotations

import os

import pytest

from protospec_gen import emit, parse_spec
from protospec_gen import emit_py

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def schema() -> dict:
    return parse_spec(os.path.join(ROOT, "schema", "mujoco.spec")).to_json()


@pytest.fixture(scope="module")
def element_names(schema: dict) -> set[str]:
    return {e["name"] for e in schema["elements"]}


@pytest.fixture(scope="module")
def child_lists(schema: dict) -> set[tuple[str, str]]:
    """The live (element name, child-list name) pairs."""
    return {
        (e["name"], c["name"])
        for e in schema["elements"]
        for c in e.get("children", [])
    }


def test_child_xml_override_keys_are_live_child_lists(child_lists):
    """emit._CHILD_XML_OVERRIDE keys are (element, child-list) pairs; each must exist."""
    for key in emit._CHILD_XML_OVERRIDE:
        assert key in child_lists, (
            f"_CHILD_XML_OVERRIDE key {key!r} is not a live (element, child-list) "
            f"pair -- schema renamed out from under emit.py?"
        )


def test_child_skip_keys_are_live_child_lists(child_lists):
    """emit_py._CHILD_SKIP entries are (element, child-list) pairs; each must exist."""
    for key in emit_py._CHILD_SKIP:
        assert key in child_lists, (
            f"_CHILD_SKIP key {key!r} is not a live (element, child-list) pair "
            f"-- schema renamed out from under emit_py.py?"
        )


def test_element_input_alias_keys_are_live_elements(element_names):
    """emit.ELEMENT_INPUT_ALIASES is keyed by element name; each must exist."""
    for key in emit.ELEMENT_INPUT_ALIASES:
        assert key in element_names, (
            f"ELEMENT_INPUT_ALIASES key {key!r} is not a live element name "
            f"-- schema renamed out from under emit.py?"
        )
