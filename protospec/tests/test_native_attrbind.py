"""Gate for the generated attribute-binding tables (Stage B).

``protospec_gen.emit_native`` generates, per converted element, an ``AttrBind``
table (``src/xml/xml_native_attrbind.inc``) that a single ``ReadAttrBinds`` loop in
``xml_native_reader.cc`` drives in place of the element's mechanical
``ReadAttr``/``ReadAttrInt``/``MapValue``/``ReadAttrTxt`` calls. The behavioral
proof that a converted parser is byte-identical to the hand-written one is the
differential corpus suite (``tests/test_path_diff.py``); this test guards the
*structure*:

  * the emitter resolves every element's fields to real, kind-compatible mjs
    fields (generation would already fail otherwise) and produces the expected
    per-element bind counts;
  * each converted ``One<Elem>`` in the reader actually calls ``ReadAttrBinds``
    with its table, and no generated attribute is still read inline (a half
    conversion would read an attribute twice or leave a stale hand line).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protospec_gen import emit_native  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
READER = REPO / "src" / "xml" / "xml_native_reader.cc"
INC = REPO / "src" / "xml" / "xml_native_attrbind.inc"

EXPECTED_COUNTS = {
    "Site": 6, "Camera": 10, "Light": 14, "Material": 8, "Pair": 8,
}


def _binds():
    return {elem: rows for elem, _struct, rows in emit_native._attr_binds()}


def test_emitter_resolves_expected_elements_and_counts():
    binds = _binds()
    assert set(binds) == set(EXPECTED_COUNTS)
    assert {e: len(r) for e, r in binds.items()} == EXPECTED_COUNTS


def test_every_bind_row_is_well_formed():
    kinds = {"kBindDouble", "kBindFloat", "kBindIntArr", "kBindInt",
             "kBindEnum", "kBindString"}
    for elem, rows in _binds().items():
        for xml, bkind, length, struct, field, exact, bmap, bmapsz in rows:
            assert bkind in kinds, f"{elem}.{xml}: bad kind {bkind}"
            if bkind == "kBindEnum":
                assert bmap != "nullptr" and bmapsz != "0"
            else:
                assert bmap == "nullptr" and bmapsz == "0"


def _fn_body(source: str, method: str) -> str:
    m = re.search(r"void mjXReader::" + re.escape(method) + r"\s*\([^)]*\)\s*\{",
                  source)
    assert m, f"{method} not found in reader"
    i = source.index("{", m.end() - 1)
    depth, j = 0, i
    while j < len(source):
        if source[j] == "{":
            depth += 1
        elif source[j] == "}":
            depth -= 1
            if depth == 0:
                return source[i + 1:j]
        j += 1
    raise AssertionError(f"unbalanced body for {method}")


ELEM_METHOD = {
    "Site": "OneSite", "Camera": "OneCamera", "Light": "OneLight",
    "Material": "OneMaterial", "Pair": "OnePair",
}


@pytest.mark.parametrize("elem", sorted(EXPECTED_COUNTS))
def test_converted_parser_uses_driver_and_drops_inline_reads(elem):
    source = READER.read_text(encoding="utf-8")
    body = _fn_body(source, ELEM_METHOD[elem])
    assert f"k{elem}Binds" in body, f"{elem}: ReadAttrBinds call missing"
    rows = _binds()[elem]
    # No generated attribute may still be read by an inline Read*/MapValue call.
    for xml, *_ in rows:
        stale = re.search(
            r'(?:ReadAttr|ReadAttrInt|ReadAttrTxt|MapValue)\s*\(\s*(?:elem|layer)'
            r'\s*,\s*"' + re.escape(xml) + r'"',
            body,
        )
        assert not stale, f"{elem}: attribute {xml!r} still read inline"


def test_inc_defines_a_table_per_element():
    text = INC.read_text(encoding="utf-8")
    for elem in EXPECTED_COUNTS:
        assert f"const AttrBind k{elem}Binds[]" in text
        assert f"const int k{elem}BindsN =" in text
