"""Extract MuJoCo's MJCF grammar tables from the vendored C++ reader into a JSON snapshot.

The MJCF grammar lives as a single static table ``MJCF[nMJCF]`` in
``src/xml/xml_native_reader.cc`` plus a set of ``static const mjMap`` keyword tables.
This module reconstructs the nested element tree exactly as ``mjXSchema`` does
(``src/xml/xml_util.cc``) and dumps every keyword map verbatim, producing the
snapshot that the ProtoSpec drift gate (plan DR-12) diffs the hand-curated IDL against.

After bootstrap this script is no longer a generator; it is retained purely as the
drift checker, so every parse failure names the offending source line.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Type codes the schema table uses in a row's second column (see mjXSchema::Check,
# xml_util.cc): required-unique, optional-unique, zero-or-more, recursive.
VALID_TYPECODES = frozenset({"!", "?", "*", "R"})

# worldbody/frame/replicate are not schema rows; they validate against the "body"
# row via a hardcoded name match in mjXSchema::NameMatch (xml_util.cc:486-498).
SPECIAL_CASES = {
    "body_row_aliases": {
        "schema_row": "body",
        "aliases": ["worldbody", "frame", "replicate"],
        "comment": (
            "worldbody/frame/replicate are not their own schema rows; "
            "mjXSchema::NameMatch (src/xml/xml_util.cc:486-498) validates them "
            "against the 'body' row (worldbody only at level 1)."
        ),
    },
}


class SchemaParseError(Exception):
    """Raised when the vendored table format has drifted from what we can parse."""


@dataclass
class Row:
    """One brace-group from the ``MJCF[]`` initializer, tagged with its source line."""

    tokens: list[str]
    line: int


@dataclass
class Element:
    name: str
    typecode: str
    attributes: list[str]
    children: list["Element"] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "typecode": self.typecode,
            "attributes": self.attributes,
            "children": [c.to_dict() for c in self.children],
        }


def _strip_line_comment(text: str) -> str:
    """Drop a trailing ``// ...`` comment, respecting string literals."""
    out = []
    i = 0
    in_str = False
    while i < len(text):
        c = text[i]
        if in_str:
            out.append(c)
            if c == "\\" and i + 1 < len(text):
                out.append(text[i + 1])
                i += 2
                continue
            if c == '"':
                in_str = False
        elif c == '"':
            in_str = True
            out.append(c)
        elif c == "/" and i + 1 < len(text) and text[i + 1] == "/":
            break
        else:
            out.append(c)
        i += 1
    return "".join(out)


def _extract_block(source: str, header_pattern: str) -> tuple[str, int]:
    """Return the text inside the first ``= { ... }`` initializer matching header_pattern.

    Also returns the 1-based line number on which the block's opening brace sits, so
    callers can offset per-row line numbers back onto the real file.
    """
    m = re.search(header_pattern, source)
    if m is None:
        raise SchemaParseError(f"could not locate initializer matching /{header_pattern}/")
    brace_start = source.index("{", m.end() - 1)
    line_at_brace = source.count("\n", 0, brace_start) + 1

    depth = 0
    i = brace_start
    in_str = False
    while i < len(source):
        c = source[i]
        if in_str:
            if c == "\\":
                i += 2
                continue
            if c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return source[brace_start + 1 : i], line_at_brace
        i += 1
    raise SchemaParseError(
        f"unterminated initializer for /{header_pattern}/ starting at line {line_at_brace}"
    )


_STRING_LITERAL = re.compile(r'"((?:[^"\\]|\\.)*)"')


def _parse_rows(block: str, base_line: int) -> list[Row]:
    """Split a brace-initializer body into its depth-1 ``{...}`` rows with line numbers.

    ``base_line`` is the file line of the initializer's opening brace, so each row's
    reported line maps back onto ``xml_native_reader.cc``.
    """
    rows: list[Row] = []
    depth = 0
    i = 0
    in_str = False
    row_start = -1
    row_start_line = -1
    line = base_line
    while i < len(block):
        c = block[i]
        if c == "\n":
            line += 1
            i += 1
            continue
        if in_str:
            if c == "\\":
                i += 2
                continue
            if c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
            if depth == 1:
                row_start = i + 1
                row_start_line = line
        elif c == "}":
            if depth == 1:
                raw = block[row_start:i]
                tokens = [
                    m.group(1) for m in _STRING_LITERAL.finditer(_strip_line_comment(raw))
                ]
                if not tokens:
                    raise SchemaParseError(
                        f"line {row_start_line}: schema row has no string tokens: {{{raw.strip()}}}"
                    )
                rows.append(Row(tokens=tokens, line=row_start_line))
            depth -= 1
            if depth < 0:
                raise SchemaParseError(f"line {line}: unbalanced '}}' in initializer")
        i += 1
    if depth != 0:
        raise SchemaParseError(f"line {base_line}: unbalanced braces in initializer")
    return rows


def _is_open(row: Row) -> bool:
    return row.tokens == ["<"]


def _is_close(row: Row) -> bool:
    return row.tokens == [">"]


def _is_bracket(row: Row) -> bool:
    return _is_open(row) or _is_close(row)


def _build_element(rows: list[Row]) -> Element:
    """Reconstruct one element (and its child block) from a slice of rows.

    Mirrors ``mjXSchema::mjXSchema`` (xml_util.cc:336-379): rows[0] is the element
    header; if the element has a child block it is bracketed as
    ``header, "<", ...children..., ">"``. A child at index ``start`` owns a nested
    block iff the following row opens a bracket; the block extends to the balanced
    closing bracket.
    """
    header = rows[0]
    if _is_bracket(header):
        raise SchemaParseError(
            f"line {header.line}: expected an element header, found bracket row {header.tokens}"
        )
    if len(header.tokens) < 2:
        raise SchemaParseError(
            f"line {header.line}: element row missing type code: {header.tokens}"
        )
    name = header.tokens[0]
    typecode = header.tokens[1]
    if typecode not in VALID_TYPECODES:
        raise SchemaParseError(
            f"line {header.line}: unknown type code {typecode!r} for element {name!r}; "
            f"expected one of {sorted(VALID_TYPECODES)}"
        )
    attributes = header.tokens[2:]
    element = Element(name=name, typecode=typecode, attributes=attributes)

    n = len(rows)
    if n == 1:
        return element
    if not _is_open(rows[1]) or not _is_close(rows[-1]):
        raise SchemaParseError(
            f"line {header.line}: element {name!r} child block is not bracket-delimited"
        )

    start = 2
    while start < n - 1:
        if _is_bracket(rows[start]):
            raise SchemaParseError(
                f"line {rows[start].line}: unexpected bracket row {rows[start].tokens} "
                f"where a child element was expected"
            )
        end = start
        if start + 1 <= n - 1 and _is_open(rows[start + 1]):
            depth = 0
            end = start
            while end <= n - 1:
                if _is_open(rows[end]):
                    depth += 1
                elif _is_close(rows[end]):
                    depth -= 1
                    if depth == 0:
                        break
                end += 1
            if depth != 0:
                raise SchemaParseError(
                    f"line {rows[start].line}: unbalanced child block for element "
                    f"{rows[start].tokens[0]!r}"
                )
        element.children.append(_build_element(rows[start : end + 1]))
        start = end + 1
    return element


# The MJCF table's array bound is either the hand-maintained ``nMJCF`` or the
# emitter's generated ``nMJCF_GENERATED`` (once the table moved into
# xml_native_schema.inc); accept either.
_MJCF_HEADER = r"std::vector<const char\*>\s+MJCF\s*\[\s*nMJCF(?:_GENERATED)?\s*\]\s*="


def parse_schema_tree(source: str) -> Element:
    """Parse the ``MJCF[...]`` table into a nested element tree."""
    block, base_line = _extract_block(source, _MJCF_HEADER)
    rows = _parse_rows(block, base_line)
    if not rows:
        raise SchemaParseError("MJCF table is empty")
    return _build_element(rows)


# The reader either defines the MJCF table inline or pulls it from a generated
# include (protospec_gen.emit_native). This names that include so the extractor
# can follow it transparently and keep working across the reader-table refactor.
_SCHEMA_INCLUDE = "xml_native_schema.inc"

# The schema-backed keyword maps were likewise lifted into a generated include
# (protospec_gen.emit_native, KEYWORD_MAPS); the reader keeps only the maps that
# are not schema-backed 1:1. The extractor gathers maps from the reader plus this
# include so the snapshot still carries the full set across that refactor.
_KEYWORDS_INCLUDE = "xml_native_keywords.inc"


def _schema_source(reader: Path, reader_source: str) -> str:
    """Return the text that carries the MJCF table: the reader itself if the
    table is inline, otherwise the generated include it pulls in."""
    if re.search(_MJCF_HEADER, reader_source):
        return reader_source
    if f'#include "{_SCHEMA_INCLUDE}"' in reader_source:
        inc = reader.parent / _SCHEMA_INCLUDE
        if not inc.exists():
            raise SchemaParseError(
                f"reader includes {_SCHEMA_INCLUDE} but {inc} is missing"
            )
        return inc.read_text(encoding="utf-8")
    raise SchemaParseError(
        "could not locate the MJCF table inline or via "
        f'#include "{_SCHEMA_INCLUDE}"'
    )


_MAP_HEADER = re.compile(r"const\s+mjMap\s+(?P<name>\w+)\s*\[[^\]]*\]\s*=\s*")
_MAP_ENTRY = re.compile(r'\{\s*"((?:[^"\\]|\\.)*)"\s*,\s*([^},]+?)\s*,?\s*\}')


def _keyword_sources(reader: Path, reader_source: str) -> list[str]:
    """Return every source text carrying keyword maps: the reader, plus the
    generated keywords include when the reader pulls it in."""
    sources = [reader_source]
    if f'#include "{_KEYWORDS_INCLUDE}"' in reader_source:
        inc = reader.parent / _KEYWORDS_INCLUDE
        if not inc.exists():
            raise SchemaParseError(
                f"reader includes {_KEYWORDS_INCLUDE} but {inc} is missing"
            )
        sources.append(inc.read_text(encoding="utf-8"))
    return sources


def parse_keyword_maps(source: str) -> dict[str, list[dict[str, str]]]:
    """Parse every ``const mjMap NAME[...] = { {"k", V}, ... }`` table, in source order."""
    maps: dict[str, list[dict[str, str]]] = {}
    for header in _MAP_HEADER.finditer(source):
        name = header.group("name")
        brace_start = source.index("{", header.end() - 1)
        block, _ = _extract_block(source[header.start() :], r"=\s*")
        entries = [
            {"key": key, "value": value.strip()} for key, value in _MAP_ENTRY.findall(block)
        ]
        if not entries:
            line = source.count("\n", 0, brace_start) + 1
            raise SchemaParseError(f"line {line}: keyword map {name!r} has no entries")
        if name in maps:
            raise SchemaParseError(f"duplicate keyword map {name!r}")
        maps[name] = entries
    return maps


def read_mujoco_version(mujoco_src: Path) -> str:
    """Read ``mjVERSION_HEADER`` from ``include/mujoco/mujoco.h`` as a dotted string."""
    header = mujoco_src / "include" / "mujoco" / "mujoco.h"
    text = header.read_text(encoding="utf-8")
    m = re.search(r"#define\s+mjVERSION_HEADER\s+(\d+)", text)
    if m is None:
        raise SchemaParseError(f"mjVERSION_HEADER not found in {header}")
    n = int(m.group(1))
    return f"{n // 1_000_000}.{(n // 1000) % 1000}.{n % 1000}"


def _count_elements(element: Element) -> int:
    return 1 + sum(_count_elements(c) for c in element.children)


def _collect_recursive(element: Element, out: list[str]) -> None:
    if element.typecode == "R":
        out.append(element.name)
    for child in element.children:
        _collect_recursive(child, out)


def run_sanity_checks(tree: Element, maps: dict[str, list[dict[str, str]]]) -> None:
    """Fail loudly if the parsed schema does not have the shape we know MuJoCo's has."""
    if tree.name != "mujoco":
        raise SchemaParseError(f"root element is {tree.name!r}, expected 'mujoco'")

    recursive: list[str] = []
    _collect_recursive(tree, recursive)
    if sorted(recursive) != ["body", "default"]:
        raise SchemaParseError(
            f"expected exactly 'body' and 'default' to be R-typed, found {sorted(recursive)}"
        )

    count = _count_elements(tree)
    if count <= 100:
        raise SchemaParseError(f"only {count} elements parsed; expected > 100")

    for required in ("angle_map", "geom_map"):
        if required not in maps:
            raise SchemaParseError(f"expected keyword map {required!r} to be present")


def build_snapshot(mujoco_src: Path) -> dict:
    reader = mujoco_src / "src" / "xml" / "xml_native_reader.cc"
    source = reader.read_text(encoding="utf-8")
    tree = parse_schema_tree(_schema_source(reader, source))
    # Keyword maps live in the reader .cc plus the generated keywords include.
    maps: dict[str, list[dict[str, str]]] = {}
    for text in _keyword_sources(reader, source):
        for name, entries in parse_keyword_maps(text).items():
            if name in maps:
                raise SchemaParseError(f"duplicate keyword map {name!r}")
            maps[name] = entries
    run_sanity_checks(tree, maps)
    return {
        "source": {
            "file": "src/xml/xml_native_reader.cc",
            "mujoco_version": read_mujoco_version(mujoco_src),
        },
        "elements": tree.to_dict(),
        "keyword_maps": maps,
        "special_cases": SPECIAL_CASES,
    }


def write_snapshot(snapshot: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n"
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--mujoco-src",
        type=Path,
        required=True,
        help="Root of the vendored MuJoCo src checkout (contains src/ and include/).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_repo_root() / "snapshots" / "mjcf_schema.json",
        help="Destination JSON path (default: snapshots/mjcf_schema.json).",
    )
    args = parser.parse_args(argv)

    try:
        snapshot = build_snapshot(args.mujoco_src)
    except (SchemaParseError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    write_snapshot(snapshot, args.out)
    elements = _count_elements(_dict_to_element(snapshot["elements"]))
    print(
        f"wrote {args.out} "
        f"({elements} elements, {len(snapshot['keyword_maps'])} keyword maps, "
        f"mujoco {snapshot['source']['mujoco_version']})"
    )
    return 0


def _dict_to_element(d: dict) -> Element:
    return Element(
        name=d["name"],
        typecode=d["typecode"],
        attributes=list(d["attributes"]),
        children=[_dict_to_element(c) for c in d["children"]],
    )


if __name__ == "__main__":
    raise SystemExit(main())
