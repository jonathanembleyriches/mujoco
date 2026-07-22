# ProtoSpec — the MJCF schema and its drift guards

This is the first, smallest slice of ProtoSpec: a single declarative schema
describing the entire MJCF model format, plus the machinery that proves it
stays truthful.

- **`schema/mujoco.spec`** — every element, attribute, type, default, enum,
  union, and reference relationship of MJCF, stated once. This file is the
  thing to read.
- **`protospec_gen/`** — the IDL parser and generator (pure Python, no
  dependencies). Later slices emit C++/Python code from the schema; this
  slice uses only the parser and the canonicalization tables.
- **`tools/bootstrap/extract_mjcf_schema.py`** — extracts the reader's
  hand-maintained `MJCF[]` table out of `src/xml/xml_native_reader.cc` into
  `snapshots/mjcf_schema.json`.
- **`tests/`** — the guards:
  - the schema parses and satisfies its structural invariants;
  - **coverage**: every element and attribute the native reader accepts is
    covered by the schema (or listed in an explicit, justified waiver) —
    walked tree-vs-tree from the root, so the schema cannot silently drift
    from the reader;
  - the extractor round-trips against the enclosing source tree.

```sh
# from this directory
uv run pytest                 # all guards
uv run python tools/bootstrap/extract_mjcf_schema.py \
    --mujoco-src ..           # refresh the snapshot from this checkout
```

Why merge this first: it gives the repository a machine-checked, single-source
description of MJCF before any code changes. The follow-up slice generates the
reader's `MJCF[]` table (and its row count) from this schema, making the
table-maintenance bug class impossible; further slices add the generated
object model and SDK.

## Generating the native reader's schema table

MuJoCo's native XML reader validates MJCF against a schema table —
`std::vector<const char*> MJCF[]` in `src/xml/xml_native_reader.cc`, rows of
`{element, occurrence, attr...}` with `{"<"}`/`{">"}` nesting markers, consumed
by `mjXSchema`. `protospec_gen.emit_native` generates that entire table (and
its row count `nMJCF_GENERATED`) from `schema/mujoco.spec` into
`src/xml/xml_native_schema.inc`, which the reader `#include`s in place of the
hand-maintained table — so the grammar the reader enforces is derived from the
IDL rather than curated by hand. The emitter owns the structural mapping the
IDL models differently (union spellings, `body`/`default` recursion,
`frame`/`replicate` aliasing, and the `<default>`-template attribute
projection), while the schema stays the single source of attribute/occurrence
content. `tests/test_native_schema_table.py` parses both the generated `.inc`
and the original hand table (recovered from git) and asserts element-tree +
occurrence + attribute-set equality.

```sh
uv run python -m protospec_gen.emit_native --write   # regenerate the .inc
uv run python -m protospec_gen.emit_native --check   # byte-gate (CI)
```
