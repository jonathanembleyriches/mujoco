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
