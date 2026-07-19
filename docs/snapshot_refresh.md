# Refreshing the MuJoCo snapshots

The generator does **not** parse MuJoCo's C/C++ sources directly. Instead, one-shot
extractors under `tools/bootstrap/` lift the facts we depend on out of a MuJoCo source
checkout into JSON snapshots under `protospec/snapshots/`, and the rest of the pipeline
(`protospec/schema/mujoco.spec`, the drafting tools, the tests) reads those snapshots. This keeps
the build reproducible and free of a MuJoCo source dependency.

Refresh the snapshots whenever you bump the pinned MuJoCo version.

## Prerequisites

Point `PROTOSPEC_MUJOCO_SRC` at the **root** of a MuJoCo source checkout — the directory
that contains both `include/` and `src/` (i.e. `include/mujoco/mjspec.h` and
`src/xml/xml_native_reader.cc` exist under it):

```bash
export PROTOSPEC_MUJOCO_SRC=/path/to/mujoco   # checkout root, contains include/ + src/
```

The same variable is what opts the extractor integration tests in (see below); with it
unset they skip and collection still succeeds.

## Regenerate the JSON snapshots

Each extractor takes `--mujoco-src` and writes its default snapshot under `protospec/snapshots/`:

```bash
# mjspec.h field inventory  -> snapshots/mjspec_fields.json
uv run python tools/bootstrap/extract_mjspec_fields.py --mujoco-src "$PROTOSPEC_MUJOCO_SRC"

# MJCF[] schema tree + keyword maps -> snapshots/mjcf_schema.json
uv run python tools/bootstrap/extract_mjcf_schema.py --mujoco-src "$PROTOSPEC_MUJOCO_SRC"

# spec-struct default values -> snapshots/spec_defaults.json
uv run python tools/bootstrap/extract_spec_defaults.py --mujoco-src "$PROTOSPEC_MUJOCO_SRC"
```

Pass `--out <path>` to any of them to write elsewhere. Each prints a one-line summary
(counts + the detected MuJoCo version) on success; a parse it cannot classify fails loudly
with a file:line rather than emitting a silently-wrong snapshot.

## Lifted-code snapshots

Verbatim-lifted symbols (under `attic/snapshots/lifted_upstream/`, tracked in the lift
registry) are refreshed separately, because they re-extract against the full studio
superproject build tree rather than a bare MuJoCo source checkout:

```bash
# Point at the tree the lifted code was taken from, then re-add drifted symbols:
export PROTOSPEC_MUJOCO_ROOT=/path/to/studio/superproject
uv run python attic/tools/lift_registry.py check      # report drift
uv run python attic/tools/lift_registry.py add ...     # refresh a symbol's snapshot + hash
```

## Verify

```bash
# Extractor integration tests run (and must pass) when the source tree is present:
PROTOSPEC_MUJOCO_SRC="$PROTOSPEC_MUJOCO_SRC" uv run pytest -q

# The generator must still be byte-identical to its checked-in output:
uv run python -m protospec_gen.emit --check
```
