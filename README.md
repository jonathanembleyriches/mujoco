# ProtoSpec

A single, clean, IDL-generated C++ object model for MuJoCo models. One schema
(`schema/mujoco.spec`) is the source of truth; build-time generators emit the C++ types,
serialization visitors, reflection tables, and MJCF XML bindings. MJCF is a wire format handled
by one IO module; `mjSpec` appears in exactly one bridge function and nowhere else.

Design document: [docs/plan.md](docs/plan.md).

## Layout

- `schema/` — the `mujoco.spec` IDL (single source of truth)
- `protospec_gen/` — Python: IDL parser and code emitters (build-time only)
- `tools/bootstrap/` — one-time extraction scripts that drafted the initial schema from the
  MuJoCo source; retained for the drift gate
- `snapshots/` — extracted JSON snapshots of the MuJoCo surface (schema table, spec fields,
  defaults), used by the drift gate on MuJoCo version bumps
- `tests/` — pytest suite for the generator toolchain

## Development

Python tooling uses [uv](https://docs.astral.sh/uv/):

```
uv sync
uv run pytest
```
