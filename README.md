# ProtoSpec

A single, clean object model for MuJoCo models. One IDL schema (`schema/mujoco.spec`) is the
source of truth; generators emit the C++ types, serialization, reflection tables, and XML
bindings. MJCF is a wire format handled by one IO module; models compile to `mjModel` through
either of two interchangeable paths behind one pure `Compile()`:

- **XML path** — form-preserving writer → `mj_loadXML` (the reference oracle, always available)
- **Native path** — a raw ProtoSpec→`mjModel` compiler (no `mjSpec`/`mjs_*` anywhere), reusing
  MuJoCo's math via a drift-gated lifted-code registry

Verified against every model in the MuJoCo repository: the XML round-trip compiles
**359/387** corpus files bit-identical to a direct load; the native path compiles **201/387**
bit-identical to the XML path (self-arming ratchet — regressions fail CI). The remaining
corpus files are enumerated skips/fallbacks with written reasons, tracked in the plans below.

Also here: **ProtoSpec Studio** (`apps/studio`) — an interactive editor (Unity/Unreal idiom:
hierarchy, generated inspector, transform gizmos, live recompile) built as plugins against
MuJoCo Studio's plugin interfaces.

## Layout

- `schema/` — the `mujoco.spec` IDL (single source of truth)
- `protospec_gen/` — Python: IDL parser + code emitters (build-time only)
- `cpp/` — the C++ library: `generated/` (emitted object model), `include/`+`sdk/` (core +
  ergonomic SDK), `io/` (MJCF reader/writer), `validate/`, `bridge/` (Compile/Binding/
  Recompile), `compile/` (native compiler + lifted code), `harness/` (mjModel differ),
  `python/` (pybind11 module), `test/`, `tools/`
- `apps/studio/` — the editor (thin vendored shell + ProtoSpec editor plugins)
- `tools/` — bootstrap extractors, lift registry, corpus study
- `snapshots/` — extracted MuJoCo-surface snapshots + lifted-code registry (drift gates)
- `tests/` — pytest suites (generator, corpus differential, native ratchet, bridge, studio smoke)
- `docs/` — the plans; start with `docs/plan.md` (living STATUS table)

## Build & test

Python tooling uses [uv](https://docs.astral.sh/uv/). MuJoCo is consumed prebuilt via the
`MUJOCO_ROOT` CMake cache variable (headers + `mujoco.lib`/`mujoco.dll` + source tree for
lifted-code verification).

```
uv sync
uv run pytest                                  # full Python suite incl. corpus gates

cmake -S cpp -B cpp/build -G "Visual Studio 17 2022"
cmake --build cpp/build --config Release
ctest --test-dir cpp/build -C Release          # object model, IO, validate, bridge, native, SDK

cmake -S apps/studio -B apps/studio/build -G "Visual Studio 17 2022"
cmake --build apps/studio/build --config Release
apps/studio/build/Release/protospec_studio.exe <model.xml>

cmake -S cpp/python -B cpp/python/build        # pybind11 module (see TRYME.md)
```

Regeneration entry points (both `--check`-reproducible, CI-gated):
`uv run python -m protospec_gen.emit` and `uv run python tools/bootstrap/draft_schema.py`.

## Documents

- `docs/plan.md` — the main plan; decision records, quirk register, living STATUS
- `docs/plan_native_compiler.md` + `docs/plan_native_compiler_impl.md` — the native compiler
- `docs/plan_studio_editor.md` — the editor
- `docs/native_compiler_survey.md` — evidence base (its mjs decode mapping is historical)
- `HANDOFF.md` — where every track stands and how to resume it
- `TRYME.md` — quickstart (Python module + Studio)
