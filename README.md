# ProtoSpec

## Where things live (fork `jonathanembleyriches/mujoco`)

- **`main` / `protospec`** — this standalone library repo (identical; `main` is the default).
  Everything under `C:\Users\jonat\Documents\protospec`.
- **`studio`** — MuJoCo Studio (real app + Filament) with the ProtoSpec editor integrated. This
  is the interactive editor you run/test. It consumes the editor sources live from this repo
  (`apps/studio/`) — no forked copy. Checkout: `C:\Users\jonat\Documents\mujoco-studio`; build
  steps in `docs/studio_build.md`.

---

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

### Prerequisites (all platforms)

- **CMake ≥ 3.20** and **Ninja** (or any generator).
- A **C++20 compiler**: MSVC 2022 (Windows), GCC ≥ 11 or Clang ≥ 14 (Linux), Apple Clang
  (Xcode ≥ 14, macOS).
- **[uv](https://docs.astral.sh/uv/)** for the Python toolchain (parser, emitters, test driver).
- **MuJoCo ≥ 3.10**, built/installed for your platform, pointed at by the `MUJOCO_ROOT` CMake
  cache variable. `MUJOCO_ROOT` must contain `include/mujoco/`, the import library
  (`lib/…/mujoco.{lib,so,dylib}`) + runtime, and the `build/_deps/` tree MuJoCo produced (the
  native compiler links MuJoCo's own `qhull`, `tinyobjloader`, and `lodepng` static libs from
  there). On this project MuJoCo is the copy vendored in the UnrealRoboticsLab plugin; on Linux/
  macOS point `MUJOCO_ROOT` at your own MuJoCo source+build tree.

The commands below use Ninja (identical on every OS). On Windows run them from a
**Developer shell** (`vcvars64.bat`, or a "x64 Native Tools" prompt) so `cl` is on `PATH`.

### Python toolchain + full test suite (cross-platform, identical)

```sh
uv sync
uv run pytest            # generator, corpus differential, native ratchet, bridge, studio smoke
```

### C++ library + tests

```sh
# Windows: run inside a vcvars64 developer shell.  Linux/macOS: a plain shell.
cmake -S cpp -B cpp/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DMUJOCO_ROOT=/path/to/mujoco
cmake --build cpp/build
ctest --test-dir cpp/build --output-on-failure   # object model, IO, validate, bridge, native, SDK
```

(Windows alternative generator: `-G "Visual Studio 17 2022"`, then add `--config Release` to the
build/ctest commands.)

### Python module (pybind11)

```sh
cmake -S cpp/python -B cpp/python/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DMUJOCO_ROOT=/path/to/mujoco
cmake --build cpp/python/build
```

Then `import protospec` — see **[TRYME.md](TRYME.md)** for the load → edit → compile → step →
save session (paths in TRYME are Windows; the Python API is identical on every OS).

### Studio (the interactive editor)

Two hosts consume one editor source (`apps/studio/`):

- **Standalone shell** — SDL2 + classic OpenGL renderer, the CI/test surface. Builds anywhere:

  ```sh
  cmake -S apps/studio -B apps/studio/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DMUJOCO_ROOT=/path/to/mujoco
  cmake --build apps/studio/build
  ctest --test-dir apps/studio/build --output-on-failure    # 8 windowless batteries
  ./apps/studio/build/protospec_studio "/path/to/model.xml"  # .exe on Windows
  ```

- **Real MuJoCo Studio (Filament)** — the full-featured app, on the `studio` branch of the
  MuJoCo fork; it compiles this repo's editor live via `PROTOSPEC_ROOT`. Because it builds
  Filament, it follows **MuJoCo's own per-platform build requirements** (Windows: Ninja + MSVC
  only — the VS generator trips Filament's resource codegen; Linux/macOS: the toolchains MuJoCo
  documents). Exact commands, flags, and the run/self-screenshot recipe are in
  **[docs/studio_build.md](docs/studio_build.md)**.

  Quote model paths that contain spaces: `--model_file="…/humanoid.xml"`.

### Regeneration (CI-gated, `--check`-reproducible on every OS)

```sh
uv run python -m protospec_gen.emit          # regenerate cpp/generated + cpp/python/generated
```

`schema/mujoco.spec` is the source of truth and is edited directly.
`tools/bootstrap/draft_schema.py` is the one-time bootstrap DRAFTER, not a
regenerator: running it over the snapshots discards the schema's hand
refinements (the BodyChildAny union, resolvers, aliases, typed refs). Keep its
tables in sync when editing the schema so future drafts start closer, but never
overwrite the schema with its output.

## Documents

- `docs/plan.md` — the main plan; decision records, quirk register, living STATUS
- `docs/plan_native_compiler.md` + `docs/plan_native_compiler_impl.md` — the native compiler
- `docs/plan_studio_editor.md` — the editor
- `docs/native_compiler_survey.md` — evidence base (its mjs decode mapping is historical)
- `HANDOFF.md` — where every track stands and how to resume it
- `TRYME.md` — quickstart (Python module + Studio)
