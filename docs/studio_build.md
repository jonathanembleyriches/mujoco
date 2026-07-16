# Building ProtoSpec Studio (the MuJoCo Studio fork)

ProtoSpec Studio is the real MuJoCo Studio app (Filament renderer) with the
ProtoSpec editor integrated. There is ONE copy of the editor sources — this repo,
under `apps/studio/`. The fork does not hold a copy; it compiles the editor and
the ProtoSpec core live from this repo via `PROTOSPEC_ROOT`.

- Fork checkout: `C:\Users\jonat\Documents\mujoco-studio`, branch `studio`.
- Core repo (this one): `C:\Users\jonat\Documents\protospec`.
- Canonical build dir: `C:\Users\jonat\Documents\mujoco-studio\build_ps`.

## What lives where

| Location | Contents |
| --- | --- |
| `protospec/apps/studio/editor/` | The editor cluster — single source. Standalone host AND fork build from here. |
| `protospec/apps/studio/test/` | The 8 editor test batteries — run in the standalone build (CI / ASan surface). |
| `protospec/apps/studio/platform/ux/{plugin,ps_plugin_ext}.h` | The plugin interface contract (ps::studio structs). |
| `mujoco-studio/.../protospec/platform/ux/*.h` | Fork host shims mapping the ps::studio plugin names onto the real `mujoco::platform` registry. |
| `mujoco-studio/.../protospec/host/shell.{cc,h}` | Host-only SE4 shell (File/Edit menu, transform toolbar, Play/Stop bridge) — binds to Studio-only plugin types. |
| `mujoco-studio/.../studio/main.cc` | Fork entry point; registers the editor cluster + host shell under `MUJOCO_STUDIO_PROTOSPEC`. |

## Build

Requires Visual Studio 2022 (`vcvars64`) and Ninja. From a plain shell:

```bat
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

set SRC=C:/Users/jonat/Documents/mujoco-studio
set BLD=C:/Users/jonat/Documents/mujoco-studio/build_ps
set PSROOT=C:/Users/jonat/Documents/protospec

cmake -S "%SRC%" -B "%BLD%" -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl ^
  -DMUJOCO_USE_FILAMENT=ON ^
  -DMUJOCO_BUILD_STUDIO=ON ^
  -DMUJOCO_STUDIO_PROTOSPEC=ON ^
  -DPROTOSPEC_ROOT=%PSROOT% ^
  -DMUJOCO_BUILD_EXAMPLES=OFF ^
  -DMUJOCO_BUILD_SIMULATE=OFF ^
  -DMUJOCO_BUILD_TESTS=OFF ^
  -DMUJOCO_TEST_PYTHON_UTIL=OFF ^
  -DMUJOCO_WITH_USD=OFF ^
  -DFILAMENT_SKIP_SAMPLES=ON

cmake --build "%BLD%" --target mujoco_studio --parallel 12
```

`PROTOSPEC_ROOT` points at this live repo (NOT a snapshot). A second cache var,
`PROTOSPEC_STUDIO_ROOT`, defaults to `${PROTOSPEC_ROOT}/apps/studio` and locates
the editor cluster; override it only if the studio tree ever moves.

The first configure fetches Filament and the other UI deps into `%BLD%\_deps`
(slow — Filament is large). Subsequent builds are incremental; editing an editor
source in this repo recompiles only `protospec_core` / `protospec_editor` and
relinks `mujoco_studio.exe`. To reuse an existing dependency checkout and skip the
Filament fetch, pass `-DFETCHCONTENT_SOURCE_DIR_FILAMENT=<path>` (and the same for
`SDL2`, `DEAR_IMGUI`, `ABSEIL-CPP`, `LODEPNG`, `MARCHINGCUBECPP`, etc.).

Output: `%BLD%\bin\mujoco_studio.exe` (window title "ProtoSpec Studio").

## Run

```bat
C:\Users\jonat\Documents\mujoco-studio\build_ps\bin\mujoco_studio.exe ^
  C:\Users\jonat\Documents\mujoco-studio\model\humanoid\humanoid.xml
```

The Hierarchy / Details / Assets / Diagnostics panels dock open by default. After a
load or a drag, the Diagnostics panel logs the compile path, e.g.
`... ok  path=native  nq=27  nbody=14`. `path=native` means the model took the fast
native compiler (no XML round-trip); large or unsupported-feature models fall back
to `path=xml`.

### Self-screenshot (headless capture)

The classic backend can capture frames without interaction (absl flags use
underscores):

```bat
mujoco_studio.exe --gfx=classic ^
  --screenshot_seq <out_dir> --screenshot_after 90 --screenshot_exit ^
  C:\Users\jonat\Documents\mujoco-studio\model\humanoid\humanoid.xml
```

## Editor tests (the 8 batteries)

The editor's tests live and run with the single source, in the standalone
`apps/studio` build (the CI / ASan surface), not in this fork:

```bat
cmake -S apps/studio -B apps/studio/build -G "Visual Studio 17 2022"
cmake --build apps/studio/build --config Release
ctest --test-dir apps/studio/build -C Release
```
