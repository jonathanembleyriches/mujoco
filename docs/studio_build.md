# Building ProtoSpec Studio (the MuJoCo Studio fork)

ProtoSpec Studio is the real MuJoCo Studio app (Filament renderer) with the
ProtoSpec editor integrated. There is ONE copy of the editor sources — this repo,
under `studio/`. The fork does not hold a copy; it compiles the editor and
the ProtoSpec core live from this repo via `PROTOSPEC_ROOT`.

Paths below use two placeholders — substitute your checkouts:

- `<studio>` — the MuJoCo fork checkout, branch `studio` (contains `src/experimental/studio`).
- `<protospec>` — this repo.
- Build into `<studio>/build_ps` (any path works; this is the convention).

## What lives where

| Location | Contents |
| --- | --- |
| `<protospec>/studio/editor/` | The editor cluster — single source. Standalone host AND fork build from here. |
| `<protospec>/studio/test/` | The 8 editor test batteries — run in the standalone build (CI / ASan surface). |
| `<protospec>/studio/platform/ux/{plugin,ps_plugin_ext}.h` | The plugin interface contract (ps::studio structs). |
| `<studio>/.../protospec/platform/ux/*.h` | Fork host shims mapping the ps::studio plugin names onto the real `mujoco::platform` registry. |
| `<studio>/.../protospec/host/shell.{cc,h}` | Host-only SE4 shell (File/Edit menu, transform toolbar, Play/Stop bridge). |
| `<studio>/.../studio/main.cc` | Fork entry point; registers the editor cluster + host shell under `MUJOCO_STUDIO_PROTOSPEC`. |

## Prerequisites

CMake ≥ 3.20, Ninja, a C++20 compiler, and MuJoCo's Filament build prerequisites for your OS
(this target *builds* Filament from source, so it inherits MuJoCo's own per-platform
requirements — see MuJoCo's `README`/`experimental/studio`). The first configure fetches
Filament + SDL2 + Dear ImGui + abseil + lodepng into `<studio>/build_ps/_deps` (slow — Filament
is large); later builds are incremental. Editing an editor source in `<protospec>` recompiles
only `protospec_core` / `protospec_editor` and relinks the executable.

`PROTOSPEC_ROOT` points at this live repo (NOT a snapshot). `PROTOSPEC_STUDIO_ROOT` defaults to
`${PROTOSPEC_ROOT}/studio`; override only if the studio tree moves.

The common configure flags (all platforms):

```
-DCMAKE_BUILD_TYPE=Release
-DMUJOCO_USE_FILAMENT=ON
-DMUJOCO_BUILD_STUDIO=ON
-DMUJOCO_STUDIO_PROTOSPEC=ON
-DPROTOSPEC_ROOT=<protospec>
-DMUJOCO_BUILD_EXAMPLES=OFF
-DMUJOCO_BUILD_SIMULATE=OFF     # avoids an rpath-codegen error on a clean configure
-DMUJOCO_BUILD_TESTS=OFF
-DCMAKE_INSTALL_PREFIX=<studio>/build_ps/install   # absolute; the rpath step needs it
```

### Windows

**Use Ninja + MSVC, from a `vcvars64` developer shell. Do NOT use the Visual Studio generator**
(MuJoCo's `filament.cmake` `/FI cstring` workaround leaks onto Filament's generated `.c` resource
files → `C1189`/`STL1003`), and **do NOT use clang-cl** (this Filament pin rejects Clang on
Windows).

```bat
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

cmake -S <studio> -B <studio>/build_ps -G Ninja ^
  -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_INSTALL_PREFIX=<studio>/build_ps/install ^
  -DMUJOCO_USE_FILAMENT=ON -DMUJOCO_BUILD_STUDIO=ON -DMUJOCO_STUDIO_PROTOSPEC=ON ^
  -DPROTOSPEC_ROOT=<protospec> ^
  -DMUJOCO_BUILD_EXAMPLES=OFF -DMUJOCO_BUILD_SIMULATE=OFF -DMUJOCO_BUILD_TESTS=OFF

cmake --build <studio>/build_ps --target mujoco_studio
```

Output: `<studio>/build_ps/bin/mujoco_studio.exe`.

### Linux

GCC ≥ 11 or Clang ≥ 14. Filament's Linux deps (per MuJoCo/Filament docs) include the
Vulkan/GL and X11/Wayland dev packages.

```sh
cmake -S <studio> -B <studio>/build_ps -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=<studio>/build_ps/install \
  -DMUJOCO_USE_FILAMENT=ON -DMUJOCO_BUILD_STUDIO=ON -DMUJOCO_STUDIO_PROTOSPEC=ON \
  -DPROTOSPEC_ROOT=<protospec> \
  -DMUJOCO_BUILD_EXAMPLES=OFF -DMUJOCO_BUILD_SIMULATE=OFF -DMUJOCO_BUILD_TESTS=OFF

cmake --build <studio>/build_ps --target mujoco_studio
```

Output: `<studio>/build_ps/bin/mujoco_studio`.

### macOS

Xcode ≥ 14 command-line tools (Apple Clang). Same invocation as Linux:

```sh
cmake -S <studio> -B <studio>/build_ps -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=<studio>/build_ps/install \
  -DMUJOCO_USE_FILAMENT=ON -DMUJOCO_BUILD_STUDIO=ON -DMUJOCO_STUDIO_PROTOSPEC=ON \
  -DPROTOSPEC_ROOT=<protospec> \
  -DMUJOCO_BUILD_EXAMPLES=OFF -DMUJOCO_BUILD_SIMULATE=OFF -DMUJOCO_BUILD_TESTS=OFF

cmake --build <studio>/build_ps --target mujoco_studio
```

Output: `<studio>/build_ps/bin/mujoco_studio` (`.app` bundle rules apply if Studio is configured
that way upstream).

> Development and CI for this project run on Windows; the Linux/macOS invocations are the
> standard MuJoCo-Studio build with the ProtoSpec flags added. If Filament itself fails to build
> on your platform, that is a MuJoCo/Filament prerequisite issue — resolve it against MuJoCo's
> own studio build first, then re-add the `PROTOSPEC_*` flags.

## Run

```sh
<studio>/build_ps/bin/mujoco_studio  "<path>/model/humanoid/humanoid.xml"
```

**Quote model paths that contain spaces**, e.g. `--model_file="…/Unreal Projects/…/humanoid.xml"` —
an unquoted path splits at the space and the model silently fails to load.

Only the Hierarchy and Details panels dock open by default (plus the central Viewport) — the
two-panel layout. Assets folds into the Hierarchy (its section's right-click and the header
"+ Asset" button create materials / textures / import meshes); Diagnostics folds into the
status-bar error chip (click it to reveal the log). Both remain toggleable from the Plugins/View
menu. After a load or a drag, the Diagnostics log records the compile path, e.g.
`… ok  path=native  nq=27  nbody=14`. `path=native` = the fast native compiler (no XML round-trip);
large or unsupported-feature models fall back to `path=xml`.

### Self-screenshot (headless capture)

The classic backend can capture frames without interaction (absl flags use underscores):

```sh
mujoco_studio --gfx=classic \
  --screenshot_seq <out_dir> --screenshot_after 90 --screenshot_exit \
  "<path>/model/humanoid/humanoid.xml"
```

## Editor tests (the 8 batteries)

The editor's tests live and run with the single source, in the standalone `studio` build
(the CI / ASan surface), not in this fork — see the **Studio** section of the top-level
`README.md` for the cross-platform commands.
