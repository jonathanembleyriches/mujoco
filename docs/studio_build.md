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
| `<protospec>/studio/editor/` | The editor cluster — single source. The plugin fork builds from here. |
| `<protospec>/studio/glue/` | The host-mount build: `CMakeLists.txt` (all build logic), the `platform/ux/plugin.h` shim mapping the `ps::studio` plugin names onto the real `mujoco::platform` registry, and `register_editor.cc` (the `mjPLUGIN_LIB_INIT` self-registration TU the fork compiles into the executable). |
| `<protospec>/attic/studio_host/test/` | The 8 editor test batteries — parked with the pre-plugin standalone host (see `attic/studio_host/README.md`); worth re-pointing at the plugin shape someday. The live windowless coverage is `protospec/tests/test_plugin_windowless.py`. |
| `<protospec>/attic/studio_host/platform/ux/{plugin,ps_plugin_ext}.h` | The old standalone `ps::studio` plugin structs — parked; stale against the 4-type retarget. |
| `<studio>/src/experimental/studio/CMakeLists.txt` | The fork's ONLY editor wiring: the `MUJOCO_STUDIO_PROTOSPEC` option and the ~15-line mount block (`add_subdirectory` of `studio/glue` + `target_sources` of `register_editor.cc`). `main.cc` is stock upstream. |

## Prerequisites

CMake ≥ 3.20, Ninja, a C++20 compiler, and MuJoCo's Filament build prerequisites for your OS
(this target *builds* Filament from source, so it inherits MuJoCo's own per-platform
requirements — see MuJoCo's `README`/`experimental/studio`). The first configure fetches
Filament + SDL2 + Dear ImGui + abseil + lodepng into `<studio>/build_ps/_deps` (slow — Filament
is large); later builds are incremental. Editing an editor source in `<protospec>` recompiles
only `protospec_core` / `protospec_editor` and relinks the executable.

`PROTOSPEC_ROOT` points at this live repo (NOT a snapshot); everything else is derived from it
(the fork mounts `${PROTOSPEC_ROOT}/studio/glue`, which locates the editor and core relative to
itself).

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

The default layout is curated plugin-side (`studio/editor/dock_layout.cc`): Hierarchy left,
Viewport centre, Details right with Layers beneath it. The host's stock Options / Explorer /
Inspector panels stay at their upstream defaults and dock as background tabs behind the curated
panels; Assets / Diagnostics / File / + Add dock hidden and are toggleable from the Plugins/View
menu. The curated layout only applies when no saved layout exists — a `~/.mujoco.ini` from a
pre-glue build (the old versioned `RootV3` dockspace) leaves windows floating; delete the ini
once to re-curate. After a load or a drag, the Diagnostics log records the compile path, e.g.
`… ok  path=native  nq=27  nbody=14`. `path=native` = the fast native compiler (no XML round-trip);
large or unsupported-feature models fall back to `path=xml`.

### Self-screenshot (headless capture)

The screenshot service is plugin-side too (`studio/editor/screenshot_service.cc`) and is
configured from the environment (no launcher flags, so it works through the stock
`LaunchStudio` path). Classic backend only. F12 captures on demand; the auto mode is:

```sh
MUJOCO_SCREENSHOT_DIR=<out_dir> MUJOCO_SCREENSHOT_AFTER=90 MUJOCO_SCREENSHOT_EXIT=1 \
  mujoco_studio "<path>/model/humanoid/humanoid.xml" --gfx=classic
```

`MUJOCO_SCREENSHOT_COUNT=N` takes N consecutive frames; shots are `shot_%04d.png`. Note the
model path must be the FIRST argument (or passed via `--model_file=`) — `main.cc` only promotes
`argv[1]` to a model path.

## Editor tests

The live windowless coverage of the plugin surface is `protospec/tests/test_plugin_windowless.py`
(builds `studio/editor/test/test_plugin_windowless.cc` against the `build_ps` libs). The 8
pre-plugin batteries are parked with the standalone host under `attic/studio_host/test/`.
