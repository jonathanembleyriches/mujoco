# Hosting the ProtoSpec editor in the real MuJoCo Studio UI

Investigation + build spike. Goal: replace our thin-shell studio (bare ImGui panels
+ classic `mjr` renderer) with the real MuJoCo Studio UI (SDL2 + ImGui docking +
their dock layout / theme / fonts / toolbar / status bar / options panels + the
Filament renderer), hosting our ProtoSpec editor inside it.

Sources are read-only:

- Vendored MuJoCo (v3.10.0): `.../UnrealRoboticsLab/third_party/MuJoCo/src`
  (studio: `src/experimental/studio`, platform: `src/experimental/platform`,
  filament: `src/render/filament` + `src/experimental/filament`).
- Our shell: `C:\Users\jonat\Documents\protospec\apps\studio`.

Spike build lives entirely OUTSIDE the repo at `C:\tmp\studio_spike\build`. Nothing
in the vendored tree was modified.

---

## 0. TL;DR

- **Spike:** the real Studio configures and builds on this machine. See §1 for the
  exact outcome (updated after the build completed).
- **Hosting option (recommended):** **(b) a `studio` branch cut from upstream on the
  owner's `jonathanembleyriches/mujoco` fork, with the ProtoSpec editor carried as a
  subdirectory + a small, reviewable patch to `app.cc`/`gui.cc`/`model_holder`/
  `renderer`.** Reasons in §3.
- **Smallest-modification map:** our four extension plugins (`ViewportPlugin`,
  `ViewportGuiPlugin`, `OverlayPlugin`, `ModelSourcePlugin`) map onto four small,
  well-localized seams in the Studio App. `GuiPlugin` / `KeyHandlerPlugin` /
  `SpecEditorPlugin` are already upstream and need **zero** engine change. See §2.
- **Migration cost:** the plugin-first design pays off. The entire `editor/` cluster
  (~5,900 LOC across 20 files: hierarchy, details, gizmos, transform math, authoring,
  asset import, undo) ports **as-is**. What is rewritten is only the thin `platform/`
  + `host/` substrate (~1,400 LOC) that we deliberately kept as a stand-in for Studio.
  See §4.

---

## 1. The spike: building the real Studio out of tree

### 1.1 Which CMake options turn Studio on

Top-level `CMakeLists.txt`:

```
option(MUJOCO_BUILD_STUDIO   "Build studio library for MuJoCo" OFF)   # default OFF
option(MUJOCO_USE_FILAMENT   "Use filament rendering"          OFF)   # default OFF
```

`MUJOCO_BUILD_STUDIO=ON` adds `src/experimental/platform` + `src/experimental/studio`
(lines 269-272). **Studio hard-requires Filament**: `platform/CMakeLists.txt` links
`mujoco::filament` and `mujoco::filament_experimental` unconditionally (lines 108-109),
and those targets only exist when `MUJOCO_USE_FILAMENT=ON` (top-level lines 173-176).
So the minimum flag set is `-DMUJOCO_BUILD_STUDIO=ON -DMUJOCO_USE_FILAMENT=ON`.

The canonical recipe is `src/experimental/studio/build.sh` (works in git-bash on
Windows). It configures with:

```
-DMUJOCO_USE_FILAMENT=ON  -DMUJOCO_BUILD_STUDIO=ON
-DMUJOCO_BUILD_EXAMPLES=OFF -DMUJOCO_BUILD_SIMULATE=OFF
-DMUJOCO_BUILD_TESTS=OFF   -DMUJOCO_TEST_PYTHON_UTIL=OFF -DMUJOCO_WITH_USD=OFF
-DFILAMENT_SKIP_SAMPLES=ON -DFILAMENT_SHORTEN_MSVC_COMPILATION=OFF
```

### 1.2 The dependency chain (what gets fetched vs built)

Everything is fetched from git and **built from source** — no prebuilt binaries:

| Dep | Source (git) | Notes |
|---|---|---|
| **Filament** | `google/filament` @ `da22932b…` | the heavy one; full clone (`GIT_SHALLOW FALSE`); builds its own shader toolchain (matc/cmgen/resgen, glslang, SPIR-V-Tools) |
| SDL2 | `libsdl-org/SDL` @ `98d1f3a…` | static (`SDL2::SDL2-static`) |
| dear_imgui | `ocornut/imgui` @ `3109131…` (docking) | same commit our shell already pins |
| implot | fetched | |
| libwebp | fetched | Filament/asset textures |
| OpenSans + Font Awesome | fetched ttf | copied to `assets/` post-build |

The imgui backends used by the platform HAL are **SDL2 + OpenGL3**
(`dear_imgui_SDL2`, `dear_imgui_OpenGL3` in `platform/CMakeLists.txt`), i.e. the same
backend pair our thin shell already uses. Filament is a *separate* renderer for the 3D
scene; ImGui is composited on top (classic-GL path) or blitted through Filament.

### 1.3 Graphics backend / GPU requirements

`GraphicsMode` (platform/hal/graphics_mode.h) offers Classic-GL, Filament-OpenGL, and
Filament-Vulkan (+ headless/software variants). Defaults:

- `app.h` Config default = `FilamentVulkan`.
- **`main.cc` desktop default = `FilamentOpenGl`** (the actual runtime default).

This machine has both backends available:

- Vulkan **1.4** runtime loader present (`vulkaninfo` reports Instance Version 1.4.309,
  real discrete GPU, `VK_KHR_win32_surface`) → `FilamentVulkan` is viable.
- OpenGL-capable GPU → `FilamentOpenGl` (the default) is viable and is the safe first
  choice. `FilamentVulkanSoftware` / `FilamentOpenGlSoftware` exist as fallbacks.

There is **no Vulkan SDK** installed (`VULKAN_SDK` unset, no `glslc`), but that is only
a build-time concern for hand-compiling shaders — Filament ships and builds its own
shader compilers, so the SDK is not required to build or run.

### 1.4 Build environment

cmake 4.2.1, Ninja 1.10.2, git 2.34, VS2022 (MSVC 14.38.33130), Python 3.9. The
pre-existing `.../MuJoCo/src/build` dir was configured with `MUJOCO_BUILD_SIMULATE=OFF`,
`MUJOCO_BUILD_STUDIO=OFF`, `MUJOCO_USE_FILAMENT=OFF` — i.e. it built **only** core
`mujoco.dll` (what our shell links). Studio/Filament had never been built here.

### 1.5 What I actually did

Configured and built out of tree (VS2022 generator, x64, Release):

```
cmake -S ".../third_party/MuJoCo/src" -B "C:/tmp/studio_spike/build" \
      -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release \
      -DMUJOCO_USE_FILAMENT=ON -DMUJOCO_BUILD_STUDIO=ON \
      -DMUJOCO_BUILD_EXAMPLES=OFF -DMUJOCO_BUILD_SIMULATE=OFF \
      -DMUJOCO_BUILD_TESTS=OFF -DMUJOCO_TEST_PYTHON_UTIL=OFF -DMUJOCO_WITH_USD=OFF \
      -DFILAMENT_SKIP_SAMPLES=ON -DFILAMENT_SHORTEN_MSVC_COMPILATION=OFF
cmake --build "C:/tmp/studio_spike/build" --config Release --target mujoco_studio --parallel
```

**Configure: SUCCESS (≈225 s).** Full Filament clone + all FetchContent deps resolved;
`Generating done`. The many "Could NOT find ZLIB/PNG/JPEG/GLUT/pthread/unistd" lines are
benign — those are Filament/host-tool probes that fall back to bundled sources on Windows.
CMake 4.x emits `FetchContent_Populate` deprecation warnings (from the vendored
`FindOrFetch.cmake`) but they are non-fatal. Two ignored cache vars (`BUILD_SHARED_LIB`,
`USE_STATIC_LIBCXX`) — harmless, they are `build.sh` leftovers.

<!-- SPIKE-BUILD-OUTCOME -->

---

## 2. Integration architecture: the smallest modification set

### 2.1 The plugin ABI already matches

Our `platform/ux/plugin.h` is Studio's `platform/ux/plugin.h` **verbatim** (only the
namespace differs: `ps::studio` vs `mujoco::platform`). Both define the same four POD
plugin structs with the same `RegisterPlugin<T>` / `ForEachPlugin<T>` dispatch:
`GuiPlugin`, `ModelPlugin`, `KeyHandlerPlugin`, `SpecEditorPlugin`.

Consequence: our **6 GuiPlugins** (Hierarchy, Details, Assets, Diagnostics, File, +Add)
and **9 KeyHandlerPlugins** (undo/redo/save, Q/W/E/R tools, Delete, Ctrl+D) register and
dispatch **unchanged** against upstream Studio. Studio even iterates GuiPlugins with the
identical Plugins-menu + `Begin(name)/update/End` pattern our host copied
(`studio/app.cc` GuiPlugin loop ≈ our `host/app.cc:432`). Zero engine change for these.

### 2.2 The four gaps — our `ps_plugin_ext.h` extensions

Everything editor-specific that stock Studio hard-codes in `app.cc`, we already surfaced
as four extension plugin types (`platform/ux/ps_plugin_ext.h`). Each maps to one small
Studio seam. The seams below are exactly what our own `host/app.cc` already implements —
so `host/app.cc` is a working reference for the required Studio patch.

**(a) `ModelSourcePlugin` — inject our compiled `mjModel*` (the important one).**
Our editor compiles ProtoSpec → `bridge::Compile` → **`mjModel` directly** (native path,
`CompilePath::Auto`), and keeps a name-based `bridge::Binding` (serial ↔ element ↔ id)
that survives every recompile. Selection identity is the creation *serial*, not an
mjModel id.

Stock Studio cannot adopt a pre-built `mjModel*`. `ModelPlugin::get_model_to_load`
returns only a serialized buffer; `App::LoadModelFromBuffer` → `ModelHolder::FromBuffer`
accepts only `text/xml` / `application/mjb` / `application/zip`, and every ModelHolder
path funnels through `PostInit()` which **always** `mj_compile`s or `mj_loadModelBuffer`s.
`ModelHolder` has **no adopt / `SetModel` / `Reset(mjModel*)` API** (members are private,
only `Release*` hand ownership *out*).

Two options:

- **XML round-trip (zero engine change):** the fallback our own header documents — a
  stock `ModelPlugin` returns `WriteMjcf(model)` as `text/xml`. Studio re-parses to a
  live `mjSpec` (its Explorer/Editor keep working), but MuJoCo re-derives the model and
  our Binding-by-id identity is thrown away (names survive, so we could re-derive the
  Binding by name each load — extra work, and it loses the native-compile artifact).
  An `application/mjb` round-trip is byte-exact and name-preserving but sets `spec_=null`
  (their Explorer shows "No mjSpec loaded").
- **Adopt seam (recommended, ~15-30 lines):** add `ModelHolder::FromModel(mjModel*,
  mjData*)` (skip `PostInit` compile) + an `App` method to swap `model_holder_`, then a
  `ModelSourcePlugin::poll_compiled` hook in `ProcessPendingLoads`. This is precisely
  what our `App::AdoptCompiledModel` does (`host/app.cc:67`: `renderer_->Init(model)`,
  `mj_makeData`, `mj_resetData`, `mj_forward`). Keeps Binding identity + the native
  artifact.

**(b) `OverlayPlugin` — append transient `mjvGeom`s (selection outlines, scene gizmos).**
Studio's `mjvScene` is private to the `Renderer` and rebuilt every frame by
`mjv_updateScene`; there is no callback between `mjv_updateScene` and render, and no
scene accessor. Our thin shell added exactly this seam: `Renderer::SetOverlayHook`, fired
between `mjv_updateScene` and `mjr_render` (`platform/hal/renderer.cc:78`), which the host
fans out to every `OverlayPlugin` (`host/app.cc:42`). Our selection outline appends 12
line geoms by bumping `scene->ngeom` (`editor/viewport_plugin.cc`). **Studio patch: ~5-10
lines** — invoke a hook (or `ForEachPlugin<OverlayPlugin>`) right after `mjv_updateScene`
in `Renderer::Render`. Filament's `SceneBridge` already iterates `scene->ngeom`, so
appended geoms render on both backends.

**(c) `ViewportGuiPlugin` — draw screen-space gizmos over the 3D viewport.**
Our gizmos are **renderer-agnostic**: drawn with `ImGui::GetForegroundDrawList()`
(`editor/gizmo.cc:460`), projected with a `ViewProj` we build from `mjvCamera` via the
same `mjv_cameraFrame`/`mjv_cameraFrustum` math MuJoCo uses (`editor/gizmo_math.cc`), over
the passthrough central dock node. This needs **no** Filament API — the foreground draw
list is available in any ImGui frame. Studio just never calls it; we pass the plugin the
camera + aspect each frame (`host/app.cc:454`). **Studio patch: dispatch
`ViewportGuiPlugin` at the end of `BuildGui` with `{model,data,camera,aspect}`** — a
handful of lines. The viewport rect is recoverable via
`DockBuilderGetCentralNode(GetID("Root"))`.

**(d) `ViewportPlugin` — viewport mouse for picking + gizmo grab.**
Studio already gates 3D input on `!io.WantCaptureMouse` (`app.cc` HandleMouseEvents) and
has pick-ray math (`interaction.cc` `Pick`/`MakePickRay`). Our `ViewportPlugin` receives a
normalized `ViewportInput` (pos/delta/buttons/modifiers + model/data/camera) and returns
`true` to consume the event (suppress camera orbit) — exactly our `host/app.cc:198-222`.
**Studio patch: build a `ViewportInput` and `ForEachPlugin<ViewportPlugin>` before the
camera-motion block in `HandleMouseEvents`** — a handful of lines.

### 2.3 Docking their layout + hiding their spec panels

- **Docking.** Studio's `ConfigureDockingLayout` (`platform/ux/gui.cc`) pre-docks a fixed
  set of window names via `DockBuilderDockWindow`: `Options` (left 22%), `Explorer` /
  `Editor` / `Inspector` (right, tabbed), `Properties` (right-bottom), `Stats`
  (left-bottom), `Profiler`, and `Dockspace` (passthrough center). A `GuiPlugin` window is
  **not** in that list, so it floats on first run (then persists via ini). To land our
  Hierarchy/Details in their layout, either (i) name our windows to match a docked node,
  or (ii) add `DockBuilderDockWindow("Hierarchy", inspector)` etc. — a **few lines** in
  `ConfigureDockingLayout`.
- **Hiding their editor.** Their Explorer + Editor + **Inspector** share one runtime bool
  `tmp_.inspector_panel` (Shift+Tab). Flipping it off also kills the Data Inspector, and
  there is no compile flag. To cleanly hide only the old-spec Explorer/Editor while keeping
  the Data Inspector, **split that flag** (2-3 lines in `app.h`/`app.cc`) or drop the
  `SpecExplorerGui`/`SpecEditorGui` calls behind a new `-DSTUDIO_HOST_EXTERNAL_EDITOR`
  guard. Either is a small, localized edit.

### 2.4 Total engine-side change

All four gaps + docking + panel-hide are small, additive, well-localized edits to
`app.cc`/`app.h`, `gui.cc`, `model_holder.*`, `renderer.*` — on the order of **~80-150
lines total**, none of it invasive, and every one already prototyped in our `host/` +
`platform/` shell. `GuiPlugin` / `KeyHandlerPlugin` / `SpecEditorPlugin` need nothing.

---

## 3. The fork question: where does the integration live

Context: the protospec repo's only git remote (`fork`) is
`github.com/jonathanembleyriches/mujoco`; branches `main` + `protospec` there currently
carry **our** repo, not a MuJoCo checkout. The vendored `third_party/MuJoCo/src` is a
read-only dependency of the UnrealRoboticsLab plugin and must stay untouched. Studio needs
~80-150 lines of edits *inside* the MuJoCo tree (§2), which the vendored-copy model
cannot absorb.

**Options weighed:**

- **(a) Patch files in protospec, applied to a Studio checkout at build time.** Keeps
  protospec self-contained and the MuJoCo tree pristine. But patches against a live
  `app.cc`/`gui.cc` are brittle across upstream churn (Studio is explicitly WIP), CI must
  clone + patch + build Filament every run, and reviewing a `.patch` is worse than a diff.
  Fine as a *transitional* mechanism, poor as the home.
- **(b) A `studio` branch on the owner's mujoco fork, cut from upstream, with ProtoSpec
  as a subdirectory + the §2 patch.** The engine edits live as normal tracked commits
  (reviewable diffs, mergeable), the ProtoSpec editor sits in e.g.
  `src/experimental/studio/protospec/` as a CMake subdir, and upstream is pulled in with
  ordinary merges. Costs: a second repo relationship (the fork's `main`/`protospec` carry
  our repo today, so the `studio` branch must be cut from **upstream** `google/mujoco`,
  not from those), and the protospec editor sources are then either vendored there or
  git-subtree'd from the protospec repo. This is how DeepMind expects Studio forks to be
  extended, and it makes the "candidate for upstreaming" note in our `ps_plugin_ext.h`
  real — the four extension seams can be PR'd upstream later.
- **(c) Vendor the full studio + platform + filament source into `apps/studio`.**
  Self-contained and no second repo, but it drags the entire Filament build into
  protospec CI, balloons the repo, and forces us to manually track upstream Studio changes
  by hand — heavier than the thin shell it replaces, and it buries the engine patch inside
  our tree where it can't be upstreamed.

**Recommendation: (b).** Cut a `studio` branch from **upstream** `google/mujoco` on the
`jonathanembleyriches/mujoco` fork; land the §2 App/gui/model_holder/renderer patch as
tracked commits; carry the ProtoSpec editor as `src/experimental/studio/protospec/`
(git-subtree from the protospec repo so the editor sources stay single-sourced). The
editor is plugin-shaped and windowless-testable, so it builds as a small CMake subdir that
links `mujoco::platform`. CI builds Filament once and caches `_deps`. Use **(a)** patch
files only as the bootstrap while branch (b) is being stood up. Keep our four extension
seams factored exactly as in `ps_plugin_ext.h` so they remain upstreamable.

---

## 4. Migration cost table (per SE0-SE3 deliverable)

Our `apps/studio` is ~36 source files. The split that matters: the `editor/` cluster
(~5,900 LOC) is deliberately **windowless-core + thin ImGui skin** and built against the
verbatim plugin ABI, so it ports as-is. The `platform/` + `host/` substrate (~1,400 LOC)
was always a stand-in for Studio and is what gets replaced.

| Deliverable | Files (apps/studio) | Ports as-is | Rework | Dies |
|---|---|---|---|---|
| **SE0 host / substrate** | `host/app.*`, `platform/hal/window.*`, `platform/hal/renderer.*`, `main.cc` | — | The frame loop, overlay-hook fan-out, model-adopt, viewport-input dispatch become the **§2 Studio patch** (~80-150 lines in their `app.cc`/`renderer.*`/`model_holder.*`). Logic is a direct port of `host/app.cc`. | The window/GL/context code (Studio's SDL2+Filament HAL supersedes our `hal/`); our classic-`mjr` `Renderer::Render`; our `main.cc` entry (Studio owns `main`). |
| **SE0 pick math** | `platform/ux/interaction.*` | Pick-ray / camera math is a copy of MuJoCo's; Studio already has `interaction.cc` `Pick`/`MakePickRay`. | Point our `ViewportPlugin` at Studio's pick helpers instead of our copy. | Our vendored `interaction.cc` copy (Studio's is the original). |
| **Plugin ABI + registry** | `platform/ux/plugin.h`, `ps_plugin_ext.*`, `registry.*` | `plugin.h` is verbatim → drop ours, use Studio's. `ps_plugin_ext.h` (4 ext structs) ports as-is (add to Studio's platform). | Registry: Studio's `plugin.cc` provides `RegisterPlugin`/`ForEachPlugin`; instantiate our 4 ext types against it (our `registry.inc.h` mechanism is equivalent). | Our `registry.inc.h`/`registry.cc` reimpl (Studio's registry replaces it). |
| **SE1 Hierarchy** | `editor/hierarchy_model.cc` (271), `editor/hierarchy_panel.*` (365) | **Yes** — a `GuiPlugin`, windowless model builder + thin ImGui skin. | Only: dock into Studio's `inspector` node (1 `DockBuilderDockWindow` line). | — |
| **SE1 Details** | `editor/details_panel.*` (868) | **Yes** — reflection-driven, windowless core (`test_details`) + `GuiPlugin` skin. | Dock placement only. | — |
| **SE1 editor core / undo** | `editor/editor_ops.*` (553), `editor/undo.*` (212), `editor/editor_context.h` (177) | **Yes** — fully windowless; the DR-S1 authority (ProtoSpec tree + `bridge::Compiled` + Binding). | None (compile path is ours, not Studio's ModelHolder). | — |
| **SE2 gizmos** | `editor/gizmo.*` (648), `editor/gizmo_math.*` (255), `editor/transform_math.*` (710) | **Yes** — ImGui-drawlist (renderer-agnostic), delta math windowless-tested (`test_gizmo`). Works over Filament unchanged (foreground draw list). | None. | — |
| **SE2 viewport plugin** | `editor/viewport_plugin.cc` (353) | **Yes** — the `ViewportPlugin` + `ViewportGuiPlugin` + `OverlayPlugin` + tool keys. | Its three ext hooks need the §2 Studio dispatch seams wired (input/overlay/gui-draw). | — |
| **SE3 authoring** | `editor/authoring_ops.*` (1015), `editor/asset_import.*` (195) | **Yes** — windowless SDK ops (add/duplicate/reparent/new-model/mesh-import+VFS), `test_authoring`. | None (operate on our ProtoSpec tree, recompile via our bridge). | — |
| **SE editor panels** | `editor/panels.cc` (235) | **Yes** — Assets/Diagnostics/File/+Add `GuiPlugin`s + undo/redo/save keys. | Dock placement. | — |
| **Tests** | `test/test_studio.cc`, `test_gizmo.cc`, `test_details.cc`, `test_authoring.cc` (~42 fns / ~319 CHECKs) | **Yes** — all windowless, link only the core + editor TUs; run headless in CI unchanged. | Rewire the 2 in-binary smoke paths (`SmokeRun`/`SmokeEditRun`) onto Studio's `App` loop. | — |
| **Studio's old editor** | (their side) | — | — | We hide (not delete) their `SpecExplorerGui`/`SpecEditorGui`/`SpecEditor`/`ModelHolder`-from-spec editing — the old-spec authoring we replace. |

**Quantified:** of the ProtoSpec editor, **~5,900 LOC (the entire `editor/` cluster + the
4 ext plugin structs + all 4 test binaries) ports as-is** — only re-homed under Studio's
CMake and pointed at Studio's registry/pick helpers. The **rework is confined to the
~80-150-line Studio engine patch** (which is a straight port of our `host/app.cc` logic)
plus dock-placement one-liners. **What dies is only the stand-in substrate** we built to
survive without Studio: our `hal/window`, `hal/renderer` (classic `mjr`), `main.cc`,
`interaction.cc` copy, and the `registry.*` reimpl — none of it editor logic. The
plugin-first architecture is what makes this lopsided in our favour: the editor never
depended on the renderer or the window, only on the verbatim plugin ABI + our four
documented extensions.

---

## 5. Answers to the brief

- **Spike outcome:** see §1.5 + the outcome block. Configure succeeds; Studio requires
  `MUJOCO_BUILD_STUDIO=ON` + `MUJOCO_USE_FILAMENT=ON`; Filament + SDL2 + imgui + implot +
  webp are all built from source (no prebuilts); the risk was Filament-on-MSVC and the
  machine has Vulkan 1.4 + an OpenGL GPU (default backend `FilamentOpenGl`).
- **Recommended hosting option:** **(b)** — a `studio` branch cut from upstream on the
  owner's mujoco fork, ProtoSpec editor as a subdir + a small reviewable engine patch;
  patch-files **(a)** only as bootstrap.
- **Smallest-modification map:** upstream already has `GuiPlugin` / `KeyHandlerPlugin` /
  `SpecEditorPlugin` (zero change). Add our four `ps_plugin_ext.h` types and wire four
  small dispatch seams — `ModelSourcePlugin` → `ModelHolder::FromModel` adopt (~15-30 LOC),
  `OverlayPlugin` → hook after `mjv_updateScene` (~5-10 LOC), `ViewportGuiPlugin` →
  dispatch in `BuildGui` (handful), `ViewportPlugin` → dispatch in `HandleMouseEvents`
  (handful) — plus dock our windows (`ConfigureDockingLayout`) and split
  `inspector_panel` to hide the old spec editor. ~80-150 LOC total, all prototyped in
  `host/app.cc`.
- **Migration cost:** the whole `editor/` cluster + tests (~5,900 LOC) ports as-is; rework
  is the ~80-150-line engine patch + dock lines; only the thin Studio-stand-in substrate
  dies.
