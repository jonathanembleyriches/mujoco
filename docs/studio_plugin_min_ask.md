<!-- Minimum-ask redesign: running the editor on upstream Studio's existing plugin surface.
     Companion to studio_plugin_feasibility.md. Investigation date 2026-07-19. -->

# ProtoSpec Studio host integration — minimum-ask redesign

Read-only investigation. Upstream = MuJoCo-fork `mujoco-studio` **below merge-base `67a1ea6d`** (all citations `file:line` at that commit unless tagged *fork HEAD*). Prior study (`docs/studio_plugin_feasibility.md`) mapped an **8-plugin-type ask**. This report shows that ask collapses to **0 hard asks + 1 optional tiny hook**, by running the entire editor on the **4 plugin types upstream already ships** plus ImGui globals, at the cost of enumerated, mostly-cosmetic degradations. Probes compiled against `build_ps/lib/libmujoco.so`.

## The two load-bearing discoveries

**1. The mouse-consume linchpin holds — cleanly, without an invisible window.** Upstream `App::HandleMouseEvents()` gates *all* camera + perturb handling on ImGui capture at the very top:

```
app.cc:439  void App::HandleMouseEvents() {
app.cc:441    if (io.WantCaptureMouse) { return; }   // ← host does nothing with the mouse
```

The vendored ImGui is **1.92.5** and exposes `ImGui::SetNextFrameWantCaptureMouse(bool)` (`imgui.h:1141`) — the official app-side override of `io.WantCaptureMouse`. So a plugin that wants the mouse (gizmo hot / active manipulation) calls `SetNextFrameWantCaptureMouse(true)` in its `GuiPlugin::update`; next frame the host's `HandleMouseEvents` early-returns and the plugin owns the drag. **No invisible full-viewport window needed** — which sidesteps the whole "our capture window shadows other host UI" hazard entirely. The plugin reads the mouse from `ImGui::GetIO()` (MousePos/MouseDown/MouseDelta/MouseWheel/KeyCtrl/…), all global inside `GuiPlugin::update`. The fork's own editor already reads `ImGui::GetIO().WantCaptureMouse` directly (fork `editor/viewport_plugin.cc:564`), proving the editor library reaches ImGui globals today.

**2. The camera — the "suspected irreducible residue" — is already delivered to a plugin every frame.** `ProcessPendingLoads()` runs each `Update()` and dispatches `SpecEditorPlugin::pre_compile` **unconditionally**, handing it the live camera pointer:

```
app.cc:404  ForEachPlugin<platform::SpecEditorPlugin>([&](auto* plugin){
app.cc:406    if (plugin->pre_compile(plugin, spec(), model(), data(), &camera_)) {
```

This is the *only* plugin conduit to `mjvCamera` (no other callback carries it), but it is **per-frame**, not compile-time. A plugin ignores the `mjSpec`, caches `&camera_` (the pointer is stable — it's `App::camera_`), always returns `false` (never triggers the host recompile), and thereafter has permanent read access to the live camera for pick-rays and gizmo projection. Because the underlying `App::camera_` is genuinely non-const, `const_cast`-ing the `const mjvCamera*` to **write** `lookat` (frame-selection) is well-defined — so even mutable-camera is reachable, if hackily. The viewport rect is **not** a separate residue: the 3D scene renders full-window behind a `ImGuiDockNodeFlags_PassthruCentralNode` dockspace (`gui.cc:317-320`) and host picking normalizes to full `io.DisplaySize` + `window aspect` — so the editor's existing `BuildViewProj` ray-cast matches by using `GetIO().DisplaySize`/full-window aspect, with no host rect needed.

## Per-capability mechanism table (old ask → existing-hook mechanism → degradation → verdict)

| Old ask (prior study) | Zero-ask mechanism on the 4 existing plugin types | Degradation | Verdict |
|---|---|---|---|
| **ViewportPlugin** (mouse + consume) | `GuiPlugin::update` reads `ImGui::GetIO()` mouse/mods; `SetNextFrameWantCaptureMouse(true)` when gizmo hot suppresses host camera+perturb (both sit below the `WantCaptureMouse` gate, `app.cc:441`) | 1-frame capture latency (standard ImGui); rare 1-frame leak if press lands the same frame the cursor first reaches a handle | **ELIMINATED** |
| **ViewportGuiPlugin** (gizmo draw + camera) | `ImGui::GetForegroundDrawList()` (`imgui.h:1062`) from `GuiPlugin::update`; camera from `pre_compile` cache; mutable camera via `const_cast` for frame-select | Camera access is a `pre_compile` *abuse* (semantic mismatch; relies on per-frame dispatch continuing) | **ELIMINATED** (fragile — see optional hook) |
| **OverlayPlugin** (selection outline as `mjvGeom`) | Draw the wireframe AABB box via foreground draw list, projected screen-space | **Loses depth occlusion** — outline draws on top even when behind geometry. Current impl is already a wireframe box (not a solid silhouette), so screen-space edges look similar | **ELIMINATED**, degraded |
| **ModelSourcePlugin + CompiledModel + ModelHolder::FromModel** (live adopt — *the crux*) | `ModelPlugin::get_model_to_load` returns **MJB bytes** (`app.cc:416`); `post_model_loaded` (`app.cc:226`, fires *inside* `OnModelLoaded` same-frame) restores camera flicker-free; `do_update` (`app.cc:279`) re-injects qpos/qvel/act/ctrl and drives live preview | Full **Filament rebuild per reload** (`renderer.cc:87` Init→Deinit→CreateContext, called at `app.cc:200`); must move to **commit-on-release** recompile + screen-space/qpos preview; welded (jointless) bodies preview as a ghost until release; editor must write qpos in `do_update` (relaxes "never touch mjData") | **ELIMINATED** at discrete-edit cadence; see cost analysis |
| **FileDialogPlugin** | Plugin self-services via `portable-file-dialogs`/`zenity` subprocess (Linux) / `IFileDialog` (Windows) | None beyond what host already accepts — upstream's own dialog is a **blocking `popen`** (`file_dialog_zenity.cc:40`), so an in-plugin blocking call is no worse; PFD is async and strictly better | **ELIMINATED** |
| **MainMenuPlugin** | `GuiPlugin::update` may itself call `ImGui::BeginMainMenuBar()/EndMainMenuBar()` (global) and add top-level menus | **Cannot suppress host's stock File menu** (host-side only) → duplicate/renamed menu ("ProtoSpec" File) | **ELIMINATED**, degraded |
| **ToolbarPlugin** | Own dockable toolbar strip drawn in a `GuiPlugin` panel | Separate panel, not fused into the host's toolbar row | **ELIMINATED**, cosmetic |
| **EditorShellPlugin** (mode/dirty/error bridge) | Editor owns its Edit↔Play mode via its own toolbar; **freezes the sim itself** in `do_update` by returning `true` (host then skips `StepControl::Advance`, `app.cc:279-292`) and resetting `data->time`; dirty/error shown in own status strip | Host Play/Stop button won't mirror editor mode; can't *drive* host pause (no API) — but doesn't need to, the freeze trick makes host pause irrelevant | **ELIMINATED**, degraded |
| **Filament fast-path** (`mjrf_flush`+`mjrf_resetMaterialCache`+`UpdateModel`) | Only needed if reloading per-frame; avoided by commit-on-release | Per-commit reload hitch remains (same hitch upstream already shows on every file open) | **ELIMINATED** if commit-on-release; otherwise a valid perf ask |

## Probe evidence

**MJB round-trip (candidate 4), warm, 200 reps, this machine:**

| Model | bodies / dof | MJB size | save+load+makeData | byte-identical roundtrip | id-stable |
|---|---|---|---|---|---|
| humanoid.xml | 17 / 27 | 1.1 MB | **0.38 ms** | YES | YES |
| humanoid100.xml | 117 / 627 | 7.2 MB | **2.48 ms** | YES | YES* |
| grid800 (synthetic) | 801 / 4800 | 2.0 MB | **0.81 ms** | YES | YES |
| grid100 | 101 / 600 | 0.25 MB | **0.12 ms** | YES | YES |

*The humanoid100 "id-stable=NO" is a **test artifact, not instability**: a `<replicate>`-generated body has an empty name and can't be looked up by name (`mj_name2id`→-1). The **byte-identical roundtrip proves index/id order is fully preserved**. **Load-bearing conclusion for the crux:** the editor's Binding must key on **element index/id (stable), not name** — name-keying breaks on any unnamed compiled element. Probe: `scratchpad/min_ask/probe_mjb.cc`.

**Serialization is not the bottleneck** — sub-3 ms even for a 728-DOF scene. The cost that kills *per-frame* live-drag reload is the **renderer**, not the bytes: `Renderer::Init` unconditionally `Deinit()`s (`filament_context_.reset()`, `scene_bridge_.reset()`) then `CreateContext()` — a **full Filament engine rebuild** — and `OnModelLoaded` calls it every load (`app.cc:200`, also `app.cc:1720`). **Candidate-7 verdict: the Filament rebuild is an upstream reality, not a fork artifact** — upstream eats it on every file open. So the fork's fast-path is a genuine optimization, but it becomes **unnecessary** the moment the editor stops reloading per-frame.

## THE minimal ask list after redesign

**Hard asks: 0.** The full editor runs on `GuiPlugin` + `ModelPlugin` + `KeyHandlerPlugin` + `SpecEditorPlugin` (all shipping today) + ImGui globals + a dialog subprocess.

**Optional asks (1–2 tiny), each buying back one degradation:**

1. **`ViewportStatePlugin` — a legitimate per-frame `{const mjvCamera*, float aspect, bool paused}` delivery** (~15 lines struct + a 5-line dispatch beside `pre_compile`). *What it fixes:* replaces the `SpecEditorPlugin::pre_compile` **abuse** — today the camera arrives only by riding a hook designed for mjSpec editing whose per-frame dispatch is incidental and could change. This is the single ask worth making; it hardens the one fragile seam. Optional because the abuse works on today's code. Make `camera` mutable *only* if frame-selection is kept (the fork already tags frame-select "punt #4", fork `viewport_plugin.cc:302`); drop it and the field is `const`.

2. **Filament fast-path** (`mjrf_flush` + `mjrf_resetMaterialCache` + `Renderer::UpdateModel`, ~50 lines) — *only if* you refuse the per-commit reload hitch. Purely performance; identical hitch already ships upstream on file-open.

**What genuinely cannot be done without a hook (i.e. the degradation is irreducible):**
- **Depth-occluded selection outline** — needs `OverlayPlugin` (append `mjvGeom` post-`mjv_updateScene`). Screen-space is the only zero-ask option.
- **Suppressing the host's stock File menu** — host-side; zero-ask leaves a second menu.
- **Live 3D preview of a *jointless* body mid-drag without recompile** — the free-joint case previews via qpos in `do_update`; welded bodies can't move in-scene without either per-frame recompile (Filament rebuild) or a scene-mutation hook, so they show a screen-space ghost until release.

None of these three blocks a native editor; all are cosmetic/interaction polish.

## UX degradations accepted (honest ranking, worst-first)

1. **Selection outline loses depth occlusion** (screen-space box drawn over occluders). Most visible; a wireframe AABB already, so mild.
2. **Two File menus / renamed editor menu** (can't suppress host's).
3. **Per-commit reload hitch** if the Filament fast-path isn't taken (bounded, ≈ what file-open costs today).
4. **Welded-body drag previews as a ghost** until release.
5. **Separate toolbar panel** instead of an inline toolbar row; host Play/Stop button doesn't mirror editor mode.
6. **Editor writes qpos in `do_update`** for live preview — relaxes the "editor never touches mjData" trust-surface claim.

## Refactor plan for our fork (toward the zero-ask shape)

**Design pivot: commit-on-release, not recompile-per-frame.** This one change unlocks the bytes route (eliminates the ModelSource crux ask) *and* drops the Filament fast-path ask.

- **`editor/viewport_plugin.cc`** — Re-target the three viewport plugins onto `GuiPlugin`:
  - Mouse: replace `ViewportPlugin::on_mouse(consumed)` with a `GuiPlugin::update` that reads `ImGui::GetIO()` and calls `SetNextFrameWantCaptureMouse(true)` when the gizmo is hot. The `ViewportInput` struct's fields all reconstruct from ImGui IO + cached model/data/camera.
  - Gizmo draw: **already** on `GetForegroundDrawList()` (`viewport_plugin.cc:774,851`) — keep as-is.
  - Selection outline: move `OnOverlay`'s `mjv_initGeom` box (`viewport_plugin.cc:403-434`) to a foreground-draw-list projected box.
  - Edit gate: replace `ctx.sim_paused = vc.edit_mode` (read from host, `viewport_plugin.cc:335`) with a **self-owned** `EditorMode`; `CanEdit()` becomes `mode==Edit`. Freeze via `do_update` returning `true` + `data->time` reset.
- **`editor/editor_ops.*` + `protospec_editor.cc`** — Replace `ModelSourcePlugin::poll_compiled → App::AdoptCompiledModel` (fork HEAD `app.cc:527`, `ModelHolder::FromModel`) with a `ModelPlugin`: `get_model_to_load` serializes the freshly compiled `mjModel*` via `mj_saveModel`→buffer (`application/mjb`); `post_model_loaded` restores the cached camera same-frame; `do_update` migrates sim state and drives live preview. Re-key the **Binding on element index/id, not name** (probe finding).
- **Camera** — Add a thin `SpecEditorPlugin` whose `pre_compile` caches `&camera_` and returns `false`; feed the cached pointer to the gizmo/pick paths. (Swap to the optional `ViewportStatePlugin` if upstream grants it.)
- **`host/shell.cc`** — `MainMenuPlugin`/`ToolbarPlugin`/`EditorShellPlugin` collapse into panels/menus drawn from the editor's own `GuiPlugin`s (own toolbar strip; own File menu via `BeginMainMenuBar`; own status chip; self-owned mode).
- **File dialogs** — `editor/*` already posts dialog requests through a windowing-free phase machine (`EditorContext::FileDialogState`); point its `deliver` at an in-plugin `portable-file-dialogs` call instead of `FileDialogPlugin`/`App::ServiceEditorFileDialogs`.

**Fork seams deleted:** the entire `src/experimental/studio/protospec/` compat layer (`platform/ux/plugin.h`, `ps_plugin_ext.h` aliasing the 8 types, `registry.*`); the 8 struct additions in the fork's `platform/ux/plugin.h`; every `ForEachPlugin<Viewport*/Overlay/ModelSource/MainMenu/Toolbar/EditorShell/FileDialog>` dispatch site added to the fork's `app.cc`/`renderer.cc`/`gui.cc`; `App::AdoptCompiledModel` + `ModelHolder::FromModel`; the `MUJOCO_STUDIO_PROTOSPEC` `#ifdef`s. `plugin_abi.h`'s guards shrink to only the 4 upstream types.

## Old-8-asks vs new-minimal

| # | Prior study ask | Prior size | New minimal | Mechanism |
|---|---|---|---|---|
| b1 | ViewportPlugin (mouse) | ~85 | **0** | `SetNextFrameWantCaptureMouse` + ImGui IO |
| b2 | ViewportGuiPlugin (gizmo+cam) | ~45 | **0** (opt: tiny ViewportState) | foreground draw list + `pre_compile` camera |
| b3 | OverlayPlugin (outline) | ~20 | **0** (degraded) | screen-space projected box |
| c | ModelSourcePlugin (live adopt) | ~80 | **0** | `get_model_to_load` MJB + `post_model_loaded` + `do_update` |
| e | FileDialogPlugin | ~95 | **0** | in-plugin subprocess dialog |
| h1 | MainMenuPlugin | ~25 | **0** (degraded) | `BeginMainMenuBar` from GuiPlugin |
| h2 | ToolbarPlugin | ~20 | **0** | own toolbar panel |
| h3 | EditorShellPlugin | ~85 | **0** (degraded) | self-owned mode + `do_update` freeze |
| g | Filament fast-path | ~50 | **0 or keep** | commit-on-release avoids it; keep as perf ask |
| — | **Total upstream surface** | **~500 lines / 8 types** | **0 required; ~15 lines / 1 optional type** | — |

**Bottom line for the champion:** the ask drops from *"accept 8 additive plugin types + `ModelHolder::FromModel` + a Filament pair"* to *"nothing required; optionally one ~15-line `ViewportStatePlugin` to de-hack per-frame camera access, and optionally the Filament fast-path for smoothness."* The redesign trades that surface for six named, mostly-cosmetic degradations, the sharpest being a screen-space (non-occluded) selection outline and a per-commit reload hitch no worse than today's file-open. The structure also lets any **partially-landed upstream hook** (a real viewport-state or scene-overlay plugin) slot straight in to buy back the matching degradation.

Probes and scratch models are in `scratchpad/min_ask/` (`probe_mjb.cc`, `gen800.py`, generated `grid*.xml`); nothing was written to either repo.
