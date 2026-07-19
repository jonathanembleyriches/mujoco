<!-- Feasibility study: reformatting ProtoSpec Studio as a native MuJoCo Studio plugin.
     Investigation date 2026-07-19; fork merge-base 67a1ea6d (MuJoCo 3.10.0-dev). -->

# ProtoSpec Studio → native Studio plugin: feasibility study

Read-only investigation across the fork (`mujoco-studio` branch `studio`, merge-base `67a1ea6d`) and the ProtoSpec repo (`mujoco` branch `fix/linux-build-and-mode-ui`, `studio/` dir).

## TL;DR verdict

**Feasibility: HIGH (A-).** The editor is *already* written as a plugin cluster against a plugin API that deliberately mirrors upstream Studio's own `mujoco::platform::` registry. Going native is not a rewrite — it is getting upstream to accept a **bounded, mostly-generic set of 8 additive plugin struct types plus their dispatch call sites**, after which the fork's compat shims collapse and the editor library links against stock Studio unchanged. The single real design question is the **live-model adoption hook** (the crux); it has a clean minimal generic form. No core-engine change is needed — only `src/experimental` plus two Filament C-API functions.

**The web/networked viewer does NOT exist in this tree yet** — only the pre-existing upstream client-side WASM port. Whether our capabilities come "for free" hinges entirely on one unknown the Studio authors must answer: server-side-streamed-ImGui vs client-side-WASM.

---

## 1. Upstream's native plugin system (ground truth, below `67a1ea6d`)

**Location:** `src/experimental/platform/ux/plugin.h` (interface) + `plugin.cc` (backing). Namespace `mujoco::platform`.

**Registration / lifecycle — static, compile-time, no dynamic loading.** `RegisterPlugin<T>(T)` stores POD structs **copied by value** (no inheritance), keyed by case-insensitive `name`; `ForEachPlugin<T>(fn)` iterates them at each host call site. Backing is a per-type function-local `std::vector<T>` with an explicit-instantiation macro `MUJOCO_SPECIALIZE_PLUGIN`. **There is no `dlopen`/`LoadLibrary`/discovery anywhere** in `src/experimental` — "plugin" today means *statically linked + registered from `main()`*. One upstream example exists: `object_launcher_plugin.cc`.

**ABI style:** C-ABI-flavoured C++ — POD structs of raw function pointers, each callback taking `self` first and carrying a `void* data` back-pointer. Not virtual, not a C header; a hybrid a shared lib *could* bind to but which is currently linked statically.

**The four upstream plugin types and what they grant TODAY:**
- `GuiPlugin` — `update(self)` draws an ImGui window titled by `name`, listed in the **Plugins** menu → **ImGui panel drawing.**
- `ModelPlugin` — `get_model_to_load` returns a **serialized model byte buffer + content-type** that Studio loads through its normal pipeline; `do_update(mjModel*,mjData*)->bool` runs each tick with **read/write model/data** → **model-as-bytes provider + per-tick model/data access.**
- `KeyHandlerPlugin` — chord + `on_key_pressed` → **keyboard input.**
- `SpecEditorPlugin` — `pre_compile(mjSpec*, const mjModel*, const mjData*, const mjvCamera*)->bool` + `post_compile` → **mjSpec editing + recompile handoff + camera/model/data read.**

**Upstream gives today:** ImGui panels ✓, keyboard ✓, read model/data ✓, write model/data ✓ (`do_update`), camera **read** ✓, model replacement **only as serialized bytes** or **only via mjSpec authoring**.
**Upstream does NOT give:** main-menu/toolbar contributions; viewport mouse/pick; per-frame viewport draw (gizmos); mutable camera; appending transient geoms to the live `mjvScene`; host-serviced file dialog; a **live-`mjModel*` adopt** (zero-copy, state-preserving); renderer control.

## 2. What our editor actually consumes

**Key finding: the fork has already reshaped every seam into the upstream idiom.** It added **8 new plugin struct types** to `src/experimental/platform/ux/plugin.h` (+187 lines) in exactly upstream's POD+function-pointer+`void* data` style, registered them through upstream's `RegisterPlugin`/`ForEachPlugin` (`plugin.cc` +16), and wired dispatch into `app.cc` (+404), `renderer.cc`, `gui.cc`. The editor side (`/home/buzz/Documents/proto/mujoco/studio/editor/plugin_abi.h`) hand-syncs POD copies and `static_assert`s field offsets — with the stated goal (verbatim): *"the ext plugin structs are HAND-SYNCED against the host's `mujoco::platform::` definitions … so a plugin loads into stock Studio."* The editor is a clean library (`mujoco::protospec_editor`) whose whole cluster shares one `EditorContext` (`editor/editor_context.h`) via each plugin's `data` pointer; nothing editor-specific lives in the host `App`.

Capability list our editor needs, and the seam carrying it:

**(a) ImGui panels + dockspace.** Panels via `GuiPlugin` (**already upstream**). The curated Unity/Unreal dock layout is a host-side edit to `gui.cc` `ConfigureDockingLayout` (versioned id `RootV3`) — **cosmetic, not a plugin API**, optional for going native.

**(b) Viewport GUI context** (`studio/editor/viewport_plugin.cc`, 665 lines):
- **Mouse/pick:** `ViewportPlugin` + `ViewportInput` (normalized mouse + modifiers + model/data/camera/vis_option; `on_mouse` returns *consumed* to suppress default camera/perturb). The editor **ray-casts itself** (`BuildViewProj`) — needs camera+viewport metrics+mouse, **not** an upstream `mjv_select`. Lowers the ask.
- **Gizmo draw:** `ViewportGuiPlugin` + `Context{model,data,mutable camera,aspect,edit_mode}`; also where the editor **latches the edit gate** (`sim_paused=ctx.edit_mode; sim_time=data->time`).
- **Selection outline:** `OverlayPlugin` — appends transient `mjvGeom` to the live `mjvScene` after `mjv_updateScene` (dispatched in `renderer.cc::Render`).
- Keys: `KeyHandlerPlugin` (**already upstream**; fork only reordered so plugin keys run before built-ins).

**(c) MODEL SOURCE / ADOPTION — the crux** (`studio/editor/protospec_editor.cc`, `editor_ops.*`): `ModelSourcePlugin{submit_load(path), poll_compiled(CompiledModel*)->bool}` + `CompiledModel{mjModel*}`. The editor owns a non-mjSpec tree (`ps::mjcf::Model`) and its own compiler; publishes a **live `mjModel*`** it keeps owning; the host **adopts** it via `App::AdoptCompiledModel` + `ModelHolder::FromModel(owns=false)`. Adoption **preserves sim/camera state** (`preserve_camera_on_load_`, `model_source_fresh_`) so gizmo-drag recompiles don't reframe the camera; fresh file loads do. Upstream's existing hooks don't suffice: `ModelPlugin::get_model_to_load` round-trips **bytes** every drag and **discards the in-memory Binding** mapping compiled elements to authored-tree serials; `SpecEditorPlugin` requires an **mjSpec**, which ProtoSpec never produces.

**(d) Sim-state observation (edit gate):** `CanEdit() = model_ready && sim_paused && sim_time==0`, latched from `ViewportGuiPlugin::Context.edit_mode` + `data->time`. Upstream has no separate "am I paused" plugin signal; this rides on (b).

**(e) File dialogs:** `FileDialogPlugin{poll,deliver,is_save/is_multi/is_folder}`. The editor carries **no windowing dependency** — it posts a request (`EditorContext::FileDialogState`, a pure phase machine) and the **host services the native dialog** (`App::ServiceEditorFileDialogs`). Host file_dialog backends got small multi-select/folder additions.

**(f) Screenshot/headless:** `App::ServiceScreenshots`/`WriteScreenshot` + `Renderer::CaptureWindowRGB` (classic backend) + `SaveToPng` (lodepng) + CLI flags. **A host feature, not a plugin capability** — orthogonal; upstream independently or drop.

**(g) Renderer fast-path:** `Renderer::UpdateModel` reuses the persistent Filament engine + swap chains across recompiles (rebuilds only the per-model scene) instead of a full `Init()` on every drag. Relies on two new Filament C-API calls `mjrf_flush` + `mjrf_resetMaterialCache` (`src/render/filament/mjrfilament.{h,cc}`, backing `FilamentContext::FlushAndWait`/`ResetMaterialCache`). **Known-additive, isolated, pure performance.**

**(h) Others** (host glue in `protospec/host/shell.cc`): `MainMenuPlugin` (File/Edit menus; **host suppresses its own File menu when present**), `ToolbarPlugin` (transform/snap/add controls), `EditorShellPlugin` (**Play/Stop mode bridge**: `set_mode(0/1)`, `is_dirty`, `error_count`, `focus_diagnostics` — the most ProtoSpec-shaped hook).

## 3. Gap analysis

Legend: HAVE (upstream already) · NEAR (near-miss) · MISSING.

| # | Capability | Status | Minimal upstream addition | Size | Generic? |
|---|---|---|---|---|---|
| a1 | ImGui panels | **HAVE** `GuiPlugin` | — | 0 | generic |
| a2 | Curated dockspace | host-cosmetic | `gui.cc` layout; not a plugin API | ~40 | n/a |
| b1 | Viewport mouse/pick | **MISSING** | `ViewportInput`+`ViewportPlugin{on_mouse->consumed}`, dispatch before camera/perturb; no `mjv_select` needed | ~85 | **generic** |
| b2 | Viewport draw + mutable camera | **NEAR→MISSING** | `ViewportGuiPlugin{draw(Context)}`; upstream exposes camera only `const` and has no viewport-draw hook | ~45 | **generic** |
| b3 | Append geoms to scene | **MISSING** | `OverlayPlugin{add_overlay(…,mjvScene*)}` | ~20 | **generic**, tiny |
| b4 | Keyboard | **HAVE** | (opt) plugin keys before built-ins | 0–5 | generic |
| c | **Live-model adopt** | **NEAR→MISSING** | `ModelSourcePlugin`+`CompiledModel`+`ModelHolder::FromModel`+`AdoptCompiledModel`; `ModelPlugin`=bytes round-trip loses Binding, `SpecEditorPlugin`=needs mjSpec | ~80 | **generic if framed** |
| d | Sim paused/time gate | **NEAR** | rides on b2's `edit_mode`+`data->time` | 0 | generic |
| e | Host file dialog | **MISSING** | `FileDialogPlugin`+`ServiceEditorFileDialogs` | ~95 | semi-generic |
| f | Screenshot/headless | host feature | not a plugin API | ~90 | orthogonal |
| g | Renderer fast-path | **MISSING**, isolated | Filament pair + `UpdateModel` | ~50 | **generic** |
| h1 | Main-menu contrib | **MISSING** | `MainMenuPlugin{draw}` (File-suppress opinionated) | ~25 | generic |
| h2 | Toolbar contrib | **MISSING** | `ToolbarPlugin{draw}` | ~20 | generic |
| h3 | Mode/dirty/error bridge | **MISSING** | `EditorShellPlugin` | ~85 | **ProtoSpec-shaped** |

**Crux (row c) design note.** Smallest generic form: *the host already owns the load→compile→adopt→render→step loop and the `mjModel*/mjData*`; add one hook by which a registered plugin hands the host a freshly-built `mjModel*` to adopt in place (retaining ownership), plus a deferred `submit_load(path)` so file-open still flows through the host.* That is `ModelSourcePlugin` verbatim; it sits beside `ModelPlugin` (bytes) and `SpecEditorPlugin` (mjSpec) as the "already-compiled native model" case, needing only `ModelHolder::FromModel(owns=false)` host-side. Only ask with real design content; the rest is mechanical.

**Total upstream surface:** ~190 lines of structs in `plugin.h`, ~16 in `plugin.cc`, ~250 lines of `ForEachPlugin` dispatch across ~12 call sites in `app.cc`/`renderer.cc`, `ModelHolder::FromModel` (~15), Filament pair + `UpdateModel` (~50). No `src/engine` changes.

## 4. Web / networked viewer readiness

**Finding: there is NO server/web/networked/streaming viewer in this tree today.** A thorough search (websocket, server, stream, remote, headless, rtc, grpc, rpc, encode/h264/jpeg, offscreen, wasm, all branches/history) found **no server, no websocket, no remote transport, no server-side ImGui streaming, no encoder**. The only "web" artifact is the **pre-existing upstream client-side WASM port** (`src/experimental/studio/emscripten.cc` + `live.{js,html,css}`): the entire Studio app (Filament WebGL + ImGui) runs **in the browser** as WASM; "network" there is only `fetch()` for assets. Offscreen rendering exists (`renderer.cc ReadPixels`, `graphics_mode.h *Headless`, `window.cc Present(pixels)`) but is **purely local** (offscreen buffer → local SDL window / WASM canvas) with no transport. The platform layer has **no client/server or render-here-display-there seam**. All web/WASM/headless code is upstream; no fork commit touches networking.

**Which capabilities survive remoting (conditional):**
- *If server-side-streamed ImGui:* panels, menu, toolbar, gizmos (`ViewportGuiPlugin`), overlays (`OverlayPlugin`), model adoption, sim-state — **all server-side, all survive.** Input arrives as client events our `ViewportPlugin`/`KeyHandlerPlugin` consume. **File dialogs are the one break** — a native OS dialog can't open on the server and server paths are meaningless remotely; Open/Save must become client file-pick + byte upload/download. Our `FileDialogPlugin` poll/deliver bridge is already the *right shape* to remote (the editor never opens a dialog itself), but its payload must change from a path to bytes.
- *If client-side WASM* (like today's `emscripten.cc`): a statically-linked editor **compiles to WASM and runs client-side unchanged** — dialogs already `#ifndef __EMSCRIPTEN__`'d out. Going native then buys correctness/maintenance, not remoting per se.

**Precise questions for the Studio authors:**
1. Is the upcoming web viewer **server-side render-and-stream** or **client-side WASM** (whole app in browser)? The plugin story differs completely.
2. In the streamed model, is **ImGui rendered server-side and streamed** so plugin panels/toolbars/gizmos surface remotely unchanged — or is the client UI native/HTML with only the 3D view streamed (ImGui plugin UI then does **not** surface, and we'd need an HTML control surface)?
3. Do plugins run **server-side** in the web build, and is registration still the **static `RegisterPlugin<T>` table**, or will there be dynamic discovery/loading (none exists today)?
4. How do **file open/save** work for a remote client — client file-picker + byte upload/download, or server-filesystem paths only? (Our editor already routes all dialog I/O through a host-serviced poll/deliver bridge — is that the shape you want?)
5. Is model **loading/adoption** still host-driven in the web build, and can a plugin publish a **live `mjModel*`** for the host to adopt, or must models cross as serialized bytes?
6. What's the **stability commitment** for the `src/experimental/platform/ux` plugin API — a candidate to become a stable public plugin ABI, or will it keep churning?
7. Does "plugins supported automatically" mean the **existing** `GuiPlugin/ModelPlugin/…` surface, and would you accept **additive** plugin types (viewport mouse, viewport-gui draw, scene overlay, model-source adopt, menu/toolbar) into that same interface?

## 5. Verdict + migration

**(a) Grade: A- (high).** Editor is already a plugin cluster over a registry mirroring upstream's, with hand-synced ABI-guarded structs whose stated purpose is loading into stock Studio. No core change. Only real design question is the adopt hook, which has a clean generic form.

**(b) Ranked upstream ask list (smallest first — pitches are quotable):**

**1. Filament fast-path (`mjrf_flush` + `mjrf_resetMaterialCache`, + `Renderer::UpdateModel`).**
> Swapping the model on a live Filament context currently forces a full engine rebuild, because there's no way to quiesce the render thread and drop the stale material cache before tearing down the old scene. These two small C-API calls — flush the render queue, reset the material cache — let the renderer keep the model-independent engine and swap chains and rebuild only the per-model scene. Any tool that reloads or hot-swaps a model benefits; it turns a visible hitch on every reload into a cheap scene rebuild. Six lines of API plus an `UpdateModel` path; no behavior change for anyone who keeps calling `Init`.

**2. `OverlayPlugin` — append transient geoms to the live scene.**
> A one-callback plugin type that runs right after `mjv_updateScene` and appends `mjvGeom`s before the scene is rendered — selection outlines, debug markers, trajectory ribbons. ~15 lines of header and a five-line dispatch. Purely additive and useful to any visualization plugin, not just an editor.

**3. `ViewportPlugin` + `ViewportInput` — viewport mouse hook.**
> Today a plugin can draw UI but can't touch the 3D viewport's mouse. This adds a per-frame snapshot of normalized mouse state, modifiers, and the current camera/model/data, delivered before the app's own camera/perturb handling; a plugin returning "consumed" suppresses the default for that frame (only when ImGui isn't already capturing the mouse). It's the generic hook every direct-manipulation tool needs — gizmos, measurement, painting — and it carries no picking API, since a plugin can ray-cast from the camera it's already given.

**4. `ViewportGuiPlugin` — per-frame viewport overlay draw with a mutable camera.**
> A hook that runs during GUI building with the live camera, viewport aspect, and an `edit_mode`/paused flag, so a plugin can draw screen-space overlays that project correctly against the rendered scene — transform gizmos, on-screen handles, HUDs. The camera is mutable so a plugin can service a "frame selection" request. It's the natural companion to the mouse hook and, together, is what makes in-viewport tooling possible for any plugin author.

**5. `MainMenuPlugin` + `ToolbarPlugin` — menu-bar and toolbar contributions.**
> Two draw-callback plugin types letting a plugin add top-level menus and toolbar controls inside the host's existing bars. Generic UI real-estate any nontrivial plugin wants. The one opinionated bit is that when a plugin contributes menus the host can suppress its stock File menu (an editor replaces it); if you'd rather keep that host-side, the toolbar half stands alone.

**6. `FileDialogPlugin` — plugin-requests-host file dialog.**
> A plugin library often can't (and shouldn't) carry a windowing/native-dialog dependency, and — importantly for the web viewer — the dialog may not even be local. This is a poll/deliver bridge: each frame the plugin posts a pending open/save/folder request with a starting hint, the host services it however it wants (native dialog on desktop, client file-picker + upload when remote), and delivers the outcome back. Designing it as request/deliver rather than a blocking call is exactly what makes it remotable. (We'd simplify our opaque `kind` ints to a small enum before proposing.)

**7. `ModelSourcePlugin` + `CompiledModel` + `ModelHolder::FromModel` — a plugin may provide/replace the live model.** *(the crux)*
> Studio already owns the load→compile→adopt→render→step loop. Today a plugin can hand Studio a model only as serialized bytes (loaded through the XML/MJB pipeline) or by authoring an mjSpec. An editor with its own compiler and model representation wants neither: it holds a freshly built `mjModel*` in memory and needs Studio to adopt *that*, in place, on every recompile — without a bytes round-trip, without losing the in-memory binding that maps model elements back to the editor's tree, and preserving sim/camera state across the swap. The ask is one plugin type (`submit_load(path)` so file-open still flows through the host; `poll_compiled -> mjModel*` which the plugin keeps owning) plus a non-owning `ModelHolder::FromModel`. It slots in beside the existing model hooks as the "already-compiled native model" case, and it's the one piece that unlocks a genuinely native external editor.

**8. `EditorShellPlugin` — editor mode/dirty/error bridge.** *(lowest priority; most editor-shaped)*
> This bridges the host's Play/Pause/Stop toolbar to an editor's Edit↔Play machine and surfaces dirty/error state in the status bar. It's the most editor-specific ask; we'd propose decomposing it (a generic status/dirty-badge provider vs. editor-specific mode transitions) or keeping it fork-side until the shape settles. Not a blocker for a mostly-native editor.

**(c) Phased migration sketch.**
- **Moves upstream:** the 8 plugin structs (`plugin.h`) + `plugin.cc` specializations; the `ForEachPlugin` dispatch sites in `app.cc`/`renderer.cc`; `ModelHolder::FromModel` + `App::AdoptCompiledModel`; `Renderer::UpdateModel` + the Filament pair. (Screenshot/headless upstreamed separately or left a fork flag.)
- **Deleted from the fork:** the entire `src/experimental/studio/protospec/` compat layer (`platform/ux/plugin.h`, `ps_plugin_ext.h`, `registry.inc.h`/`registry.cc`, `ps_plugin_ext.cc` — pure `ps::studio::`→`mujoco::platform::` shims); the `MUJOCO_STUDIO_PROTOSPEC` `#ifdef`s in `main.cc`/CMake collapse into normal plugin registration.
- **Stays as-is:** everything in `/home/buzz/Documents/proto/mujoco/studio/editor/*` and `host/shell.cc` — they already code against the plugin structs; once those are the upstream ones, the editor library links against stock Studio unchanged. `plugin_abi.h`'s ABI guards keep protecting the boundary.

**(d) What we can do NOW to converge (mostly already done):**
1. Keep the fork's 8 struct definitions **byte-identical** to whatever we propose upstream (the ABI-guard discipline already enforces this) so the swap is delete-shim-and-relink.
2. **Split the ask** in our tree: keep the generic hooks (Overlay/Viewport/ViewportGui/ModelSource/MainMenu/Toolbar/Filament) cleanly separable from the editor-shaped `EditorShellPlugin`, so upstream can take the easy 7 first.
3. **De-ProtoSpec-ify** the two shaped hooks now: replace `FileDialogPlugin`'s opaque `kind` ints with a small documented enum; decompose `EditorShellPlugin` into a generic status/dirty provider + editor-specific mode calls.
4. Confirm/document the editor **never writes `mjData` directly** (all changes go through recompile→adopt) — shrinks the trust surface of the model-source ask.
5. Write a **one-page RFC per ask** (the pitches above are the seed) so the champion can drop them into upstream discussion individually.

**(e) Risks.**
- **Experimental-API churn.** `src/experimental` carries no stability promise; an upstream rename breaks our hand-synced shims. Mitigated (not eliminated) by ABI guards — hence question 6.
- **Remoting unknowns dominate the "free web support" thesis.** If the web viewer is HTML-client + streamed-3D (not streamed-ImGui), our ImGui panels/toolbars/gizmos do **not** surface remotely and we'd need a parallel control surface — large, currently unscoped. Biggest open risk; rides on questions 1–2.
- **Static-only registration.** No dynamic plugin loading exists; if the web build expects server-side dynamic discovery, that's new upstream work (question 3).
- **Remote file-I/O semantics** (paths vs byte upload/download) need real design; our bridge is the right shape but the payload contract is unspecified (question 4).
- **`EditorShellPlugin`** is the least likely to be accepted verbatim; plan to carry it fork-side or redesign.

Note: I was unable to save `REPORT.md` to the scratchpad (the harness blocks subagents from writing report files) — the full content is above for you to relay or persist.
