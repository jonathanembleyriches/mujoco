# Upstream asks — ProtoSpec Studio editor

Prepared 2026-07-19 for discussion with the MuJoCo / Studio team. Context and
credibility first, then two small API asks, one optional build hook, and three
bug reports with paste-ready drafts.

## Context (why these asks are small)

ProtoSpec Studio is a model editor built **entirely as plugins of MuJoCo
Studio's existing plugin surface** — the four shipping types
(`GuiPlugin`, `ModelPlugin`, `KeyHandlerPlugin`, `SpecEditorPlugin`) plus the
recently-landed `ScenePlugin`, against upstream main `3990305` (3.10.1). After
deliberately migrating everything migratable into plugin-side implementations
(selection outlines ride `ScenePlugin`; the curated dock layout and
screenshot/headless capture are pure plugins; registration rides
`mjPLUGIN_LIB_INIT` exactly like `object_launcher`), our entire fork of the
Studio tree is **three files, +45/−8**:

- a ~19-line CMake mount block (build wiring for an out-of-tree plugin dir),
- one ~18-line functional hunk in `app.cc` (Ask 1 below),
- `.gitignore` hygiene.

`src/engine`, `src/xml`, `src/user`, `src/render`, `include/` are untouched.
Nothing below asks you to accept editor-specific machinery — each item is a
generic affordance any tool-plugin author would want.

## Ask 1 — Dispatch plugin `KeyHandlerPlugin` chords before built-in shortcuts

**The one functional hunk left in our fork; retires it entirely.**

Today (`app.cc`, `HandleKeyboardEvents`) registered plugin key chords are
checked only in the trailing `else` of the built-in chord chain, so built-ins
always win. Any direct-manipulation tool wants the standard editor bindings —
Q/W/E/R tools, Ctrl-S save, Ctrl-D duplicate, Space transport — and today every
one of those is swallowed by a built-in (Q/E camera-vis toggles, Ctrl-S
SaveXml, Ctrl-D PrintData, Space pause; Delete dies in an empty
`// TODO: SpecDeleteElement` branch). Proposal: check registered plugin chords
first and skip the built-in for that frame when a plugin consumed the chord
(or equivalently, give `on_key_pressed` a handled/consumed return). ~13 lines;
zero behavior change when no plugin binds the key. We carry exactly this patch
and can PR it as-is.

## Ask 2 — `ViewportStatePlugin`: per-frame `{const mjvCamera*, float aspect, bool paused}`

**Not a blocker — a de-hack.** Our editor already runs without it, by caching
the `mjvCamera*` that `SpecEditorPlugin::pre_compile` happens to deliver every
frame and always returning false from that hook. It works because
`ProcessPendingLoads` dispatches `pre_compile` unconditionally and
`App::camera_` is address-stable — but that is a semantic abuse of an
mjSpec-editing hook, and it is the single most fragile seam we have left. A
~15-line plugin type delivering read-only viewport state per frame (camera,
aspect, paused flag) beside the existing dispatches gives every viewport-aware
plugin (gizmos, measurement tools, on-screen handles, pick rays) a legitimate
conduit. Mutable camera only if you want plugins to service frame-selection;
`const` covers us.

## Ask 3 (optional) — an out-of-tree plugin mount point in Studio's CMake

Our remaining ~19-line CMake block does one generic thing: `option(...)` +
`add_subdirectory(<external plugin dir>)` + link + compile one registration TU
into the executable. A first-class
`MUJOCO_STUDIO_PLUGIN_DIRS` list variable doing exactly that would let any
external editor/tool ship as `-DMUJOCO_STUDIO_PLUGIN_DIRS=/path/to/plugin`
with a zero-line fork. Lowest priority — the mount block is tiny and stable —
but it makes the "Studio supports out-of-tree plugins" story complete.

## Bug reports (paste-ready drafts in `docs/EXCEPTIONS.md`, section "Issue drafts")

1. **§I-1 — cylinder actuator: 3-valued `bias` overruns a scalar, zeroing
   `gainprm` area.** `xml_native_reader.cc` reads the documented 3-valued
   `bias` attribute into a stack `double` (`ReadAttr(..., 3, &bias, ...)`);
   the overrun clobbers the adjacent `area` value. Re-verified live at
   `3990305` (today's main). Minimal XML repro + suggested fix in the draft.
   We deliberately do not replicate this in ProtoSpec's spec-authoring path
   and track the intentional divergence as a named exception.
2. **§I-2 — `mjs_setName` is O(n²·log n) via per-set `CheckRepeat`.**
   `CheckRepeat` sorts the full name vector on every `mjs_setName`; building
   an 800-body fully-named scene through the mjs API spends ~300 ms in the
   scan — and the same per-element scan taxes your own XML parse path.
   Benchmark repro + hash-set suggestion in the draft.
3. **§I-3 — Studio `LoadModelFromBuffer` null-spec crash on `application/mjb`
   (already fixed on main).** `spec_editor_.Reset(*spec())` dereferenced a
   null spec after an MJB buffer load at 3.10.0-era revisions; current main
   guards it. Framed as: confirmed fixed, here is a regression repro
   (a `ModelPlugin` returning MJB bytes) you are welcome to keep.

## Related, for the same conversation (not asks)

- **Web/networked viewer**: seven architecture questions we need answered
  before scoping any remoting work — `docs/studio_plugin_feasibility.md` §4
  (server-streamed ImGui vs client WASM decides whether plugin UIs surface
  remotely for free).
- **`include/mujoco/mjrfilament.h`**: if a render-flush + material-cache-reset
  pair ever lands in the now-public Filament C API, model hot-swap tools get a
  cheap scene rebuild; we currently avoid needing it (commit-on-release), so
  this is a nice-to-have, not an ask.
- A **context-local `mju_user_warning`** (handler with a userdata pointer)
  would let embedded compilers capture warnings without borrowing the
  process-global hook; we serialize our borrows under a mutex today.
