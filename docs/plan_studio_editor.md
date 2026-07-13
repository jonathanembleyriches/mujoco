# ProtoSpec Studio: design for a ProtoSpec-native model editor

Companion to `docs/plan.md`. This forks MuJoCo's experimental **Studio** app
(`src/experimental/studio` + `src/experimental/platform`, vendored tree) and replaces its
editing substrate — mjSpec — with the ProtoSpec object model. It is deliberately the first
consumer-grade test of ProtoSpec: the SDK, the reflection tables, the bridge
(Compile/Binding/Recompile), and the form-preserving writer all get exercised by a real
interactive tool. Unreal adoption remains out of scope; nothing here touches the UE plugin.

## STATUS (living section)

| Milestone | State | Evidence |
|---|---|---|
| SE0 vendored shell builds | queued | |
| SE1 ProtoSpec explorer + generated inspector | queued | |
| SE2 modeling mode: selection, gizmos, recompile loop | queued | |
| SE3 authoring ops: add/delete/import mesh | queued | |
| SE4 polish: undo depth, diagnostics panel, save UX | queued | |

---

## 1. What Studio already provides (verified against the vendored tree)

Studio is further along than expected — it is already an interactive mjSpec editor:

- Single-threaded SDL2 + Dear ImGui (docking) + ImPlot app; renderer HAL with two backends:
  classic `mjr`/OpenGL and Filament (`platform/hal/renderer.cc:89-115`, default FilamentVulkan).
- `ModelHolder` owns `mjVFS/mjSpec/mjModel/mjData`; deferred load/swap plumbing
  (`pending_load_` → `ProcessPendingLoads`, `app.cc:385-396`); pause/run/step + a frame-history
  scrubber (`StepControl`, `SimHistory`).
- An **Explorer** tab walking the spec into a Body Tree / Elements / Assets outliner
  (`app.cc:1118-1192`, `platform/ux/gui_spec.cc:217-271`).
- A **SpecEditor** engine: undo/redo via full-spec snapshots, add/delete for most element
  types, "Compile and Reload" (`platform/ux/spec_editor.{h,cc}`, `app.cc:1194-1296`).
- An editable property inspector: a per-type switch casting `mjs_as*` and binding struct
  fields to ImGui widgets (`gui_spec.cc:273-712`, `imgui_widgets.h`).
- Custom ray picking (`platform/ux/interaction.cc:553-575`) wired to selection and to
  `mjvPerturb` body nudging. **No transform gizmos** — the one big UI gap.
- Plugin hooks incl. `SpecEditorPlugin::pre_compile` for programmatic edit + recompile
  (`platform/ux/plugin.h:103-122`).

What we keep: the shell (window/renderer/dock/step control/picking/model swap). What we
replace: everything that touches mjSpec.

## 2. The core inversion

Studio's editing stack is mjSpec-native and inherits its pathologies: two parallel specs
(`model_holder_->spec()` vs `SpecEditor::active_spec_`) that go stale against each other;
undo via `mj_copySpec` snapshots; a hand-written 900-line inspector switch; save that emits
the *compiled* model's normalized XML (`mj_saveLastXML`) — orientation forms, class
structure, and shorthands destroyed.

ProtoSpec Studio replaces that substrate:

| Studio today | ProtoSpec Studio |
|---|---|
| `SpecEditor` holding `mjSpec* active_spec_` | `PsEditor` holding `ps::Model` (single authority; the dual-spec problem is deleted, not fixed) |
| Undo = `mj_copySpec` snapshot deque | Undo = generated deep `Clone()` snapshot deque (same 256-deep design; our clone is a plain-data copy with fresh serials) |
| `ElementKey` map to survive spec copies | **Creation serials** — already on every element, stable by construction; selection = serial |
| `ElementSpecGui` per-type switch (886 lines, hand-maintained) | **Generated inspector** driven by the reflection tables: all 142 element types, zero per-type UI code (§5) |
| `mj_recompile` in place | Pure `Compile()` from the bridge; old `Compiled` dropped, new one swapped in (§6) |
| Save = `mj_saveLastXML` (normalizing) | Save = `WriteMjcf` — authored forms, classes, macros round-trip byte-faithfully |
| Explorer walks `mjs_firstElement/firstChild` | Explorer walks the ProtoSpec tree (SDK traversal); shows frames, replicate/composite/attach as authored, defaults classes, assets |

`mjSpec` disappears from the editor entirely. MuJoCo appears exactly twice: the compiled
`mjModel/mjData` consumed by rendering/physics, and the picking/perturb helpers.

## 3. Dataflow: the one-way rule (the perturb lesson)

The previous experiment failed by reading world-space results out of `mjModel`/`mjData` and
writing them back into spec fields. That inversion is lossy by construction: `xpos/xquat`
are world-frame, compiled outputs; ProtoSpec fields are parent-relative, possibly
frame-wrapped, possibly class-inherited, authored in any of five orientation forms. There is
no well-defined inverse, so "sync issues" are guaranteed, not incidental.

**DR-S1: strict one-way dataflow.**

```
ps::Model (authority)  --Compile-->  Compiled{mjModel, Binding}  --mjv-->  screen
      ^                                                             |
      |                                                             v
   edits (gizmo deltas, inspector commits, SDK ops)  <--input--  selection/picking
```

- Nothing ever flows from `mjModel`/`mjData` back into the tree. Not poses, not masses,
  not anything.
- Gizmo drags do NOT move the compiled model; they compute a ProtoSpec field edit (§4),
  apply it to the tree, and the view catches up by recompiling.
- `mjvPerturb` stays available in **simulate mode** for physics poking, explicitly
  firewalled: perturb state dies with the `mjData` it touched; there is no "bake perturb
  into model" path. (A deliberate *tool* — "write current pose to keyframe N" — may come
  later; it is an explicit user action into the `Key` element, still authored-side.)

**DR-S2: two modes, one authority.**

- **Simulate mode** (Studio's current behavior): physics runs, perturb allowed, inspector
  is read-only-ish (edits allowed but batched — they mark the model dirty and prompt
  recompile). The displayed state is `mjData` at simulation time.
- **Modeling mode**: physics paused; the displayed state is the *spec pose* — `mjData`
  freshly reset to `qpos0` after every recompile. This removes all ambiguity about what a
  gizmo edit means: you are editing the model definition, and you see exactly the model
  definition. Entering modeling mode: pause + recompile + reset. Exiting: recompile
  (already current), then either reset-and-play or `Recompile`-style state migration if the
  user opts to keep the pre-modeling simulation state (only meaningful when the structure
  didn't change; offered as a toggle, default off = clean reset).

## 4. Modeling mode: selection, gizmos, and the recompile loop

**Selection.** Reuse Studio's ray pick (`platform::Pick`) → `PickResult.geom/body` →
`Binding` reverse lookup (`GeomAt/BodyAt`) → ProtoSpec element pointer + serial. Tree click
→ element → `Binding.Id()` → highlight. Selection state = `{serial, element ptr}`; the
serial survives recompiles (the Binding is rebuilt each compile, the serial is forever).
Studio's fragile `mjs_getId == picked.body` matching is deleted.

**World transform of the selected element.** Read from the *compiled* model (body: `body_pos/
body_quat` chain, or `xpos/xquat` at qpos0; geom/site: `geom_xpos` etc. at qpos0). That is
display-side data — allowed, because it never writes back; it only seeds gizmo rendering.

**Gizmo drag math (the correct inverse).** For a drag producing a desired new world pose
`W_new` of element E with parent chain world pose `P` (also from the compiled model at
qpos0):

```
local_new = inverse(P) * W_new          // parent-relative
E.pos    = local_new.translation
E.orient = Quat(local_new.rotation)     // materialized as quaternion
```

Rules:
- If E's pose is class-inherited (unset), the edit **materializes** the field on the element
  (presence-first semantics make this natural; the inspector shows it switching from
  "inherited" to "authored").
- If E's orientation was authored as euler/axisangle, a gizmo rotation rewrites it as a
  quaternion (form is lost for that field, by explicit policy — a gizmo is a quaternion
  instrument; the inspector still allows editing the authored form directly). A
  status-line note flags the conversion.
- Elements inside a `<frame>`: the frame's transform is part of `P` (frames are just parent
  chain). Editing a frame itself moves its subtree, which is exactly MJCF semantics.
- Elements inside `<replicate>`/`<composite>`: gizmo editing the *generated* copies is
  impossible by design (they don't exist in the tree); picking a generated body selects the
  macro element (via `Binding.Find` name-pattern reverse mapping) and gizmos apply to the
  macro's own pos/offset fields. Phase 1 may simply mark macro-generated picks
  "select-only".
- Scale gizmo maps to `size` for geoms/sites (per-axis where the geom type allows,
  uniform otherwise); bodies have no scale (disabled).

**Preview cadence (DR-S3).** During a drag, recompile is debounced (~30-60 ms) on the
native path; measured NC-phase compiles for rigid models are low single-digit
milliseconds, so drag-preview via *actual recompile* is the baseline plan — no proxy
transforms, no divergence between preview and truth. Two fallbacks if profiling on big
models demands them, in order: (a) recompile only on mouse-release, with a cheap overlay
ghost (mjv_initGeom axes/wire box at the target pose) during the drag; (b) a
subtree-transform of `mjvScene` geoms as pure eye candy. The fallbacks affect smoothness
only — correctness always comes from the post-drag compile.

**Overlay rendering.** Gizmos, selection highlight, and ghosts are appended to the
`mjvScene` after `mjv_updateScene` via `mjv_initGeom` (works on both classic and Filament
backends since `SceneBridge` iterates `scene->ngeom`). Requires one small hook in
`Renderer::Render` between update and draw — the only renderer change.

## 5. The generated inspector (reflection tables earn their keep)

One generic widget function renders *any* element:

- Walk `reflect::Describe(elemtype)` field descriptors: name, kind (Bool/Int/Double/
  String/Enum/Ref/Variant/array kinds), arity, optional, has_default.
- Kind → widget mapping: scalars → `InputDouble/Int`; fixed arrays → N-wide rows; InlineVec
  → row + filled-count; enums → combo from the generated keyword tables; keyword-set enums
  → checkbox row; `Ref<T>` → combo populated from the model's elements of the target
  type(s) (union targets expand) with a "dangling" warning icon (validation tier-2 rule
  surfaced inline); variants → a tag selector + the active arm's fields; strings → text.
- **Presence-aware**: unset fields render grayed with the *effective* value from
  `sdk::Effective()` (class chain + IDL default) and an "inherit" badge; typing materializes
  the field; a per-field ⟲ clears presence (revert to inherited). This surfaces DR-1 in the
  UI and is something the mjSpec inspector fundamentally cannot do (mjSpec has no
  presence).
- Edits go through `PsEditor::SetField` (command pattern: field id + old/new value) so undo
  is per-edit, not per-snapshot, for scalar changes; structural ops still snapshot-clone.
- The existing `ImGui_SpecElementTable` widget layer (`imgui_widgets.h`) is reused for
  look-and-feel; only its data source changes.

Net effect: Studio's 886-line hand-written `gui_spec.cc` becomes ~200 lines of generic
code, and every future schema change is picked up with zero UI work.

## 6. Compile/reload loop and error surfacing

- `PsEditor::Commit()` (debounced during drags, immediate on inspector commit/structural
  op): `Compile(model_, options)` → on success, swap the app's `Compiled` (new
  `mjModel` + `Binding`), `mj_makeData`, reset to qpos0 (modeling mode), re-init renderer
  scene (Studio's existing `OnModelLoaded` path, minus its spec reseeding). `CompilePath::
  Auto` — rigid models go native (fast), flex/plugin models fall back to the XML path
  transparently; `report.taken` shown in the status bar.
- **Purity makes this safe**: compile failure leaves the tree untouched — the app keeps
  simulating the last good `Compiled` while the editor shows the errors. No mjSpec-style
  "spec poisoned by failed compile" states.
- **Diagnostics panel**: `Validate(model, tiers 1-2)` runs before every compile (fast);
  tier results + compile errors render in a dock panel, each row carrying SourceLoc
  (file:line for loaded models) and the element path; clicking a row selects the element.
  Tier-3 lint runs on demand.
- Structural edits invalidate the Binding by contract; `Commit()` after every structural op
  keeps the window where a stale Binding exists to a single frame, inside one function.

## 7. Authoring operations (SDK earns its keep)

| Op | Mechanism |
|---|---|
| Add primitive (sphere/box/capsule/...) | `sdk::AddBody` + `sdk::AddGeom(type)` at a camera-ray drop point; defaults from IDL tables |
| Add joint/site/camera/light/actuator/sensor | SDK builders via the existing "+" popup pattern, filtered by valid parent |
| Import mesh | File dialog → `sdk::AddMesh(asset, file)` (+ copy into the model's meshdir or register via `CompileOptions.vfs_assets` for unsaved models) → `AddGeom(type=mesh, mesh=ref)`; name defaults to file stem (MuJoCo's own convention). Convex hull is automatic at compile; a "maxhullvert" field is right there in the inspector |
| Delete | `sdk::DeleteRecursive` — the referrer check surfaces dangling refs as a confirm dialog listing them (cascade or cancel), instead of mjSpec's silent-compile-error-later |
| Rename | `sdk::Rename` — all referrers updated atomically |
| Reparent (drag in tree) | Detach subtree + reattach under new parent; pose fixup optional ("keep world pose" checkbox does the §4 parent-frame math) |
| Class/default editing | Defaults classes appear in the explorer as a section; editing a class field live-updates every inheriting element on next compile — the presence-aware inspector makes the inheritance visible |

## 8. Where the code lives and what it builds against

**DR-S4: `apps/studio/` inside the protospec repo**, vendoring the ~6k lines of Studio
shell sources we keep (app shell, `platform/hal` window+renderer, `platform/ux` widgets/
interaction/step-control, `platform/sim/model_holder` reduced), the same way tinyxml2 was
vendored — with provenance headers and lift-registry entries so MuJoCo bumps diff cleanly.
Build: links the protospec libs + prebuilt `mujoco.lib` via `MUJOCO_ROOT`, includes the
vendored `src/` tree (picking uses private engine headers — already our situation in
`cpp/harness`), FetchContent for SDL2 + Dear ImGui (docking, pinned to Studio's commit) +
ImPlot.

**DR-S5: classic `mjr` renderer first.** Filament + webp is by far the heaviest dependency
cluster and irrelevant to editor correctness; Studio's HAL already abstracts both backends,
so we compile the classic path only (`GraphicsMode::Classic`) in SE0-SE4 and keep the HAL
seam so Filament can be reinstated later. This cuts the vendored surface and build time
dramatically.

Alternative considered and rejected: a branch on the MuJoCo fork carrying protospec as a
subdirectory — couples the editor to MuJoCo's build, drags Filament in by default, and
makes our CI/corpus tooling awkward. The repo stays standalone; the fork branch remains the
publish target.

## 9. What this tests (why it's the right ProtoSpec proving ground)

- **SDK**: builders, traversal, rename/delete-with-referrers, Effective — all UI-driven.
- **Reflection**: the inspector is generated; any gap in the tables becomes visible as a
  missing widget.
- **Bridge**: pure Compile under rapid-fire recompiles (drag debounce = compile stress
  test), Binding stability via serials, Auto-path fallback UX, CompileReport surfacing.
- **Writer**: save from the editor must round-trip authored form — the corpus differential
  already proves the writer; the editor proves it *matters* (diffs of saved files stay
  human-readable).
- **Validation**: tiers 1-2 as live editor diagnostics with SourceLoc navigation.

## 10. Deferred (explicitly out of scope for SE0-SE4)

- **Flex/flexcomp editing**: tree shows them, the generic inspector edits their fields,
  compile works (Auto → XML path), but no gizmos/no flex-specific tooling. Same for
  composite beyond select-only.
- **Plugins**: models requiring plugin registration are load-blocked in the harness today;
  the editor inherits that until the plugin wave lands.
- **Python Studio**: the Python app is a parallel, less capable shell with no editor; not
  worth mirroring. Python users get the pybind `protospec` module; the editor is C++.
- **USD, WASM/live build, keyframe timeline editing, CoACD decomposition on import.**

## 11. Milestones

- **SE0 — shell builds standalone.** Vendored Studio shell compiles in `apps/studio`
  against MUJOCO_ROOT + FetchContent deps, classic renderer, loads a corpus model via the
  existing ModelHolder path (still mjSpec-free? No — SE0 may temporarily keep `mj_parse`
  loading; the swap lands in SE1). Exit: humanoid loads, simulates, picks, perturbs.
- **SE1 — ProtoSpec substrate.** `PsEditor` (Model authority, Clone-based undo, serial
  selection), explorer walks the ProtoSpec tree, generated inspector replaces
  `ElementSpecGui`, load path = `ParseMjcf` + `Compile`, save = `WriteMjcf`. mjSpec is gone
  from the app. Exit: load → browse → edit fields → compile-and-reload → save, with
  authored-form-preserving output, on the 359-file corpus subset that loads.
- **SE2 — modeling mode.** Mode state machine, gizmos (translate/rotate/scale) via overlay
  geoms, drag→parent-frame→tree math, debounced recompile preview, exit-mode reset/migrate.
  Exit: the user story — load, enter modeling, move/resize things with gizmos, exit,
  simulation plays the edited model.
- **SE3 — authoring ops.** Add-primitive drop, mesh import (+vfs assets), delete/rename
  with referrer UX, reparent-in-tree.
- **SE4 — polish.** Diagnostics panel with SourceLoc navigation, presence badges/revert,
  status-bar compile report, undo depth/perf tuning on large models.

Agent decomposition per Section-12 conventions: SE0 single-owner (build surgery); SE1 two
parallel owners (editor substrate vs generated inspector — disjoint files); SE2 single-owner
(input/gizmo/loop are one organism); SE3 parallelizable per-op; verification throughout by a
different agent driving the app headlessly (SDL offscreen) where feasible, plus scripted
PsEditor-level tests that don't need a window.

## 12. Open questions (owner input wanted)

1. **Gizmo rotation vs authored euler**: silently materialize quat (proposed), or attempt
   euler-preserving conversion when the authored form was euler and the rotation is
   axis-aligned? Proposed: quat + status note; euler editing stays available in the
   inspector.
2. **Exit-modeling-mode default**: clean reset to qpos0 (proposed) vs state migration when
   structure unchanged. Proposed: reset, with a "keep state" toggle.
3. **SE0 renderer**: confirm classic-only is acceptable visually for the first iteration
   (Filament later), since it trades visual quality for a far smaller build.
