# ProtoSpec Studio: a new ProtoSpec-native model editor (Unity/Unreal idiom)

Companion to `docs/plan.md`. A new interactive editor for MuJoCo models whose data model is
the ProtoSpec tree — the first consumer-grade test of ProtoSpec (SDK, reflection tables,
pure Compile/Binding, form-preserving writer). Unreal adoption remains out of scope;
nothing here touches the UE plugin.

**What this is NOT.** MuJoCo's experimental Studio ships an mjSpec-based spec editor
(`SpecExplorerGui`/`SpecEditorGui`/`ElementSpecGui`, `platform/ux/spec_editor.*`). That
code is the *old-spec* editing model — dual specs that go stale against each other, a
hand-written 886-line per-type inspector, `mj_copySpec` undo, normalizing save. **None of
it is reused.** It serves only as reference anatomy (what panels exist, how picking works,
how model swap is plumbed). The ProtoSpec editor UI is written new, clean, on top of a thin
retained shell.

## STATUS (living section)

| Milestone | State | Evidence |
|---|---|---|
| SE0 thin shell builds (no old editor code) | queued | |
| SE1 Hierarchy + Details (generated) + load/save/compile | queued | |
| SE2 Edit mode: gizmos, viewport editing, recompile loop | queued | |
| SE3 authoring: add primitives, mesh import, delete/rename/reparent/duplicate | queued | |
| SE4 Play mode + polish (diagnostics, snapping, framing, multi-select) | queued | |

---

## 1. What we take, what we build

**Retained from Studio (the thin shell, ~infrastructure only):**
- SDL2 window + Dear ImGui (docking) + ImPlot context (`platform/hal/window`).
- Renderer HAL over `mjvScene` with classic `mjr` and Filament backends
  (`platform/hal/renderer`) — classic first (DR-S5).
- `StepControl` (pause/run/speed/single-step) and the deferred model-swap plumbing pattern
  (`pending load → apply at frame top`).
- Ray picking primitives (`platform/ux/interaction.cc` `MakePickRay`/`Pick` — geometry
  math only; its selection/perturb wiring is not kept).

**Explicitly dropped (old-spec editor):** `spec_editor.{h,cc}`, `gui_spec.{h,cc}`, the
Explorer/Editor/Inspector spec tabs in `app.cc`, `ElementKey`, the S/M/D property views,
`mjs_*` everywhere. The fork's `App` is rewritten; mjSpec appears nowhere.

**Built new (the editor):** everything the user touches — Hierarchy, Details, Viewport
editing, Assets, toolbar/mode machine, undo — ProtoSpec-native from the first line.

## 2. Data model: one authority, one-way flow

**DR-S1: the ProtoSpec `ps::Model` is the only model state.** `mjModel`/`mjData` are
disposable compiled artifacts held in a `Compiled` (bridge). Dataflow is strictly one-way:

```
ps::Model (authority) --Compile--> Compiled{mjModel, Binding} --mjv--> viewport
      ^                                                          |
      +---- edits (gizmos, Details commits, SDK ops) <-- input --+
```

Nothing is ever read back from `mjModel`/`mjData` into the tree — not poses, not masses.
This is a hard lesson from the previous experiment: writing world-space `xpos/xquat`
results back into spec fields has no well-defined inverse (parent-relative fields, frames,
class inheritance, five orientation forms) and guarantees sync drift. Gizmos therefore
compute *tree edits* (§5) and the viewport catches up by recompiling — the pure `Compile`
makes this cheap, safe, and impossible to corrupt (a failed compile leaves the tree
untouched and the last good `Compiled` running).

**DR-S2: Unity mode semantics.**
- **Edit mode (default).** Physics paused. The viewport shows the *spec pose*: `mjData`
  reset to qpos0 after every recompile. What you see is the model definition. All editing
  lives here.
- **Play mode.** Toolbar ▶ compiles (if dirty) and runs the simulation; ⏸ pauses; ⏹ stops
  and **discards all simulation state**, returning to Edit mode at qpos0 — exactly Unity's
  play-mode contract. Mouse perturb (spring-drag bodies) is available in Play mode as a
  physics tool and dies with the `mjData`; there is no path from perturb to the tree.
  Editing fields during Play is allowed but marks the model dirty and takes effect on the
  next compile (stop→play or an explicit "apply & restart").
- A deliberate escape hatch (later, explicit): "Save current pose to keyframe N" — a user
  action that writes `qpos` into a `Key` element. The only sanctioned data crossing, and it
  goes *through* the authored model, not around it.

## 3. Editor layout (Unity/Unreal idiom)

Dockable ImGui layout, defaults:

```
+--------------------------------------------------------------+
| Toolbar:  [▶ ⏸ ⏹]  [Move Rotate Scale]  [Local/World] [Snap] |
+---------------+---------------------------+------------------+
| Hierarchy     |        Viewport           |  Details         |
| (tree,search) |  (scene, gizmos, camera)  |  (properties)    |
|               |                           |                  |
+---------------+---------------------------+------------------+
| Assets (meshes/textures/materials/hfields)| Diagnostics      |
+--------------------------------------------------------------+
```

**Hierarchy** — the ProtoSpec tree, not mjModel: worldbody subtree with typed icons
(body/geom/joint/site/camera/light/frame), authored macros shown as single nodes
(`<replicate>`, `<composite>`, `<attach>`), plus sections for Actuators, Sensors, Tendons,
Equality, Contact, Defaults (the class tree), Keyframes, Custom. Search/filter box;
drag-to-reparent; right-click context menu (Add child ▸, Duplicate, Rename, Delete, Copy
path); double-click focuses viewport (F frames selection). Multi-select
(ctrl/shift-click).

**Details** — the generated inspector (§4). Categorized like Unity components: Transform
first (pos/orient/size), then the element's remaining fields grouped by the schema's field
order, collapsible.

**Viewport** — mjv scene render + editor overlays (gizmos, selection outline, grid). Orbit/
pan/dolly camera (alt-drag idiom) + fly mode (RMB+WASD, Unreal-style). Click picks
(ray-cast → `Binding` reverse → element); click-through cycles overlapping geoms; W/E/R
switch gizmo, Q selects, F frames, ctrl+D duplicates, Del deletes.

**Assets** — the model's assets as a browsable grid (meshes/textures/materials/hfields)
with import button (§6); selecting an asset shows it in Details; usages listed via the
SDK's `FindReferrers`.

**Diagnostics** — live `Validate()` tiers 1-2 + last `CompileReport`; rows carry
SourceLoc + element path, click-to-select. Status bar shows compile path taken
(native/xml), compile time, and dirty state.

## 4. The generated Details panel (reflection tables earn their keep)

One generic renderer draws *any* of the 142 element types — zero per-type UI code:

- Walk `reflect::Describe(elemtype)`: field name, kind (Bool/Int/Double/String/Enum/Ref/
  Variant/arrays), arity, optional, defaults.
- Kind → widget: scalars/arrays → numeric rows (InlineVec shows filled count); enums →
  combos from the generated keyword tables; keyword-sets → checkbox rows; `Ref<T>` →
  combo of the model's matching elements (union targets expand) with a dangling-ref
  warning inline; variants → tag selector + active arm fields; orientation → quat row
  *plus* the authored-form arm if euler/axisangle (editable in authored form).
- **Presence-aware (the mjSpec editor cannot do this):** unset fields render grayed with
  the *effective* value from `sdk::Effective()` (class chain + IDL default) and an
  "inherited" badge; editing materializes the field; per-field ⟲ reverts to inherited.
  DR-1 made visible.
- Edits are commands (`SetField{element serial, field id, old, new}`) → undo/redo is
  per-edit; structural ops snapshot via generated deep `Clone()` (bounded deque).

## 5. Edit mode: selection, gizmos, recompile loop

**Selection identity = creation serial** (survives every recompile by construction; the
Binding is rebuilt per compile, serials are forever). Pick: viewport ray → geom/body id →
`Binding.GeomAt/BodyAt` → element. Tree click → element → `Binding.Id()` → viewport
outline. Macro-generated pick (a replicate clone's geom): `Binding.Find` name-pattern maps
back to the macro element — selecting it edits the macro's own fields; phase 1 may mark
these select-only.

**Gizmos** (translate/rotate/scale, local/world, snapping) are drawn as overlay geoms
appended to the `mjvScene` after `mjv_updateScene` via `mjv_initGeom` (renders on both
backends; needs one small hook in the renderer between update and draw). No `mjvPerturb`
involvement in Edit mode.

**Drag math — the delta rule (DR-S6).** Compiled poses are NOT invertible into authored
fields, and mesh geoms prove it: the compiler *bakes the mesh frame into the geom frame* —
`mjuu_frameaccum(pos, quat, meshpos, mesh->GetQuatPtr())` at `user_objects.cc:4048-4054`
composes the mesh's re-centering/principal-axis transform (stored per-mesh in
`mjModel.mesh_pos/mesh_quat`, `mjmodel.h:636-637`) into `geom_pos/geom_quat`. Reading a
mesh geom's compiled pose and writing it back into the authored field double-applies that
transform: the mesh jumps on every round trip (the exact historical "never saved back
nicely" failure — it was structural, not a code bug).

Therefore gizmos never reconstruct absolute authored poses. They apply the drag **delta**,
conjugated into the parent frame, onto the existing authored value:

```
W = P ∘ L_authored ∘ M        // M = any compiler-baked suffix (mesh frame, or identity)
W_new = D ∘ W                 // D = world-space gizmo delta
⇒ L_new = (inv(P) ∘ D ∘ P) ∘ L_authored      // M cancels exactly; never inverted
```

`P` (parent world pose at qpos0) and the gizmo anchor (`W`, on the visible geometry) come
from the compiled model — display-side only. `L_authored` comes from the tree (via
`Effective()` when class-inherited, materializing the field on first edit — Details shows
the badge flip). One uniform rule for every element type; mesh geoms need no special case
because the baked suffix cancels algebraically.

- Euler/axisangle-authored orientations are rewritten as quat by gizmo rotations
  (status-line note); the authored form remains editable in Details.
- Frames are part of `P`; editing a frame moves its subtree (MJCF semantics for free).
- Scale gizmo → geom/site `size` (per-axis where the type allows); bodies have no scale.
  Mesh geom "scale" maps to the mesh asset's `scale` (affects all users — Details warns),
  not geom size.

**Recompile cadence (DR-S3).** Debounced native recompile (~30-60 ms) *is* the drag
preview — measured rigid-model native compiles are low single-digit ms, so preview and
truth are the same artifact. Fallbacks if large models demand them: compile on
mouse-release with a ghost overlay during the drag; never a divergent proxy transform as
the source of truth.

**Commit loop.** `Editor::Commit()`: `Validate(tiers 1-2)` → `Compile(model,
CompilePath::Auto)` → swap `Compiled`, fresh `mjData` at qpos0, renderer re-init (Studio's
model-swap plumbing pattern). Failure: tree untouched, last good `Compiled` keeps
rendering, Diagnostics shows the errors. Structural edits invalidate the Binding by
snapshot contract; `Commit()` immediately after keeps any stale window to one frame inside
one function.

## 6. Authoring operations (SDK-backed)

| Op | Mechanism |
|---|---|
| Add primitive | Context menu / toolbar ▸ sphere/box/capsule/cylinder/ellipsoid/plane: `sdk::AddBody`+`AddGeom(type)` at the camera-ray drop point; IDL defaults |
| Add joint/site/camera/light/actuator/sensor/tendon/equality | Type-filtered add menus via SDK builders (valid-parent aware) |
| Import mesh | File dialog → `sdk::AddMesh(file)` (copy into meshdir, or `CompileOptions.vfs_assets` for unsaved models) → auto-create `AddGeom(type=mesh, mesh=ref)`; name = file stem (MuJoCo's convention); convex hull automatic at compile; maxhullvert editable in Details |
| Delete | `sdk::DeleteRecursive` — dangling referrers surface as a confirm dialog (cascade/cancel) instead of a later compile error |
| Rename | `sdk::Rename` updates all referrers atomically |
| Duplicate (ctrl+D) | Generated `Clone()` + name uniquing + insert as sibling |
| Reparent (drag in Hierarchy) | Detach/attach; optional "keep world pose" does the §5 parent-frame math |
| Defaults editing | The class tree is in the Hierarchy; editing a class field updates every inheritor on next compile — visible through the presence badges |
| Save / Save As | `WriteMjcf` — authored forms, classes, macros preserved byte-faithfully (the mjSpec editor saves normalized soup; ours saves what you wrote) |

## 7. Where the code lives, what it builds against

**DR-S4: `apps/studio/` in the protospec repo.** Vendored thin-shell sources (~2-3k lines:
hal window/renderer, step control, pick math) with provenance headers + lift-registry
entries so MuJoCo bumps diff cleanly; the editor itself is new code under
`apps/studio/editor/`. Builds against protospec libs + prebuilt `mujoco.lib` via
`MUJOCO_ROOT` (+ vendored `src/` includes for the pick math's engine headers — same
situation as `cpp/harness`). FetchContent: SDL2, Dear ImGui (docking, Studio's pinned
commit), ImPlot.

**DR-S5: classic `mjr` renderer first.** Filament(+webp) is the heaviest dependency
cluster and irrelevant to editor correctness; the HAL seam stays so it can be enabled
later. Alternative (branch inside the MuJoCo fork tree) rejected: couples the editor to
MuJoCo's build, drags Filament in, complicates our CI.

## 8. What this tests (the point of the exercise)

SDK builders/traversal/referrers/Effective driven by real UI; the reflection tables as a
complete UI generator (any schema gap becomes a visibly missing widget); pure Compile under
drag-rate recompile stress; Binding/serial stability across hundreds of swaps; Auto-path
fallback UX; the form-preserving writer producing human-readable diffs of edited models;
validation as live diagnostics with SourceLoc navigation.

## 9. Deferred

Flex/flexcomp editing (visible in Hierarchy, generically editable in Details, compiles via
Auto→XML fallback; no gizmos/tooling). Composite beyond select-only. Plugin-requiring
models (until the plugin wave). Python Studio (the pybind `protospec` module is the Python
surface). USD, WASM, keyframe timeline, CoACD decomposition on import.

## 10. Milestones

- **SE0 — thin shell.** `apps/studio` builds standalone: window+imgui+classic renderer+
  step control+pick math vendored, none of the old editor code. Loads a corpus model via
  `ParseMjcf`+`Compile` (ProtoSpec from day one — no temporary mjSpec loading), renders,
  simulates, ray-pick prints the hit element. Exit: humanoid loads, runs, picks.
- **SE1 — Hierarchy + Details.** ProtoSpec tree panel with search/icons; generated Details
  with presence badges + Effective ghosts; selection sync both directions; undo (command +
  snapshot); save via `WriteMjcf`. Exit: load → browse → edit fields → compile-and-reload
  → save round-trips authored form on the loadable corpus.
- **SE2 — Edit mode.** Mode machine (Edit/Play per DR-S2), gizmos with local/world+snap,
  drag→parent-frame→tree math, debounced recompile preview, viewport hotkeys (QWER/F/Del).
  Exit: the user story — load, move/resize with gizmos, hit ▶, simulation plays the edited
  model, ⏹ returns to the edited definition.
- **SE3 — authoring.** Add-primitive drop, mesh import (+vfs), delete/rename referrer UX,
  duplicate, reparent. Exit: build a small scene from an empty model entirely in-editor,
  simulate it, save it, reload it.
- **SE4 — Play-mode polish + QoL.** Perturb in Play (firewalled), diagnostics panel
  navigation, multi-select ops, framing, grid/snap settings, compile-report status bar,
  large-model perf pass.

Agent decomposition (Section-12 conventions): SE0 single-owner; SE1 two parallel owners
(Hierarchy+editor core vs generated Details — disjoint); SE2 single-owner (input/gizmo/
loop is one organism); SE3 parallel per-op; verification by a different agent via scripted
`Editor`-level tests (no window needed for the command/undo/commit logic) + offscreen
smoke.

## 11. Resolved decisions (owner, 2026-07-08)

1. Gizmo rotations materialize quat (+ status note); euler stays editable in Details. No
   euler-preservation heuristics.
2. Play→Stop discards simulation state (Unity semantics); "save pose to keyframe" is the
   later explicit escape hatch.
3. Classic `mjr` renderer for SE0-SE4; Filament later behind the HAL.
4. **Gizmos are the core deliverable** — SE2 is the milestone that matters; everything
   else serves it.
