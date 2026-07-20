# Interactive Joint Rigger + Visualization — Investigation & Implementation Plan

Repo: `/home/buzz/Documents/proto/mujoco` (branch `protospec`, HEAD `4d48f859`);
host fork: `/home/buzz/Documents/proto/mujoco-studio` (branch `studio`, pinned MuJoCo 3.10.1).
All paths below are absolute unless obviously repo-relative.

The user's problem: adding a joint today means typing axis/pos/range numbers with no
visual feedback on what the joint will *do*. Hard requirement: everything shown must be
**correct** — no hand-derived joint math that can drift from what MuJoCo computes.

---

## 1. What exists today (code-grounded)

### 1.1 Joint visualization — screen-space, no depth occlusion, range arc DROPPED

- `studio/editor/joint_overlay.h/.cc` — `CollectJointVis(m, d, binding, selected_serial, show_all)`:
  the windowless collect step. Ground truth is already right: world anchor/axis come from
  the forwarded preview data (`d->xanchor`, `d->xaxis`, joint_overlay.cc:68-71), range from
  `m->jnt_range` gated on `m->jnt_limited` (joint_overlay.cc:72-77). Radians/metres, compiled.
- `studio/editor/viewport_plugin.cc:100-155` — `DrawJointsScreen`: the CURRENT overlay is a
  screen-space simplification on the ImGui foreground draw list: axis segment + anchor dot
  (+ triad for free joints). The comment at viewport_plugin.cc:107-108 is explicit:
  *"Screen-space simplification of the old mjvScene rig: axis segment + anchor dot; **the
  range arc is dropped**."* `JointVis.has_range/range` is collected but **unused** by the
  current draw code.
- The **old** depth-occluded overlay (pre plugin-retarget, `git show a1b4cf32^:studio/editor/viewport_plugin.cc`,
  `DrawJoint`) drew mjvScene geoms: axis arrow, anchor sphere, hinge **range arc**
  (24-segment mjGEOM_LINE fan) and slide travel line. Two important observations:
  - It was dropped in `a1b4cf32` ("operate exclusively through upstream Studio's four plugin
    types") because the editor lost its mjvScene hook; the ScenePlugin has since landed
    upstream (5637f743) and is already adopted for the selection outline
    (viewport_plugin.cc:156-209, `AddSelectionOutlineGeoms`/`EnhanceScene`), so the *channel
    to restore it depth-occluded now exists*.
  - **Correctness gap in the old arc**: its plane basis came from `PlaneBasis(axis, u, v)` — an
    arbitrary seed vector. The arc had the right angular *extent* but its zero direction was
    meaningless: it did not show where the limb actually is at qpos, nor where range min/max
    actually put the limb. This is exactly the "hand-derived drift" class the user forbids.
    The restored arc must anchor its reference direction to the real child pose (§3).

### 1.2 Joint manipulation — the joint gizmo (pos + axis), commits through the normal flow

- `studio/editor/transform_math.h:214-257` — the deliberately separate joint-rigging section:
  `IsJointSerial`, `JointDragFrame` (parent world pose P at qpos0, authored `pos`/`axis` in
  parent frame, derived `world_anchor = P.pos`, `world_axis = P.quat·axis`),
  `BuildJointDragFrame`, `ApplyJointTranslate` (W tool: move authored anchor by world delta),
  `ApplyJointAxisRotate` (E tool: reorient authored axis, optional snap to ±X/Y/Z). Ball/free
  have no axis handle; all types translate.
- `studio/editor/gizmo.cc:103-118, 255-256, 408-477` — the gizmo enters `joint_mode_` when the
  selection is a joint serial; drags run the delta rule on the authored tree, then
- `gizmo.cc:607-690` — `LivePatch`/`LivePatchJoint`: per-frame preview WITHOUT recompiling —
  writes compiled `m->jnt_pos`/`m->jnt_axis` from the authored values mapped through the
  `<frame>` chain, then **`mj_kinematics(m, d)`** on the editor's preview data. This is the
  standing precedent that a per-frame forward-kinematics pass is cheap and is the accepted
  preview mechanism.
- Release → `ctx.CommitEdit(label)` (gizmo.cc:720-729) → undo step + debounced recompile.

### 1.3 The preview/mirror channel (the load-bearing mechanism for the rigger)

- The editor owns its own compiled artifact + preview data: `ctx.compiled.model`,
  `ctx.sim_data` — rebuilt on every successful compile and `mj_forward`'d
  (`studio/editor/editor_ops.cc:160-178`).
- During gizmo drags, `MirrorDragKinematics` (viewport_plugin.cc:501-527) copies the
  patched kinematics (`xpos/xquat/xmat/xipos/ximat/geom_x*/site_x*/cam_x*/light_x*/
  xanchor/xaxis`) from `ctx.sim_data` onto the HOST's `mjData` each tick, guarded by
  `DragDimsMatch` (viewport_plugin.cc:531-535). `mjv_updateScene` reads those arrays
  directly, so the host renders editor-driven poses without any authored-state change.
- Ordering contract: the model ModelPlugin ("ProtoSpec Model", registered first in
  `plugins.h:57-65`) dispatches before the viewport ModelPlugin; while `ctx.gizmo_active`
  its `DoUpdate` **defers all host-data writes** (protospec_editor.cc:250-258) so the
  viewport's mirror (running later the same tick, viewport_plugin.cc:567-575) is
  authoritative. Otherwise, in Edit mode, `DoUpdate` resets time≠0 to qpos0 and runs
  `mj_forward(host_model, host_data)` every tick (protospec_editor.cc:283-287) — i.e. any
  scrub mirror MUST get the same deferral flag or the host forward at qpos0 will fight it.
- Edit gating: `EditorContext::CanEdit()` (editor_context.h:431-436) = model ready + Edit
  mode + paused + `sim_time == 0.0`. `sim_time` is the HOST data time; a scrub that only
  touches `ctx.sim_data` and mirrors kinematics never advances host time, so CanEdit
  stays true by construction.

### 1.4 What the host offers (and what the plugin surface cannot reach)

- **ScenePlugin**: `app.cc:463-472` — `plugin_scene_.ngeom = 0`, every registered
  `enhance_scene(plugin, model(), data(), &plugin_scene_)` runs, then the geoms are handed to
  `renderer_->Render(..., extra_geoms)`. `plugin_scene_` budget is **2000 geoms**
  (`mjv_makeScene(nullptr, &plugin_scene_, 2000)`, app.cc:100). The renderer appends them to
  its scene AFTER `mjv_updateScene` (platform/hal/renderer.cc:161-166) and renders through
  the normal path — **full depth occlusion and standard transparency**; mesh-type `mjvGeom`s
  with a valid `dataid` render like any scene mesh. This is the ghost/arc channel.
- **mjVIS_JOINT**: the host toggles `vis_options_.flags[mjVIS_JOINT]` on `J` (app.cc:790).
  Stock MuJoCo joint viz = per-joint arrows at the current pose, uniform scale, no ranges, no
  selection, no interactivity. The editor has NO access to the host `vis_options_` (accepted
  R1 degradation, noted at viewport_plugin.cc:627-632), so it can neither force it on nor
  restyle it. Worth *mentioning in docs* (press J for stock joint arrows) but useless as the
  rigger's substrate.
- **Perturb**: `app.cc:337-354` — `mjv_applyPerturbPose/Force` run only in the
  `!plugin_stepped` branch. Our Edit freeze returns true from `do_update`
  (protospec_editor.cc:289), so host perturb is structurally DISABLED in Edit mode. Riding it
  would need a fork change — ruled out.
- The plugin surface candidly cannot: read/write host `vis_options_`; draw depth-occluded
  *text* (angle readouts stay screen-space ImGui, or use `mjvGeom.label`); influence host
  picking (irrelevant — `mj_ray` runs on model geoms, so plugin geoms never pollute pick).

### 1.5 Editing joints today (panel side)

- Details panel is fully reflection-generated (`details_panel.cc`); a Joint shows raw
  `type/pos/axis/range/limited/ref/springref/damping/...` fields as bare numeric rows
  (`DrawNumericRow`, details_panel.cc:170-183) with the gesture commit protocol
  `GestureShouldCommit` → `BeginEdit`/`CommitEdit`/`CancelEdit` (details_panel.cc:56-64).
  **No units are shown** and nothing tells the user what the numbers mean spatially.
- Toolbar already has a "Joints" toggle for `ctx.show_all_joints` (panels.cc:975-980).
- Joint picking in the viewport exists (`TryPickJoint`, viewport_plugin.cc:259-300) —
  clicking near a drawn joint selects it.

### 1.6 Units: where the degree↔radian conversion actually lives

- The authored tree stores the MJCF **wire form**: `range`, `ref`, `springref` are declared
  `(unit=angle)` in the schema (`protospec/schema/mujoco.spec:998,1002-1003`) but the flag is
  *metadata only* — no reader/writer conversion exists (the `unit_angle` field in
  `generated/xml_binding.h:31` has zero consumers in `lib/`). So authored values are in
  compiler-angle units (degrees by MJCF default).
- Both compile paths hand MuJoCo the authored numbers plus the angle selector:
  `protospec/lib/compile/mjs_builder.cc:295-320` sets `spec_->compiler.degree` from the tree
  and lets **MuJoCo apply deg→rad**; XmlPath likewise via `mj_loadXML`. There is exactly ONE
  conversion point and it is MuJoCo's compiler. `m->jnt_range`/`qpos` are always radians
  (hinge/ball) or metres (slide) — `unit=angle` applies to hinge/ball only; slide range is
  metres and must never be converted.
- `ReadOrientContext(model)` (transform_math.h:64-68) already exposes the tree's
  `compiler.angle` for the one place the rigger needs it: converting a handle-drag result
  (radians, compiled space) back into the authored field's units on commit.
- `limited` is a TriState with auto semantics; the compiled pin is `m->jnt_limited`
  (already what `CollectJointVis` keys on). Never re-derive "is it limited" from the
  authored field.
- `ref` shifts what qpos=0 means spatially; compiled `m->qpos0` holds it. Range limits are
  enforced against qpos directly, so posing `qpos[jnt_qposadr] = jnt_range[k]` IS the limit
  pose. No further reasoning required — that is the whole point of scrubbing through
  mj_forward.

### 1.7 Test culture to follow

`studio/editor/test/test_plugin_windowless.cc` + `protospec/tests/test_plugin_windowless.py`:
splice the plugin TU into a test TU with a stub plugin registry, drive the callbacks
directly against a real `mj_loadXML` model, g++ against prebuilt `libprotospec_core.a` +
`libmujoco.so` from `build_ps` (env: `PROTOSPEC_BUILD_PS_LIB`, `PROTOSPEC_MJ_INCLUDE`,
`PROTOSPEC_STUDIO_SRC`). The rigger's windowless core follows this recipe exactly. (The old
gizmo/joint-overlay tests live in `attic/studio_host/test/` and are not built — the live
suite currently has NO joint-overlay coverage; this plan restores some.)

---

## 2. The design question: interaction models evaluated

### (a) DOF scrubbing — select a joint, drag a slider (or the limb), watch the subtree move

- **Mechanism**: write `q` into `ctx.sim_data->qpos[m->jnt_qposadr[jid]]`, `mj_forward`
  (or `mj_fwdPosition`) the preview data, mirror kinematics to the host via the existing
  `MirrorDragKinematics` channel under a new `rig_preview_active` deferral flag (same
  contract as `gizmo_active`, protospec_editor.cc:250-258). Release/deselect → restore
  `qpos0` + forward (snap-back), with an optional "hold" pin while the panel section is open.
- **Correctness**: perfect by construction — MuJoCo computes the articulation; the editor
  computes nothing. Works for every joint type, every frame nesting, every ref/springref
  configuration, multi-DOF stacks, for free.
- **Cost**: one `mj_forward` per scrub frame. Precedent: `LivePatchJoint` already runs
  `mj_kinematics` per drag frame (gizmo.cc:688); `DoUpdate` runs `mj_forward` on the host
  model every Edit tick already (protospec_editor.cc:286). Editor-scale models: microseconds.
- **Cons**: shows one pose at a time; doesn't by itself show the *range*.
- Effort: **S** (the channel, the freeze interplay, and the commit protocol all exist).

### (b) Limit ghosts — transparent subtree poses at range min/max

- **Mechanism**: per selected limited joint, build two ghost qpos vectors
  (`qpos0` with dof := `jnt_range[0]` / `jnt_range[1]`), run `mj_forward` on a dedicated
  scratch `mjData` (owned next to `ctx.sim_data`, rebuilt per compile), then emit one
  `mjvGeom` per subtree geom into `plugin_scene_` from the ScenePlugin hook:
  type/size/dataid from `m->geom_*`, pos/mat from the scratch `d->geom_xpos/geom_xmat`,
  rgba with α≈0.25, `category = mjCAT_DECOR`. Depth-occluded, transparent, excluded from
  pick (`mj_ray` never sees plugin geoms). Subtree = bodies under `m->jnt_bodyid[jid]`
  via `body_parentid`. Cache: recompute only when (compile_generation, selected serial,
  jnt_range) changes — but even per-frame is affordable.
- **Correctness**: poses are mj_forward outputs at the exact compiled limits; shapes are the
  compiled geom tables. Range edits in the Details panel instantly re-pose the ghosts after
  the debounced recompile lands — the meaning of the numbers becomes visible.
- **Cons**: geom budget (2000; guard per-geom like `AddOutlineBox` does); exotic geom types
  (plane/hfield/SDF) skipped in v1; visual clutter on big subtrees (cap + fade with subtree
  size). Unlimited joints have no ghosts (that absence is itself honest signal).
- Effort: **S/M**.

### (c) In-viewport handles

- Restore the **depth-occluded axis arrow + anchor sphere + hinge arc + slide travel line**
  through the ScenePlugin (the old `DrawJoint` code is recoverable from
  `a1b4cf32^`, modernized to append to `plugin_scene_`), with the arc's reference direction
  FIXED (see §3 arc pinning) so its endpoints coincide with the ghost poses.
- **Axis re-orientation + anchor drag already exist** (E/W gizmo on a selected joint) — keep.
- New draggable handles: range endpoints (spheres at the arc ends / travel-line ends).
  Dragging one maps cursor → angle about the compiled axis (hinge) or distance along it
  (slide) → converts to authored units via `ReadOrientContext` → writes the authored
  `range` field inside a `BeginEdit`/`CommitEdit` gesture. The *interaction* math (mouse →
  candidate value) may be approximate; the *display* is regenerated from the recompiled
  model + mj_forward, so nothing shown can drift.
- **Cons**: most implementation surface (picking on plugin-scene handles is editor-side
  screen-space math like `TryPickJoint`); needs care during the drag: preview by patching
  `m->jnt_range` on the editor's compiled copy + re-posing ghosts (same LivePatch spirit —
  the on-release compile reconciles from ground truth).
- Effort: **M**.

### (d) Riding host-native affordances — evaluated and mostly ruled out

- `mjVIS_JOINT`: not reachable from the plugin (host-owned `vis_options_`), shows no range,
  no selection emphasis. Document the `J` key; don't build on it.
- Host perturb: structurally disabled under the Edit freeze (`!plugin_stepped` branch,
  app.cc:346-354). Would require a fork change (against the min-ask doctrine). Ruled out.
- Verdict: build on ScenePlugin + the preview/mirror channel — both proven, both ours.

### Recommendation: (a) + (b) + restored (c)-viz first, then (c)-handles

Scrub answers "what does this joint DO", ghosts answer "what does this RANGE mean", the
depth-occluded axis/arc answers "where is it and which way" — together they cover the
user's problem with the strongest possible correctness story (two of the three render
nothing but mj_forward output). Handles come second: they are ergonomics on top of the
same visuals.

### Interaction flow

- **No new mode.** Edit mode + a selected joint = rig affordances live, exactly like the
  gizmo today (gated on `CanEdit()`; `TryPickJoint` and hierarchy selection both work).
  The Details panel Joint view grows a **"Rig" section** at the top:
  - a scrub slider per DOF (hinge: degrees displayed, radians underneath; slide: metres;
    display conversion only — the value written to qpos is always compiled units),
    bounded by `jnt_range` when `jnt_limited`, else a model-extent/±180° span with an
    "unlimited" badge;
  - readouts: compiled `jnt_range` (in display units), `limited` (with "(auto)" when the
    authored TriState is auto), `ref`/`springref` if set;
  - a "Ghosts" checkbox (default ON when limited) and "Reset pose" (clear preview).
  - Scrub is **preview-only**: it never calls BeginEdit/CommitEdit, never sets `dirty`,
    never recompiles. Snap-back on deselect / Edit-exit / compile-generation change.
- Viewport: ghosts + axis/arc always drawn for the selected joint (and for all joints when
  the existing toolbar "Joints" toggle is on — axis/anchor only for unselected ones);
  scrub-by-limb-drag (P2) grabs any subtree geom while a joint is selected and maps the
  cursor to the joint's dof.
- **Multi-joint bodies** (e.g. free+hinge stacks, 3-hinge gimbals): the Rig section lists
  one slider row per joint on the selected body (via `CollectJointVis` body scoping), each
  independently scrubbable; ghosts follow the *selected* joint only. Scrubbing joint A with
  joint B previously scrubbed composes naturally (both are just qpos writes).
- **Ball joints**: P1 shows anchor sphere + readout only (range[1] = max rotation angle);
  scrub via three axis-angle sliders and a swing-limit visual land in P3. **Free joints**:
  no rig affordances (nothing to rig; the gizmo already ignores them, gizmo.cc:668).
- **Degrees vs radians**: display degrees for rotational DOFs (suffix °), metres for slide.
  The scrub/qpos side is always radians/metres (compiled); the *only* place authored units
  appear is the commit path of a range-handle edit (P2), via `ReadOrientContext` +
  joint-type branch (slide never converts).

---

## 3. Correctness guarantees (ground truth + pinning test per visual)

| Visual element | Ground truth (compiled/forwarded quantity) | Pinning windowless test |
|---|---|---|
| Scrubbed pose (subtree) | `mj_forward(ctx.compiled.model, sim_data)` after `qpos[jnt_qposadr]=q` | Load fixture twice (editor pipeline vs raw `mj_loadXML`); scrub q; assert `sim_data->xpos/xquat/geom_xpos` == reference `mj_forward` at same qpos, exact (same code path ⇒ bitwise) |
| Host-rendered scrub | mirror copy of the above | Drive `MirrorDragKinematics` under the new flag; assert host `mjData` arrays == preview arrays; assert `DoUpdate` leaves host data untouched while `rig_preview_active` (extend the existing gizmo-defer test, test_plugin_windowless.cc:251-256) |
| Snap-back | `qpos0` + forward | After ClearPreview: `sim_data->qpos == m->qpos0`, xpos == reference forward at qpos0; `ctx.dirty` unchanged; no recompile requested |
| Ghost poses | `mj_forward` at qpos0-with-dof:=`m->jnt_range[k]`, geoms from `m->geom_*` | For each ghost mjvGeom: pos/mat equal scratch `d->geom_xpos/geom_xmat` from an independent forward at that qpos; ghosts exist iff `m->jnt_limited`; count == subtree geom count; α < 1; category == mjCAT_DECOR |
| Hinge arc + endpoints | anchor/axis: `d->xanchor/d->xaxis`; endpoints pinned to real limit poses | Choose reference point p = a subtree geom centre at current qpos. Assert `Rot(xaxis, jnt_range[k] − q_now)·(p − xanchor) + xanchor` equals the same point under `mj_forward` at qpos=`jnt_range[k]` (tol 1e-12). This pins the one piece of derived geometry (the arc fan) to MuJoCo's kinematics and kills the old PlaneBasis arbitrary-zero defect |
| Slide travel line | `xanchor + xaxis·(range[k] − q_now)` | Same pin via mj_forward at the endpoint qpos (child translates by exactly Δq along xaxis) |
| Range/limited readouts | `m->jnt_range`, `m->jnt_limited` | Fixture with `limited` unset + range set (autolimits): assert readout says limited=auto-on because `m->jnt_limited==1`; fixture with `limited="false"`: no ghosts/arc |
| Units (display) | display° = rad·180/π of compiled values, display-only | Fixture `<compiler angle="degree">` `range="-30 45"`: `m->jnt_range == {−30π/180, 45π/180}`; panel string shows −30/45; a `<compiler angle="radian">` twin shows the radian values verbatim. Slide fixture: no conversion anywhere |
| Range-handle commit (P2) | authored `range` field, converted once via tree `compiler.angle` | Simulated drag to rad target → authored field == expected degrees (or radians per fixture; metres for slide) → recompile → `m->jnt_range` == drag target (round-trip, no double-convert) → undo restores prior authored value |

Double-conversion is structurally prevented: the render/scrub layer reads ONLY compiled
radians/metres; the authored layer is touched ONLY by the P2 commit helper, whose single
conversion is tested round-trip through a real recompile.

---

## 4. Edit-commit path

- **Scrub = preview only, never commits.** No BeginEdit, no dirty, no recompile, authored
  tree untouched. Cleared on: slider release (default snap-back; "hold" keeps the preview
  qpos while the joint stays selected), joint deselect, Edit-mode exit
  (`ExitEditToHost`), and compile-generation change (re-apply or drop).
- **Range-handle drag (P2)** = a normal gesture edit, identical to a details field:
  grab → `BeginEdit()`; per-frame preview patches `m->jnt_range` on the editor's compiled
  copy (LivePatch spirit — ghosts/arc re-pose live; on failure fall back to debounced
  recompile); release → write authored `range` (unit-converted) → `CommitEdit("joint
  range")` → undo step + debounced recompile → fresh artifact + new `sim_data`
  (editor_ops.cc:160-178) → host adopts bytes next frame → ghosts rebuilt from the new
  `m->jnt_range`. Esc mid-drag → `CancelEdit()` (deferred revert, editor_context.h:459-474).
- **Axis/pos edits**: unchanged — the existing joint gizmo path (grab → per-frame authored
  write + `LivePatchJoint` → `CommitEdit` on release).
- **What recompiles when**: only commits recompile (debounced, coalesced by
  `ServiceEditorModel`); scrubbing recompiles nothing. A field edit to range in the panel
  recompiles as today — and now visibly re-poses the ghosts, which is the instant feedback
  the user asked for.

---

## 5. Risks (candid)

1. **Scrub vs Edit freeze**: in Edit, `DoUpdate` forwards host data at qpos0 every tick
   (protospec_editor.cc:283-287). The scrub mirror MUST be paired with a deferral flag in
   `DoUpdate` exactly like `gizmo_active` (protospec_editor.cc:250-258); relying on
   dispatch order alone would work today but is fragile and against the documented
   contract. Pinned by extending the existing windowless defer test.
2. **Adoption races**: a debounced recompile can land mid-scrub (user edited a field, then
   scrubbed within the debounce window). Guards: re-resolve jid by serial each frame
   (serials are stable), `DragDimsMatch` before mirroring (viewport_plugin.cc:531-535),
   and on `compile_generation` change re-apply the scrub q onto the fresh `sim_data`
   (clamped to the new range) or clear the preview. The `AdoptDimsMatch` freeze
   (protospec_editor.cc:213-238) already protects the host side.
3. **Exit-to-Play mid-scrub**: host qpos is never written by the mirror (kinematics arrays
   only), so Play always starts from authored qpos0 — inherently safe; still clear the
   preview + flag in `ExitEditToHost` consumers so the last mirrored frame doesn't linger.
4. **Geom budget**: `plugin_scene_` maxgeom = 2000 (app.cc:100) shared by all ScenePlugins
   (selection outline uses ~12). Two ghosts × subtree geoms + arc segments; guard every
   append (as `AddOutlineBox` does) and cap ghost subtree size (e.g. first N geoms + a
   diagnostic) for pathological models.
5. **Mesh ghosts across two mjModels**: ghost `dataid` comes from the EDITOR's model but
   renders in the HOST's scene. The models are compiled from identical bytes (adoption
   contract; `AdoptDimsMatch` enforces dims), so mesh ids correspond; additionally gate
   ghost emission on `compile_generation == emitted_generation` (the state `DoUpdate`
   already tracks) to skip the one-frame adoption window.
6. **Multi-DOF stacks (free+hinge)**: pure qpos writes compose; the only UX risk is
   confusion about which slider moves what — mitigated by per-joint rows + ghosting only
   the selected joint. Ball qpos is a quaternion (4 slots, 3 dofs) — excluded from P1
   scrub to avoid inventing a parametrization prematurely.
7. **Performance**: per-frame `mj_forward` during scrub + 2 cached ghost forwards per
   invalidation. Precedent says fine (`mj_kinematics` per drag frame in `LivePatchJoint`;
   `mj_forward` every Edit tick on the host, protospec_editor.cc:286). If a giant model
   ever hurts, `mj_fwdPosition` (skips sensors/actuation) is a drop-in for preview needs.
8. **Plugin-surface hard limits**: no depth-occluded text (angle readout is screen-space
   ImGui near the anchor, or `mjvGeom.label`); no host vis-option access; plugin geoms
   don't participate in host pick (good here). Nothing in this plan needs a fork change
   or a new upstream ask.

---

## 6. Phased implementation plan

### P1 — scrub + limit ghosts + restored depth-occluded joint viz  (Effort: M overall)

The smallest slice that solves the stated problem, correct by construction.

Files touched (all inside the plugin; **zero** new SDK/host surface):
- **new** `studio/editor/joint_rig.h/.cc` — the windowless core:
  - `RigPreview` state (serial, q, active, hold) — struct lives in `editor_context.h`;
  - `SetJointPreview(ctx, serial, q)` / `ClearJointPreview(ctx)`: qpos write +
    `mj_forward` on `ctx.sim_data` (re-resolve jid by serial via Binding);
  - `GhostQpos(m, jid, end)`; `CollectGhostGeoms(m, scratch_d, jid) -> std::vector<mjvGeom>`
    (subtree walk over `body_parentid`, mjv_initGeom + dataid, DECOR, α);
  - depth-occluded joint glyph builder (axis arrow / anchor sphere / hinge arc / slide
    travel line as mjvGeoms), arc reference direction pinned per §3;
  - display-unit helpers (`DisplayAngle`, joint-type aware).
- `studio/editor/editor_context.h` — `RigPreview rig_preview;` + scratch ghost `mjData*`
  (owned/rebuilt beside `sim_data`); flag accessor used by DoUpdate.
- `studio/editor/protospec_editor.cc` — `DoUpdate`: defer host writes while
  `rig_preview.active` (mirrors the gizmo_active branch, ~4 lines).
- `studio/editor/editor_ops.cc` — rebuild/clear ghost scratch data with `sim_data`;
  clear preview on recompile adopt.
- `studio/editor/viewport_plugin.cc` — `EnhanceScene`: append joint glyphs + ghosts
  (replacing the screen-space `DrawJointsScreen` for the occluded parts; keep the
  screen-space layer only for the free-joint triad label/readout text); `ViewportUpdate`:
  mirror while `rig_preview.active` (reuse `MirrorDragKinematics`).
- `studio/editor/details_panel.cc` — Joint "Rig" section: per-DOF scrub slider
  (preview-only, deliberately NOT a gesture edit), compiled range/limited/ref readouts
  with units, ghost toggle, reset. Multi-joint body: one row per joint on the body.
- **tests**: **new** `studio/editor/test/test_rigger_windowless.cc` +
  `protospec/tests/test_rigger_windowless.py` (recipe cloned from
  test_plugin_windowless.py). Gates: scrub == reference mj_forward; snap-back; ghost
  pose/count/alpha/category pins; arc/travel endpoint pins vs mj_forward; unit fixtures
  (degree/radian/slide); jnt_limited auto pin; DoUpdate deferral under rig_preview;
  no-commit invariants (dirty/undo untouched by scrubbing).
- Out of scope for P1: any commit-from-viewport interaction; ball/free scrub; labels.

### P2 — interactive handles  (Effort: M)

- Range endpoint handles (spheres at hinge-arc ends / slide-line ends): hover + drag in
  `viewport_plugin.cc`/`gizmo.cc` style (screen-space proximity like `TryPickJoint`);
  drag preview patches `m->jnt_range` + re-poses ghosts live; release commits authored
  `range` via BeginEdit/CommitEdit with the single unit conversion
  (`ReadOrientContext`, hinge/ball only); Esc cancels. Snap: 5° / snap_translate.
- Direct limb scrub: drag any subtree geom of the selected joint's body → cursor-to-dof
  mapping (hinge: plane projection angle; slide: axis projection) → `SetJointPreview`.
  Display remains pure mj_forward, so the mapping being approximate is cosmetic.
- Angle/travel readout while scrubbing/dragging (screen-space text near anchor, shows
  q in display units + range).
- Files: `joint_rig.cc` (+ handle geometry/picking as pure functions), `viewport_plugin.cc`,
  `gizmo.cc` (or a sibling `rig_handles.cc` to keep gizmo upstream-mergeable, matching the
  transform_math.h:214 separation rationale).
- Tests: commit round-trip (drag target → authored units → recompile → jnt_range equals
  target; undo restores); handle-pick hit math; cursor→dof mapping monotonicity; cancel
  restores authored range; scrub-by-limb produces identical state to slider scrub.
- Out of scope: ball handles, multi-select rigging.

### P3 — polish  (Effort: M/L)

- **Ball joints**: swing-limit visual (fan of limit poses: mj_forward at quaternions of
  angle `jnt_range[1]` about sampled axes ⊥/∈ the joint frame — still mj_forward-grounded,
  no analytic cone), 3-axis scrub sliders (axis-angle → qpos quat → mj_forward).
- **Tendon/actuator awareness**: badge joints referenced by actuators/tendons (Binding +
  tree refs); optional ctrl-scrub through an attached position actuator later (needs
  mj_step-free actuator kinematics — investigate, may stay out).
- **Multi-joint UX**: "pose all joints" mini-table; per-joint ghost color coding; keyframe
  capture of a scrubbed pose into a `<keyframe>` (a real authored commit — reuses the
  normal commit path).
- `springref`/`ref` visualization (a third, differently-tinted ghost at qpos_spring).
- Screen-space → `mjvGeom.label` readouts if occluded labels prove desirable.
- Tests per feature, same harness; ball-pose pins vs mj_forward at constructed quats.

### Explicitly out of scope (all phases)
- Any fork/host change, any new upstream ask (everything above rides
  ModelPlugin/ScenePlugin/GuiPlugin/KeyHandler already in place).
- IK ("drag the end-effector, solve the chain") — different feature, different correctness
  story.
- Editing `jnt_range` by dragging ghosts themselves (ambiguous target; the arc endpoints
  are the handle).
- Play-mode rigging (CanEdit gate stays authoritative).
