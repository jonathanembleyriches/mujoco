# Gizmo-drag performance: incremental mjModel pose patching

Investigation into replacing the per-frame full recompile that ProtoSpec Studio
runs during a gizmo drag with an incremental `mjModel` pose patch + `mj_forward`,
recompiling only on drag release. Report-only; no production code changed.

## TL;DR

- **Pose drags (translate/rotate of body/geom/site/camera/light, and joint
  anchor/axis) are incrementally patchable.** The compiled field for every
  spatial element is `field = A ∘ L_authored ∘ B`, where `A` (a left-prefix) and
  `B` (a right-suffix) are baked frames the compiler already computes and which
  are **constant** across a pose-only edit. Capture `A` and `B` per element at
  compile time into the Binding; during a drag write `A ∘ L_new ∘ B` into
  `body_pos/geom_pos/...` and call `mj_forward`. No recompile.
- **The read-back that "behaves weirdly for mesh geoms" (DR-S6) fails only
  because it tries to *invert* `B`.** Write-forward never inverts anything — it
  re-applies the *same* captured `B` to the freshly authored `L_new`, which is
  exactly correct for mesh geoms.
- **Size/scale edits should recompile on release** (they propagate into inertia
  and mass). They can be *previewed* live by patching `geom_size/geom_rbound/
  geom_aabb`, but the inertia will be stale until release.
- **Native recompile is NOT uniformly ~3 ms.** Measured on the current core:
  tiny primitive models 0.3-3 ms, a single mesh geom ~18 ms (qhull runs every
  compile), 100-DOF / muscle / particle models 30-57 ms, and any XML-fallback
  model 100-660 ms. So per-frame full recompile is smooth only for the smallest
  primitive models; incremental patching helps the *general* case, not just XML
  fallback.

---

## 1. How the compiler bakes a pose (the `A ∘ L ∘ B` structure)

`attic/compile/build.cc` builds each spatial element's compiled pose by starting
from the authored `pos/quat` and accumulating frames. Two accumulation
primitives (`attic/compile/lifted/mjuu_util.cc:481,495`):

- `mjuu_frameaccum(T, Child)` → `T = T ∘ Child` — **Child is a right-suffix**.
- `mjuu_frameaccumChild(P, Child)` → `Child = P ∘ Child` — **P is a left-prefix**.

### Geom (`build.cc`)
Starting from authored `cg.pos/cg.quat`:
1. **Mesh / fitted-mesh bake (SUFFIX `B`)** — `build.cc:834` (fitted) and
   `:857` (mesh): `mjuu_frameaccum(cg.pos, cg.quat, meshpos, mesh_quat)`. This
   folds `mesh_pos/mesh_quat` (the compiler's mesh recentering / principal-axis
   frame) into `geom_pos/geom_quat`. **This is the DR-S6 culprit** — the
   recentering offset lives inside `geom_pos` after compile.
2. **Enclosing `<frame>` chain (PREFIX `A`)** — `build.cc:3199,3361`:
   `if (xf.present) mjuu_frameaccumChild(xf.pos, xf.quat, cg.pos, cg.quat)`.
   Frames are flattened away in `mjModel`, so the frame chain between the owning
   body and the geom is folded into `geom_pos` as a left-prefix.
3. **Free-joint alignment (PREFIX `A`, conditional)** — `build.cc:3440`:
   for a body whose free joint is aligned, every child geom gets
   `mjuu_frameaccumChild(ipos_inv, iquat_inv, geoms_[j].pos, geoms_[j].quat)` —
   the inverse inertial frame prepended.

Net: `geom_pos_field = A ∘ L_authored ∘ B`, with
`A = InvInertial(align_free)? ∘ FrameChain`, `B = mesh/fit frame`.

Final write to `mjModel`: `build.cc:3812` `geom_pos/geom_quat`.

### Site / camera / light (`build.cc` `FillVisual` @3843)
Same as geom but **no mesh suffix**: enclosing-frame prefix (`build.cc:1071,
1116,1222`) and the align-free inverse-inertial prefix (`:3445,3448,3452`).
`B = identity`. Writes: `site_pos` @3861, `cam_pos` @3883, `light_pos` @3898.
(Lights carry a `dir` vector rotated by `iquat_inv`, not a quat — the existing
gizmo already special-cases this in `RotateElem`.)

### Body (`build.cc` `Collect`)
1. **Enclosing frame (PREFIX `A`)** — `:3425`
   `if (body_xf.present) mjuu_frameaccumChild(body_xf.pos, body_xf.quat, cb.pos, cb.quat)`.
2. **Free-joint inertial fold (SUFFIX `B`, conditional)** — `:3436`
   `mjuu_frameaccum(cb.pos, cb.quat, cb.ipos, cb.iquat)` (align_free only).
Writes: `body_pos/body_quat` @3674.

So bodies also fit `A ∘ L ∘ B`. For the overwhelmingly common case — a body
directly under another body, no enclosing frame, no free-joint alignment —
`A = B = identity` and `body_pos = L_authored` verbatim.

### Why write-forward works where read-back fails
`transform_math.cc` already computes the new authored local pose `L_new` (it
writes it into the tree in `ApplyTranslate`/`ApplyRotate`). Reading `geom_pos`
back and trying to recover `L` requires inverting `B` (the mesh recentering) —
that is what double-applies and corrupts mesh geoms. Writing forward composes
`A ∘ L_new ∘ B` and never inverts `B`; because `A` and `B` are compile-time
constants of the model *topology/mesh*, and a pose edit changes neither, they
remain valid for the whole drag.

---

## 2. The PosePatch design

### What the bridge/Binding must expose

Capture, at compile time, a per-element patch descriptor:

```cpp
// bridge/binding.h — captured during the build, one per spatial element.
struct PosePatch {
  int objtype = 0;         // mjOBJ_BODY / GEOM / SITE / CAMERA / LIGHT
  int id = -1;             // mjModel id (already in Binding::Entry)
  ps::Rigid prefix;        // A: left frame (frame chain [∘ inv-inertial])
  ps::Rigid suffix;        // B: right frame (mesh/fit; body free-joint inertial)
  bool has_suffix = false; // false => B == identity (sites/cams/lights/plain bodies)
  // Free/ball-jointed body: patching body_pos alone does not move the rest pose;
  // qpos0 (and the qpos slice) for this joint must also be reseeded. See §3.
  int reseed_qposadr = -1; // >=0 => also write qpos0[adr..]/qpos[adr..]
  int reseed_width = 0;    // 3 (free translation) or 7 (free) / 4 (ball)
};
```

`A` and `B` are already computed inside `build.cc` while accumulating the pose;
the change is to *record the two accumulators separately* (the running left
accumulation and the running right accumulation) per element instead of only the
fused product, and hand them to `BindingBuilder`. On the XML path (where the
element poses come from `mj_loadXML`, not our accumulators), the same descriptor
can be reconstructed from the tree: `A` = composed authored frame chain to the
owning body; `B` = mesh `mesh_pos/mesh_quat` for a mesh geom (readable from
`mjModel` by `geom_dataid` → `mesh_pos/mesh_quat`), identity otherwise. Both
paths converge on the same `PosePatch`.

Add one query + one applier to the bridge (keeps `mujoco.h` in the quarantine
zone):

```cpp
// Binding
std::optional<PosePatch> PosePatchFor(const void* elem) const;

// A new mujoco-side helper (compile.cc or a small patch.cc):
// Writes A ∘ L_new ∘ B into the element's mjModel pose field. Pure array write;
// caller runs mj_forward afterwards. Returns false if the element is unpatchable.
bool ApplyPosePatch(mjModel* m, const PosePatch& p, const ps::Rigid& L_new);
```

`ApplyPosePatch` is ~15 lines: compose `A ∘ L_new ∘ B` with the same
`Compose` used in `transform_math`, then `mjuu_copyvec` into
`m->{body,geom,site,cam}_pos/_quat` (or set `light_pos` + rotate `light_dir`).

### Editor wiring (`gizmo.cc` / `editor_ops.cc`)

`GizmoController::UpdateDrag` already does, each frame:
`ApplyTranslate/ApplyRotate(tree, ...)` (writes `L_new` into the authored tree)
then `ctx.RequestRecompile()`. Replace the `RequestRecompile()` on the *pose
tools* with:

```cpp
// After ApplyTranslate/ApplyRotate has written L_new into the tree:
if (auto pp = ctx.compiled.binding.PosePatchFor(elem)) {
  Rigid L_new = ReadAuthoredLocal(elem);           // same L transform_math wrote
  ApplyPosePatch(model, *pp, L_new);
  mj_forward(model, data);                          // recompute xpos/xquat/geom_xpos
  ctx.dirty = true;                                 // authored tree changed
} else {
  ctx.RequestRecompile();                           // fallback: unpatchable elem
}
```

On **release** (`HandleMouse` → `CommitEdit`), do exactly one real
`RequestRecompile()` (already happens via `CommitEdit`). That reconciles the
patched `mjModel` with a freshly compiled truth (picks up any second-order
effects: contacts, BVH, sameframe flags, qpos0 for jointed bodies), so patch
drift never accumulates past one drag.

The joint gizmo (`ApplyJointTranslate`/`ApplyJointAxisRotate` edit
`jnt_pos`/`jnt_axis`) is patchable the same way — write `jnt_pos`/`jnt_axis`
(no bake; body-frame vectors) + `mj_forward`.

### mj_forward correctness and the qpos0 caveat
Edit mode holds `mjData` at qpos0 (`app.cc:78,98` `mj_resetData` + `mj_forward`).
`mj_forward`'s kinematics stage recomputes `xpos/xquat/geom_xpos/geom_xmat/
site_xpos/cam_xpos` from `body_pos/quat`, `geom_pos/quat`, `qpos`, so a patched
field is reflected immediately. **Caveat (free/ball-jointed bodies):** such a
body's world pose at qpos0 is driven by `qpos` (seeded from `body_pos/quat` at
compile), not directly by `body_pos`. Patching `body_pos` alone will *not* move
it — the patch must also reseed `qpos0[adr..]` and `d->qpos[adr..]` for that
joint (hence `reseed_*` in `PosePatch`). Hinge/slide bodies are fine: their
qpos0 is 0 and `body_pos` is the anchor, so kinematics moves them from
`body_pos` directly.

---

## 3. Exact predicate: "pose-only, incrementally patchable"

A gizmo edit is incrementally patchable iff **all** hold:

1. **Tool is Translate or Rotate** (not Scale) — see §4 for scale.
2. **Target is a single Body / Geom / Site / Camera / Light / Joint / FreeJoint**
   resolved by serial (the families `FindSpatial`/`IsJointSerial` already handle).
3. **No structural change**: no add / delete / reparent. (Drags never do this;
   the Binding-snapshot contract in `binding.h:14-21` already states pose edits
   never invalidate ids, structural edits do.)
4. **No size / fromto-length change** (those are Scale, handled separately) and
   **no endpoint edit that changes a fromto capsule's length** (translate/rotate
   of a fromto geom move/rotate *both* endpoints rigidly, preserving length — the
   existing `is_fromto` branches in `transform_math.cc` already do this, so a
   rigid fromto move is still pose-only and patchable via `geom_pos/quat`).
5. **The element bound** (`Binding::Id(elem)` present) and a `PosePatch` was
   captured for it. Elements dropped by `discardvisual`/`fusestatic`, or produced
   by macro expansion (replicate/attach/composite/flexcomp — not gizmo-selectable
   anyway), have no stable single id → fall back to recompile.
6. **Mesh scale unchanged** (mesh-geom translate/rotate is fine; the mesh *asset*
   `scale` is a Scale edit that changes `B` itself → recompile).

Everything else — size edits on inertia-participating geoms, mesh-asset scale,
any add/delete/reparent, edits to defaults/classes or references, compiler-flag
changes — **must recompile**.

---

## 4. Scale edits

`geom_size` feeds a chain of derived quantities: `geom_rbound`, `geom_aabb`
(`build.cc:3810,3811,3834`), and — for inertia-participating geoms — the body
inertial frame `body_ipos/iquat`, `body_inertia`, and `body_mass`
(`InertiaFromGeom`, `build.cc:3396-3419`), which then also shift `body_pos` when
a free joint is aligned. Because a size change propagates into mass/inertia and
can move the inertial frame (and thus the align-free body pose and every child's
inverse-inertial prefix), **scale is not a local patch**.

Verdict: **recompile scale on release.** For live feedback during the scale drag
you may optionally patch `geom_size` + `geom_rbound` + `geom_aabb` and
`mj_forward` to preview the *shape* (visually correct), accepting that inertia /
mass / contacts are stale until the release recompile. Mesh-geom scale edits the
mesh asset (model-wide) and always recompiles.

---

## 5. Native compile speed on the current core

Timed with a throwaway probe linking the current bridge libs (Jul 16, matching
HEAD `e2dfe7a0`) against the current MuJoCo core, median of N compiles of an
already-parsed `Model` (parse excluded), `mujoco.dll` warm. `auto` = the path
Studio actually uses (`CompilePath::Auto`, native when supported else XML
fallback); `xml` = forced XML path.

| model | dofs / feature | auto (path) | forced xml |
|---|---|---|---|
| car.xml | tiny primitive | **0.7 ms** (native) | 1.9 ms |
| slider_crank.xml | tiny | **0.3 ms** (native) | 1.0 ms |
| balloons.xml | small | **1.1 ms** (native) | 1.8 ms |
| humanoid.xml | ~27 DOF primitive | **3.2 ms** (native) | 3.1 ms |
| bunny.xml | 1 mesh geom | **18.4 ms** (native) | (n/a) |
| humanoid100.xml | ~100 DOF | **31.8 ms** (native) | 40.8 ms |
| particle.xml | many bodies | **49.5 ms** (native) | 62.4 ms |
| arm26.xml | muscles/tendons | **57.1 ms** (native) | 28.5 ms |
| 22_humanoids.xml | falls back to XML | **109 ms** (xml) | 136 ms |
| 100_humanoids.xml | falls back to XML | **659 ms** (xml) | 523 ms |

Findings:
- The "~3 ms native" assumption holds **only for humanoid-class primitive
  models**. A single mesh geom already costs ~18 ms because the native mesh
  pipeline re-runs qhull convex-hull + inertia **every compile** (`build.cc`
  mesh branch, `protospec_lifted`). Muscle models re-run `mj_setLengthRange`
  every compile (arm26: native *slower* than XML, 57 vs 28 ms).
- 100-DOF and particle models are 30-50 ms; XML-fallback models are 100-660 ms.
- At 60 fps the frame budget is ~16 ms. So per-frame full recompile is smooth
  only for the sub-~3 ms tier (tiny/humanoid primitive models). For mesh, large,
  muscle, or any XML-fallback model, dragging visibly stutters or drops to a few
  fps.

**Conclusion:** incremental patching is **not** an XML-fallback-only concern.
It benefits every non-trivial model, and is essential for mesh and large models
even on the native path. (It does not need to be gated on `report.taken`.)

---

## 6. Recommendation

Implement **pose-drag incremental patching with commit-on-release reconcile**:

1. **Capture `PosePatch{A, B, field-id, reseed}` per spatial element at compile**
   (native path records the two frame accumulators separately in `build.cc`; XML
   path reconstructs from the tree + `mesh_pos/mesh_quat`). Store on the Binding.
2. **During a Translate/Rotate drag**: after `transform_math` writes `L_new` into
   the authored tree (unchanged), compute `A ∘ L_new ∘ B`, write it into the
   `mjModel` pose field via `ApplyPosePatch`, reseed qpos0/qpos for a free/ball
   body, and `mj_forward`. Skip the recompile. `ctx.dirty = true`.
3. **On release** (`CommitEdit`): one real `Compile` reconciles the patched model
   with ground truth — bounding drift to a single drag. Undo/save already read
   the authored tree, which the drag kept correct throughout.
4. **Scale**: recompile on release; optional live `geom_size/rbound/aabb` preview.
5. **Fallback**: any element with no captured `PosePatch`, or any non-pose edit,
   keeps today's `RequestRecompile()` path — behaviour is never worse than now.

### API surface (minimal, keeps the quarantine)
- `bridge/binding.h`: `struct PosePatch`, `Binding::PosePatchFor(const void*)`.
- `bridge` (new small TU or in `compile.cc`): `bool ApplyPosePatch(mjModel*,
  const PosePatch&, const Rigid& L_new)`.
- `build.cc`: record the left/right accumulators per element → `BindingBuilder`.
- `gizmo.cc`: swap the pose-tool `RequestRecompile()` for patch + `mj_forward`.

### Effort
- Bridge `PosePatch` capture (native accumulators + XML reconstruction) +
  `ApplyPosePatch`: ~1-1.5 days.
- `build.cc` plumbing of the two accumulators through to the binding: ~0.5 day
  (the accumulation sites are already localized — §1 line refs).
- Editor wiring + free/ball qpos0 reseed + tests (extend `test_gizmo.cc`, which
  already builds `mjData` at qpos0 and drives the delta math): ~1 day.
- **Total ≈ 2.5-3 days.**

### Risks / mitigations
- **Patch drift vs recompiled truth** → bounded to one drag by the
  commit-on-release recompile; the authored tree (source of truth for save/undo)
  is written every frame regardless, so a drift can never be *saved*.
- **Free/ball-jointed body rest pose** → handled by the `reseed_*` fields; if
  deemed not worth the complexity initially, exclude jointed-body pose drags from
  the fast path (recompile them) — geoms/sites/cameras/lights/static bodies (the
  common gizmo targets) need no reseed.
- **XML-path `PosePatch` reconstruction** → if reconstructing `A`/`B` on the XML
  path is fiddly, ship native-path patching first (native already covers ratchet
  353 features); XML-fallback models still recompile until the reconstruction
  lands, and they are the minority.
- **Second-order visuals** (contacts, BVH-dependent overlays) are stale mid-drag
  → acceptable in edit mode at qpos0; the release recompile restores them.
