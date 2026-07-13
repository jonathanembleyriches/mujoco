# ProtoSpec Studio: editor certification checklist

Companion to `docs/plan_studio_editor.md` (design, DR-S1/S2/S6, SE0-SE4) and
`docs/studio_ui_migration.md` (real-Studio hosting). Unreal adoption is gated on the model
editor being **certified** — this document defines certified so it can be signed, not felt.

Certified = **all Section 1 rows green + every Section 1 gap closed or owner-waived + every
Section 2 step PASS in one sitting + the Section 3 statement signed.**

**Canonical code under test:** the `studio` branch of `C:\Users\jonat\Documents\mujoco-studio`
(real Studio host + editor cluster at `src/experimental/studio/protospec/`). The batteries also
live in the protospec repo at `apps/studio/test/`; note `test_gizmo.cc` has **diverged** — the
studio-branch copy adds the joint-rig battery and the Q-ORIENT sync (7ebe3a08) and is the copy
that counts. Sync-back is a gap item (G9).

---

## 1. Automated certification

All five batteries plus the core-repo suites must exit 0 at the certification commit.

| Battery | Binary | Test fns |
|---|---|---|
| Core (load, pick, hierarchy, rename/delete, undo, save) | `test_studio` | 11 |
| Gizmo (delta rule, joint rig, projection, mode machine) | `test_gizmo` | 20 |
| Movability audit (inline fixtures + 5 corpus models) | `test_movability` | 4 + audit engine |
| Details (reflection coverage, presence, refs, enums) | `test_details` | 8 |
| Authoring (add/duplicate/reparent/import/exit story) | `test_authoring` | 9 |

### 1.1 Guarantee → evidence map (must be green)

| # | Editor guarantee | Covering evidence |
|---|---|---|
| A1 | **Movability**: every body/geom/site/camera/light in the corpus moves under gizmo translate by exactly the world delta, or is classified (worldbody / class-template = nonscene; compiler-rejected large delta = excused **loudly** with the diagnostic). Expected-immovable list is **empty** — any silent no-op (the fromto class) fails. | `test_movability.cc` — 5 inline fixtures (fromto, class-inherited fromto, frame, mocap, world elems) + humanoid, car, arm26, slider_crank, cards |
| A2 | **Delta rule (DR-S6)**: translate on rotated body / nested body / inside `<frame>`; **mesh-frame bake cancels exactly** (compiled pose moves by D; second drag shows no accumulating jump); rotate = left-applied world delta, position invariant; local-vs-world axis mapping; class-inherited pose materializes via `Effective()` on first drag | `test_gizmo.cc`: TestTranslate*, TestMeshFrameCancellation, TestRotateGeom, TestLocalVsWorldTranslate, TestClassInheritedPoseMaterialize |
| A3 | **fromto semantics**: rotate pivots the midpoint (endpoints rewritten, pos/orient never materialized); twist about the axis is a documented no-op; scale maps to endpoints + radius | `test_movability.cc`: TestFromToRotate / TwistIsNoOp / FromToScale |
| A4 | **Scale mapping**: per-axis on box, uniform on sphere, mesh geom scale → mesh **asset** scale (geom size untouched) | `test_gizmo.cc` TestScaleMapping; fromto in A3 |
| A5 | **Joint-rig math**: anchor translate moves the compiled global anchor by exactly the world delta (authored pos = parent-frame conjugate); axis reorient hand-verified through body rotation; cardinal snap; ball = anchor-only, free = triad-only; joint-vis collection (selected body → own joints, selected joint flagged, show-all) | `test_gizmo.cc` (studio branch): TestJointTranslateAnchor / ReorientAxis / AxisSnap / BallFreeNoAxis / CollectJointVis |
| A6 | **Reflection coverage**: every field of every element descriptor (1,444 fields / 142 types) classifies to a real widget — zero Unhandled, so a schema change cannot silently drop a field from the UI; named mappings (checkbox/enum/ref/variant/quat-row/…) pinned | `test_details.cc` TestWidgetCoverage + TestWidgetMappingSpecifics |
| A7 | **Presence layers (DR-1 visible)**: authored / class-inherited / IDL-default / required / unset resolve correctly; per-field revert restores inheritance | `test_details.cc` TestPresence* |
| A8 | **Ref combos**: target-type resolution incl. union expansion (TendonAny → Spatial+Fixed), whole-tree candidate population, dangling-ref detection | `test_details.cc` TestRefTargetsAndCandidates |
| A9 | **Enum/keyword tables**: every label round-trips label ↔ value; InlineVec grow/shrink bounds | `test_details.cc` TestEnumLabels, TestInlineVecBounds |
| A10 | **Undo/redo incl. serial preservation**: CloneWithSerials reproduces every serial (vs generated Clone minting fresh); selection survives undo across the model swap; redo branch cleared by a fresh commit; CancelEdit reverts; **gesture granularity** — one drag = one undo entry, Esc-cancel = zero entries; depth bound 128 | `test_studio.cc` TestCloneWithSerials + TestUndoRedoBattery; `test_gizmo.cc` TestGestureUndoGranularity |
| A11 | **Authoring ops compile**: every add (body/geom×types/joint/site/camera/light/frame/actuator/sensor) followed by a real compile; drop-add lands at the click point; frame-nested adds work | `test_authoring.cc` TestAddPrimitivesCompile, TestDropAdd |
| A12 | **Name uniquing**: repeated adds never collide; UniqueName stable/suffixing | `test_authoring.cc` TestNameUniquing |
| A13 | **Duplicate ref remap**: fresh serials throughout, uniquified names, internal refs remapped to the clone, external refs preserved to the original, subtree shape preserved, result compiles | `test_authoring.cc` TestDuplicate |
| A14 | **Reparent**: keep-world-pose invariant (plain and frame-wrapped), cycle rejection leaves the model untouched | `test_authoring.cc` TestReparent* |
| A15 | **Delete/rename referrer safety**: rename rewrites every typed referrer atomically; delete previews the referrers that would dangle (read-only) and cascades on confirm | `test_studio.cc` TestRenameUpdatesReferrers, TestDeleteSurfacesReferrers |
| A16 | **Mesh import**: unsaved model → compile-VFS bytes, auto mesh geom, compiles; save externalizes bytes to disk next to the .xml and a fresh disk load compiles | `test_authoring.cc` TestMeshImportVfsAndExternalize |
| A17 | **Save round-trip fidelity**: editor save is a byte-stable WriteMjcf fixpoint + deep-equal reload; the SE3 exit story (blank → build → simulate → save → reload → deep compare); the writer's own fixpoint battery + differential-vs-MuJoCo harness | `test_studio.cc` TestSaveRoundTripFixpoint; `test_authoring.cc` TestExitStory; `cpp/test/test_io.cc`; `cpp/harness` |
| A18 | **Load/pick/registry core**: clean failure on bad path (tree untouched, diagnostics populated); humanoid loads via ParseMjcf→Compile; every bound geom/body id resolves back to its authored element with the same serial; plugin registry order + name-dedup + pointer mutation | `test_studio.cc` TestLoadFailureIsClean, TestLoadHumanoid, TestPickResolvesToElement, TestRegistryOrderAndDedup |
| A19 | **Mode-machine recompile gate (DR-S2)**: Edit services queued recompiles; Play defers them; the one-shot `apply_edits` (Play transition) forces one | `test_gizmo.cc` TestModeMachineRecompileGate |

### 1.2 Gaps — guarantees with no automated test yet

Each row is a checklist item: **add the test** (or the owner initials the Waived column).

| ID | Gap | What to add | Closed/Waived |
|---|---|---|---|
| G1 | **Play/Stop state-discard end-to-end in the real host.** A19 tests only the windowless gate; the host wiring (`app.cc` Play → `set_editor_mode(1)` + unpause, Stop → mode 0 + `ResetPhysics()` to qpos0, dirty edits compiled on Play) has zero coverage. | Scripted host-level test (offscreen `App` loop): edit → Play → step N → assert qpos moved → Stop → assert qpos0 and tree keeps the edit | ☐ |
| G2 | **Key routing in the real host.** Q/W/E/R deliberately shadow Studio's camera-vis toggles and must win only while the viewport is hovered; Ctrl+S/Z/Y/D and Del routing vs ImGui text-input capture. Untested. | ImGui-injected key-dispatch test over the hosted app: keys with viewport hovered vs Details focused vs Hierarchy focused | ☐ |
| G3 | **Overlay picking through the real renderer.** Windowless tests cover Binding reverse-lookup only ("full Pick needs a live mjvScene/camera"); the ray-pick → geom id path and gizmo-handle hit priority over camera orbit run only in the live app. | Offscreen render + synthetic-click test: click at the projected screen pos of a known geom → assert selection serial; click a gizmo handle → assert drag consumed (no orbit) | ☐ |
| G4 | **Save As + externalization in the real host.** A16/A17 are windowless; the menu path (`SaveExternalize`: inline path field → SaveModel → ExternalizeVfsAssets → dirty flag/title clear) is untested. | Host-level scripted Save As: unsaved model with VFS mesh → Save As → assert file + sibling mesh on disk, `vfs_assets` empty, dirty cleared | ☐ |
| G5 | **Layout persistence.** Curated dock layout (Hierarchy/Viewport/Details, spec panels hidden) on first run + imgui.ini restore across restarts. Untested. | ini round-trip check, or owner-waive to manual step M26 | ☐ |
| G6 | **Performance bound.** Recompile perf is informational only (`ReportHumanoidRecompilePerf` prints ~3 ms, never fails); no threshold enforced; nothing larger than humanoid ever measured. | Perf gate: humanoid ≤ 10 ms/compile hard fail; add one large corpus model (e.g. a scene-scale MJCF) with a measured, recorded budget | ☐ |
| G7 | **Delete-confirm dialog flow.** A15 tests preview/cascade logic; the `delete_request_serial` → dialog → confirm/cancel UI path is untested. | Host-level test or fold into manual step M25 (owner call) | ☐ |
| G8 | **Diagnostics navigation.** Failed-compile errors reach the Diagnostics panel (movability's excused path proves the report exists) but click-to-select-from-diagnostic has no test. | Windowless test: diagnostic row → SourceLoc → selection serial | ☐ |
| G9 | **Battery divergence.** `test_gizmo.cc` in the protospec repo lacks the joint-rig battery (studio branch is ahead). One source of truth or CI on both. | Sync studio-branch tests back to `apps/studio/test/` (or subtree them) and run both in CI | ☐ |

---

## 2. Manual sign-off script

Owner walks this in one sitting, ~25 minutes. Every step gets PASS or FAIL; any FAIL stops
certification.

**Setup.** Build `mujoco_studio.exe` from the `studio` branch with the ProtoSpec editor cluster
enabled (Windows: **Ninja + MSVC**, never the VS generator or clang-cl — see
`docs/studio_ui_migration.md` §1.6). Launch from **PowerShell** (not git-bash):

```
.\mujoco_studio.exe --model_file=<corpus>\model\humanoid\humanoid.xml
```

### Load and inspect

| # | Action | Expected | P/F |
|---|---|---|---|
| M1 | Launch as above | Studio window, curated layout: Hierarchy left, Viewport center, Details right; humanoid rendered at qpos0; old Explorer/Editor spec tabs absent; status bar shows compile path + time; title has no dirty marker | ☐ |
| M2 | Hierarchy: expand Body Tree, click `torso` | Viewport outlines the torso; Details shows Body fields, Transform group first | ☐ |
| M3 | Select a geom that inherits from a class; find a grayed field with the inherited badge; edit it; then use the per-field revert | Badge flips to authored on edit; revert restores the badge and the class value; title/status show dirty | ☐ |
| M4 | Type a body name fragment in the Hierarchy search box | Matches + their ancestors only; clearing restores the full tree | ☐ |
| M5 | Viewport: click a limb geom; click again on overlapping geometry; press F | Click selects (Hierarchy syncs); repeat click cycles overlaps; F frames the selection | ☐ |

### Gizmos on the humanoid (incl. fromto)

| # | Action | Expected | P/F |
|---|---|---|---|
| M6 | Select a **fromto capsule limb** (e.g. an upper arm), press W, drag an axis handle | Limb follows the drag live (debounced recompile is the preview); Details shows the fromto endpoints updating — pos/quat stay unset; release: no jump, no drift | ☐ |
| M7 | Press E, rotate the same limb; then try rotating about the capsule's own long axis | Limb rotates about its midpoint (midpoint fixed); twist about its own axis visibly does nothing (documented axisymmetric no-op) | ☐ |
| M8 | Undo (Ctrl+Z) twice | Both gestures revert as single steps each; limb back to original | ☐ |

### Joint rig

| # | Action | Expected | P/F |
|---|---|---|---|
| M9 | Toolbar: toggle **Joints**; then select a single body with the toggle off | Toggle on: joint overlays (axes/arcs) on every body; off + body selected: only that body's joints; selecting a joint highlights it | ☐ |
| M10 | Hierarchy: right-click a geom-bearing body → Add child → Joint (hinge). With the joint selected, drag its **anchor** gizmo, then its **axis** handle; repeat the axis drag with Snap on | Anchor follows the mouse exactly; axis reorients; with snap the axis lands exactly on a cardinal; Details `pos`/`axis` match what you see | ☐ |
| M11 | Toolbar +Add → Actuator → Motor (joint from M10 selected) | Motor appears under the Actuators section wired to the joint; model recompiles clean | ☐ |

### Authoring via all three surfaces + mesh

| # | Action | Expected | P/F |
|---|---|---|---|
| M12 | Hierarchy: right-click a body → Add child → Geom → Sphere | Sphere appears in viewport + tree, auto-named (`geom`, `geom_1`, …); compiles | ☐ |
| M13 | Viewport: right-click an empty floor spot → Add here → Box | New world-parented body+geom sitting at the clicked point | ☐ |
| M14 | Toolbar +Add → Body / geom (world) → Capsule | Capsule added at world level; no name collision with M12/M13 | ☐ |
| M15 | File → Import Mesh… → type the path to any `.obj` → Import | Status line confirms; Mesh appears in Assets; auto-created mesh geom renders; W-drag it — it moves cleanly (no mesh-bake jump) | ☐ |

### Duplicate / reparent / undo depth

| # | Action | Expected | P/F |
|---|---|---|---|
| M16 | Select `torso`, Ctrl+D | Full `torso_1` subtree with uniquified names; original untouched; compiles; the copy's internal refs point at `_1` names | ☐ |
| M17 | Hierarchy: drag a geom onto a different body (keep world pose) | Tree shows the new parent; the geom **does not move** in the viewport | ☐ |
| M18 | Make 20+ mixed edits (drags, Details edits, adds, a delete), then Ctrl+Z back through all of them, then Ctrl+Y forward several | Every step reverts/reapplies in order; selection keeps resolving (Details never blanks on a live element); no crash, no drift | ☐ |

### Play / Pause / Stop (DR-S2)

| # | Action | Expected | P/F |
|---|---|---|---|
| M19 | Press **Play** with dirty edits pending | Edits compile first, then physics runs — humanoid moves/falls | ☐ |
| M20 | **Pause**, then Play again; while playing, edit a field in Details | Pause freezes, Play resumes; the edit marks dirty but does NOT change the running sim | ☐ |
| M21 | Press **Stop** | Back to Edit mode at qpos0 of the **edited definition** — not the fallen pose; the M20 edit is in the tree and takes effect on the next Play | ☐ |

### Save / reload / diff

| # | Action | Expected | P/F |
|---|---|---|---|
| M22 | File → Save As… → a scratch path → Save | File written; dirty marker clears from title + status bar; imported mesh bytes externalized next to the .xml | ☐ |
| M23 | File → Open… the saved file; separately diff it against the original `humanoid.xml` (`git diff --no-index` / `fc`) | Reload renders identically; the diff shows **only your edits** — classes, `euler`, `fromto`, comments/structure preserved (form-preserving writer, not normalized soup) | ☐ |

### New-from-blank build-a-robot

| # | Action | Expected | P/F |
|---|---|---|---|
| M24 | File → New. Build a minimal robot entirely in-editor: add a body + box geom, rig a hinge (M10 flow), add a Motor, Play, Stop, Save As, File → Open it back | Starter scene (ground + light); robot simulates under Play; Stop returns to the definition; reload matches what you built | ☐ |

### Error paths

| # | Action | Expected | P/F |
|---|---|---|---|
| M25 | Load `model/tendon_arm/arm26.xml`; select a tendon-routing site; W-drag it very far | Compile fails **loudly**: Diagnostics shows the tendon error, viewport keeps the last good compile, nothing silently no-ops; Ctrl+Z recovers to a compiling model | ☐ |
| M26 | Select a joint that an actuator references; press Del | Confirm dialog lists the referrers; Cancel leaves everything intact; Del again + Confirm cascades — model still compiles, no dangling refs | ☐ |
| M27 | Rearrange one panel, quit, relaunch | Layout restored from imgui.ini (covers G5 until automated) | ☐ |

**27 steps.**

---

## 3. Exit criteria and known exclusions

### Exit criteria

1. All five batteries + `cpp/test` suites exit 0 at the certification commit (both repos).
2. Every §1.2 gap row marked Closed (test added and green) or Waived with owner initials.
3. All 27 manual steps PASS in one sitting on a fresh Ninja+MSVC build.
4. The statement below signed.

### Known exclusions — certified does NOT cover

| Exclusion | Why |
|---|---|
| Flex / flexcomp editing | Deferred by design (visible in Hierarchy, generic Details only, Auto→XML compile; no gizmos) |
| Composite beyond select-only; plugin-requiring models | Deferred until the plugin wave |
| Multi-select operations | SE4 QoL item, not landed |
| Keyframe timeline, USD, WASM, CoACD decomposition | Deferred (plan §9) |
| OS-native file dialogs | Open / Save As / Import use inline path fields **by design** (no dialog dependency in the editor library) |
| Python Studio | Separate surface (`protospec` pybind module) |
| Filament-vs-classic visual parity | Rendering look, not editor correctness |
| Perturb-to-tree | There is deliberately no path from Play-mode perturb to the model (DR-S2 firewall) |

### Performance bounds

| Metric | State |
|---|---|
| Humanoid drag recompile | ~3 ms/compile measured (informational print in `test_gizmo`); preview = truth well inside a 60 Hz drag budget |
| Enforced bound | none — G6 |
| Large-model behavior | unmeasured — G6 |

### Certification statement

> I certify that the ProtoSpec Studio model editor meets every guarantee in Section 1, that all
> Section 1.2 gaps are closed or waived above, and that I walked all 27 steps of Section 2 on
> the build identified below with zero failures.
>
> protospec commit: `________________`  mujoco-studio (`studio`) commit: `________________`
>
> Signed: ______________________  Date: ____________

