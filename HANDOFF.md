# HANDOFF — track index and resume guide

Thin index; the living state lives in the STATUS tables of the plan docs. Verify health first:

```
uv run pytest                                     # ~1660 passed
ctest --test-dir cpp/build -C Release             # 6/6
ctest --test-dir apps/studio/build -C Release     # 8/8
uv run python -m protospec_gen.emit --check
uv run python tools/lift_registry.py check
```

## ProtoSpec Studio (the MuJoCo Studio fork)

One source for the editor: `apps/studio/editor/` + `apps/studio/test/`. Two hosts
consume it — no copies:

- **Standalone** `apps/studio/` — thin SDL2 + classic-renderer host. Builds the
  editor and the 8 test batteries; this is the CI / ASan surface.
- **Fork** `mujoco-studio` (branch `studio`) — the real MuJoCo Studio app (Filament).
  Compiles the editor + ProtoSpec core live from this repo via `PROTOSPEC_ROOT`
  (the live repo, not a snapshot). It keeps only host glue: the `platform/ux` plugin
  shims and `host/shell.{cc,h}` (the SE4 File/Edit menu + toolbar + Play/Stop bridge,
  which bind to Studio-only plugin types the standalone host lacks). The error-chip
  is rendered host-side; the editor cluster exposes the `DiagnosticErrorCount`
  predicate it reads.

Build dir: `C:\Users\jonat\Documents\mujoco-studio\build_ps`
(`PROTOSPEC_ROOT=C:/Users/jonat/Documents/protospec`). Exact commands, run, and
self-screenshot in `docs/studio_build.md`. On the live core, humanoid.xml takes the
fast native compile path (`path=native`); verify with
`cpp/build/Release/ps_compile.exe <model.xml>` -> `"path_taken": "native"`.

## SDK save surface (cpp/sdk/protospec/save.h)

First-class model persistence on the SDK (previously only the studio could save):

- `sdk::Save(const Model&, path)` — `WriteMjcf` -> file on disk.
- `sdk::SaveAs(const Model&, path, std::vector<InMemoryAsset>*)` — Save plus
  externalize in-memory assets (mesh/texture/hfield bytes) next to the target,
  honoring `<compiler meshdir>`, then clear the list. `nullptr` == Save. The
  model is `const` (deviates from the owner's `Model&` sketch on purpose: saving
  never mutates the tree, matching CDR-14 / rule 3; the asset list is the only
  mutable output).
- Shared asset externalization lives ONCE here: `ModelAssetDir` + `WriteAssetFile`
  (primitives) and `ExternalizeAssets` (convenience). Built as `protospec_sdk_io`
  (STATIC, links protospec_io); the pure-tree `sdk.h` stays MuJoCo/disk-free.
- Tests in `test/test_sdk.cc`: `TestSaveRoundtrip` (build -> Save -> ParseMjcf ->
  deep-equal `operator==` + byte-fixpoint) and `TestSaveAsExternalizesAssets`
  (in-memory bytes -> file under meshdir, byte-exact, list drained, reload parses
  + mesh resolves). `protospec_sdk_tests` needs `/bigobj` now.
- **FOLLOW-UP (editor agent, apps/studio):** point `ExternalizeVfsAssets`
  (apps/studio/editor/asset_import.cc) at the shared impl — loop
  `sdk::WriteAssetFile(sdk::ModelAssetDir(*ctx.tree, xml), a.name, a.bytes.data(),
  a.bytes.size())` over `ctx.vfs_assets` (zero-copy; `bridge::VfsAsset` has the
  same `.name`/`.bytes` shape), then keep the studio-only `vfs_assets.clear()` +
  `base_dir` repoint. Link `protospec_sdk_io`. Not done here (studio is off-limits
  to this track).

## ASan (tools/run_asan.ps1 + .sh; `pytest -m asan`)

Separate instrumented tree in `cpp/build_asan` (never touches `cpp/build`):
`cmake -DCMAKE_CXX_FLAGS="/fsanitize=address /Zi" -DCMAKE_TRY_COMPILE_CONFIGURATION=Release`
(the last flag keeps `/RTC1`, incompatible with ASan, out of CMake's Debug-default
compiler probe). The clang_rt ASan runtime DLL is copied beside the test exes
from the CMAKE_LINKER toolset dir. Run: `pwsh tools/run_asan.ps1`
(or `PROTOSPEC_RUN_ASAN=1 pytest -m asan`). Result today: **CLEAN — 0 reports**
across the 4 instrumented suites (676 checks) + 60 diverse corpus files under
`ps_roundtrip`.

- **Scope = MuJoCo-free only** (object model, io, core, validate, SDK+save).
  The MuJoCo-linked suites (bridge, native/compile, ps_native_diff) **cannot** be
  built under MSVC `/fsanitize=address`: any TU including `<mujoco/mujoco.h>`
  pulls in vendored `mujoco/mjsan.h`, whose `ADDRESS_SANITIZER` path (a) uses
  GCC/Clang `__attribute__`/`asm` cl.exe rejects, and (b) calls
  `mj__markStack`/`mj__freeStack`, which exist only in a libmujoco built under
  ASan — the vendored `mujoco.dll` is a normal build. Instrumenting them needs an
  ASan-built MuJoCo, or the ClangCL VS toolset (parses mjsan.h; not installed on
  this box). NOT a ProtoSpec bug and NOT in our files — routed, not fixed.

## Standing rules (owner)

1. **Never integrate anything into UnrealRoboticsLab.** This repo stays standalone.
2. **No mjSpec/mjs_*/mjC* in the native path or the editor.** Reuse = public API + lifted code
   (registered in `snapshots/lifted_code.json`, drift-gated).
3. **Compile is pure; Binding is a snapshot object** (no mutation counter — rejected; see
   `cpp/bridge/binding.h`).
4. **Zero-divergence discipline**: native claims require bit-identity vs the XML path
   (`tests/native_ratchet.json` is the floor); descopes need written reasons (gates in
   `cpp/compile/native.cc`).
5. **Minimal representation**: typed semantics preserved (actuator/equality spellings);
   redundant encodings canonicalized at read (orientation → quat decided; inventory in
   `docs/plan_canonicalization.md`). Joint angle fields stay authored-form (per-consumer
   conversion — the proven boundary case).
6. Studio: plugin-first (Studio's interfaces verbatim + `ps_plugin_ext.h`), build-as-you-go,
   windowless tests per milestone, DR-S1 one-way dataflow, DR-S6 delta rule.
7. Execution: code by Opus-class agents; sequential waves on shared files; harness authored by
   a different agent than the code it tests; commit per green wave; push to the fork
   (`git push fork master:protospec master:main`).

## PAUSED 2026-07-13 (resume point)

Both repos clean and pushed. protospec @ efd5b7a5 (fork branches `protospec` + `main`);
mujoco-studio checkout at C:\Users\jonat\Documents\mujoco-studio, branch `studio` @ 5dba7038
(fork). Studio exe: `C:\tmp\studio_spike\build_ps\bin\mujoco_studio.exe --model_file=...`
(build dir pinned to worktree C:\tmp\ps_snapshot_b @ a1ab7eb4 — REPOINT to a fresh snapshot
after resuming native-compiler work, since the editor doesn't need it but consistency does).
The two ADOPTION GATES (owner): Gate 1 = native compiler 100% parity; Gate 2 = editor
certification signed. Nothing touches UnrealRoboticsLab until both pass.

| Track | State | Queue lives in |
|---|---|---|
| Core library (M1-M7) | complete | `docs/plan.md` STATUS |
| Canonicalization Waves A+B | complete (minimal repr landed) | `docs/plan_canonicalization.md` |
| Native compiler | NC0-NC5 + NC6 assets + NC6b attach slice + **NC7a long-tail (sleep, partial-size/nkey, full sensor family, tendon-armature, muscle/lengthrange, springdamper, light-texture, site/refsite/slidercrank transmission)** + **NC6c attach FULL-IMPORT (defaults merge + assets + referencing + keyframes + whole-model)** + **NC5 wave 6 interpolated FE + strain equality (trilinear/quadratic nodal mesh, FE stiffness/bending, mjEQ_FLEXSTRAIN)** + **NC7b plugins (extension/instance/config machinery + actuator/sensor plugins) + actuator/sensor delay-history**, **ratchet 339/387** | queue below |
| Studio editor SE0-SE4 + real-Studio migration | complete; running in real MuJoCo Studio | `docs/plan_studio_editor.md`, `docs/studio_ui_migration.md` |
| Editor certification | automated side DONE (7 batteries both trees, gaps G1-G9 closed/rescoped) | `docs/editor_certification.md` — **WAITING ON OWNER: 27-step manual walk + signature** |

## Native compiler remaining queue (Gate 1)

NC5 flex waves 1-4 (procedural + elasticity + direct), **wave 5 (gmsh)**, **wave 5b (mesh
file)**, **wave 6b (vert flex equality)**, the **reduced-dof slice of wave 6 (radial/2d)**, and
now **wave 6 full interpolated FE + strain equality** are DONE and on the ratchet.

1. **NC5 wave 6 — interpolated FE (trilinear/quadratic) + strain equality — LANDED (ratchet
   323 -> 335, +12 files, 0 divergences).** The nodal finite-element machinery is native.
   **Make side** (`ExpandFlexcompInto`): interpolated flexcomps attach all vertices to the parent
   and synthesize a separate nodal mesh of slider bodies (`name_gi_gj_gk`, trilinear mass=1 /
   quadratic Simpson weights, normalized to total mass), pinning nodes exclusively in empty cells
   via a reproduced `MarkEmptyCells`+`FlexComputeCellEmpty`, and set `<flex>` node/dof/cellcount;
   node local offsets + the cell_empty mask flow out-of-band through the FlexcompSink. **Compile
   side** (`FlexCompile`): the interpolated branch resolves nodebodyid, computes nodexpos +
   `FlexComputeUnrotatedNodePositions` (inverse grid rotation R0), per-cell/per-face FE stiffness
   (`ComputeLinearStiffness` 3D volume / `ComputeLinearStiffness2D` shell membrane +
   `ComputeWarpMode`/`ComputeWarpStiffness`), `EigendecomposeStiffness` for strain constraints
   (K_young=1e1, K_poisson=0.3), `ComputeInterpBending` for shell bending, and vert0/node0 in the
   local frame; fill emits flex_node/node0/nodebodyid/interp/cellnum. **Strain equality**
   (mjEQ_FLEXSTRAIN): one constraint per FE cell (volume, data=[ci,cj,ck]) / boundary face (shell,
   data=[fe,-1,-1]); `has_strain_eq` is scanned from the compiled equalities before FlexCompile.
   **Mesh fidelity fix**: the flexcomp mesh path now applies `ProcessVertices(remove_repeated)` --
   position-dedup + face remap -- matching `LoadFromResource(...,true)` (fixed bunny_with_uv;
   textured_torus_flex unaffected). **11 registry lifts** (ComputeLinearStiffness/2D, WarpMode,
   WarpStiffness, EigendecomposeStiffness, ComputeInterpBending, quadrature/phi/dphi,
   ComputeCellEmpty, ComputeUnrotatedNodePositions, MarkEmptyCells). **Landed**: trilinear,
   quadratic, strain, hollow_vs_solid (+sleep/), bunny, bunny_multicell, bunny_shell,
   bunny_quadratic, bunny_with_uv, gripper_trilinear, sphere_trilinear. **New precise sub-gates**:
   `flexcomp.document_order` (a flexcomp inside an earlier child body must get a lower flex id than
   a later flexcomp sibling, but the collector expands a body's own flexcomps first -> reorders;
   the ONE remaining descope: `test/testdata/flex.xml`), `flexcomp.constraint_elasticity` (reader
   hard error: equality + young>0 needs elastic2d==bend), `flexcomp.interpolated_pingrid` (non-grid
   pin grid uses an adjusted count check).
2. **NC6 assets — DONE** (ratchet 239 -> 259). File textures (mjCTexture single-file 2D/cube:
   PNG via lifted lodepng `DecodePNG`, KTX raw, custom binary; gridsize/gridlayout composition;
   hflip/vflip; colorspace=AUTO from the PNG sRGB chunk; texturedir+strippath; file-stem naming;
   `tex_pathadr`), file hfields (PNG grey row-reversed + custom binary via meshdir), skins
   (`mjCSkin::Compile`/`LoadSKN`: inline `<bone>` + `.skn`, weight normalization, bone/material id
   resolution, `skin_*` arrays), and mesh-fit geoms (`mjCMesh::FitGeom`: inertia-box/aabb size +
   fitscale, `geom_dataid` retains the source mesh). Still gated: texture separate cube-faces +
   authored content_type; light.texture; `hfield`/`skin` authored content_type.
3. **NC6c attach FULL-IMPORT — LANDED (ratchet 315 -> 323, +8 attach files, 0 divergences).** The
   full `mjs_attach` import is now reproduced into the synth model
   (`ExpandAttaches`/`ImportChildSections`/`ImportChildKeyframes` in `cpp/compile/native.cc`):
   child `<default>` tree grafted as a **prefix-named subclass under the parent root** (a class-free
   child element resolves `prefix -> root` == child-main -> parent-main; a named child class
   `prefix_cls -> prefix -> root`; graft bodies get an injected `childclass=prefix` and class-free
   world-level / referencing elements a `dclass=prefix` -- matching `mjCBase::NameSpace`'s always-true
   cross-model guards); child **assets** (meshes/textures/materials/hfields/skins) appended prefixed;
   **referencing sections** (tendons/equalities/actuators/sensors/contacts) appended prefixed+tagged in
   `operator+=` order; **whole-model attach** (empty body) fabricates an identity frame over the whole
   child worldbody. Uniform cross-model NameSpace prefixes every name + typed ref (incl. `dclass`/
   `childclass` -- both are `opt<Ref<Default>>`, so the existing `RefPrefixer` reaches them).
   **Serial pitfalls fixed** (the predecessor's flag): `Clone` regenerates serials, so (a) parent
   unnamed elements get their ORIGINAL serials copied back in lockstep (their `_ps:<family>:<serial>`
   auto-name must match the XML oracle) and (b) imported child unnamed elements are given an authored
   EMPTY name (`mj_loadXML` pulls the child raw -> unnamed, never `_ps`-auto-named).
   **Keyframes:** child keys import as prefixed top-level keys placed at the graft's qpos offset
   (joints before it, depth-first, counting `<replicate>` as `count x subtree-nq`) with the surrounding
   dofs NaN-gapped -> `qpos0` in `FillKeyframes` (mirrors `ResolveKeyframes`/`RestoreState`; a NaN slot
   defaults to qpos0). **Lands** ten_armature_0/1_compare, humanoid100, contact_subtree, 2humanoid100
   x3 (benchmark/island/sleep-perf), engine-testdata/hammock. **Still gated (honest sub-gates):**
   `attach.keyframe_macro_offset` (a flexcomp/composite before the attach makes the qpos offset
   uncomputable -> model/hammock), `attach.replicate_referencing` (an attach nested in `<replicate>`
   whose child carries referencing elements or keyframes needs per-clone cloning -> 22/100_humanoids,
   sleep/100_humanoids), `attach.keyframe_state` (child key with qvel/act/ctrl/mpos/mquat -- a separate
   dof/act/mocap offset), and `attach.child_multidefault`/`child_deformable`/`child_custom`/
   `child_plugin`/`child_nested_model`.

3b. **NC6b attach/`<model>` — self-contained-child slice (ratchet 259 -> 262), superseded by NC6c
   above.** ProtoSpec stores `<model file=...>` as a `ModelAsset` (file ref only) and
   `<attach model=.. body=.. prefix=..>` as an `Attach` element; the reader passes both unexpanded,
   so leg B relies on `mj_loadXML` to recursively parse the child and run `mjs_attach`.
   **Landed** (`ExpandAttaches` in `cpp/compile/native.cc`, "attach"/"model" admitted in
   `native_supported.h`): at native compile, parse the child model (`io::ParseMjcfFile`; child path
   = parent-model dir + `file`, `..` normalized; a nameless `<model>` is indexed by its child
   `<mujoco model=..>` name per reader :3624), deep-clone the named child body, prefix-namespace
   every element NAME (`mjCBase::NameSpace`, name=prefix+name — cross-model, so real `mjs_attach`
   with empty suffix), splice where the `<attach>` stood (the reader's fabricated identity frame
   :3935 folds away). Orientation is quat-resolved in the child's own compiler context at parse, so
   the graft is context-safe. Lands **parent.xml, parent_model.xml, many_dependencies.xml** (the
   last a lone unreferenced `<model>` asset, now admitted+ignored), all bit-identical.
   **NOT reproduced (routes to XML fallback with a written `attach.*` reason)** — the full
   `mjs_attach` machinery: (a) child asset deepcopy-import (prefixed meshes/textures/materials/
   hfields/skins into the parent asset space) → `attach.child_assets`; (b) child default-class
   NameSpace-merge (`mjCFrame::operator+= :2969-2974`, `*model += *subdef`) → `attach.child_defaults`;
   (c) keyframe `StoreKeyframes`/`ExpandAllKeyframes` resize+prefix → `attach.child_keyframes`;
   (d) referencing-element copy with the drop rule and bit-exact `operator+=` append ordering
   (`user_model.cc:452-503` copies meshes,skins,hfields,textures,materials, then flexes,pairs,
   excludes,tendons,equalities,actuators,sensors,tuples — each dropped if its refs don't resolve
   into the graft) → `attach.referencing_elements`; plus `attach.whole_model` (empty `body`:
   fabricated world frame over every top body, `user_api.cc:397-412`), `attach.subtree_refs` (graft
   subtree carries any cross-ref — would dangle without an imported asset/class), `attach.child_
   deformable`/`child_custom`/`child_nested_model`. The **humanoid/hammock family** (humanoid100,
   2humanoid100, 22/100_humanoids, hammock, sleep perf — ~13 files) all gate on `attach.child_assets`
   (humanoid.xml carries 5 assets + 21 defaults + actuator/tendon/keyframe): they need the FULL
   import wave, which is the genuinely large, high-FP-divergence remainder (a single mismatched ref
   or one-off in the append order or keyframe resize diverges the whole model). Build on the landed
   `ExpandAttaches` seam (it already produces a mutable synth `Model` from `Clone(m)`); the next
   wave imports child assets/defaults/keyframes/referencing-sections into that synth with the exact
   `operator+=` order and prefix rules, then reuses the existing pipeline. Note: `Clone` regenerates
   serials, so a full-import wave handling models with UNNAMED referenceable elements must bake the
   original-serial auto-name before cloning (the `BakeName` precedent in `build.cc`).
6. **NC7a long tail — LANDED (262 -> 315), 0 divergences, all baselines green.** Wave by wave:
   per-body `sleep` (tree_sleep_policy incl. world-body demotion); geom/site partial-size
   eager-copy tail (`EagerSizeArray`, per-slot highest-priority-wins over the mjCGeom {0,0,0} /
   mjCSite {0.005} constructor base) + `<size nkey>` keyframe pre-allocation (pad to max(authored,
   nkey) with empty-name default keys); the full sensor family — rangefinder (dataspec bitmask in
   `sensor_intprm[0]`, `RaydataSize`), camprojection (site obj + camera ref, dim 2), insidesite
   (typed obj + site ref), geom-distance distance/normal/fromto (geom|body obj/ref, AXIS datatype
   for normal), and contact (dataspec/reduce/num in intprm[0..2], `dim = num*CondataSize`);
   tendon-armature `body_simple`/`nC` demotion (site/cylinder/sphere wrap bodies, incl. world);
   muscle actuators via the **public `mj_setLengthRange`** post-build forward-sim pass (runs for
   every model after `mj_setConst`; default LRMODE_MUSCLE + useexisting make it a no-op elsewhere;
   `<compiler><lengthrange>` LRopt overrides); joint `springdamper` (AutoSpringDamper: jnt_stiffness
   / dof_damping from `dof_invweight0` after mj_setConst); light image-texture `light_texid`; and
   site/refsite/slidercrank transmission (`trnid[0/1]`/gear/cranklength + `nJmom` via a reproduced
   `mj_mergeChain`, actuator_length0 from mj_setConst's mj_transmission). **Fixed a real compiler
   bug**: a dynamic hfield (`nrow`/`ncol` only, no file/data) now allocates zeros like the reader
   (was `native.build_failed`). New precise sub-gates: `rangefinder.camera` (dim scales with camera
   resolution), `sensor.delay` (nsample history ring), `material.class_layers` (child-list class
   inheritance unmodeled by Effective — also masks a latent PBR material bug). **Final remaining
   histogram (61 fallback files, reason -> #files, a file may carry several):** config/extension/
   plugin 17, geom.sdf 14, flexcomp.interpolated 13, instance 12, geom.plugin/mesh.plugin 11,
   attach.child_assets 10, actuator.cross_spelling_default 5, flexcomp.equality_kind/mesh.builtin 4,
   composite/replicate.referencing_element 3, attach.whole_model 2, then singletons: tactile,
   replicate.childclass, dcmotor, rangefinder.camera, actuator.delay, compiler.discardvisual,
   freejoint.align, compiler.alignfree, material.class_layers.
   **Still gated (tractable, next):** plugin models (config/extension/plugin/instance +
   geom.plugin/mesh.plugin/composite/tactile — need native plugin-registry resolution for
   actdim/sensor dims + mjModel plugin arrays), geom.sdf (marching cubes + octree), mesh.builtin
   (procedural sphere/cone/wedge/supertorus... generators + full mesh pipeline), dcmotor (stateful
   mjs_setToDCMotor + na/actdim), actuator.delay (nsample history ring), compiler.discardvisual
   (delete visual geoms + id compaction pre-id-assignment), alignfree/freejoint.align (free-joint
   frame alignment, user_objects.cc:2793), replicate.childclass/referencing_element, and
   actuator.cross_spelling_default (the shared per-class actuator default accumulated across every
   spelling — assessed: actuator_group_disable + python model.xml would flip, robot_arm has a real
   `general`->`velocity` ctrllimited leak; deferred as the gain/bias/dyn interplay across the
   shared default is high-divergence-risk to reproduce bit-exact). **Excluded (separate waves):**
   attach full-import (attach.* gates), flexcomp interpolated FE + strain (flexcomp.interpolated/
   equality_kind -- SINCE LANDED in NC5 wave 6; the histogram's "flexcomp.interpolated 13" line is
   now retired, leaving only the single-file flexcomp.document_order descope).
7. **Plugins** — DONE (XML route). First-party engine plugins registered at harness startup
   (`cpp/harness/plugin_registry.{h,cc}`, shared by mj_model_diff/ps_native_diff/ps_compile;
   default DLL dir beside mujoco.dll, `--plugin-dir`/`PROTOSPEC_PLUGIN_DIR` override). The 17
   plugin corpus models now round-trip byte-identical: **differential 376/387**, floor guarded
   by `test_xml_parity_floor`. Remaining 11 skips are non-loadable fixtures (10 malformed + 1
   engine-fail). Native plugin support (flex.plugin etc.) is still gated → NC7+.
8. **NC7b native plugins + delay — LANDED (ratchet 335 → 339, 0 divergences).** The
   `<extension><plugin><instance><config>` machinery + actuator/sensor plugin refs compile
   native (`CollectPlugins`/`FillPlugins`/`PackPluginAttr` in `cpp/compile/build.cc`): instances
   resolve to registry slots via the PUBLIC `mjp_getPlugin`/`mjp_getPluginAtSlot` (the harness
   loads the first-party plugin DLLs process-wide), config packs into `plugin_attr` in the
   plugin's declared attribute order (mjCPlugin::Compile), and `nplugin`/`npluginattr`/`plugin[]`/
   `plugin_attradr`/`name_pluginadr` + `actuator_plugin`/`sensor_plugin` are filled; `nstate`
   (`npluginstate`) and sensor-plugin `dim`/`needstage` (`nsensordata`) are queried from the
   plugin callbacks AFTER allocation (mirroring CopyPlugins — plugin sensor dim is 0 at the size
   census, so `sensor_adr`/`nsensordata` are recomputed post-callback). Explicit `<instance>`s
   come first (extension order; unreferenced ones dropped = RemovePlugins), implicit inline-config
   instances follow in element order. The actuator/sensor **delay+history** buffer is also native
   (`nhistory` sizing + `actuator_history`/`historyadr`/`delay` + `sensor_history`/`historyadr`/
   `delay`/`interval`, one shared cursor across actuators-then-sensors); the former
   `actuator.delay`/`sensor.delay` gates are removed (the latter also closed a latent gap — basic
   sensors with `nsample` had no delay sub-gate and would have diverged). **Lands** pid (4 explicit
   pid-actuator instances, actdim/na), touch_grid + touch_grid_test (implicit sensor-plugin
   instance, dim from nsensordata), delay. **Descoped (precise gates, honest reasons):**
   `geom.sdf`/`mesh.plugin` — SDF meshes need marching-cubes (`MC::marching_cube`) + octree at
   compile time (mjCMesh::LoadSDF, user_mesh.cc:356) — the genuinely large remaining lift (14
   files: torus/gear/nutbolt/bowl/cow/mesh/primitives/mug/cloth_sdf/ray sdf...); `composite`
   passive plugins (elasticity cable/belt/coil are `<composite>`-gated, a separate family); `tactile`
   (dim = 3·mesh.nvert); `mesh.builtin` (procedural generators); `rangefinder.camera` (rfcamera
   also needs mesh.builtin). No verbatim lifts (packing/CopyPlugins reproduced in ProtoSpec style),
   so no `lifted_code.json` delta. **Next-largest bucket = the SDF pipeline** (marching cubes +
   octree + the SDF-mesh geom AABB/BVH), which would unlock ~14 files.
9. **NC7c FINAL wave — LANDED (ratchet 339 -> 354, +14 files, 0 divergences), all baselines green
   (pytest, cpp 6/6, differential, registry 141 entries drift-clean, emit --check).** Five buckets
   burned down:
   - **SDF plugin pipeline (+8: primitives/torus/bowl/gear/nutbolt/mesh + ray/sdf + cloth_sdf).**
     `mjCMesh::LoadSDF` reproduced (`LoadSdfMesh`, build.cc): resolve the plugin instance, call the
     public `sdf_attribute`/`sdf_aabb`/`sdf_staticdistance` callbacks, sample a regular grid
     (`floor(300/total*aabb)+1` per axis), marching-cube it via the **vendored MarchingCubeCpp**
     single header (`cpp/compile/lifted/marching_cube.{h,cc}`, `MC_IMPLEM_ENABLE` instantiated once),
     rescale the verts back to geom-local, and feed the generated vert/normal/face to the ordinary
     mesh pipeline with a new `MeshInput.needreorient=false` (nulls the CoM/principal-axis transform,
     `mjCMesh::Process` :1511). geom `type="sdf"` binds bounds/inertia exactly like a mesh geom
     (upstream treats `mjGEOM_SDF` as `mjGEOM_MESH`); `CollectMeshHullRefs` counts sdf collision
     geoms for the hull. Plugin arrays extended to geoms: `CollectPlugins` walks the body tree for
     geom plugin refs (+ mesh refs) into the `referenced` set (RemovePlugins) and binds each geom;
     `FillPlugins` fills `geom_plugin`. **Precise remaining gate (`geom.sdf`, 3 files cow/mug/shapes
     + solver/testdata model):** an sdf geom over a PLAIN (non-plugin) mesh is a mesh-DERIVED SDF
     that needs the octree / `mjCOctree::ComputeSdfCoeffs` (`needsdf=true`), which this wave does not
     build (plugin-SDF sets `needsdf=false`, so no octree) -- gated by resolving the geom's mesh and
     checking it is a plugin (SDF) mesh.
   - **Builtin procedural meshes (+2: makemesh/spheremesh).** The `mjCMesh::Make{Sphere,Hemisphere,
     Supersphere,Supertorus,Wedge,Rect,Cone}` generators + the `mjs_makeMesh` dispatch/validation
     lifted verbatim into `cpp/compile/lifted/builtin_mesh.{h,cc}` (with the file-local
     Fovea/LinSpace/BinEdges/SphericalToCartesian/TangentFrame/aux_c/aux_s helpers); wired into
     MeshCompile (generate vert/normal/face -> pipeline; plate sets `mjMESH_INERTIA_SHELL`).
   - **Free-joint alignment (+2: dzhanibekov/freejoint).** `mjCBody::Compile` phase 1/2 (user_objects.cc
     :2795-2897) in `BodyCollector`: when exactly one free joint, no child bodies, and `align="true"`
     (or `align="auto"` + `<compiler alignfree>`), fold the inertial frame into the body frame
     (`mjuu_frameaccum`), null it, and inverse-transform child geoms (phase 1) then sites/cameras/
     lights (phase 2, lights rotate their dir), before xpos0/BVH.
   - **Camera-target rangefinder (+1: rfcamera).** dim = `RaydataSize(dataspec) * resolution[0]*
     resolution[1]` (one ray per pixel, `mjs_sensorDim` :1758); `CameraResolutionByName` resolves the
     target camera's effective resolution.
   - **Tactile sensor (+1: tactile).** obj = mesh, ref = geom (`mjOBJ_MESH`/`mjOBJ_GEOM`),
     dim = `3 * mesh.nvert`; `SensorMaps` gains a mesh id + nvert map; `native_supported.h` admits the
     `tactile` family. (tactile.xml also exercised the new SDF gear plugin mesh + builtin wedge/plate
     meshes, all native now.)
   **Registry delta (+10 entries, 141 total, drift-clean):** `marching_cube` (vendored MC.h, whole
   file), `mesh_loadsdf` (`mjCMesh::LoadSDF`), and `builtin_make{sphere,hemisphere,supersphere,
   supertorus,wedge,rect,cone}` + `builtin_makemesh_dispatch`.
   **FINAL remaining histogram (23 fallback files after NC7c; reason -> #files):** `geom.sdf` 5
   (mesh-derived SDF octree: cow/mug/shapes + solver/testdata model), `actuator.cross_spelling_default`
   5, `attach.keyframe_macro_offset` 4 (hammock, one file x4 keys), `attach.replicate_referencing` 3
   (100/22 humanoids, sleep/100), `composite` 3 (belt/cable/coil), `replicate.referencing_element` 3
   (newton_cradle/tendon/lengthrange), then singletons `replicate.childclass` 1 (inertia),
   `dcmotor` 1, `flexcomp.document_order` 1 (flex), `compiler.discardvisual` 1, `material.class_layers`
   1 (pbr). **Per-bucket reasons for what remains (each a deliberate, documented sub-gate):**
   (a) **mesh-derived `geom.sdf`** needs the ~750-line mjC-entangled `mjCOctree`/`ComputeSdfCoeffs`
   octree (only the plugin-SDF slice, which skips the octree, was tractable). (b)
   **`cross_spelling_default`** = MuJoCo shares ONE actuator mjCDef per class across every spelling,
   so a `<general ctrllimited>` in a parent leaks into a child `<velocity>`; ProtoSpec keeps
   per-spelling partials and `Effective` merges only the matching one -- reproducing the shared merge
   (type/gain/bias/dyn set by the last spelling, common fields last-write-wins across the class chain)
   is high-divergence-risk, deferred (robot_arm has a real general->velocity ctrllimited leak).
   (c) **`composite` cable** needs body-level passive plugins (the elasticity plugin attaches to
   every synthesized body), but ProtoSpec `Body` has no plugin field and the native binder only does
   actuator/sensor plugins -- new infrastructure with no precedent. (d) **`attach.*`** are attach-in-
   replicate referencing/keyframe cloning and flexcomp-before-attach qpos offsets (existing NC6c
   gates). (e) **`replicate.referencing_element`/`childclass`** need per-clone referencing-element
   cloning / a ParentMap that reaches arena clones (the clones live outside the source tree, so
   `Effective` can't resolve an unclassed clone descendant's childclass). (f) **`dcmotor`** is a
   ~154-line stateful `mjs_setToDCMotor` (na/actdim + electrical/thermal/lugre models) for 1 file.
   (g) **`flexcomp.document_order`** is the single flex-id-ordering descope from NC5. (h)
   **`compiler.discardvisual`** needs the visual-geom delete + id compaction + inertia-baking pre-pass
   (CDR-8) for 1 file. (i) **`material.class_layers`** (inherit the `<layer>` child-list from a class
   default) masks a latent PBR material bug per NC7a, so it is left gated rather than risk a divergence.

## Editor remaining queue (Gate 2)

1. OWNER: walk the 27-step manual script (docs/editor_certification.md §2) against the studio
   build; report pass/fail; sign the certification statement at the tested commit.
2. Small feature queued from G8 re-scope: diagnostics rows carry SourceLoc/serial ->
   click-to-select navigation.
3. Owner-waivable: G5 layout persistence (manual step M27 covers it).
4. Known punts if wanted later: OS file dialogs for editor Save As, multi-select ops,
   keyframe timeline, perturb-in-Play firewall polish.
5. Editor source of truth: protospec repo apps/studio == mujoco-studio protospec/ (7 identical
   batteries; keep them in lockstep — sync direction protospec -> mujoco-studio).

## Upstream MuJoCo bugs (owner files these personally — do not fix here)

- Cylinder actuator `bias` stack overflow: `xml_native_reader.cc:2557` (`double bias`) +
  `:2561` (`ReadAttr(..., 3, &bias, ...)`); `mjs_setToCylinder` takes scalar (`mujoco.h:1737`).
- Replicate `prefix` read nowhere: `:3865` hardcodes `""`; corpus authors it
  (`collision_primitive/spheres_cylinders.xml`).
- Stale keyframe: `test/engine/testdata/solver/humanoid.xml` qvel has 25 values, nv is 27.
- `mjspec.h` comment bugs: `:455` ("errorsx"), `:644-645` (exclude bodynames say "geom").
- UNVERIFIED: plugin-ordering check reachability (`xml_native_reader.cc:3148-3151`) — trace
  `Parse()` section order before filing.
- Studio build breaks under the Visual Studio generator: `cmake/third_party_deps/filament.cmake`'s
  `/FI cstring` CXX workaround leaks onto Filament's generated `.c` resource files
  (C1189/STL1003 in `fxaa.c`). Ninja+MSVC is unaffected (CXX flags stay off C files). Windows
  build rule for Studio: `vcvars64` + `-G Ninja`; never the VS generator; clang-cl is rejected
  by this Filament pin ("Building with Clang on Windows is no longer supported").
