# HANDOFF — how to continue this project

State as of 2026-07-08 (~mid-morning), last commit on this branch. Everything below is green and
pushed; there is NO uncommitted work anywhere. This file is the strict continuation guide; the
design authority is docs/plan.md (main plan, with living STATUS table), docs/plan_native_compiler.md
(native-compiler design, reuse ledger), docs/plan_native_compiler_impl.md (implementation
contracts; its mutation-counter sections are superseded — see the SUPERSEDED paragraph).

## Current state (verify before doing anything)

```
uv sync
uv run pytest                                  # expect ~1658 passed / ~84 skipped
cmake -S cpp -B cpp/build -G "Visual Studio 17 2022" && cmake --build cpp/build --config Release
ctest --test-dir cpp/build -C Release          # expect 6/6
uv run python -m protospec_gen.emit --check    # generated tree in sync
uv run python tools/lift_registry.py check     # 74+ lifted functions match vendored MuJoCo
```

- MJCF IO (XML path): all 142 element types; differential harness 359/387 corpus files identical.
- Native compiler (raw ProtoSpec -> mjModel, zero mjSpec): **ratchet 201/387 bit-identical**
  (tests/native_ratchet.json — the floor; regressions fail CI).
- Python: TRYME.md at repo root is the verified quickstart.
- MuJoCo dependency: the vendored tree in the UE plugin at
  `<plugin>/third_party/MuJoCo/src`, pinned 67a1ea6d (3.10.0), prebuilt libs in its build/.
  CMake cache var MUJOCO_ROOT points there.

## Standing rules (non-negotiable, from the owner)

1. **DO NOT integrate anything into UnrealRoboticsLab.** This repo stays standalone.
2. **No mjSpec/mjs_*/mjC* anywhere in the native path.** Reuse = public API + free-standing
   utils + LIFTED code (registered in snapshots/lifted_code.json via tools/lift_registry.py).
3. **Compile is a pure function** (never mutates the tree; purity gate test enforces).
   **Binding is a snapshot object** — no mutation counter (owner rejected it; see
   cpp/bridge/binding.h header comment for the contract).
4. **Zero-divergence discipline**: a corpus file is claimed native ONLY when bit-identical to
   the XML path via mj_model_diff; ratchet (tests/native_ratchet.json) updated incrementally;
   descoping needs a written reason (sub-feature gate in cpp/compile/native.cc).
5. Angles are form-preserved (stored as authored; MuJoCo converts at compile). Defaults are
   data, never applied outside Effective()/FlattenDefaults.
6. Execution policy: code by Opus-class agents; sequential waves when sharing cpp/compile
   files; test harnesses authored by a different agent than the code they test; commit after
   each green wave; push to this fork branch (`git push fork master:protospec` — remote `fork`
   = jonathanembleyriches/mujoco).

## WORK QUEUE (in order)

### 1. NC5 flex/flexcomp (in progress — reconnaissance done, implementation NOT started)
The full landing plan with upstream line ranges is proven and approved. Corpus ceiling: 49
clean flex files (+10 co-blocked by other gates, +10 oracle-failing fixtures excluded).
Waves, each independently diff-verified then ratcheted:
1. **flex-direct foundation** (no corpus movement; verified by test_native.cc batteries):
   lift mjCFlex::Compile (user_mesh.cc:4630-5191 + ComputeUnrotatedNodePositions,
   ComputeCellEmpty, CreateBVH, CreateShellPair, ResolveReferences, edge hashing, flap
   stencil) -> cpp/compile/lifted/flex_compile.{h,cc}; the 65-field CopyObjects flex fill
   (user_model.cc:3444-3600); nflex* sizing (:2182-2193); flex_vertedge adjacency
   (:2219-2224); flex BVH appended after body/mesh BVH; name tables (:2330).
2. **flexcomp grid/box/circle/disc, young=0 edge-only** (~15 files: flag, sleep/flex_*,
   hammock, pulley, strain, sphere_full, sphere_passive, skingroup...): lift
   mjCFlexcomp::Make driver (user_flexcomp.cc:148-846) + MakeGrid (:861-1073) +
   MakeSquare (:1073-1110) + MakeBox (:1110-1325) -> lifted/flexcomp_expand.{h,cc}; bodies
   generated through the existing pipeline using the pure-expansion arena pattern (NC4
   replicate precedent, cpp/compile/build.cc:1229+). Add sub-feature gates: flexcomp.gmsh,
   flexcomp.interpolated, flexcomp.mesh (temporarily), flex.plugin (permanent).
3. **Linear elasticity** (ComputeStiffness<Stencil2D/3D> + ComputeBending, user_objects.h
   templates): hollow_vs_solid, jelly, plate, press, basket, pancake, floppy (~7 files).
4. **mesh + direct flexcomp types** (MakeMesh user_flexcomp.cc:1325-1453 reusing the NC3
   mesh pipeline; direct inline points): bunny*, gripper, rigid_flex, flex_line_obj (~7).
5. **gmsh** (LoadGMSH41/LoadGMSH22 .msh ASCII+binary parser, user_flexcomp.cc:1453+): 12 files.
6. **interpolated FE** (trilinear/quadratic: ComputeLinearStiffness/2D,
   EigendecomposeStiffness, ComputeWarpStiffness, ComputeInterpBending,
   user_mesh.cc:3826-4500): 5 files.
Also: un-gate equality flex/flexvert/flexstrain as referenced flexes go native.

### 2. Un-gate `default.duplicate_class`
The SDK DefaultIndex multi-block fix landed (commit 76e9f7a) with mj_loadXML-verified
semantics. Remove the gate in cpp/compile/native.cc, verify the poncho files (and any other
multi-default-block models) go native bit-identical, ratchet.

### 3. NC6 — attach/model + file assets
- **attach + <model> asset native expansion**: same clone-arena pattern as replicate
  (cpp/compile/build.cc:1229+), prefix namespacing of ALL internal refs; ~41 files
  including 2humanoid100. NC4 descoped it for time, not difficulty.
- **PNG file textures** via lodepng (already wired in CMake, reserved): 18 files.
- **File hfields** (PNG + custom binary format per mjCHField::LoadFile): 4 files.
- **Skins** (file + inline; mjCSkin compile): 4 files.
- **mesh-fit geoms** (mjCMesh::FitGeom, primitive-referencing-mesh): 4 files.

### 4. NC7 — the long tail (each a small focused wave)
- muscle actuators (needs mj_setLengthRange — PUBLIC API, runs post-build; wire into the
  native finalize path), dcmotor lowering, site/refsite/slidercrank transmissions (lift
  mj_mergeChain — engine-internal), actuator delay/nhistory buffers.
- remaining sensors: rangefinder, contact (SensorContact), user, tactile, camprojection,
  insidesite, geom-distance sensors (variable dims + intprm).
- compiler flags: discardvisual (delete visual geoms + id compaction pre-id-assignment),
  alignfree (free-joint frame alignment, user_objects.cc:2793 — one corpus file:
  dzhanibekov.xml), per-body sleep policy, cross-spelling actuator class defaults,
  geom/site partial-size-default eager-copy semantic (see NC2 report notes: MuJoCo copies
  the class's full size array then overwrites the authored prefix — needs a compile-side
  special case), replicate.childclass + replicate.referencing_element gates,
  mesh.builtin (spheremesh/makemesh builtins).
- sdf plugin meshes: stays gated until plugins (below).
- M5 leftover: implement Expand() (macro materialization; MuJoCo-round-trip per DR-7).
- Debug config: add /bigobj for cpp/io/mjcf_reader.cc (Release unaffected).

### 5. Plugins (the asymptote — also unlocks the 28 harness skips)
The 28 skipped corpus files need first-party plugin registration (mujoco.elasticity.*,
mujoco.sdf.*, mujoco.pid, mujoco.sensor.touch_grid). Check the vendored build tree for the
built plugin libs; if present, register them in the harness (mj_loadPluginLibrary) so those
files become testable on BOTH paths, then decide native handling (plugin config passthrough
is mjs-free: mjModel plugin arrays are fillable directly).

### 6. Corpus expansion + upstream tracking
- The drift gates (schema surface, lifted code) run against the vendored pin. On any MuJoCo
  bump: re-run tools/bootstrap extractors, tools/lift_registry.py check, and the corpus —
  they will enumerate exactly what changed.
- Upstream bug follow-ups are the OWNER's (do not fix in MuJoCo): cylinder bias overflow
  (xml_native_reader.cc:2557/2561 + scalar mjs_setToCylinder mujoco.h:1737), stale keyframe
  (test/engine/testdata/solver/humanoid.xml qvel 25 vs nv 27), replicate prefix ignored
  (:3865 vs spheres_cylinders.xml), mjspec.h comment bugs (:455, :644-645), and the
  UNVERIFIED unreachable-plugin-ordering claim (:3148-3151 — verify Parse() section order
  before filing).

### 7. After parity: plugin adoption (SEPARATE, owner-gated)
UnrealRoboticsLab adoption of ProtoSpec (replacing Scripts/codegen) happens ONLY when the
owner asks. Design note: the UE mirror should consume the reflection tables
(cpp/generated/reflect.h) — see docs/plan.md milestone 8.

## Quick try-out
See TRYME.md (verified). Native compiler check on any corpus file:
`cpp/build/Release/ps_native_diff.exe <path-to-xml>` -> JSON verdict
(native_supported / identical / fallback_reasons).
