# HANDOFF — track index and resume guide

Thin index; the living state lives in the STATUS tables of the plan docs. Verify health first:

```
uv run pytest                                     # ~1660 passed
ctest --test-dir cpp/build -C Release             # 6/6
ctest --test-dir apps/studio/build -C Release     # 4/4
uv run python -m protospec_gen.emit --check
uv run python tools/lift_registry.py check
```

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

## Tracks

| Track | State | Queue lives in |
|---|---|---|
| Core library (M1-M7) | complete | `docs/plan.md` STATUS |
| Native compiler | NC0-NC4 done, ratchet 201/387 | `docs/plan.md` STATUS + NC5/NC6/NC7 queue below |
| Studio editor | SE0-SE3 done, handed to owner | `docs/plan_studio_editor.md` STATUS |
| Canonicalization | audit in flight | `docs/plan_canonicalization.md` (inventory + wave) |
| Studio real-UI migration | spike in flight | `docs/studio_ui_migration.md` |

## Native compiler remaining queue (unchanged priorities)

1. **NC5 flex/flexcomp** — reconnaissance complete with upstream line ranges (49-file ceiling;
   waves: flex-direct foundation → grid/box edge-only (~15 files) → linear elasticity → mesh/
   direct → gmsh parser → interpolated FE). See the NC5 wave plan preserved in git history
   (commit message trail) and `docs/plan_native_compiler.md`.
2. Un-gate `default.duplicate_class` (SDK multi-block fix landed).
3. **NC6** — attach/`<model>` native expansion (clone-arena pattern), PNG file textures
   (lodepng wired), file hfields, skins, mesh-fit.
4. **NC7 long tail** — muscle (`mj_setLengthRange` is public/post-build), dcmotor, site/
   refsite/slidercrank transmissions (`mj_mergeChain` lift), remaining sensors, discardvisual,
   alignfree, per-body sleep, partial-size-default eager-copy semantic, `Expand()`.
5. **Plugins** — register first-party plugin libs in the harness to unlock the 28 skips.

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
