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
| Native compiler | NC0-NC5 waves 1-5b + 6b + 6-partial done, **ratchet 239/387** | queue below |
| Studio editor SE0-SE4 + real-Studio migration | complete; running in real MuJoCo Studio | `docs/plan_studio_editor.md`, `docs/studio_ui_migration.md` |
| Editor certification | automated side DONE (7 batteries both trees, gaps G1-G9 closed/rescoped) | `docs/editor_certification.md` — **WAITING ON OWNER: 27-step manual walk + signature** |

## Native compiler remaining queue (Gate 1)

NC5 flex waves 1-4 (procedural + elasticity + direct), **wave 5 (gmsh)**, **wave 5b (mesh
file)**, **wave 6b (vert flex equality)**, and the **reduced-dof slice of wave 6 (radial/2d)**
are DONE and on the ratchet (239). The one remaining flex descope:

1. **NC5 wave 6 — full interpolated FE (trilinear/quadratic) + strain equality** (~12 files:
   bunny*, gripper_trilinear, sphere_trilinear, trilinear, quadratic, bunny_quadratic, strain,
   hollow_vs_solid). This is the nodal finite-element machinery: lift ComputeLinearStiffness/2D,
   EigendecomposeStiffness, ComputeWarpStiffness, ComputeInterpBending (user_mesh.cc:3826-4500),
   the node-body generation in Make (:631+, ComputeUnrotatedNodePositions/MarkEmptyCells), and
   the interpolated branch of FlexCompile (nodebody/nodexpos/per-cell stiffness assembly). The
   `strain` flex equality (mjEQ_FLEXSTRAIN, one constraint per FE cell) is coupled to this wave —
   it needs cellcount/cell_empty from the interpolated path (edge + vert equalities already land).
   Gated as `flexcomp.interpolated` (trilinear/quadratic only now) and `flexcomp.equality_kind`
   (strain only now). Descoped for size + FP-divergence risk; a self-contained NC5 wave 6.
2. **NC6** — attach/`<model>` native expansion (clone-arena pattern), PNG file textures
   (lodepng wired), file hfields, skins, mesh-fit.
6. **NC7 long tail** — muscle (`mj_setLengthRange` public/post-build), dcmotor, site/refsite/
   slidercrank transmissions (`mj_mergeChain` lift), remaining sensors, discardvisual,
   alignfree, per-body sleep, partial-size-default eager-copy, `Expand()`, flexcomp
   document-order interleaving (known limitation note in wave-2 report).
7. **Plugins** — DONE (XML route). First-party engine plugins registered at harness startup
   (`cpp/harness/plugin_registry.{h,cc}`, shared by mj_model_diff/ps_native_diff/ps_compile;
   default DLL dir beside mujoco.dll, `--plugin-dir`/`PROTOSPEC_PLUGIN_DIR` override). The 17
   plugin corpus models now round-trip byte-identical: **differential 376/387**, floor guarded
   by `test_xml_parity_floor`. Remaining 11 skips are non-loadable fixtures (10 malformed + 1
   engine-fail). Native plugin support (flex.plugin etc.) is still gated → NC7+.

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
