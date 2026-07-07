# ProtoSpec Native Compiler Plan: ProtoSpec tree → mjModel, directly

Companion to `docs/plan.md` (the main plan) and `docs/native_compiler_survey.md` (evidence base).
This document **replaces** the previous plan of the same name, which proposed decoding ProtoSpec
into a transient mjSpec via the `mjs_*` C API. **That approach is rejected.** The mandate: mjSpec
is the thing ProtoSpec exists to escape — one source of truth, one clean compiler, no shims. The
native compiler consumes the ProtoSpec tree directly and produces an `mjModel`.
mjSpec / `mjs_*` / `mjCModel` appear **nowhere** in the native path.

Reuse, under the mandate, means exactly three things:

1. **Public engine API** — `mju_*` math, model/data lifecycle (`mj_makeData`, `mj_setConst`,
   `mj_setLengthRange`, ...), the plugin registry (`mjp_*`), the resource/VFS layer (`mju_openResource`).
2. **Free-standing user-layer utilities** — the `mjuu_*` pool in `src/user/user_util.{h,cc}`
   (verified: zero `mjC*` references in the whole file).
3. **Lifted code** — we vendor and build MuJoCo from source
   (`third_party/MuJoCo/src`, pin 3.10.0-dev / `mjVERSION_HEADER 3010000`); copying its compiler
   algorithms into our tree with provenance attribution is legitimate and encouraged. Calling into
   the internal `mjC*` classes is not — they are mjSpec's guts.

All MuJoCo file references below are relative to `third_party/MuJoCo/src` (so `src/user/...`,
`src/engine/...`, `include/mujoco/...`, `plugin/...`), same convention as the survey.

## STATUS (living section — update on every milestone commit)

Last updated: 2026-07-07.

Implementation plan: `docs/plan_native_compiler_impl.md` (public API, CDR-14 purity/binding
architecture, CompileContext side tables, lifted-code registry mechanics, NC0/NC1 work breakdown).

| Milestone | State | Evidence |
|---|---|---|
| NC0 infrastructure: lifted-code registry + gate, harness factoring, fallback skeleton | queued | |
| NC1 rigid-body pathfinder (blocks, defaults, tree, primitive geoms, sizes, names) | queued | |
| NC2 constraint + drive (contact, equality, tendon, actuator, sensor, keyframe, custom) | queued | |
| NC3 assets (mesh pipeline, hfield, texture, material, skin) | queued | |
| NC4 macros-as-tree-ops + plugins + compiler transforms (fusestatic/discardvisual) | queued | |
| NC5 flex + flexcomp; composite decision | queued | |
| NC6 native default + perf note | queued | |

---

## 0. Direction correction and evidence base

**What was rejected.** The previous version of this file (mjs-walker plan, NDR-1..15) built a
transient mjSpec per compile and let `mj_compile` do everything. It inherited mjSpec's compile
pipeline for free — and with it mjSpec's semantics: eager default copies, sentinel-encoded
presence, authored bitmasks, deep/shallow attach hazards, a `void*` plugin contract, id assignment
we could observe but not own. The owner's verdict: that is a shim on top of the thing ProtoSpec
exists to replace. Rejected in full.

**What survives from it** (adopted below, with new numbering):

- the three-way golden harness (original XML vs ProtoSpec-XML-path vs native), CDR-2;
- the per-model fallback gate with loud `path_taken` reporting, CDR-2;
- the drift-gate extension concept — repointed from the mjs API surface to the lifted-code
  registry, CDR-3;
- the XML path as differential oracle, CDR-2.

**Survey status.** `docs/native_compiler_survey.md` remains the evidence base with one carve-out:
its §2 (the family-by-family `mjs_*` decode-surface mapping) is **moot** — it maps an API we will
not call. It is retained as history, not deleted; its warts catalog is part of why the walker was
rejected. Its §1 (attach semantics), §3 (compile-pass inventory, public-API and `mjuu_*` utility
tables), §5 (harness assets), and §6 (risks R1-R10, several of which simply evaporate on the
native path — see CDR-4/CDR-5) remain valid and are cited below. This plan's reuse ledger
(Section 3) supersedes the survey's §3.3 "internal C++ we might link" table: under the mandate we
lift, we do not link internals.

**New evidence.** The reuse ledger below is grounded in a fresh pass-by-pass read of
`mjCModel::TryCompile` (`src/user/user_model.cc:4976-5388`), the per-object `Compile` methods in
`src/user/user_objects.cc`, the mesh pipeline in `src/user/user_mesh.cc`, and the actuator/plugin/
keyframe surfaces in `src/user/user_api.cc` + `src/engine/`. The decisive finding: **MuJoCo's
compiler passes are thin `mjC*` orchestration over a free-standing math kernel.** Almost every
numeric algorithm lives in `mjuu_*` free functions, static file-local helpers, or classes that
consume plain arrays (BVH, octree, qhull graph, mesh inertia). The entanglement is in the driver
loops and field plumbing — exactly the part that must be rewritten against ProtoSpec types anyway.

---

## 1. Goal and non-goals

**Goal.** A raw compiler behind the existing signature:

```
Compile(const Model&, CompilePath path) -> Compiled          // Compiled{mjModel*, Binding, report}
enum class CompilePath { XmlPath, NativePath };
```

`NativePath` walks the ProtoSpec tree, runs our own compile passes (lifting MuJoCo's algorithms
where the ledger says so), allocates the `mjModel`, fills its arrays, and finishes with MuJoCo's
public post-build passes (`mj_setConst`, sparsity builders, `mj_setLengthRange`, validation).
ProtoSpec is the only authoring and source representation; there is no intermediate model object
of any kind between the ProtoSpec tree and the `mjModel`.

**The XML path is retained — as scaffolding, honestly labeled.** `XmlPath`
(ProtoSpec → canonical MJCF → `mj_loadXML`) remains: (a) the differential **oracle** — every
native compile is answerable to it until the corpus exit bar is met, and it stays in the harness
permanently; (b) the **per-model fallback** while native feature coverage grows (CDR-2); (c) the
writer half of `Expand()`. It is scaffolding for correctness, not a shim in the runtime
architecture: no runtime component depends on XML-path behavior, and long-term the native path is
the default (NC6) with the XML path demoted to a harness flag and the MJCF interchange writer it
always was.

**Non-goals:**

- No mjSpec, `mjs_*`, or `mjC*` anywhere in the native path — including the tempting conveniences
  (`mjs_resolveOrientation`, `mjs_sensorDim`, the OBJ/STL decoder plugins that emit an `mjSpec`).
  Their underlying logic is lifted instead (ledger rows below). `mjs_*` entry points remain
  legitimate **inside test binaries only**, as oracles for lifted-function goldens (Section 4.3).
- No `mjSpec → ProtoSpec` importer, no persistent compiled-side state, no URDF (unchanged from
  plan.md Section 6).
- No incremental compile. Like MuJoCo itself (plan.md DR-11 investigation), recompile is a full
  compile plus state migration.
- Phase 1 makes no attempt at compile-time perf work; correctness first, perf note at NC6.

**Why this is worth a long job** (the owner knows it is one): the compiler becomes ours —
diagnosable end to end with SourceLoc on every error, id assignment we control (CDR-4), defaults
that are a generic merge instead of an eager-copy trap (CDR-5), recompile without a serialize/
parse hop, and no dependency on the least stable, most convoluted layer of MuJoCo (`src/user/`
mjC* lifecycle) — only on its most stable layers (public API, plain math) plus lifted code that a
registry pins and audits per bump (CDR-3).

---

## 2. Decision records

Numbered CDR-n (compiler decision records; the retired mjs-walker plan used NDR-n — that
numbering is dead, and plan.md's pointers to it get one-line amendments, Section 8).

**CDR-1: Three-tier reuse boundary; non-exported symbols are lifted, never linked.**
The native path may (tier 1) call anything `MJAPI` — whether declared in `include/mujoco/mujoco.h`
or in an engine header like `src/engine/engine_io.h` (e.g. `mj_makeDofDofSparse` at
`engine_io.h:90`, `mj_validateReferences` at `:87` — exported but not surfaced in `mujoco.h`; we
carry our own declarations, drift-gated); (tier 2) use the free-standing `mjuu_*` utilities —
lifted as one unit (`user_util.cc` has zero `mjC*` references) rather than linked, since
non-`MJAPI` symbols are not exported from the mujoco DLL we build against; (tier 3) lift anything
else per the registry (CDR-3). Nothing in the native path includes `mjspec.h` or `user/user_*.h`.
Exactly three engine-internal, non-exported functions are needed and lifted: `mj_makeModel`
(`src/engine/engine_io.c:227` — compact, driven by the public `mjxmacro.h` X-macros, so the lift
tracks upstream field additions structurally), `mj_hashString` (`src/engine/engine_name.c:229-236`,
7 lines, djb2-xor), and `mj_mergeChain` (`src/engine/engine_core_util.h:40`, used by the
actuator-moment nnz count).

**CDR-2: Per-model gate, loud fallback, XML oracle — carried over from the rejected plan.**
`Compile(m, NativePath)` pre-scans the tree against a native-supported feature manifest
(`bridge/native_supported.json`); any unsupported feature routes the **whole model** through the
XML path. `Compiled.report` carries `path_taken` and
`fallback_reasons[{feature, count, first SourceLoc}]`; the differential suite keys its ratchet off
`path_taken`. No half-native models: one model, one compiler, so a diff failure indicts exactly
one implementation. The gate is feature-granular, not tag-granular — e.g. "geom with mesh
reference" is a separate gate item from "geom", so primitive-geom models compile natively before
the mesh pipeline lands (NC1 before NC3).

**CDR-3: The lifted-code registry — THE mechanism that keeps a raw compiler honest across bumps.**
Every lifted function is recorded in `tools/lifted/registry.json`:

```
{ "ours": "cpp/compile/...", "upstream": "src/user/user_model.cc",
  "symbol": "mjCModel::ComputeSparseSizes", "lines": "906-1119",   // human reference only
  "snapshot": "tools/lifted/upstream/ComputeSparseSizes.cc",       // exact upstream text at lift time
  "lifted": "2026-07-..", "notes": "plumbing retargeted to ProtoSpec; algorithm verbatim" }
```

Extraction is anchored by symbol signature + brace matching, not raw line numbers (lines shift
every bump). The DR-12 drift gate grows a lifted-code panel: on every MuJoCo bump it re-extracts
each registered symbol from the new vendored tree and diffs against the stored snapshot. Any
divergence fails the gate with the upstream diff attached — the reviewer then re-lifts (or
documents why our copy intentionally stays behind) and refreshes the snapshot. Additionally, every
lifted file in our tree carries a provenance header (upstream path + symbol + pin version +
Apache-2.0 attribution; MuJoCo's license permits this with notice — NOTICE file updated once at
NC0). This registry, plus the corpus goldens (Section 4), is how upstream semantic drift gets
caught instead of silently accumulating.

**CDR-4: We assign ids — Binding becomes direct. A structural win over both the XML and mjs paths.**
MuJoCo's id rule is `id = position in per-type list` (`ProcessList_`, `user_model.cc:4566-4588`).
In the native compiler *we* run that assignment over *our* lists, so the DR-10 Binding is built
**during compile**: as each element receives its id, `Binding` records element-pointer → id
directly. Consequences:

- No `mj_name2id` lookups on the native path; no reliance on auto-generated names for
  bindability. (Names are still emitted into the model's name tables per DR-10 auto-naming, so
  `mj_name2id` parity holds for downstream users and the XML path's Binding is unchanged.)
- The two hazards that forced DR-10 to choose name-based binding — `discardvisual`/`fusestatic`
  id compaction and pair/exclude re-sorting (`user_model.cc:5143-5146`) — evaporate: we perform
  those transforms ourselves *before* id assignment (CDR-8), and we sort pairs ourselves, so the
  binding is constructed after all reordering, never patched.
- Debug cross-check: for every named element, `mj_name2id(m, type, name) == binding.Id(elem)` —
  a free structural audit of our name tables + hash fill (ledger 3.7).
- plan.md DR-10 gets an amendment note (Section 8): name-based lookup remains the mechanism on
  the XML path and the public Binding semantics are unchanged; the native path is a second,
  direct constructor of the same Binding.

**CDR-5: Defaults resolution is ProtoSpec's generic presence merge. mjCDef is replaced entirely — a win, not a port.**
Confirmed by source read: MuJoCo has **no deferred merge algorithm to lift**. `mjCDef` holds one
fully-resolved element per family; a child class is constructed by copying its parent wholesale
(`AddDefault` → `CopyWithoutChildren`, `user_model.cc:1533-1559`), and elements copy the whole
family struct at construction (`*this = _def->Joint()`, `user_objects.cc:3064`; geom `:3323`).
ProtoSpec's presence-tracked `opt<T>` fields make resolution a generic generated merge
(`Effective()`, plan.md Section 7) evaluated at compile time per element: element → class →
nearest childclass → ancestors → `main` → IDL defaults, first authored value wins. The eager-copy
trap (survey R2), the label-only `mjs_setDefault` trap, and the authored-bitmask machinery
(survey R9) all cease to exist. The IDL default values themselves were already harvested from the
`mjs_default*` tables (`src/user/user_init.c:25-391`) at bootstrap; the drift gate already
watches them.

**CDR-6: Recompile and state migration are fully ours (DR-11, minus every hop).**
`Recompile(model, prevCompiled, d)` = read state slices via the previous Binding, run the native
compile, write slices back via the new Binding, absent → `qpos0`/zeros (unchanged semantics from
plan.md DR-11). No XML hop, no mjSpec `SaveState`/`RestoreState` (mjC*, banned, and unnecessary —
DR-11's binding-keyed design was chosen precisely because it is representation-independent).
`qpos0` needed for migration comes from our own reference-configuration pass (ledger 3.3). The
recompile-equivalence test (plan.md 10.7) compares against `mjs_*` edit + `mj_recompile` as an
oracle inside the test binary.

**CDR-7: Macros. Replicate/attach are our tree clones — native early. Composite/flexcomp fall back per-model first, generators lifted later.**
- `Replicate` and `Attach` are ProtoSpec-tree operations: clone the sub-tree (DR-2 plain values
  make Clone trivial), apply prefix/suffix via the SDK's rename-with-referrers (DR-8), accumulate
  pos/quat offsets with `mjuu_frameaccum` math. This replaces MuJoCo's `NameSpace`/`CopyList`
  machinery — including its silent drop of dangling referencers (`user_model.cc:266-282`), which
  becomes a hard tier-2 validation error instead. Lands at NC4; 42 replicate + 14 attach corpus
  files unlock (measured, Section 5).
- `Flexcomp`: the point/element **generators** (`MakeGrid`/`MakeBox`/`MakeSquare`/`MakeMesh`/
  `LoadGMSH`, `user_flexcomp.cc:861-1500, 2100+`) are near-free-standing geometry code producing
  plain `point`/`element` vectors — liftable to emit ProtoSpec elements directly. Blocked on flex
  compile itself (NC5). 69 corpus files.
- `Composite`: the vendored pin has deprecated every composite type except `cable`
  (`user_composite.cc:202-238` returns errors pointing at replicate/flex/flexcomp); 3 corpus
  files. Per-model XML fallback until NC5, where the `MakeCable` lift-vs-permanent-fallback
  decision is made with usage data (Open Q1).

**CDR-8: fusestatic / discardvisual become ProtoSpec-level pre-passes — simpler than upstream's.**
MuJoCo applies both *after* object compilation, then patches ids (`FuseStatic`
`user_model.cc:4364-4524`; `IndexAssets(discard)` `:2021-2058`). We apply them to our own lists
**before id assignment** (discardvisual = filter visual geoms/meshes/materials/textures;
fusestatic = fold jointless non-mocap bodies into parents, with the frame/inertia accumulation
math lifted), so no compaction or fixup pass exists and CDR-4's direct binding stays valid —
elements removed by a transform simply report unbound with the responsible flag named, exactly
DR-10's contract. Until implemented (NC4), models with these flags set take the fallback gate.

**CDR-9: Diagnostics contract — SourceLoc-native, strictly better than MuJoCo's.**
One diagnostic type (`Diagnostic{SourceLoc, severity, message}`) replaces `mjCError` in all lifted
code (a mechanical rewrite of throw sites, part of each lift). Contract, tested: (a) every
diagnostic for a file-originated element carries file:line through include expansion (DR-9);
(b) **parity** — any model that fails native compile also fails XML compile and vice versa, over
the corpus and the broken-model suite (a one-sided failure is by definition a native-compiler
bug); (c) warnings surfaced, never swallowed; (d) errors name the compile pass that raised them.
Where MuJoCo detects an error only at compile time via an `info` string, we typically detect it
at validation with a typed ref — earlier and better-located.

**CDR-10: Equality lowering is owned and lifted from the XML reader.** The typed→`data[11]`
packing lives only in `src/xml/xml_native_reader.cc:2192-2311` (connect anchor→data[0..2]; weld
anchor+relpose+torquescale→data[0..10]; joint/tendon polycoef→data[0..4]; flex variants). On the
native path this is simply the fill rule for `m->eq_data` — one table-driven module, lifted with
a registry entry, golden-tested against reader-parsed specs (Section 4.3). plan.md's Q-EQ
amendment stays true; its pointer is re-aimed (Section 8).

**CDR-11: Actuator lowering math is lifted; `mjs_setTo*` are off the table as calls.**
The shortcut→general math lives in `src/user/user_api.cc:1133-1468` (motor `:1133-1142`; position
incl. the biasprm[2]-sign kv-vs-dampratio encoding `:1147-1181`; intvelocity `:1186-1198`;
velocity `:1203-1211`; damper `:1216-1231`; cylinder `:1236-1248`; muscle `:1253-1294`; dcmotor
incl. the full actdim/state-slot derivation `:1314-1468`). Deferred pieces live in
`mjCActuator::Compile`: `inheritrange` resolution `user_objects.cc:7139-7181`, actdim rules
`:7211-7227`, muscle param validation `:7236-7266`. All lifted (registry entries), applied when
lowering ProtoSpec's typed actuator variants into `m->actuator_{gain,bias,dyn}prm` at fill time.
`mju_muscleGain/Bias/Dynamics` are runtime engine functions (public, `mujoco.h:1363-1371`) — the
compiler only sets the type enums. Length range is **not** compile-entangled: it is a post-build
pass over `(mjModel*, mjData*)` via public `mj_setLengthRange` (`mujoco.h:299`) — reused as-is
(ledger 3.8).

**CDR-12: Threading — phase 1 single-threaded, by decision not accident.** MuJoCo threads exactly
one thing: per-asset mesh/texture compilation (`CompileMeshesAndTextures`,
`user_model.cc:4760-4840`, one task per mesh or texture) and optionally length range. Our NC1-NC5
compiler is single-threaded: simpler to make bit-identical, and authoring-scale models compile in
milliseconds. The upstream ThreadPool (`user_threadpool.cc`, plain std::thread, no mjC*) is a
registered lift candidate if a measured need appears; per-asset granularity ports directly since
our mesh compiles are equally independent. Revisit with numbers at NC6.

**CDR-13: Signature is lifted for bit-identity.** `m->signature` is filled by
`mjCModel::Signature` (`user_model.cc:5403-5482`: `PrintTree` topology string + per-type counts +
sensor types, hashed with `mj_hashString(str, UINT64_MAX)`). The three-way harness diffs every
field, so we reproduce it exactly (a small lift) rather than carve a diff exception. ProtoSpec's
own edit detection stays the DR-10 edit counter; the signature is write-only output for us.

**CDR-14: Compile is a pure function; Binding is a separate wrapper object; all intermediate
state lives in compiler-internal side tables.** (Owner directive, 2026-07-07; supersedes any
reading of CDR-4/CDR-6 that implies writing into the tree.)
`Compile(const Model&, path)` MUST NOT modify the ProtoSpec tree in any way: no resolved values
written back, no ids stamped on elements, no name mutations, no lazy caches on elements. Every
intermediate/resolved quantity — effective defaults after class merge, resolved orientations,
computed masses/inertias, id assignment, the interned name blob, keyframe padding — lives in
side tables inside a compile-scoped `CompileContext`, keyed by element identity. This is the
deliberate divergence from MuJoCo, whose compiler mutates its spec heavily (`CopyFromSpec`
lifecycle, id stamping, keyframe resizing); the side-table architecture is named as such and is
not negotiable per lift. Purity is enforced mechanically: a standing test deep-compares the tree
(generated `operator==` plus a serial/SourceLoc sweep) before and after Compile, and the compiler
is const-correct end to end — any `const_cast` (or `mutable` state on elements) in `cpp/compile/`
is a review-rejected defect, backed by a grep gate. Documented caveat: Compile reads asset FILES
from disk (mesh/texture/hfield), so it is pure with respect to the model object, not the
filesystem.
`Compile` returns `Compiled{ unique_ptr<mjModel>, Binding, CompileReport }`. `Binding` wraps the
ProtoSpec model: it holds a const pointer to the Model plus side maps (element identity →
`mjtObj` type + id, the reverse map, and address sugar: qposadr/dofadr/sensor_adr/actuator ids).
Binding may not rely on Compile having stamped anything into the tree. Staleness mechanism: the
Model root carries an SDK-maintained mutation counter (bumped by SDK mutators, never by Compile,
which only READS it); Binding stores the counter value at compile time and every accessor asserts
it is unchanged. Reconciliation with DR-10: on the native path the Binding is direct (we assigned
the ids, CDR-4); on XML-path compiles it stays name-based (`mj_name2id` + auto-name serials) —
two constructors behind one Binding interface. Full API, identity-key decision, and enforcement
details: `docs/plan_native_compiler_impl.md`.

---

## 3. The reuse ledger

The heart of the plan. Classification:

- **A** — called as-is: public API (`mujoco.h`) or exported engine API (`MJAPI` in engine headers;
  own declarations, drift-gated).
- **B** — lifted verbatim or near-verbatim: free-standing over plain data; registry entry; only
  error-type/plumbing touches.
- **C** — reimplemented against ProtoSpec types: the algorithm is lifted (registry entry cites the
  source) but the data plumbing — mjC* field reads, list walks, `CopyFromSpec` lifecycle — is
  rewritten. "What we write" is the plumbing; "what we lift" is the logic and constants.
- **D** — not needed: XML/spec-lifecycle concern with no counterpart in a spec-tree→mjModel path.

Reference pass order: `mjCModel::TryCompile` (`src/user/user_model.cc:4976-5388`). Our compiler
keeps the same pass order unless a CDR says otherwise (CDR-8 moves two transforms earlier).

### 3.1 Front matter: checks, ids, bookkeeping

| Pass | Upstream | Class | Disposition |
|---|---|---|---|
| Global guards (joint-in-world, body+flex < 65534) | user_model.cc:4997-5011 | C | trivial validation, ours |
| Name fill / `CheckEmptyNames` (asset names mandatory) | user_model.cc:2064-2092 | C | folded into tier-2 validation + DR-10 auto-naming |
| Id assignment + duplicate check (`ProcessLists`/`CheckRepeat`) | user_model.cc:4550-4618 | C | ours by construction (CDR-4); duplicate names caught at validation with SourceLoc, earlier than upstream's compile-time throw |
| Asset name→id resolution (`IndexAssets`, non-discard half) | user_model.cc:1917-2020 | C | replaced by DR-8 typed refs — resolution already done by tier-2 validation; the fill step reads resolved ids |
| discardvisual (delete + id compaction) | user_model.cc:5060-5068, 2021-2058 | C | reimplemented as pre-id filter pass (CDR-8); upstream's compaction/fixup not ported |
| fusestatic (`FuseStatic`/`FuseReindex`) | user_model.cc:4364-4524 | C | reimplemented as ProtoSpec tree transform pre-id (CDR-8); inertia/frame accumulation math lifted |
| `SetNuser` (auto `nuser_*` from max user-array length) | user_model.cc:1865-1914 | C | trivial |
| Hull-needed marking (contact-participating mesh geoms) | user_model.cc:5074-5092 | C | trivial predicate, logic lifted |
| Defaults machinery (`mjCDef` tree, eager copies) | user_objects.cc:1303-1457 | **D** | replaced by generic presence merge (CDR-5) — the headline win |
| `CopyFromSpec`/`PointToLocal` tri-storage sync | throughout user_* | **D** | ProtoSpec has one storage |
| Attach/`NameSpace`/`CopyList` merge machinery | user_model.cc:261-511, user_objects.cc:1381-1496 | **D** | replaced by tree clone + rename-with-referrers (CDR-7) |
| Mesh/asset cache (`mjCCache`) | user_cache.cc; user_mesh.cc:446-514, 1022-1093 | **D** | pure optimization; skipped (correctness unaffected) |

### 3.2 Resolvers: orientation, fromto, frames, limits

| Pass | Upstream | Class | Disposition |
|---|---|---|---|
| Orientation resolution (euler/axisangle/xyaxes/zaxis → quat) | `ResolveOrientation` user_objects.cc:241-349 | B | pure function over plain args; unify with the existing `core/resolve.cc` resolver (one implementation, not two); `mjs_resolveOrientation` (its public wrapper, user_api.cc:1649) is a test oracle, never a call |
| Frame flattening (frame pose into children) | `mjCFrame::Compile` user_objects.cc:3028-3047 → `mjuu_frameaccumChild` | B math, C walk | the single most reused primitive across body/joint/geom/site/camera/light; the walk is ours over ProtoSpec Frame containers |
| fromto → pos/quat/size (geom + site variants) | user_objects.cc:3978-4016, 4238-4272 | B | closed-form (`mjuu_z2quat`); unify with the `GeomShape` resolver |
| Full inertia → principal (quat + diag) | `mjuu_fullInertia` user_util.cc:861 + `mjuu_eig3` :662 | B | `mjuu_eig3` is non-MJAPI — rides with the `mjuu_*` unit |
| Tri-state autolimits (`checklimited`/`islimited`) | user_objects.cc:173-191 | B | lifted verbatim (consumers: joints :3203, tendons :6709, actuators :7186-7194); matches Q-AUTO |
| The whole `mjuu_*` pool (~45 fns: quat/frame/inertia math, path/content-type utils) | user_util.{h,cc} (0 mjC* refs) | B | lifted as one unit, one registry entry; `mjuu_eigendecompose` is MJAPI (A) but rides along |

### 3.3 Kinematic tree compilation

| Pass | Upstream | Class | Disposition |
|---|---|---|---|
| Body compile orchestration (order: child frames → weldid → orientation → inertial variant → geom compile → inertia inference → bounds → frame accum → free-joint alignment phase 1 → BVH → joints/dofs → qpos0 pose → sites/cams/lights → alignment phase 2) | `mjCBody::Compile` user_objects.cc:2673-2898 | C | the ordering is semantics and is lifted exactly; plumbing ours |
| Body inertia from geoms (mass-weighted COM, parallel-axis, principal re-diag) | `InertiaFromGeom` user_objects.cc:2458-2527 | C | math is `mjuu_globalinertia`/`offcenter`/`fullInertia` (B); trigger predicate (inertiafromgeom tri-state × explicitinertial × inertiagrouprange, :2728-2740) lifted |
| boundmass / boundinertia / balanceinertia (A+B≥C) | user_objects.cc:2749-2771 | B | predicates lifted verbatim |
| Joint compile (per-consumer degree conversion: hinge/ball range, hinge ref/springref; axis rotation + normalization; autolimits; pos flatten) | `mjCJoint::Compile` user_objects.cc:3179-3292 | C | the Q-ANGLE conversion sites (:3217-3224, :3279-3282) lifted exactly — this pass is why read-time conversion was wrong |
| Geom compile (validity, fromto, mesh-fit hook, size semantics, mass/density→inertia, aabb, fluid coefs) | `mjCGeom::Compile` user_objects.cc:3936-4135 | C | driver ours |
| Geom volume/inertia/AABB closed forms per type×(volume\|shell) | `GetVolume`/`SetInertia`/`ComputeAABB`/`checksize` user_objects.cc:3415-3713, 3873-3926, 141-170 | B | lifted verbatim; mesh branches consume ledger-3.4 outputs |
| Site / Camera / Light compile | user_objects.cc:4218-4292, 4387-4457, 4562-4582 | C | thin; camera intrinsics arithmetic (six arrays → `intrinsic[4]` + fovy override, :4419-4456) lifted as a B sub-block — the compile-side half of the `CameraIntrinsics` variant |
| BVH build (leaf transform into inertial frame, median-split, AABB accumulation) | `mjCBoundingVolumeHierarchy` user_objects.cc:357-546 (+ distance queries :636-828) | B | consumes plain id/aabb/pos/quat/flag arrays, emits `bvh_*` arrays 1:1; per-body glue (`ComputeBVH` :2623-2635) trivial C |
| Octree + SDF coeffs (adaptive build, balancing, hanging nodes, BFS corner SDF) | `mjCOctree` user_objects.cc:589-1258, 831-921 | B | plain vert/face arrays + the B BVH; no plugin dependency |
| Reference configuration (`qpos0`/`body_pos0`/`body_quat0` from joint refs) | `ComputeReference` user_model.cc:4853-4883 | C | trivial; feeds keyframes + CDR-6 |

### 3.4 Asset pipelines

| Pass | Upstream | Class | Disposition |
|---|---|---|---|
| Resource open/read (file + VFS + providers) | `mju_openResource`/`mju_readResource` (public) | **A** | called directly; `mjCBase::LoadResource` glue not needed |
| OBJ parse (tinyobjloader; quad triangulation, V-flip, line segments) | plugin/obj_decoder/obj_decoder.cc (167 lines) | B | parsing core lifted **minus its mjSpec-emission tail** (it emits via `mjs_addMesh` — banned); tinyobjloader already vendored |
| STL parse (binary, header/size validation, vertex dedup) | plugin/stl_decoder/stl_decoder.cc (144 lines) | B | same treatment |
| MSH parse (binary blocks, winding swap) | `LoadMSH` user_mesh.cc:1101-1187 | B | lifted |
| Vertex weld/validate (`ProcessVertices`, `CheckInitialMesh`) | user_mesh.cc:539-610, 1670-1707 | B | lifted |
| Normals (`MakeNormal`, area-weighted + sharp-edge removal) | user_mesh.cc:2503-2610 | B | lifted |
| **Convex hull + graph** (qhull `"qhull Qt"`, maxhullvert → `Q9 TA<n>`, degeneracy guards, flat graph `[nv, nf, vert_edgeadr, vert_globalid, edge_localid, face_globalid]`) | `MakeGraph` user_mesh.cc:1726-1980 (+ `CopyGraph` :1985-2005) | B | self-contained qhull invocation over `double*`; qhull already vendored; the single most reusable heavy algorithm |
| Polygon merge (`MeshPolygon` + `MakePolygons`: hull tris → coplanar polys → `mesh_poly*`) | user_mesh.cc:2674-2958 | B | helper class pure; driver trivially retargeted |
| Volume / surface / centroid / inertia (exact/legacy/convex/shell variants) + principal reorientation + inertia box | user_mesh.cc:1192-1637 | B | pure math over verts + faces/graph; `mjuu_eig3` per 3.2 |
| Primitive fit (`FitGeom`: inertia-box or AABB modes, fitscale) | user_mesh.cc:944-1018 | B | writes into our geom struct instead of `mjCGeom*` |
| Mesh compile orchestration (load → check → process → BVH → octree) | `mjCMesh::TryCompile`/`Process` user_mesh.cc:689-807, 1350-1570 | C | driver ours; stage order lifted |
| SDF-plugin meshes (`LoadSDF`, marching cubes) | user_mesh.cc:356-442 | C | deferred with plugins (NC4); MarchingCubeCpp vendored |
| HField (custom binary `[nrow,ncol,f32...]`, PNG grey, [0,1] normalization) | user_objects.cc:4691-4870 | B core, C driver | lifted |
| Texture builtins (gradient/checker/flat + marks, 2D + cube) | user_objects.cc:4974-5218 | B | pure array fills |
| Texture/hfield PNG + KTX decode (lodepng wrapper) | `PNGImage` user_objects.cc:57-155; loaders :5221-5631 | B core, C dispatch | lodepng vendored |
| Skin (.skn binary, bind validation, per-vertex weight normalization) | user_mesh.cc:3094-3311 | C | normalization/validation blocks lifted |
| Material compile + texture role array | user_objects.cc (trivial) | C | fill only |

### 3.5 Constraint and drive objects

| Pass | Upstream | Class | Disposition |
|---|---|---|---|
| Contact pair combination (margin/gap max; priority winner; solmix-weighted solref/solimp; condim/friction max; direct-solref min) | `mjCPair::Compile` user_objects.cc:5952-6058 | B rules, C glue | combination arithmetic lifted verbatim; ref resolution ours (typed refs) |
| Exclude signature packing (`(body1<<16)+body2`, ordered) + pair/exclude stable sort | user_objects.cc:6131-6173; user_model.cc:5142-5146 | B | trivial; we own the sort **and** the ids assigned after it (CDR-4) |
| Equality (objtype selection, hinge/slide guard) + typed→data[11] | user_objects.cc:6267-6348 + xml_native_reader.cc:2192-2311 | C + B packer | CDR-10 |
| Tendon (wrap-path validation: pulley/site adjacency, sphere→cylinder, sidesite, springlength order, autolimits) | user_objects.cc:6470-6736, 6800-6874 | C | validation rules lifted; path items are already real ProtoSpec structs (no `mjsWrap` stub to fight) |
| Actuator (trntype resolution, typed lowering, inheritrange, actdim, muscle checks) | user_objects.cc:7027-7295 + user_api.cc:1133-1468 | C + B math | CDR-11 |
| Sensor (per-type validation switch; obj/ref resolution) | user_objects.cc:7393-7954 | C | switch logic lifted; `sensorDatatype`/`sensorNeedstage` tables (:7481-7604) B; **dim table** lifted from the `mjs_sensorDim` impl (user_api.cc:1694) B — `sensor_adr` itself is a fill cursor (3.7) |
| Keyframe compile (empty vector → defaults from qpos0/body pose; wrong size → hard error) | `mjCKey::Compile` user_objects.cc:8353-8433 | B/C | logic lifted; runs against the built model exactly as upstream does; `core/sizes.cc` predictions + tier-3 lint stay the front line |
| Keyframe expansion for edited trees | `ExpandKeyframe` user_model.cc:4903-4948 | C | ours (the attach-pending keyframe machinery is D — CDR-7 clones keyframes with the tree) |
| Custom numeric/text/tuple | trivial | C | fill only |
| Plugin registry (name→slot, plugin table) | `mjp_getPlugin`/`mjp_getPluginAtSlot`/`mjp_pluginCount` mujoco.h:1518-1524 | **A** | called directly |
| Plugin attribute flattening (declared-order NUL-separated → `plugin_attr`) + state sizing (`nstate`/`nsensordata` callbacks → `plugin_stateadr`, sensor dims) | user_objects.cc:8489-8527; user_model.cc:3122-3203, 5848-5899 | B flatten, C wiring | ProtoSpec's ordered (key,value) pairs feed it — the `void*` C++-map contract (survey R6) has no counterpart here |

### 3.6 Sizes and addresses

| Pass | Upstream | Class | Disposition |
|---|---|---|---|
| Size census (`SetSizes`: every n*; mostly trivial counts + four real algorithms: `ntree` static-ancestry walk, `nJfe`/`nJfv` flex-Jacobian nnz, `nnames` accumulation, `nhistory`) | user_model.cc:2109-2368 | C | count plumbing ours; the four algorithms lifted |
| qposadr/dofadr/actadr assignment (list order) | `SaveDofOffsets` user_model.cc:329-367 | C | trivial cursors over our lists |
| **Dof-tree + sparse sizes** (`nM` ancestor chains, `nD = 2nM − nv`, `nB` via subtreedofs, `nC` simple-body analysis incl. joint-based and tendon-armature demotion, `nJten`) | `ComputeSparseSizes` user_model.cc:906-1119 | C | the crown-jewel lift: algorithm verbatim, accessors retargeted; cross-validated at fill time (3.7) |
| nmocap / mocapid | user_model.cc:5157-5164 | C | trivial |
| Moving-body mass guard | `CheckBodyMassInertia` user_model.cc:5486-5508 | C | validation |

### 3.7 mjModel construction and array fill

| Pass | Upstream | Class | Disposition |
|---|---|---|---|
| Allocation (`mj_makeModel`: ~80 sizes → one buffer, xmacro-driven) | engine_io.c:227 (decl engine_io.h:50, non-exported) | B lift | CDR-1; must stay layout-compatible with `mj_deleteModel`/`mj_copyModel` (round-trip tested) |
| `m->opt`/`m->vis` copy; statistics override (upstream NaN sentinels → our presence checks) | user_model.cc:5190-5191, 5308-5320 | C | trivial |
| **Name tables + hash** (fixed 23-list order; `name_*adr`; `names_map` sized `mjLOAD_MULTIPLE·n`, djb2-xor + linear probing) | `CopyNames`/`namelist`/`addtolist` user_model.cc:2546-2667; `mj_hashString` engine_name.c:229-236 | B | lifted; must be byte-exact for `mj_name2id` parity — the CDR-4 cross-check enforces it |
| Path table | `CopyPaths` user_model.cc:2670-2691 | B | lifted |
| **Tree fill** (`CopyTree`: all body/jnt/dof/geom/site/cam/light arrays; `dof_parentid` via running lastdof, `dof_treeid`/`body_treeid`, `body_rootid`, sameframe detection, simple-body finalization incl. slider promotion, `qpos0`/`qpos_spring`, `dof_Madr` + nM/nB/nD cross-validation) | user_model.cc:2696-3119 | C | the widest fill surface; topology algorithms lifted verbatim, hundreds of field copies retargeted — the mjxmacro diff harness is the safety net |
| **Object fill** (`CopyObjects`: assets/constraints/drive/keys; `sensor_adr` cursor, `actuator_actadr` cursor, shared history cursor, tendon wrap ranges) | user_model.cc:3325-4043 | C | address-cursor arithmetic lifted; per-family copies mechanical |
| Simple-dof finalization + actuator-moment nnz (`FinalizeSimple`; `CountNJmom`/`CountTendonDofs`/`CountNJten`) | user_model.cc:4048-4092, 3209-3324 | B | operate on the **built mjModel only** — near-verbatim lifts; need `mj_mergeChain` (CDR-1) |
| Signature | user_model.cc:5403-5482 | B | CDR-13 |
| narena heuristic | user_model.cc:5219-5252 | B | lifted verbatim |

### 3.8 Post-build finalization — MuJoCo keeps all of it (public API)

| Pass | Entry point | Class |
|---|---|---|
| Sparsity structures (D, B, M rows + M↔D index maps) | `mj_makeDofDofSparse`/`mj_makeBSparse`/`mj_makeDofDofMaps` engine_io.h:90-102 (MJAPI) | **A** |
| Scratch data for setConst | `mj_makeRawData` engine_io.h:116 + `mj_resetData` mujoco.h:259 | **A** |
| Keyframe quat normalization | `mj_normalizeQuat` mujoco.h:647 | **A** |
| **All derived constants**: `body_invweight0`, `dof_invweight0`, `tendon_length0`/`invweight0`, `flexedge_invweight0`, `actuator_acc0`, `cam_pos0`/`cam_mat0`, `light_pos0`, tree/actuator cross-indexes, `stat.*` (extent/center/mean*) | `mj_setConst` mujoco.h:295 (impl engine_setconst.c:1217) | **A** |
| Auto spring-damper (springdamper[2] → stiffness/damping via `dof_invweight0`) | lifted from user_model.cc:2373-2410 **minus** its write-back into `mjCJoint` (XML-save-only) | B |
| Actuator length range | `mj_setLengthRange` mujoco.h:299 per actuator; our loop replicates the option-save/flag-disable wrapper (user_model.cc:2446-2487), skips the ThreadPool | **A** + trivial C |
| Total mass scaling (settotalmass) | `mj_setTotalmass` mujoco.h:657 | **A** |
| Reference validation | `mj_validateReferences` engine_io.h:87 (MJAPI) | **A** |
| Smoke test (data + one step, contact/sleep flags handled as upstream) | `mj_makeData`/`mj_step`/`mj_deleteData` mujoco.h | **A** |

### 3.9 Ledger totals and the hardest passes

Row counts over 3.1-3.8 (a row = one coherent pass/function group):

| Class | Rows | Share of rows | Share of *effort* (judgment) |
|---|---|---|---|
| A — call as-is | 11 | ~18% | ~5% — the entire post-build numeric finalization is free |
| B — lift verbatim | 24 | ~39% | ~30% — copy, retarget error type, golden-test each |
| C — reimplement, algorithm lifted | 22 | ~35% | ~65% — the fill surfaces and drivers dominate schedule |
| D — not needed | 5 | ~8% | 0 — and several are *negative* effort: whole hazard classes (eager defaults, authored bitmasks, id-compaction fixups, `void*` plugin map, attach silent-drop) cease to exist |

**The five hardest passes**, by correctness risk × volume:

1. **Dof-tree topology and sparse sizes** (3.6 `ComputeSparseSizes` + 3.7 `CopyTree` topology
   fields): the simple-body demotion cascade, `dof_parentid`/`dof_Madr`, nM/nB/nC/nD — subtle,
   interlocking, must be bit-exact. Mitigation: upstream itself cross-validates the
   `dof_Madr`-derived counts against the pre-computed sizes (`user_model.cc:3085-3118`) — we lift
   the validation too.
2. **The mesh pipeline** (3.4): the largest lift volume (qhull graph, four inertia variants,
   polygon merge), and it feeds geom→body inertia, so errors propagate into every mass property.
3. **The fill-surface breadth** (`CopyTree`/`CopyObjects`, 3.7): hundreds of mechanical field
   copies; individually trivial, collectively where omissions hide. The mjxmacro full-field diff
   exists specifically as the net for this.
4. **The body-compile ordering chain** (3.3): geom inertia → `InertiaFromGeom` → bounds/balance →
   two-phase free-joint alignment → frame flattening — order is semantics; lifted exactly and
   documented per step.
5. **Flex + flexcomp** (NC5): the one genuinely large C item (~570-line driver + elasticity
   kernels, `user_mesh.cc:3418-5201`), gating 69 corpus files.

---

## 4. Golden / verification strategy

Existing assets: `cpp/harness/mj_model_diff.cc` (mjxmacro full-field diff + fk invariant),
`tests/test_differential.py` (corpus-wide, manifest-gated), the 387-model corpus.

1. **Three-way harness, unchanged in mechanics.** A = `mj_loadXML(original)`; B = ProtoSpec →
   canonical XML → load (the DR-5 path, the permanent arbiter); C = native. Factor mj_model_diff's
   comparison core into a library target; `ps_native_diff <model.xml>` produces all legs
   in-process. A==B green isolates any A≠C failure to the native compiler. **Corpus exit bar:
   bit-identical** — both paths end in the same engine finalization over data produced by the same
   (lifted) algorithms; a tolerance would be a symptom, not a policy.
2. **Pass-level goldens.** Two mechanisms:
   - *Field-attribution diffing*: most "intermediate" quantities are final `mjModel` fields
     (`qpos0`, `dof_parentid`, `body_simple`, `mesh_graph`, `sensor_adr`, ...). The harness gains
     a pass-attribution map (mjxmacro field → owning ledger row) so a diff names the pass that
     produced the field, not just the field.
   - *True intermediates* (per-geom mass before body accumulation, hull graph pre-`CopyGraph`,
     per-class resolved defaults): harvested per corpus slice by an instrumented dump build of the
     vendored compiler (a local build flag, never shipped) into checked-in golden JSON under
     `tests/golden/native/`; our passes are unit-asserted against them.
3. **Lifted-function unit goldens.** Per registry entry, input/output pairs harvested from the
   original running inside a test binary — e.g. `mjs_setToPosition` on a scratch spec vs our
   lifted lowering; `ComputeInertia` via tiny meshes loaded through `mj_loadXML`; the equality
   packer vs `mj_parseXMLString`-produced `mjsEquality.data`. Test binaries may use `mjs_*`
   freely — they are oracles, not the product (the CDR-1 boundary applies to shipped code).
4. **The fidelity-chase risk, named.** MuJoCo's compiler semantics move upstream (recent examples
   in this very pin: composite deprecation, OBJ/STL extraction into decoder plugins, sleep/tree
   fields). Two tripwires: the lifted-code registry diff (CDR-3) flags every changed upstream
   function we lifted from, at bump time, *before* behavior drifts; and the corpus goldens (leg A
   vs leg C on the new pin) catch semantic changes in passes we did **not** lift from. Both fire
   in the same drift-gate run; a bump PR is not green until lifted code is re-reviewed and the
   corpus is re-identical.
5. **Regression pins.** Degree-vs-radian pair (per-consumer conversion, Q-ANGLE);
   defaulted-field goldens per family (CDR-5); inertiafromgeom tri-state × explicitinertial
   matrix; discardvisual/fusestatic binding reports; replicate suffix numbering + offset
   accumulation; keyframe wrong-size errors; the broken-model suite through both paths for the
   CDR-9 parity contract.
6. **Fuzz.** The plan.md 10.8 mutation fuzzer runs mutants through both paths: identical
   accept/reject, identical models on accept.
7. **Ratchet.** `native_supported.json` grows per milestone; the suite counts native-path models
   via `path_taken` (CDR-2) and the count only goes up.

---

## 5. Milestones

Measured corpus-unlock numbers (tag-whitelist scan over the 387-model corpus, 2026-07-07;
approximate — the gate measures the real number at each exit):

| Feature set | Models | Cumulative |
|---|---|---|
| blocks + defaults + body tree + primitive geoms (no mesh refs) | 39 | 10% |
| + contact/equality/tendon/actuator/sensor/keyframe/custom | +97 | 35% |
| + assets (mesh/hfield/texture/material/skin) | +105 | 62% |
| + replicate (42 files), attach (14), plugins/extension (17) | ~+55 | ~75-80% |
| + flex/flexcomp (69 files), composite (3), deformable (2) | rest | ~100% |

Execution conventions (plan.md Section 12): all code by Opus subagents; the compiler core is one
shared file set, so families land in **sequential waves, one owner per wave**; harnesses and
golden fixtures are authored by a **different agent** than the code they test; every wave ends
with an adversarial review pass before its exit criterion is declared met. Lifts are their own
reviewable units: one commit per registry entry (or tight group) with the upstream snapshot
included, so review is "diff against upstream", not "read 500 new lines cold".

This is a long job and is sized as such: NC1 and NC2 are multi-wave milestones on the scale of
the M3 IO effort; NC3 (mesh) and NC5 (flex) are each the size of a major IO wave on their own.
Nothing here starts until the M3 IO waves cover the corpus (leg B must be green to arbitrate) and
the M5 bridge surface exists (`Compiled{mjModel*, Binding}`, auto-naming serials, `Recompile`,
`Expand`) — the native path plugs in behind that surface and must not co-evolve with it.

**NC0 — Infrastructure.** Lifted-code registry format + extraction/diff tooling + drift-gate
panel (CDR-3), armed with its first entry (the `mjuu_*` unit) to prove the loop end to end;
NOTICE/attribution done once; mj_model_diff comparison core factored into a lib +
`ps_native_diff` three-way harness (harness agent ≠ compiler agent); `Compile(path)` gate +
`Compiled.report` plumbing (CDR-2), everything falling back; decision spike for Open Q2
(`mj_makeModel` lift vs own xmacro allocator), proven by an
allocate→`mj_copyModel`→`mj_deleteModel` round trip.
*Exit: corpus runs all-fallback with correct `path_taken`; registry gate green against the
current pin; allocation spike merged with its layout-compat test.*

**NC1 — Rigid-body pathfinder** (the shape-setter; single owner, like the M3 IO pathfinder).
Blocks (option/visual/statistic/size; presence → NaN-sentinel statistics); defaults via
`Effective()` (CDR-5); body tree with frames, joints, primitive geoms, sites, cameras, lights
(ledger 3.2-3.3 lifts: resolvers, checklimited, the body-compile chain, geom closed forms, BVH);
sizes/addresses (3.6 incl. `ComputeSparseSizes`); reference configuration; name/path tables +
hash (3.7); allocation + tree fill; post-build finalization calls (3.8); direct Binding + name
cross-check (CDR-4).
*Exit: the 39-model primitive slice bit-identical native vs XML path; pass-attribution harness
live; degree/radian and inertiafromgeom regression pins green; Binding cross-check green.*

**NC2 — Constraint + drive.** Contact pairs/excludes (combination rules + our sort), equality
(CDR-10 packer), tendons + wrap validation, actuators (CDR-11 lowering + inheritrange + muscle
checks + `mj_setLengthRange` loop + AutoSpringDamper), sensors (switch + datatype/needstage/dim
tables), keyframes, custom, user arrays.
*Exit: cumulative 136-model slice (~35%) bit-identical; lifted-function goldens for every
registry entry in the wave; recompile-equivalence third leg (CDR-6) green.*

**NC3 — Assets.** Mesh pipeline (parsers, weld, normals, qhull graph, polygons, inertia
variants, FitGeom, maxhullvert, octree), hfield, textures (builtins + PNG/KTX), materials,
skins; single-threaded (CDR-12).
*Exit: cumulative ~241 models (~62%) bit-identical — including `mesh_graph`/`mesh_poly*`
byte-equality and mesh-derived inertia bit-equality; per-stage mesh unit goldens in place.*

**NC4 — Macros as tree ops + plugins + compiler transforms.** Replicate/attach as ProtoSpec
clone + rename (CDR-7) with tier-2 pre-validation (upstream's silent drop becomes an error,
tested); plugins (registry calls + attribute flattening + state sizing); fusestatic/discardvisual
as pre-passes (CDR-8); SDF meshes.
*Exit: cumulative ~75-80% (gate measures); replicate/attach property tests (suffix numbering,
offset accumulation, keyframe cloning) green; fusestatic/discardvisual binding-report tests
green.*

**NC5 — Flex + flexcomp; composite decision.** Flex compile (elasticity kernels lifted, driver
ours); flexcomp generators emitting ProtoSpec elements (CDR-7); the composite decision (Open Q1)
made with corpus data.
*Exit: fallback list empty or exactly the documented composite set; ratchet at ~100% minus that
set.*

**NC6 — Native by default.** `NativePath` becomes `Compile()`'s default; XML path demoted to
oracle flag + `Expand()` writer; DR-11 `Recompile` rewired natively; threading decision revisited
with numbers; perf note (native vs XML-hop compile times over the corpus); plan.md amendments
(Section 8) landed.
*Exit: differential suite runs native-primary with XML as oracle; perf note committed.*

---

## 6. Risk register

| # | Risk | Owner | Teeth |
|---|---|---|---|
| N1 | Lifted code silently diverges from upstream across bumps | CDR-3 | registry snapshot diff fails the drift gate; bump PR blocked until re-reviewed; corpus goldens double-lock |
| N2 | Fill-surface omissions (a field nobody noticed) | §4.1-4.2 | mjxmacro full-field diff is exhaustive by construction; pass attribution names the culprit |
| N3 | Dof-tree/simple-body math subtly wrong on exotic topologies | ledger 3.6/3.7 | lifted cross-validation (`dof_Madr` vs sizes) + corpus + synthetic topology fixtures (tendon-armature demotion, slider promotion, static-ancestry trees) |
| N4 | Mesh numerics differ (qhull options, inertia variant selection, principal reorientation) | NC3 | byte-equality on `mesh_graph`; bit-equality on inertia; per-stage goldens harvested from the instrumented reference |
| N5 | Ordering-as-semantics missed (body-compile chain, id order, name-table order, adr cursors) | ledger 3.3/3.7 | orders lifted explicitly and documented per pass; the three-way diff surfaces any slip as an id/adr mismatch |
| N6 | Angle-unit regressions (per-consumer conversion) | ledger 3.3 | Q-ANGLE regression pair, both paths |
| N7 | Error-path divergence (a model compiles on one path only) | CDR-9 | parity contract over corpus + broken-model suite |
| N8 | Upstream restructures `src/user/` wholesale (more decoder-plugin extractions, file splits), orphaning registry anchors | CDR-3 | symbol-anchored extraction fails loudly → manual re-anchor at bump; a review cost, not a correctness hole |
| N9 | Scope creep: the fallback gate makes it painless to never finish | §5 | the ratchet is an exit criterion of every milestone; NC6 flips the default only when the fallback list is empty-or-composite |
| N10 | The `mj_makeModel` lift falls behind a new mjModel field | CDR-1, Open Q2 | the xmacro-driven lift picks up new fields structurally; allocate/copy/delete round-trip test + mjxmacro diff catch layout skew |

---

## 7. Open questions (genuinely undecidable today)

1. **Composite: lift `MakeCable` or permanent documented fallback?** 3 corpus files; the pin has
   already deprecated every other composite type in favor of replicate/flexcomp. Decide at NC5
   with usage data; the fallback set is documented either way.
2. **`mj_makeModel`: lift verbatim vs write our own xmacro-driven allocator?** Both are small
   thanks to the public `include/mujoco/mjxmacro.h`; the binding constraint is byte-layout
   compatibility with `mj_deleteModel`/`mj_copyModel`. Decided by the NC0 spike, not by argument.

---

## 8. Pending one-line amendments to plan.md (landed at NC0/NC6, not from this doc)

- STATUS row "Native compiler": repoint from "mjs_* walk ... NDR-1..15" to this plan
  (CDR-1..13, NC0-NC6, direct ProtoSpec→mjModel, no mjSpec anywhere in the native path).
- DR-5: the escape-hatch sentence ("a direct `mjs_*` walker can be added behind the same function
  signature") is superseded — the second path behind the signature is the native compiler, and it
  becomes the default at NC6.
- DR-10: add the CDR-4 note — native compiles construct the Binding directly at id-assignment
  time; name lookup remains the XML-path mechanism and the debug cross-check.
- Q-EQ: re-aim "(plan_native_compiler.md NDR-6)" at CDR-10.
- Q-ACT: note that the native path lifts the `mjs_setTo*` math per CDR-11 (the dormant
  `Lower()`/`Raise()` helpers and their differential tests against `mjs_setTo*` are unchanged —
  they now double as the lift's golden oracle).
