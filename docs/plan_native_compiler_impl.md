# ProtoSpec Native Compiler: Implementation Plan

Companion to `docs/plan_native_compiler.md` (the design: reuse ledger, CDR-1..14, NC0-NC6) and
`docs/plan.md` (conventions; Section 12 agent policy applies verbatim: all code by Opus
subagents, harnesses by a different agent than the code they test, sequential waves for shared
compiler files, adversarial review before any exit criterion is declared met). This document
turns the design into contracts and agent-sized tasks for NC0 and NC1, with an NC2 sketch. It
contains no code; signatures below are normative shapes, not compiled text.

CDR-14 (purity + separate Binding) is the architectural spine of everything here.

**Prerequisites** (design §5, restated): M3 IO waves cover the corpus (leg B arbitrates) and the
M5 bridge surface exists (`Compile` via XML, auto-name serials, `Recompile`, `Expand`). Where NC0
starts before M5 is fully landed, the XML leg inside `Compile` is the minimal
`WriteMjcf → mj_loadXML → name-based Binding` core that M5 formalizes — same signature, one
implementation, no fork.

---

## 1. Public API

One public header pair under `cpp/compile/` (layout in §3). `compile.h` forward-declares
`mjModel` (`struct mjModel_`) so no consumer transitively includes `mujoco.h`; `cpp/compile/` and
`cpp/bridge/` are jointly the MuJoCo quarantine zone (amends plan.md §3's dependency rule from
"bridge/ only" to "bridge/ + compile/ only").

```cpp
namespace ps::mjcf::compile {

enum class CompilePath { Auto, XmlPath, NativePath };
// Auto: native when the CDR-2 gate admits the whole model, else XML fallback.
// XmlPath: force the oracle path. NativePath: force native; unsupported
// features are a hard CompileError (no silent fallback) — the harness and
// ratchet use this to make fallbacks loud.

struct Diagnostic {                       // CDR-9, one type for all lifted code
  enum class Severity { Error, Warning };
  Severity severity;
  std::string pass;                       // stage name (S-table, §2) that raised it
  std::string message;
  ps::SourceLoc loc;                      // empty file for programmatic elements
  std::string Render() const;             // "file:line: [pass] message"
};

struct FallbackReason {                   // CDR-2
  std::string feature;                    // manifest feature key, e.g. "geom.mesh_ref"
  int count;
  ps::SourceLoc first;
};

struct CompileReport {
  CompilePath requested;
  CompilePath taken;                      // path_taken; the ratchet keys off this
  std::vector<FallbackReason> fallback_reasons;   // empty when taken == requested
  std::vector<Diagnostic> errors;
  std::vector<Diagnostic> warnings;       // surfaced, never swallowed (CDR-9c)
  bool ok() const;                        // errors.empty()
};

struct ModelDeleter { void operator()(mjModel* m) const; };  // mj_deleteModel, out-of-line

class Binding;                            // below

struct Compiled {
  std::unique_ptr<mjModel, ModelDeleter> model;   // null on failure
  Binding binding;                        // empty on failure
  CompileReport report;
};

Compiled Compile(const Model& m, CompilePath path = CompilePath::Auto);

}  // namespace ps::mjcf::compile
```

Errors: `Compile` never throws. Failure = `model == nullptr` + at least one error diagnostic in
the report. The CDR-9 parity contract (fails native ⇔ fails XML) is tested over the corpus and
the broken-model suite; pass names in diagnostics satisfy CDR-9(d).

### Binding (CDR-14, reconciling DR-10 and CDR-4)

```cpp
class Binding {
 public:
  // Identity lookup. E is any generated element type; the E -> mjtObj mapping
  // is a compile-time trait table (§3). Returns empty for elements compiled
  // away (discardvisual/fusestatic), with the responsible flag recorded in the
  // report — exactly DR-10's contract.
  template <class E> std::optional<int> Id(const E& elem) const;

  // Reverse: model id -> tree element (typed accessors per family).
  const Joint*  JointAt(int id) const;    // ... GeomAt, BodyAt, SiteAt, etc.

  // Address sugar over the id maps.
  std::optional<int> QposAdr(const Joint&) const;   // jnt_qposadr
  std::optional<int> DofAdr(const Joint&) const;    // jnt_dofadr
  std::optional<int> SensorAdr(const SensorAny&) const;
  std::optional<int> ActId(const ActuatorAny&) const;
  std::optional<int> ActAdr(const ActuatorAny&) const;

  // Pattern queries for macro-generated elements (XML path; DR-10).
  std::vector<int> Find(int objtype, std::string_view glob) const;

  bool valid() const;                     // mutation counter still matches
};
```

Storage per entry: `{const void* elem, uint64_t serial, mjtObj type, int id}` — the pointer is
the lookup key, the serial is recorded for the ABA assert (§2). Binding also stores
`const Model*` and the Model root's mutation-counter value observed at compile time.

**Staleness mechanism, made concrete.** The generated `Model` root gains one field,
`std::uint64_t mutations = 0` (emitter change, NC0 task T0.3 — the DR-10 "cheap edit counter",
now a hard prerequisite rather than an M5 nicety). Contract: every SDK mutator (builders, attach,
rename, delete, `ApplyDefault` on tree elements, any future setter surface) bumps it; readers
never do; **Compile reads it and never bumps it** (purity). Binding accessors assert
`model->mutations == stored` and the `serial` recorded per entry as a second line of defense.
Documented residual: raw field pokes on the structs bypass the counter (they bypass every
invariant; the SDK is the supported mutation surface).

Two constructors, one interface: the native path fills the maps directly at id-assignment time
(stage S10, CDR-4); the XML path fills the same maps via `mj_name2id` over authored +
auto-generated (`_ps:<family>:<serial>`) names. Consumers cannot tell which path built it except
via `report.taken`.

### Thread safety (pure-function corollary)

Concurrent `Compile` calls on the same `const Model&` are safe. What that demands, itemized:

| Demand | Mechanism |
|---|---|
| No compiler globals | all state lives in a stack-owned `CompileContext`; a static/global in `cpp/compile/` is a review-rejected defect (same grep gate as `const_cast`, §5) |
| No lazy caches on elements | CDR-14 purity; the memoization that would tempt one (effective defaults) lives in the context (§2) |
| Serial counter | `ps::detail::next_serial()` is already a process-global atomic; moreover Compile never constructs ProtoSpec elements (internal shadow lists are plain compiler structs, §2), so it never touches the counter |
| Lifted code statelessness | `mjuu_*` and the ledger's B/C lifts are stateless over plain args (verified per lift; a static local in lifted code fails review) |
| MuJoCo public API | `mj_makeData`/`mj_setConst`/... operate on distinct objects per call — safe; the plugin registry (`mjp_*`) is process-global and read-mostly: plugin registration must complete before concurrent compiles (MuJoCo's own rule, documented in `compile.h`) |
| qhull (NC3) | lift targets the reentrant `libqhull_r` exactly as upstream `MakeGraph` does |
| Caller contract | the tree must not be mutated concurrently with Compile — ordinary C++ const semantics, stated in the header |

---

## 2. CompileContext internal architecture

### Identity keys: pointers, with serials recorded — decision and justification

Candidates: element pointer (stable per DR-2 — children are `unique_ptr`, identity survives
sibling mutation) vs creation serial (`types.h` stamps `serial = ps::detail::next_serial()` on
every element; process-unique, never reused). Neither survives `Clone` — core.h explicitly mints
fresh serials on Clone — so neither buys cross-Clone identity; a Binding is bound to the tree
instance it was compiled from, by design (DR-11 migrates state through same-instance keys).

**Decision: side tables are pointer-keyed for the duration of the compile; Binding entries
additionally record the serial.**

- During compile the tree is const and single-instance: pointers are unique, stable, and free to
  obtain (no hash of a lookup key that must itself be found). Serial-keyed tables would need a
  serial→element index built from a full walk anyway — pure overhead inside the compile.
- Across the Binding's longer lifetime, pointer reuse is possible (delete element, allocate a new
  one at the same address). The mutation counter catches every SDK-visible edit first; the
  recorded serial is the ABA backstop (asserted on every accessor) and doubles as the stable,
  loggable identity in diagnostics and in DR-11 state-migration maps.

### Side-table shape: slot vectors, not per-element hash maps

Stage S1 (Collect) walks the tree once and builds per-family **ordered pointer lists**
(`std::vector<const Geom*>`, document order — which is id order until the NC4 transforms filter
them). Every later per-element quantity is a parallel vector indexed by that slot; one
`unordered_map<const void*, Slot>` per family (built during Collect) serves ref resolution and
Binding fill. Consequences: linear, cache-friendly memory for 10^5-element models; id assignment
after transforms is "slot index within the filtered list" — CDR-4 falls out of the data
structure.

`CompileContext` owns (grown milestone by milestone):

| Member group | Contents | Producing stage |
|---|---|---|
| lists | per-family `vector<const T*>` + slot maps; post-NC4 also the transform-filtered lists | S1, S4 |
| effective | resolved-default family values (memoized per class chain, overlay per element — see memory note below) | S3 |
| resolved | orientation quats, geom pose/size from fromto, principal inertias, tri-state limit resolutions | S5 |
| tree | per-body compiled state: masses/inertias after inference, frame-flattened poses, BVH arrays, per-joint dof records | S7 |
| assets | compiled mesh/hfield/texture products (NC3) | S6 |
| lowered | pair combinations, `eq_data[11]` packs, actuator gain/bias/dyn prm, sensor dims | S8 |
| sizes | the full n* census struct + sparse sizes (nM/nB/nC/nD) + address cursors (qposadr/dofadr/actadr/sensor_adr) | S9 |
| names | interned name blob, `name_*adr`, hash table, path table | S10 |
| binding | the maps handed to `Binding` at the end | S10 |
| diag | diagnostics sink (drained into the report) | all |

**Memory note (risk R-I2).** The one fat table is `effective`: a fully-merged family struct per
element would be ~1-2 KB × n. Instead the class-chain merge (`element-class → ancestors → main →
IDL defaults`) is memoized **per `Default*` chain** (dozens of classes, not 10^5 elements), and
the element's own authored-field overlay is applied into a reusable scratch value inside the
consuming stage. Per-element storage holds only what later stages actually need (resolved pose,
inertia, sizes), not whole merged structs.

**Shadow lists, not shadow trees.** Where upstream mutates (fusestatic folds bodies, keyframe
compile resizes vectors, `mjCGeom` writes resolved size back), we keep compiler-plain structs in
the context: NC4 transforms produce filtered/adjusted *lists + per-slot pose fixups*; keyframe
padding materializes directly into `m->key_*` at fill time. No ProtoSpec elements are ever
constructed inside Compile.

### The pass pipeline

`NativeCompile` is a fixed sequence of stages, each `Stage(const Model&, CompileContext&)` —
pure with respect to the Model, mutating only the context (and, from S11 on, the mjModel under
construction). Stage order mirrors `mjCModel::TryCompile` except where CDR-8 moves transforms
earlier. Mapping to the design's reuse-ledger rows:

| # | Stage | Ledger rows | Lands |
|---|---|---|---|
| S0 | Gate scan against `native_supported.json`; route or record fallback | CDR-2 | NC0 |
| S1 | Collect: single tree walk → per-family lists + slot maps | 3.1 (id precursor) | NC1 |
| S2 | Compile validation: tiers 1-2 mandatory + 3.1 guards (joint-in-world, counts, duplicate names, moving-body mass) | 3.1, 3.6 guard | NC1 |
| S3 | Effective defaults: generated presence merge over class chains | CDR-5 | NC1 |
| S4 | Tree transforms: discardvisual filter, fusestatic fold (identity pass until NC4) | CDR-8, 3.1 | NC4 |
| S5 | Resolvers: orientation, fromto, full→principal inertia, tri-state limits, `SetNuser`, hull-needed marking | 3.2, 3.1 | NC1 |
| S6 | Asset compile: mesh pipeline, hfield, textures, materials, skins | 3.4 | NC3 |
| S7 | Body-tree compile: the `mjCBody::Compile` ordering chain, inertia inference, joint compile (Q-ANGLE sites), geom closed forms, site/camera/light, BVH, reference configuration | 3.3 | NC1 |
| S8 | Constraint/drive lowering: pair combination + our sort, exclude packing, equality packer (CDR-10), tendon path validation, actuator lowering (CDR-11), sensor switch + tables | 3.5 | NC2 |
| S9 | Sizes + addresses: n* census, `ComputeSparseSizes` lift, dof/act/sensor cursors, nmocap | 3.6 | NC1 |
| S10 | Id assignment over (filtered) lists + name blob/hash/path tables + Binding map fill | CDR-4, 3.7 names | NC1 |
| S11 | Allocate (`mj_makeModel` lift or own allocator, Open Q2) + tree fill + object fill + simple-dof finalization + signature + narena | 3.7 | NC1 (tree) / NC2 (objects) |
| S12 | Post-build finalization: sparsity, `mj_setConst`, spring-damper, length range, settotalmass, `mj_validateReferences`; keyframe fill against the built model | 3.8, 3.5 keyframe | NC1 (+NC2 keyframes/LR) |
| S13 | Smoke: `mj_makeData` + one `mj_step` | 3.8 | NC1 |

Lifted code plugs in as free functions under `cpp/compile/lifted/` (one file per registry entry
or tight group), called from stages; stages own plumbing, lifts own algorithms — the ledger's
B/C split made physical.

### Purity enforcement mechanics (CDR-14, testable)

1. **Deep-compare gate**: before Compile, `Clone` the model and snapshot, via a `Visit` walk,
   every element's `(serial, loc, child-list sizes)` (generated `operator==` deliberately
   excludes `serial` and `loc`, so the sweep covers what equality does not). After Compile:
   `*model == *clone` and the sweep re-matches exactly. Runs as a standing ctest over fixtures +
   a corpus slice, and `ps_native_diff` re-runs it per model in CI mode.
2. **Const end to end**: stages receive `const Model&`; there is no non-const path to the tree
   inside `cpp/compile/`.
3. **Grep gate** (pytest): `const_cast`, `mutable`, and namespace-scope mutable state in
   `cpp/compile/**` fail the suite — the mechanical tripwire ahead of the review-rejection rule.
4. **Filesystem caveat**: asset reads go through `mju_openResource` (ledger 3.4/A) only from S6;
   documented in `compile.h` as the one impurity (w.r.t. the filesystem, never the model).

---

## 3. Module/file layout and build

```
cpp/compile/
  compile.h                     public surface (§1); fwd-declares mjModel
  compile.cc                    Compile() driver: gate -> native | xml fallback
  report.h                      Diagnostic, FallbackReason, CompileReport
  binding.h / binding.cc        Binding, both constructors; objtype trait table
  gate.h / gate.cc              S0 manifest scan
  native_supported.json         CDR-2 feature manifest (relocated here from the
                                design sketch's bridge/ path; compile/ owns it)
  context.h                     CompileContext, Slot, per-family list types
  stages/
    collect.cc  validate.cc  effective.cc  resolve.cc  body_tree.cc
    sizes.cc    names.cc     alloc_fill_tree.cc  finalize.cc
    (NC2+) constraints.cc actuators.cc sensors.cc fill_objects.cc keyframe.cc
  lifted/
    mjuu_util.h/.cc             first registry entry (the whole mjuu_* unit)
    make_model.cc               Open Q2 spike outcome
    ...                         one file per registry entry/group, provenance header each
cpp/harness/
  model_diff_lib.h/.cc          comparison core factored out of mj_model_diff
  field_attribution.json        mjxmacro field -> owning stage/ledger row (§4.2 of design)
  mj_model_diff.cc              thin CLI over the lib (behavior unchanged)
  ps_native_diff.cc             three-way harness: legs A/B/C in-process, JSON verdict
tools/lift_registry.py          registry tooling (§4)
snapshots/lifted_code.json      the registry
snapshots/lifted_upstream/      exact upstream text per symbol at lift time
tests/test_native_differential.py
tests/test_lifted_drift.py
cpp/test/test_compile_purity.cc
cpp/test/test_compile_units.cc  per-stage/lifted-function goldens (grows per wave)
tests/golden/native/            harvested intermediate goldens (§5)
```

**CMake additions** (root `cpp/CMakeLists.txt`): the MUJOCO_ROOT discovery block moves from
`harness/CMakeLists.txt` to the top level (same cache variable, same layout expectations:
`include/`, `build/lib/Release/mujoco.lib`, `build/bin/Release/mujoco.dll`, DLL copied next to
exes). New targets, all gated on MuJoCo being found so the MuJoCo-free core still builds without
it:

| Target | Kind | Links |
|---|---|---|
| `protospec_compile` | static lib | `protospec`, `protospec_io` (XML fallback), MuJoCo |
| `model_diff_lib` | static lib (harness) | MuJoCo |
| `ps_native_diff` | exe | `protospec_compile`, `model_diff_lib` |
| `protospec_compile_tests` | exe + ctest | `protospec_compile`, `model_diff_lib` |

`/W4 /permissive-` like every project target; lifted files included (they are ours now — warnings
fixed at lift time, noted in the registry entry when nontrivial).

### Generated vs handwritten (the class-merge decision)

Checked against the current tree: the generated defaults surface is `ApplyDefault(T&)`
(`cpp/generated/defaults.h`) — it **mutates an element in place** with IDL defaults. Usable by
the compiler only on scratch values, never on tree elements (purity). There is **no generated
presence-merge today**, and no `core/resolve.cc` yet either (plan.md §3 lists it; nothing under
`cpp/core/` exists — the resolvers land with T1.2).

**Decision: the class merge is generated, per family — a new emitter output `merge.h/.cc` with
`MergeUnset(T& dst, const T& src)`** (per field: `if (!dst.f) dst.f = src.f;`), same shape and
ownership as the existing `ApplyDefault` emitter. Rejected alternative: reflect-driven runtime
merge — it needs a generic typed accessor per field kind (opt/InlineVec/variant/ref/array),
which is a new reflection capability with real surface area, to save one small emitter that
follows an existing pattern; the generated form is also trivially reviewable against the IDL.
Composition (handwritten, `stages/effective.cc`): scratch = default-constructed family struct →
`MergeUnset` from the element → `MergeUnset` up the class chain (element class or nearest
childclass → ancestors → `main`) → `ApplyDefault` last (IDL defaults) — first authored value
wins, all on scratch, memoized per class chain (§2).

Also generated vs handwritten, for the record: the Binding element→`mjtObj` trait table is
handwritten in `binding.cc` (small, stable); `field_attribution.json` is handwritten data owned
by the harness agent; everything in `stages/` and `lifted/` is handwritten per the ledger.

The `Model.mutations` counter (§1) is one more emitter touch: emitted on the root element only.
Regeneration ripples into `test_emit` goldens — regenerate in the same commit; mind the
codegen-gate CRLF note below.

---

## 4. Lifted-code registry mechanics (CDR-3, concretized)

Paths here supersede the design sketch's `tools/lifted/registry.json` naming; mechanism
unchanged.

**`snapshots/lifted_code.json`** — one entry per lifted symbol (or tight group):

```json
{ "symbol":   "mjCModel::ComputeSparseSizes",
  "upstream": "src/user/user_model.cc",
  "lines":    "906-1119",
  "snapshot": "snapshots/lifted_upstream/ComputeSparseSizes.cc",
  "sha256":   "<hash of LF-normalized snapshot content>",
  "ours":     "cpp/compile/stages/sizes.cc",
  "lifted":   "2026-07-XX",
  "notes":    "plumbing retargeted to CompileContext lists; algorithm verbatim" }
```

`lines` is a human reference only; extraction anchors on **symbol signature + brace matching**
(lines shift every bump). Hashes and snapshots are LF-normalized, and
`snapshots/lifted_upstream/*` gets a `.gitattributes` `-text` entry — otherwise Windows autocrlf
produces false drift-gate failures (already bitten by this on the codegen gate).

**`tools/lift_registry.py`** (Python, cross-platform, no shell wrappers needed):

| Subcommand | Behavior |
|---|---|
| `add --symbol S --upstream F --ours P [--notes N]` | extract S from the vendored tree by signature + brace matching, write the snapshot, hash it, append the entry |
| `check [--mujoco-root R]` | re-extract every registered symbol from the vendored tree, diff against stored snapshots; nonzero exit lists divergent symbols with unified diffs attached |
| `verify-headers` | every file under `cpp/compile/lifted/` carries the provenance header and appears in the registry, and vice versa |
| `list` | table of entries |

**Drift gate**: `tests/test_lifted_drift.py` runs `check` + `verify-headers` against the vendored
tree (located via the same MUJOCO_ROOT/env convention as `test_differential.py`). It is part of
the normal suite, so it also fires on every MuJoCo bump (the DR-12 gate's lifted-code panel). A
divergence means: re-lift or document intentional lag in `notes`, refresh the snapshot — reviewer
decision, never silent.

**Lifting procedure (agent checklist, binding for every lift task):**

1. `lift_registry.py add ...` — snapshot first, so the diff-vs-upstream review is possible.
2. Copy the code into `ours` under a provenance header: upstream path, symbol, pin version
   (`mjVERSION_HEADER 3010000`), Apache-2.0 attribution (NOTICE updated once at NC0).
3. Adapt plumbing ONLY: includes; `mjCError`/throw sites → `Diagnostic`; mjC* field reads →
   plain args / CompileContext reads. Algorithm, constants, and iteration order stay verbatim;
   any semantic deviation goes in `notes` and gets its own test.
4. Golden-test: IO pairs against the original running in a test binary (`mjs_*` and internal
   observation allowed there — oracles, not product; CDR-1 applies to shipped code only).
5. One commit per registry entry (or tight group), snapshot included, so review is
   "diff against upstream", not "read 500 new lines cold".

---

## 5. Test scaffolding (harness-first, different author than the compiler)

Build order — all of this exists and is green **before** the first compiler stage lands:

1. **Three-way differential: `tests/test_native_differential.py` + `ps_native_diff`.** New file;
   `test_differential.py` (legs A/B) stays untouched and green. `ps_native_diff <model.xml>`
   produces legs A (`mj_loadXML(original)`), B (ProtoSpec→XML→load), C (native) in-process and
   emits a JSON verdict: `{path_taken, fallback_reasons, identical_ab, identical_ac, diffs[],
   pass_attribution[]}` — each diff annotated with the owning stage via
   `field_attribution.json`. The pytest driver reuses `test_differential.py`'s conventions
   (corpus discovery, `_find_binary`, supported-tag prefilter, dotfile-in-source-dir asset
   strategy, size cap). **Ratchet, self-arming**: for every corpus model whose feature scan is
   within `native_supported.json`, assert `path_taken == native` AND leg C bit-identical to
   leg B (exit bar per design §4.1: bit-identical, no tolerance); models outside the manifest
   must report `path_taken == xml` with non-empty `fallback_reasons`. Growing the manifest IS
   moving the ratchet.
2. **Purity gate: `cpp/test/test_compile_purity.cc`** (mechanics in §2). Fixtures + corpus
   slice; runs both `Auto` and forced paths. Plus the pytest grep gate over `cpp/compile/**`
   (`const_cast`, `mutable`, namespace-scope state).
3. **Per-stage unit goldens — harvest mechanism, concretely.** Two sources:
   - *Final-field attribution* (most stages): the stage's outputs ARE mjModel fields; the
     three-way diff plus `field_attribution.json` already names the guilty stage. No extra
     harvest.
   - *True intermediates* (per-class resolved defaults, per-geom mass pre-accumulation,
     equality `data[11]`, actuator prm vectors, hull graph pre-copy): harvested by
     `ps_golden_harvest`, a test-only binary (never shipped) that loads corpus slices through
     MuJoCo's own surfaces and dumps JSON to `tests/golden/native/<slice>/`:
     `mj_parseXML` + `mjs_getSpec`/`mjs_firstElement`/`mjs_getDefault` walks for resolved
     defaults and parsed `mjsEquality.data`; `mjs_setTo*` on scratch specs for actuator
     lowering; `mj_loadXML` + direct `m->mesh_*`/`m->qpos0`/`m->dof_*` reads for everything
     that is a final field consumed mid-pipeline. Goldens are checked in; a
     `tools/harvest_goldens.py` driver regenerates them (rerun documented per MuJoCo bump).
     `mjs_*` in test binaries is explicitly legitimate (design §4.3).
4. **Lifted-function IO-pair goldens**: per registry entry, a `test_compile_units.cc` case
   comparing the lifted function against its oracle — in-binary `mjs_*`/public API where
   callable, checked-in harvested pairs where not. The dormant `Lower()/Raise()` differential
   tests double as the CDR-11 oracle per design §8.
5. **Regression pins** (design §4.5) land with their owning waves; the CDR-9 parity run
   (broken-model suite through both paths) lands with T1.7.

Authorship: items 1-3 by the harness agent(s); item 4's cases by the same agent that lifts (the
oracle makes self-testing safe), reviewed adversarially; item 5 split per wave.

---

## 6. Work breakdown: NC0, NC1, NC2 sketch

Conventions: Opus for all code; **worktree-parallel only when file sets are disjoint**; anything
touching `context.h`/`stages/`/`compile.cc` is sequential (the IO-waves rule). Every task ends
with build + full test run before commit; adversarial review before a milestone exit is declared.

### NC0 — scaffolding

| Task | Owner files | Depends | Parallel? | Exit criteria |
|---|---|---|---|---|
| T0.1 registry tooling | `tools/lift_registry.py`, `snapshots/lifted_code.json`, `snapshots/lifted_upstream/`, `tests/test_lifted_drift.py`, `.gitattributes` entry, NOTICE, first entry: `cpp/compile/lifted/mjuu_util.*` | — | yes (worktree) | `check` green vs pin; `verify-headers` green; mjuu smoke goldens (hand cases + `mju_*` analogues) pass |
| T0.2 harness factoring + three-way | `cpp/harness/model_diff_lib.*`, `mj_model_diff.cc` (thin-out), `ps_native_diff.cc`, `field_attribution.json` (skeleton), `tests/test_native_differential.py` | T0.3 header shape (compile.h signature only) | yes (worktree; different author than T0.3/T0.4 compiler code) | `mj_model_diff` behavior unchanged (existing suite green); corpus runs all-fallback with `path_taken == xml` everywhere |
| T0.3 API skeleton + gate + counter | `cpp/compile/{compile.h,compile.cc,report.h,binding.h,binding.cc,gate.h,gate.cc,context.h,native_supported.json}`, CMake edits, emitter change adding `Model.mutations` + regeneration | — | no (defines shared files) | `NativePath` returns an UnsupportedNatively error for every model; `Auto` falls back through the XML core with correct report; XML-path Binding functional (name-based, auto-name serials); counter emitted + `test_emit` goldens regenerated |
| T0.4 purity gate | `cpp/test/test_compile_purity.cc`, pytest grep gate | T0.3 | yes after T0.3 (different author) | gate green on the all-fallback compiler; deliberately-mutating fixture (test double) proves the gate fails when it should |
| T0.5 allocation spike (Open Q2) | `cpp/compile/lifted/make_model.*` (or own allocator note), round-trip test in `test_compile_units.cc`, registry entry if lifted | T0.1, T0.3 | yes (worktree) | allocate → `mj_copyModel` → `mj_deleteModel` round trip green; decision recorded in the design plan's Open Q2 |

Order: T0.3 first (one agent), then T0.1/T0.2/T0.4/T0.5 fan out. NC0 exit = design plan's NC0
exit bar, plus the purity gate standing.

### NC1 — rigid-body pathfinder (sequential waves, single owner each)

| Task | Scope (stages / ledger) | Owner files | Exit criteria |
|---|---|---|---|
| T1.1 collect + effective defaults | S1, S3; CDR-5; emitter `merge.h/.cc` (`MergeUnset`) | `stages/collect.cc`, `stages/effective.cc`, `context.h`, emitter + regen | per-class resolved-default goldens (harvested via `mjs_getDefault`) match over the defaults-exercising corpus slice; memoized class-chain merge in place |
| T1.2 resolvers | S5; ledger 3.2 lifts: ResolveOrientation, fromto, fullInertia/eig3, checklimited | `cpp/core/resolve.*` (NEW — the single implementation plan.md §3 lists but which does not exist yet; MuJoCo-free, shared by SDK queries and compiler), `stages/resolve.cc`, `lifted/` entries | lifted-fn goldens vs `mjs_resolveOrientation` + hand cases green; registry entries committed one-per-lift |
| T1.3 body tree | S7; ledger 3.3: body-compile ordering chain, InertiaFromGeom + bound/balance, joint compile (Q-ANGLE), geom closed forms, site/camera/light, BVH, reference configuration | `stages/body_tree.cc`, `lifted/` entries | stage goldens (per-geom mass, qpos0, bvh arrays via attribution) green on fixtures; splits into T1.3a (joints/geoms/sites) then T1.3b (inertia inference + BVH) if one agent-session cannot hold it — still sequential |
| T1.4 sizes + addresses | S9; ledger 3.6: census, `ComputeSparseSizes` lift + its cross-validation, cursors | `stages/sizes.cc`, `lifted/compute_sparse_sizes.cc` | synthetic-topology fixtures (slider promotion, static-ancestry trees) green vs harvested sizes |
| T1.5 ids + names + Binding | S10; CDR-4; ledger 3.7 name tables + `mj_hashString` + path table | `stages/names.cc`, `binding.cc` (native constructor), `lifted/` entries | `mj_name2id(m, type, name) == binding.Id(elem)` for every named element over the slice; name tables byte-exact vs leg B |
| T1.6 alloc + fill + finalize | S11-S13; blocks (option/visual/statistic presence→sentinels), CopyTree topology fields, signature, narena, post-build A-calls, smoke; flip manifest features on (blocks, defaults, body tree, joints, primitive geoms, sites, cameras, lights) | `stages/alloc_fill_tree.cc`, `stages/finalize.cc`, `compile.cc`, `native_supported.json`, `lifted/` entries | **NC1 exit bar**: 39-model primitive slice `path_taken == native` and bit-identical vs leg B |
| T1.7 verification wave (different author) | fill `field_attribution.json` for NC1 fields; regression pins: degree/radian pair, inertiafromgeom×explicitinertial matrix, statistics presence; CDR-9 parity over broken-model suite; purity gate over the native slice | harness + tests files only | all pins green; ratchet asserts 39; adversarial review notes resolved |

T1.1-T1.6 are strictly sequential (shared `context.h`/stages). T1.7 runs in a worktree in
parallel with T1.5/T1.6 (disjoint files) but its exit is the milestone gate. Golden harvests
(`ps_golden_harvest` slices) for each wave are produced by the harness author before the wave
starts — harness-first, per wave.

### NC2 — sketch (constraint + drive)

Sequential waves, same shape: T2.1 contact pairs/excludes (combination lift + our sort), T2.2
equality (CDR-10 packer), T2.3 tendons + wrap validation, T2.4 actuators (CDR-11 lifts,
inheritrange, actdim, muscle checks, `mj_setLengthRange` loop, AutoSpringDamper), T2.5 sensors
(switch + datatype/needstage/dim tables), T2.6 keyframes/custom/user arrays + `fill_objects.cc`
(S8 + S11 object half), T2.7 verification wave incl. the CDR-6 recompile-equivalence leg.
Exit: cumulative 136-model slice bit-identical; a lifted-function golden for every registry
entry in the milestone.

---

## 7. Implementation risks (new; the design risk register N1-N10 is not repeated)

| # | Risk | Mitigation |
|---|---|---|
| I1 | Identity ABA: Binding pointer keys aliased after delete-then-allocate SDK edits | mutation counter asserts first; per-entry serial asserts second; residual (raw field pokes bypass the counter) documented in `binding.h` — the SDK is the supported mutation surface |
| I2 | Side-table memory blowup on 10^5-element models (a merged family struct per element ≈ hundreds of MB) | class-chain merges memoized per `Default*` chain, element overlays computed into reusable scratch in the consuming stage; per-element storage limited to slot vectors of the quantities later stages read (§2) |
| I3 | CompileReport becomes a de-facto plugin API and then breaks | declared append-only from NC0: fields are never removed/renamed, enum values never renumbered, `Render()` text is non-contractual; JSON emission (via `ps_native_diff`) carries a schema version field |
| I4 | `Model.mutations` emitter change churns generated goldens and trips the codegen drift gate on Windows CRLF | regenerate in the same commit as the emitter change; known autocrlf false-failure: run the generator once before judging the gate, don't stage EOL churn |
| I5 | Purity pressure: stages want to memoize on elements (effective defaults, resolved quats) as the compiler grows | the memoization home is the context by construction (§2); grep gate + review rule are the backstop; the purity test runs per corpus model in `ps_native_diff`, so a violation cannot land quietly |
| I6 | Resolver duplication: plan.md's `core/resolve.cc` does not exist yet; a compiler-local resolver would create the second implementation the design forbids | T1.2 creates `cpp/core/resolve.*` as the single MuJoCo-free implementation; `stages/resolve.cc` is plumbing over it; SDK queries adopt the same module later |
| I7 | Concurrent-compile hazards hiding in dependencies (plugin registry writes, qhull state) | registration-before-concurrency rule documented in `compile.h`; NC3 lifts target `libqhull_r`; a two-thread same-model compile smoke test lands with T1.6 |
| I8 | The XML fallback inside `Compile` and the M5 bridge drift apart while M5 is unfinished | one implementation behind one signature from T0.3; M5 formalizes, never forks; `test_differential.py` (leg A/B) keeps arbitrating it independently |
| I9 | Slot/id skew once NC4 transforms filter lists (off-by-one between slot maps and post-filter ids) | ids are computed from the filtered lists only (never from slots); S4 is an identity pass until NC4 so NC1-NC3 cannot hide the bug; NC4 lands with binding-report tests for removed elements |
| I10 | `ps_golden_harvest` goldens go stale across MuJoCo bumps and mask drift | goldens carry the pin version; `test_lifted_drift.py` failing on a bump triggers the documented reharvest step in the bump procedure |
