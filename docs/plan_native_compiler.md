# ProtoSpec Native Compiler Plan: mjs_* walker behind Compile()

Companion to `docs/plan.md` (the main plan) and `docs/native_compiler_survey.md` (the evidence
base — all survey citations below are to that document's sections; MuJoCo source citations reuse
its conventions). This document exercises the escape hatch DR-5 reserved: "a direct `mjs_*` walker
can be added behind the same function signature."

## STATUS (living section — update on every milestone commit)

Last updated: 2026-07-07.

| Milestone | State | Evidence |
|---|---|---|
| N0 harness factoring + walker skeleton | queued | |
| N1 pathfinder families (defaults, body tree, assets) | queued | |
| N2 constraint + drive families | queued | |
| N3 macros native (attach/replicate/flexcomp; composite decision) | queued | |
| N4 default switch + upstream readiness | queued | |

---

## 1. Goal and non-goals

**Goal.** A second implementation of the compile boundary: ProtoSpec tree → `mjSpec` via the
`mjs_*` C API → `mj_compile`, eliminating the XML serialize/parse hop of DR-5. It lives entirely
behind the existing signature:

```
Compile(const Model&, CompilePath path = CompilePath::XmlPath) -> Compiled
enum class CompilePath { XmlPath, NativePath };
```

Selectable per call. The walker is the only new code; every compiler pass — mesh/qhull, BVH,
octree, inertia, defaults resolution, lengthrange, fusestatic, keyframe sizing — is inherited
unchanged through `mj_compile` (survey §3.1).

**Non-goals, permanent:**

- **The XML path is never removed.** It remains the reference implementation, the differential
  oracle (leg B of the three-way harness, §5), the fallback for unsupported families, and the
  writer half of `Expand()`.
- No `mjSpec → ProtoSpec` importer (unchanged from plan.md Section 6 non-goals).
- No persistent mjSpec. The walker builds a fresh spec per `Compile()` call and discards it after
  `mj_compile` (see NDR-10). ProtoSpec's tree remains the single source of truth; we do not adopt
  mjSpec's tri-storage lifecycle.
- No exposure of any `mjs_*` type outside `bridge/`. The dependency rule (plan.md Section 3)
  stands: `bridge/` is the quarantine zone.

**Why bother** (the honest case): removes the serialize/parse hop from `Compile()` and
`Recompile()` (DR-11's budgeted cost drops), moves duplicate-name and several structural errors
from compile-time to decode-time with SourceLoc attached (survey §4.5), enables in-memory mesh /
texture / hfield buffers without mjVFS registration (survey §2 asset rows), and positions us for
the upstream attach conflict-policy work (survey §1.2).

---

## 2. Decision records

Numbered NDR-n to avoid collision with plan.md's DR-n. Each traces to a survey finding.

**NDR-1: Per-model path gate; fallback is loud, never silent.** `Compile(m, NativePath)` first
runs a pre-scan of the tree against the native-supported family set (a manifest,
`bridge/native_supported.json`, mirroring `cpp/io/supported.json`'s role in the differential
harness — survey §5.5). If any element family is outside the set, the whole model routes through
the XML path. Mechanics and reporting:

- `Compiled` gains `report.path_taken : CompilePath` and
  `report.fallback_reasons : vector<{family, count, SourceLoc first}>` — populated on every call,
  empty when native ran. Requesting `NativePath` and getting XML is visible in the return value,
  in one log line, and assertable in tests. There is no code path where the caller cannot tell
  which compiler ran.
- The gate is per-model, not per-element: no half-native models. One model = one spec-building
  strategy, so a diff failure always indicts exactly one walker.
- The differential suite keys off `path_taken` so the corpus ratchet (§5) counts only genuinely
  native compiles.

**NDR-2: Deep-copy always.** `mjs_setDeepCopy(spec, 1)` immediately after `mj_makeSpec()`, and
again defensively before every `mjs_attach` in native `Expand()`. Rationale (survey §1.1):
shallow attach — the `mj_makeSpec` default — mutates the source child in place (re-parents,
prefixes names), refcounts it alive, and marks it attached, after which compiling the source
throws "cannot compile child spec if attached by reference". ProtoSpec-owned sub-specs (attach
children, replicate templates) must remain untouched and independently compilable. Deep copy costs
extra allocation at authoring scale (10^3 elements) — irrelevant. Note the XML reader itself
forces deep copy while parsing `<worldbody>` (survey §1.1), so this also matches XML-path
semantics exactly.

**NDR-3: `compiler.degree` is an explicit first-write from the ProtoSpec compiler block.**
Survey §4.2's top silent-failure risk (R1): `mj_makeSpec()` defaults `compiler.degree = 1`
(degrees), and euler/axisangle/joint-range/ref conversion happens at compile against that flag.
Since the Q-ANGLE amendment (M3 wave 1), ProtoSpec is form-preserving: angles are stored exactly
as authored and the model's `angle` unit round-trips verbatim — there is no radians-always
invariant. The walker therefore:

- writes `spec->compiler.degree` from the ProtoSpec compiler block's authored angle unit (or the
  MJCF default, degrees, when unauthored) as its **first** compiler-block write, never trusting
  the `mj_makeSpec` default — the default happening to match degrees-authored models is exactly
  how a radians-authored model corrupts silently by 57.3x;
- copies `eulerseq` verbatim;
- passes all angle-carrying values (euler, axisangle, joint range/ref/springref) exactly as
  authored — MuJoCo's compile performs the same per-consumer conversion it performs for the XML
  path, so both paths run identical conversion code on identical inputs.

Guard: a dedicated regression pair — one euler-authored model in degrees, one in radians — diffed
native vs XML, plus a walker assertion that the compiler block is decoded before any posed
element.

**NDR-4: Defaults-first, strict two-pass decode.** Survey §2 defaults row + R2: class values are
copied **eagerly** at element construction (`mjs_addGeom(body, def)` copies the whole family
struct), later class edits do not propagate, and `mjs_setDefault(elem, def)` after creation sets
only the class *label*, no values. Therefore:

- **Pass 1**: create the entire default-class tree via `mjs_addDefault(s, name, parent)`
  (parent before child — a new class starts as a copy of its parent), then write every authored
  class field by mutating the returned `mjsDefault`'s per-family pointers. The tree is complete
  and frozen before pass 2 begins.
- **Pass 2**: create elements, passing the resolved class pointer
  (`mjs_findDefault`/`mjs_getSpecDefault`) as the `def` argument to every `mjs_add*` call —
  `childclass` resolution is ours, using the same `Effective()` walk order as plan.md Section 7.
- Invariant, assert-enforced: no `mjsDefault` field write after any element referencing that
  class (directly or via childclass) exists. `mjs_setDefault` is never used to convey values.

This mirrors ProtoSpec's own `Effective()` semantics exactly (survey §2), which is why the
two-pass ordering is a decode rule rather than a semantic change.

**NDR-5: Union-list decode order = MuJoCo id order.** `ProcessLists_` assigns
`id = position in list`, and per-type lists append in `mjs_add*` call order (survey §2 general
mechanics). ProtoSpec's ordered union child lists (actuators, sensors, equalities, tendons,
spatial-tendon path items — plan.md Section 6 "Ordering") already exist to preserve interleaved
document order; the walker decodes each union list **in list order**, one `mjs_add*` per item,
reproducing MuJoCo's id and `sensor_adr` assignment exactly as the XML reader does. Tendon path
items append via `mjs_wrapSite/Geom/Joint/Pulley` in list order (no reorder API exists; a decoder
needs none). Frames decode as: create frame, create children on the owning body in document
order, `mjs_setFrame` each (survey §2 frames row). Contact pairs/excludes stay order-insensitive
(re-sorted by signature at compile), matching plan.md's decision to keep them non-union.

**NDR-6: Equality typed→`data[11]` lowering is a new, owned module — and an honest amendment to
Q-EQ.** Plan.md's Q-EQ row claims "no data-vector handling anywhere — MJCF is the interchange
form on both read and compile paths." That was true when the XML path was the only compile path;
it is **not** true for the native path. The packing from typed spellings into
`mjsEquality.data[mjNEQDATA]` lives in MuJoCo's XML reader, not in the mjs API
(xml_native_reader.cc:2192-2311; survey §2 equality row), so the walker must own it:

| ProtoSpec equality variant | data[] packing | objtype selection |
|---|---|---|
| Connect | anchor → data[0..2] | site form → mjOBJ_SITE, body form → mjOBJ_BODY |
| Weld | anchor → data[0..2], relpose[7] → data[3..9], torquescale → data[10] | site vs body as above |
| Joint | polycoef[5] → data[0..4] | mjOBJ_JOINT |
| Tendon | polycoef[5] → data[0..4] | mjOBJ_TENDON |
| Flex | — | mjOBJ_FLEX |
| FlexVert / FlexStrain | cell → data[0..2] (flexstrain) | per reader |

One small table-driven module in `bridge/` (the only consumer), golden-tested per NDR-13's
equivalence recipe. Q-EQ's row in plan.md gets a one-line amendment: "no data-vector handling on
the XML path; the native path owns a golden-tested packer (see plan_native_compiler.md NDR-6)."
The typed IDL representation is unchanged — this is lowering at the compile boundary, the same
pattern Q-ACT already established.

**NDR-7: Actuator lowering = call `mjs_setTo*`, never reimplement.** The native path calls the
very functions MuJoCo's own XML reader calls (`mjs_setToPosition` etc., survey §2 actuator row),
so typed-variant lowering is byte-identical to the XML path **by construction**. This preserves
the spirit of plan.md Resolved Decision 1 ("lowering exists only as a query helper and inside no
critical path"): the critical path now contains lowering *calls*, but the lowering *logic*
remains MuJoCo's. Our dormant `Lower()`/`Raise()` query helpers are unaffected. General-form
actuators write gaintype/biastype/dyntype/prm arrays directly.

**NDR-8: Composite = hybrid fallback; no internal linkage in v1.** Composite has **no public
mjs API** — the only entry is internal `mjCComposite::Make` (survey §2 composite row, R5).
Decision: models containing any `Composite` element route through the XML path via the NDR-1
gate. This is a designed, permanent-until-upstream-moves state, not a hack:

- Mechanics: the NDR-1 pre-scan flags `Composite` as unsupported; `fallback_reasons` names the
  family and cites the first composite element's SourceLoc. The differential suite tracks the
  composite-model count as an explicit corpus subset so scope is visible, not forgotten.
- The link-`mjCComposite` alternative (internal C++, highest-churn area per survey §3.3) is
  re-evaluated once at milestone N3 with corpus usage data; if declined, the fallback set is
  documented as exactly the composite corpus subset and the decision is closed.
- Flexcomp is the softer sibling: `mjs_makeFlex` is public but covers only the common parameter
  subset (survey §2 flex row). Flexcomp models whose authored attributes fit `mjs_makeFlex`
  compile natively; the rest fall back per-model, same reporting. Full fidelity
  (link `mjCFlexcomp` vs permanent partial fallback) is the second N3 decision.

**NDR-9: Plugin config `void*` = one isolated adapter, drift-gated.**
`mjs_setPluginAttributes(plugin, void*)` requires the pointer to be a
`std::map<std::string, std::string, std::less<>>*` — a C++-shape contract behind a void*
(survey §2 plugins row, R6). Works for us since we build MuJoCo from source, but pin-fragile.
Exactly one function in `bridge/` constructs the map from ProtoSpec's ordered (key, value) pairs
and makes the call; the drift gate (NDR-12) watches the reader-side construction and the impl's
move-out site for shape changes. The explicit-instance-before-implicit ordering rule remains a
tier-3 validation lint (Q-PLUGIN), enforced before decode.

**NDR-10: Attach and Replicate decode via `mjs_attach` loops; native `Expand()` follows.** The
recipes are read straight off MuJoCo's own reader (survey §1.1):

- `Attach` → `mjs_attach(frame, child_spec_element, prefix, "")`, child ProtoSpec sub-model
  decoded into its own spec first (deep copy per NDR-2 keeps it ours).
- `Replicate` → decode template subtree + frame, then `count` iterations of
  `mjs_attach(body, frame, "", zero_padded_suffix)` with accumulated pos/quat offsets, then
  delete the template — the literal `<replicate>` expansion loop.
- Pre-attach, ProtoSpec tier-2 referential validation runs on the child: MuJoCo's merge
  **silently drops** referencing elements whose targets do not resolve (`CopyList` skip,
  survey §1.1) — we refuse to hand it a model where that can happen, and we test the silent-drop
  case explicitly (R3).
- Known upstream semantics we inherit and test rather than fix: attach-pending keyframes are
  materialized only at compile; attaching twice without an intervening compile loses the first
  attachment's keyframes (survey §1.1). Since the walker builds spec → compile in one shot per
  `Compile()` call, this cannot bite the compile path; it is documented for `Expand()`.
- Native `Expand()` becomes: decode (macros execute during decode via the loops above) →
  `mj_saveXMLString` → `Read()` — the XML writer half of DR-7's round trip is retained, the
  parse-XML front half is dropped.

**NDR-11: Binding stays name-based; native adds a debug cross-check only.** DR-10's reasoning is
path-independent: name lookup via `mj_name2id` is immune to `discardvisual`/`fusestatic` id
compaction and pair re-sorting, and uniformity means one Binding implementation, one invalidation
model, one test suite for both paths. The native path *could* read ids off spec objects
(`mjs_getId` post-compile), but a native-only fast path would fork Binding semantics for a lookup
that is O(1) hashed and built once per compile — no measurable win, real divergence risk.
Decision: no public native binding path. In debug builds, the walker asserts
`mjs_getId(elem) == mj_name2id(m, type, name)` for every named element after a native compile —
a free structural cross-check of the whole decode (survey §4.5).

**NDR-12: Recompile keeps DR-11's binding-keyed migration; `mjCModel::SaveState/RestoreState` is
rejected.** Three reasons (survey §3.3, §4.7):

1. Spec-keyed save/restore requires a **persistent** mjSpec whose C++ object identities survive
   between compiles. The walker deliberately builds a fresh spec per call (non-goal §1); keeping
   the spec alive would create a second source of truth beside the ProtoSpec tree — precisely the
   tri-storage duality we quarantined.
2. `SaveState`/`RestoreState` are internal C++ with no stability promise; DR-11's design already
   achieves identical semantics (surviving elements keep state, deleted drop, new get qpos0/zeros)
   through the public binding.
3. DR-11 is path-agnostic: state is keyed by ProtoSpec element pointer via the previous Binding,
   so `Recompile` works identically whichever path compiled either side. The native path simply
   deletes the XML serialize/parse cost DR-11 budgeted for. The recompile-equivalence test
   (plan.md 10.7) gains a third leg: mjs-native edit + `mj_recompile` vs ProtoSpec edit + native
   `Recompile` — state vectors must match (survey §5.6).

**NDR-13: Authored bitmasks are generated writes, not hand code.** For option/visual/compiler/
size fields, presence-tracked decode also sets the corresponding `spec->authored.*` /
`compiler.authored` bits, exactly as the XML reader does (survey §2 model-root row, R9). Generated
from the same field tables as the rest of the walker. This is load-bearing twice: faithful
`mj_saveXML` of built specs (unit goldens, §5), and the prerequisite for the upstream conflict
policy doing anything useful (NDR-14). Statistic fields left unauthored keep their NaN sentinels
untouched.

**NDR-14: Upstream conflict-policy adoption plan + DR-12 drift-gate extension to the mjs
surface.** Upstream (post-pin) adds `mjtConflict` + a `conflict` field in `mjsCompiler`, attach
self-attachment and a `frame` attribute, and an `mjtByte → mjtBool` migration (survey §1.2). Plan:

- **Now**: ProtoSpec's `Attach` element grows `conflict : ConflictPolicy` (opt tri-state:
  Warning/Merge/Error) and `frame` (mutually exclusive with `body`) — authored, round-tripped,
  dormant against the vendored pin. NDR-13 bitmask fidelity is already in place.
- **At the next MuJoCo bump**: the drift gate fails loudly on the `mjsCompiler` layout change;
  wiring `conflict` through is then a one-field walker addition plus tests of the documented
  merge table (min/max/OR/error per field). The `mjs_attach` C signature itself is unchanged
  upstream — policy rides on the parent spec's compiler block, which the walker already owns.
- **DR-12 gate extension** — the gate grows an mjs panel watching, per bump:

| gate item | what is diffed | catches |
|---|---|---|
| `mjspec.h` struct surface | field names/types/order of every `mjsXxx` we decode into, incl. `mjsCompiler`, `mjsAuthored` | `conflict` insertion, `mjtBool` migration, any layout churn behind our direct writes |
| `mjs_*` signatures | the `MJAPI mjs_*` prototype set in `mujoco.h` | added/removed/changed decode functions (e.g. a future `mjs_attach` policy arg) |
| mjs default values | re-dump of the `mjs_default*` probe (bootstrap tooling, retained) diffed against schema defaults | silent default changes (`degree`, deepcopy flag, sentinel values) |
| plugin-attr contract | the map type at the reader construction + impl move-out sites | NDR-9 shape break |
| upstream watchlist | named items: `mjtConflict`, attach `frame` attr, self-attach, `mjtBool` | arrival of the tracked post-pin features |

**NDR-15: Decode-time API misuse is pre-checked; the walker never triggers `mju_error`.** Some
mjs setters abort the process on misuse (`mjs_setInStringVec` OOB, survey §2 material row). Every
such call site bounds-checks first and reports a ProtoSpec diagnostic instead. `mjs_setName`'s
-1 return (duplicate) is surfaced immediately with SourceLoc — strictly earlier and better-located
than the XML path's compile-time `CheckRepeat` throw (survey §4.5). The known O(n log n)-per-call
`CheckRepeat` cost is accepted for v1 and measured at N1 (R8); if hot at scale, names are set
locally and verified once in batch.

---

## 3. Error mapping: the diagnostics contract

The wire between ProtoSpec provenance (DR-9) and MuJoCo's compiler diagnostics (survey §2 general
mechanics, §4.6):

1. **Every element decode writes its `info` string** with the SourceLoc rendering
   (`file.mjcf:123`; `_ps:<type>:<serial>` for programmatically built elements). MuJoCo appends
   `info` to its own compiler errors, so MuJoCo-detected failures arrive file:line-cited with
   zero mapping code — the same mechanism the XML reader uses for line numbers.
2. **Decode-time failures** (duplicate names, pre-checked bounds, unresolvable class refs) are
   ProtoSpec diagnostics with SourceLoc, raised before `mj_compile` is ever called.
3. **Compile-time failures**: `mj_compile` returns NULL; the walker wraps `mjs_getError(spec)` /
   `mjs_isWarning` into a ProtoSpec diagnostic. MuJoCo guarantees the spec remains usable after a
   failed compile (survey §2), but the walker discards it regardless (fresh spec per call) — the
   guarantee matters only in that no cleanup hazard exists on the error path.
4. **Contract, tested**: (a) every native diagnostic for a file-originated element carries
   file:line; (b) any model that fails native compile also fails XML compile — a model that
   compiles on one path and not the other is a walker bug by definition, and the broken-model
   suite (plan.md milestone 4) runs through both paths to enforce it; (c) warnings are surfaced,
   never swallowed.

---

## 4. Binding and recompile interplay (summary of NDR-11/NDR-12)

Recommendation, with reasoning consolidated: **name-based Binding, unchanged, both paths**
(uniformity, id-compaction immunity, single test surface; `mjs_getId` demoted to a debug
assertion) and **DR-11 binding-keyed state migration, unchanged, both paths**
(`SaveState/RestoreState` requires persistent spec identity we deliberately do not keep, and is
internal API). Net effect of the native path on this area: `Recompile` gets faster (no XML hop),
gains a third equivalence-test leg, and changes zero public API.

---

## 5. Golden / verification strategy

Existing assets: `cpp/harness/mj_model_diff.cc` (mjxmacro full-field diff + fk invariant),
`tests/test_differential.py` (corpus-wide, `supported.json`-gated), 387-file corpus
(survey §5 preamble).

1. **Harness factoring first**: extract mj_model_diff's comparison core into a small library
   target; add `ps_native_diff <model.xml>` producing all legs in-process (no temp XML; asset-dir
   strategy inherited since all loads share the model dir). Same tolerance policy: sizes exact,
   floats rtol/atol, fk spot-check.
2. **Three-way diff**: A = `mj_loadXML(original)`; B = ProtoSpec → canonical XML → load (the DR-5
   path, already green per corpus ratchet); C = `mj_compile(walk(Read(M)))`. A==B isolates any
   A≠C failure to the walker, not the reader. B stays in the harness **permanently** as arbiter.
3. **Corpus exit bar**: every non-composite corpus model **bit-identical** XML-path vs
   native-path. Expectation grounded in survey §5.1: both paths run the same compiler on
   near-identical specs; no tolerance-consuming resolution is planned on our side (compile
   resolves everything).
4. **Unit goldens per family**: minimal ProtoSpec fragment → walk → `mj_saveXMLString(spec)`
   (works pre-compile) → snapshot. Catches decode drift human-reviewably before it becomes an
   mjModel diff. Depends on NDR-13 bitmask fidelity for faithful output.
5. **Targeted lowering equivalence**: equality packer (NDR-6) — parse tiny XML per type via
   `mj_parseXMLString`, compare `mjsEquality.data` against walker output; actuator `mjs_setTo*`
   — reader-identical by construction, one smoke case per spelling.
6. **Regression pins**: NDR-3 degree pair (degrees-authored + radians-authored euler models);
   NDR-4 defaulted-field goldens per family; NDR-10 attach silent-drop and double-attach keyframe
   cases; replicate suffix-numbering and template-deletion properties.
7. **Fuzz**: the planned attribute-mutation fuzzer (plan.md 10.8) runs mutants through **both**
   paths, requiring identical accept/reject and identical models on accept.
8. **Incremental landing**: NDR-1's per-model gate means the suite runs the native path exactly
   on the models it claims; `native_supported.json` grows family-by-family and the corpus
   native-count is the ratchet. Composite models are the designed long tail.

---

## 6. Risk register

Condensed from survey §6; each row's mitigation is owned by a numbered NDR.

| # | risk | owner | teeth |
|---|---|---|---|
| R1 | `mj_makeSpec` degree default (1) diverges from authored unit → silent 57.3x corruption | NDR-3 | first-write rule + assert + degree regression pair |
| R2 | eager default copy; label-only `mjs_setDefault` | NDR-4 | two-pass invariant + defaulted-field goldens |
| R3 | shallow attach mutates/locks source; `CopyList` silent drop; keyframe pending loss | NDR-2, NDR-10 | deepcopy-always + pre-attach tier-2 validation + explicit drop tests |
| R4 | strict keyframe size errors | (unchanged) | `core/sizes.cc` prediction + tier-3 lint stay the front line; error-mapping test |
| R5 | composite has no public API | NDR-8 | loud per-model fallback + N3 decision point with corpus data |
| R6 | plugin `void*` C++ map contract | NDR-9 | one adapter fn + drift-gate item |
| R7 | mjs API churn on bumps | NDR-14 | gate extension table |
| R8 | `mjs_setName` O(n log n)/call | NDR-15 | measure at N1; batch-verify fallback |
| R9 | authored-bitmask infidelity | NDR-13 | generated writes + saveXMLString goldens |
| R10 | silent behavioral deltas in unexercised corners | §5 | 387-corpus three-way + synthetic fixtures for corpus gaps (flex/composite variants) |

---

## 7. Prerequisites from current work

Nothing in this plan starts until:

1. **M3 IO waves complete enough that the XML reference path covers the corpus.** Leg B of the
   three-way harness must be green for a model before a native failure on it is attributable to
   the walker. Practically: waves 1–3 (defaults/assets, contact/equality/tendon/actuators,
   sensors/custom/keyframe/extension/macros) landed and the corpus differential ratchet at or
   near full coverage.
2. **M5 bridge in place**: the `Compile() -> Compiled{mjModel*, Binding}` signature, auto-naming
   with stable creation serials (DR-10/DR-11 prerequisite — the walker reuses the same serials
   for `info` strings and `mjs_setName`), `Recompile`, and `Expand()`. The native path plugs in
   behind this surface; it must not co-evolve with it.
3. **Drift gate operational** (DR-12, exists since M1) — NDR-14 extends it, does not create it.
4. `mj_model_diff` exists (M3, done) — N0 factors it, does not rewrite it.

---

## 8. Milestones

Sequential waves throughout: the walker is one shared file set (`bridge/mjs_walker.cc` + the
generated decode tables), so families land in order, one Opus owner per wave, per plan.md
Section 12 conventions. Harnesses are authored by a different agent than the code they test;
every wave ends with an adversarial review pass before its exit criterion is declared met.

**N0 — Harness factoring + walker skeleton.**
Work: mj_model_diff comparison core → reusable lib target; `ps_native_diff` three-way harness
(harness agent A); walker skeleton decoding only compiler/option/size/statistic/visual blocks —
including NDR-3 degree handling and NDR-13 authored bitmasks — with NDR-1 gate falling back for
everything else, plus the `Compiled.report` plumbing (walker agent B ≠ A).
*Exit: harness runs the whole corpus in fallback mode with correct `path_taken` reporting;
block-only synthetic models bit-identical XML vs native; degree regression pair green.*

**N1 — Pathfinder families** (mirrors the M3 IO wave order: blocks → body tree → defaults →
assets).
Work: defaults tree (NDR-4 two-pass), body tree with frames (NDR-5 frame recipe),
joints/geoms/sites/cameras/lights (variant lowering: `alt.type` + authored array, fromto/inertia
NaN-sentinel forms), meshes/hfields/textures/materials incl. buffer paths (NDR-15 pre-checks).
Single walker owner; unit-golden fixtures per family by a second agent.
*Exit: every corpus model whose families are within this set compiles natively bit-identical
(expected: the large majority of 387); R1/R2 regression suites in place; `mjs_setName` cost
measured (R8).*

**N2 — Constraint + drive families.**
Work: contact, equality (NDR-6 packer + equivalence tests — packer and its golden tests by
different agents), tendons + wrap paths (NDR-5 order rule), actuators (NDR-7 `mjs_setTo*`),
sensors, keyframes, custom, plugins (NDR-9 adapter), skin, flex (`mjs_addFlex` direct fields).
*Exit: full corpus native except macro-containing models; recompile-equivalence third leg
(NDR-12) green; diagnostics contract tests (§3.4) green through both paths.*

**N3 — Macros native.**
Work: Attach + Replicate via NDR-10 recipes; native `Expand()`; flexcomp via `mjs_makeFlex` with
a fidelity audit against the XML path on the flexcomp corpus subset; **the two deferred
decisions**, made with corpus usage data: composite (link `mjCComposite` vs permanent documented
fallback) and flexcomp full fidelity (link `mjCFlexcomp` vs partial-fallback).
*Exit: full corpus native, with the fallback list empty or exactly equal to the documented
composite set; attach/replicate property tests (suffix numbering, template deletion, silent-drop,
double-attach keyframes) green.*

**N4 — Default switch + upstream readiness.**
Work: `NativePath` becomes `Compile()`'s default (XML path retained as the oracle flag and
`Expand()`'s writer half); DR-11 `Recompile` rewired off the XML hop; NDR-14 gate extension
landed and armed; ProtoSpec `Attach` grows `conflict`/`frame` fields (dormant against the
vendored pin); perf note comparing compile-time vs the XML hop using `mjs_getTimer`.
*Exit: differential suite runs native-primary with XML as oracle; gate panel green against the
current pin; perf note committed.*

---

## 9. Open questions (genuinely undecidable today)

1. **Composite: link internal `mjCComposite` or permanent fallback?** Undecidable without corpus
   usage counts and a churn read on `user_composite.cc` at N3 time (it is the survey's
   highest-churn area, and the upstream pin will have moved by then). Decision point: N3.
2. **Flexcomp full fidelity route** — same shape as (1), softer stakes since `mjs_makeFlex`
   covers the common subset. Decision point: N3.
3. **ProtoSpec-side default for `conflict` once the upstream policy lands** — Warning (MuJoCo's
   default, maximally compatible) vs Error (fail-loud, more ProtoSpec-flavored). Cannot be judged
   until the merge-table semantics are exercisable against a real pin. Decision point: first
   MuJoCo bump past the conflict-policy commit.
