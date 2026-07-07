# ProtoSpec: Implementation Plan (Audited)

## STATUS (living section — update on every milestone commit)

Last updated: 2026-07-07. Execution: all code by Opus subagents; sequential waves for shared-file
IO work; test harnesses authored by a different agent than the code they test.

| Milestone | State | Evidence |
|---|---|---|
| M1 schema + IDL + drift gate | **DONE** | commits 03d1451..1326c1c; corpus study 100% on elements/attributes/enum-values/pairs over all 387 vendored MuJoCo MJCF files |
| M2 emitters + generated C++ | **DONE** | 660eeab (+recursion fix 5451b46); ~22k generated lines, /W4 clean, zero deps; 74 ctest checks |
| M3 differential harness | **DONE** | ca512e8; mj_model_diff (mjxmacro full-field diff + mj_forward xpos/xquat invariant, vendored MuJoCo 3.10.0); 387-file pipeline |
| M3 IO pathfinder (blocks + body tree) | **DONE** | 41001b6; schema-complete xml_binding tables; ps_roundtrip (exit 3 = unsupported-skip); 26 corpus files passing live differential |
| M3 wave 1: defaults + assets/include | **DONE** | 604c042; 89 corpus files identical / 0 differ; Q-ANGLE amended to form preservation (see quirk register); FreeJoint.align tri-state schema fix; dangling class refs = validation tier-2, not read errors |
| M3 wave 2: contact/equality/tendon + actuators | **DONE** | 7f1d4cb; 162 identical / 0 differ; 8 arity schema fixes + ActuatorPlugin.plugin type fix |
| M3 wave 3: sensors + custom/keyframe/extension + macros/deformable | **DONE** | a7fbd1d; FINAL: 359/387 identical, 0 differ, 28 honest skips (plugins/malformed fixtures); all 142 element types supported; Body/Frame/Replicate children unified into ordered `subtree : BodyChildAny *` (mutual document order is id-semantic) |
| M4 validation, M5 bridge+binding+recompile, M6 SDK, M7 pybind | queued | |
| Native compiler (RAW ProtoSpec->mjModel; NO mjSpec/mjs_*/mjC* anywhere) | **planned** | cd0998a: docs/plan_native_compiler.md — 62-row reuse ledger (18% call-as-is public API, 39% lift-verbatim with provenance registry + drift gate, 35% re-plumb, 5 rows deleted-hazard-classes); direct id-assignment Binding; phases NC1(10% corpus)→NC2(35%)→NC3(62%)→NC4(~80%)→NC5(~100%); XML path retained as oracle/fallback; survey: docs/native_compiler_survey.md (its mjs decode mapping is historical) |

Key facts a fresh session needs: repo is standalone at C:\Users\jonat\Documents\protospec (this
file is the canonical plan; the copy in the UE plugin's docs/ is a stale scratch draft). Tests:
`uv run pytest` (Python) and cmake+ctest under cpp/ (VS2022 generator; MUJOCO_ROOT points at the
vendored prebuilt 3.10.0 in the UE plugin's third_party/MuJoCo/src). supported.json under cpp/io/
is the IO coverage manifest the differential harness keys off. Regeneration entry points:
`python -m protospec_gen.emit --check` and `python tools/bootstrap/draft_schema.py --check`.


A single, clean, IDL-generated C++ object model for MuJoCo models. The plugin (and any other
consumer) works only with ProtoSpec objects. MJCF XML is a wire format handled inside one IO
module. mjSpec appears in exactly one bridge function and nowhere else.

Scope note: Unreal adoption is a downstream task and is deliberately out of scope here. This plan
covers the standalone library only: IDL, generator, object model, MJCF IO, canonicalization,
validation, the mjModel bridge + binding, SDK, and Python bindings. ProtoSpec lives in its own
repository while it iterates; the plugin consumes it later as a third-party dependency.

All MuJoCo file references below are into the vendored tree at `third_party/MuJoCo/src`.

---

## 1. Audit of the original plan

### Keep (sound ideas)

| Original idea | Verdict |
|---|---|
| Single IDL source of truth, generate everything from it | Keep. This is the core of the design. |
| Generated visitor/serializer hook (`Serialize(S&)`) instead of per-format hand code | Keep. One traversal, N formats. |
| Reflection index descriptors per field | Keep, generalized into full field tables (needed later for UE detail panels and Python). |
| Aggressive canonicalization (downstream code never sees euler/axisangle) | Keep, but refined: canonicalize *values*, preserve *authoring form* as data (DR-3). |
| XML name-divergence dictionaries (IR name vs XML tag) | Keep, folded into IDL annotations rather than a separate dictionary layer. |
| Include flattening | Keep, with MuJoCo's exact rules (once-per-file, relative-to-includer) and provenance tracking. |
| SDK builders, recursive delete, traversal | Keep. |
| pybind11 bindings | Keep, but as a late milestone. |
| Python drives codegen at build time, C++ at runtime | Keep. |

### Change (right instinct, wrong shape)

1. **"Optional scalars" as a type-system footnote → presence tracking as the central mechanism.**
   mjSpec's single worst wart is that "unset" is encoded with sentinels (`mjNAN` for `ipos`,
   `fullinertia`, `fromto`, geom `mass`; `-1` for `settotalmass`; `*_AUTO` enums), patched over
   with bolt-on `mjsAuthored` bitmasks (`mjspec.h:181-192`), and the XML writer *reconstructs*
   defaults by diffing values against class defaults (`xml_util.cc:1020-1042`). In ProtoSpec every
   defaultable field is presence-tracked by construction. This one decision eliminates sentinels,
   authored bitmasks, and diff-based writing simultaneously. See DR-1.

2. **Canonicalization must not destroy authoring intent.** MuJoCo's writer always emits radians and
   quaternions and always collapses `<position>`/`<motor>`/... to `<general>`; euler/axisangle/
   fromto/fullinertia forms are unrecoverable. That is exactly the lossiness we complained about
   (the biastype/gaintype round-trip bug came from this class of problem). ProtoSpec stores
   aliasing field groups as tagged variants and provides resolvers, instead of resolving eagerly
   and discarding the form. Actuators in particular stay in their authored typed form: loading an
   XML `<position>` yields a `Position{kp, kv, dampratio, ...}` you mutate in those terms; the
   general gaintype/biastype form is a lowering that happens only at the compile boundary. See
   DR-3 and Q-ACT.

3. **Custom IDL: keep, but minimal.** A hand-rolled schema language is justified here (comments,
   terseness, annotations that YAML would make ugly), but the grammar must stay tiny and the
   parser must emit a canonical JSON AST that all emitters consume. Emitters never touch the text
   format. Bootstrap the first draft mechanically from MuJoCo's own tables (Section 5).

4. **"Engine Divergence Hooks / CopyBodyFields"** — premature. Downstream consumers (UE) will be
   built on the reflection tables later. The library ships reflection; it does not know about any
   engine.

### Cut (with reasons)

1. **The InwardBuffer bidirectional memory engine.** Cut entirely. A model description tree is
   built once at import, mutated occasionally, and holds 10^3–10^5 elements. Arena packing with
   front/back write heads and integer offsets buys nothing measurable and costs: unstable element
   identity under mutation, a custom allocator to debug, and non-trivial recursive delete. mjSpec's
   own worst behaviors come precisely from clever storage (the `mjCXxx : private mjsXxx` tri-storage
   with `PointToLocal()` pointer re-aiming in `user/user_objects.h` — the struct you edit and the
   struct that compiles can silently diverge). ProtoSpec uses plain owned values:
   `std::string`, `std::vector`, `std::unique_ptr` children, stable pointers for identity. If a
   packed binary dump is ever needed, it is a *serialization format* behind the visitor, not the
   live representation.

2. **"Recurser Modules" as a DOM-ingestion framework.** Include expansion is ~100 lines with three
   rules (Section 8, Q-INC). It does not need a framework.

3. **The `base`/inheritance keyword in the IDL.** Field sharing across elements (pos/quat/alt
   appear on body/geom/site/camera/frame) is better expressed as a named *mixin* the generator
   flattens, not a C++ inheritance hierarchy. Inheritance in generated code creates exactly the
   aliasing ambiguity we are escaping. (Taxonomy note: `class` vs `table` vs `struct` distinctions
   collapse to one thing: `element`, plus plain `struct` for POD clusters like solref pairs.)

### Blind spots in the original plan (now covered)

The original plan never mentions, and the design must cover:

- **The defaults class system** — the `<default>` tree, `class`/`childclass` references, layered
  resolution. This is the most structurally important MJCF feature after the body tree. Section 7.
- **Frames, attach, replicate, composite, flexcomp** — which persist and which are parse-time
  macros. Section 8, Q-MACRO.
- **Actuator shortcut lowering** (`position` → `general` gaintype/biastype/dynprm math lives in
  `mjs_setTo*`, `src/user/user_api.cc`, not in the XML layer). Section 8, Q-ACT.
- **Sensor taxonomy**: ~50 XML element names mapping to `(type, objtype, objname, reftype,
  refname, intprm)` tuples. Section 8, Q-SENS.
- **Keyframes** whose vector lengths depend on model sizes (`nq`, `nv`, `na`, `nu`, `nmocap`).
  Section 8, Q-KEY.
- **Spatial tendon paths** — mjSpec's `mjsWrap` is a stub whose real data hides in C++
  `mjCWrap`; ProtoSpec makes path items real structs. Section 6.
- **Plugins/extension** (opaque key-value configs, instance-ordering rule), **custom**
  numeric/text/tuple, **user arrays** (`nuser_*`), **option/visual/statistic** blocks.
- **How ProtoSpec becomes an mjModel at all**, and how elements bind to compiled ids afterwards
  (mjSpec's `.id` workflow). The original plan never states either. DR-5, DR-10.

---

## 2. Ground truth: what we are escaping

Condensed from the vendored source; this is the quirk inventory the design answers to.

**mjSpec warts** (`include/mujoco/mjspec.h`, `src/user/*`):
- `mjString*`/`mjDoubleVec*`/... are pointers to `std::string`/`std::vector` (`mjspec.h:35-55`);
  the "C" struct is C++-only and every variable-length field needs a dedicated `mjs_set*` call.
- Each `mjsXxx` fronts a hidden `mjCXxx` storing every resolvable field three ways
  (`spec_foo_`, `foo_`, base field) synced by `CopyFromSpec()`/`PointToLocal()`.
- Unset-vs-set encoded with NaN/-1 sentinels (`src/user/user_init.c`) plus `mjsAuthored` bitmasks.
- Aliasing field groups with runtime mutual-exclusion checks instead of type invariants:
  `quat` vs `alt` (5 orientation encodings), `fromto` vs pos/quat/size, `fullinertia` vs
  `inertia`+`iquat` vs `ialt`, four texture load methods in one struct, six overlapping camera
  intrinsic arrays.
- Cross-references by name string (geom→mesh/material, actuator→target, sensor→obj/ref, ...),
  resolved only at compile; dangling names surface late.
- `mjsWrap` carries no data; tendon path data lives only behind accessor functions.
- Keyframe vector sizes unknowable pre-compile; `mjs_attach` mutates the *source* child and
  namespaces by string concatenation; refcounted shallow/deep copy toggled by global
  `mjs_setDeepCopy`.

**MJCF grammar facts** (`src/xml/`):
- The whole grammar is one 248-row static table `MJCF[]` (`xml_native_reader.cc:169-596`) plus
  ~40 keyword maps (`:600-1039`). `worldbody`, `frame`, and `replicate` are not schema rows; they
  validate against the `body` row via a special case (`xml_util.cc:486-498`).
- Variable-arity attributes via `ReadAttr(..., exact=false)`: `size` 0–3 (meaning depends on geom
  type), `friction` 1–3, `solimp` ≤5, `gear` ≤6, etc.
- `limited`/`inertiafromgeom`/`align` are false/true/**auto** tri-states.
- Degree→radian and euler-seq resolution deferred to compile time; the reader stores raw numbers
  plus `compiler.degree` and `eulerseq[3]`.
- `<include>`: expanded in a DOM pre-pass (`xml.cc:101-240`), each file at most once globally,
  resolved relative to model dir then including file, allowed anywhere.
- `composite`/`flexcomp`/`replicate`/`attach` expand at parse time into ordinary elements; nothing
  persists. `frame` persists.
- Writer (`xml_native_writer.cc`): requires a compiled model, always emits radians + quats +
  `<general>`, reconstructs default classes by value-diffing, conditional inertial emission.
  Read→write is semantically stable but not form-preserving.

---

## 3. Architecture

```
mujoco.spec (IDL, hand-curated, versioned)         <- THE single source of truth
        |
   spec_parser.py  -> schema AST (canonical JSON)
        |
   emitters (Python, build-time)
        |----> types.h / types.cc          generated element structs, variants, enums
        |----> visit.h                     generated Serialize/Visit hooks per element
        |----> reflect.h / reflect.cc      field tables: name, kind, offset-free accessors
        |----> keywords.cc                 enum <-> XML string tables
        |----> xml_binding.cc              per-element attribute binding tables (tag names,
        |                                  arity, unit flags, variant group routing)
        |----> defaults.cc                 global default values table
        '----> py_bindings.cc              pybind11 (late milestone)

   handwritten C++ (small, quirk-focused):
        io/mjcf_reader.cc                  tinyxml2 -> ProtoSpec (drives xml_binding tables;
                                           hand code only for Q-* quirks in Section 8)
        io/mjcf_writer.cc                  ProtoSpec -> tinyxml2 (same tables)
        core/resolve.cc                    orientation/fromto/inertia resolvers, unit conversion
        core/defaults.cc                   default-class layering / effective-value queries
        core/validate.cc                   3-tier validation (Section 9)
        core/sizes.cc                      nq/nv/na/nu/nmocap computation from the tree
        bridge/mj_bridge.cc                ProtoSpec -> mjModel + Binding (the ONLY module
                                           including mujoco.h)
        sdk/*.cc                           builders, find/traverse, delete, attach, rename
```

Dependency rule: everything above `bridge/` is MuJoCo-free (tinyxml2 only). `bridge/` is the
single quarantine zone for `mujoco.h`.

---

## 4. Design decisions

**DR-1: Presence-first fields.** Every field that MJCF treats as optional/defaultable is
`opt<T>` (a thin `std::optional` alias) in generated structs. "Authored" == `has_value()`.
Consequences: no sentinels anywhere; the writer emits exactly the authored fields (no value
diffing); default classes are just elements whose fields are all `opt<T>`; merging a class into an
element is a generic generated operation (`if (!dst.f) dst.f = src.f;` per field). Required
structural fields (e.g. joint `type`) are plain `T`.

**DR-2: Plain owned-value storage.** Elements are ordinary structs. Children are
`std::vector<std::unique_ptr<Child>>` so element identity (the pointer) is stable across sibling
mutation. Strings are `std::string`, numeric arrays are `std::array`/`InlineVec<T,N>` (fixed
capacity + filled count, for variable-arity attributes). No arenas, no offsets, no refcounts.

**DR-3: Variant + resolver pattern for aliasing groups.** Every mutually-exclusive field group
becomes one tagged variant field; "exactly one form" is a type invariant, not a runtime check.
Canonical *values* (radians, normalized quats) are enforced at the IO boundary; canonical *form*
is available through resolvers but the authored form is data:

| Variant field | Alternatives | Resolver |
|---|---|---|
| `Orientation` | Quat, Euler(seq-dependent), AxisAngle, XYAxes, ZAxis | `ToQuat(eulerseq)` |
| `GeomShape` | Explicit{pos,quat,size}, FromTo{fromto,size} | `ResolvePose(geomtype)` |
| `InertiaSpec` | Diagonal{inertia,iquat-or-ialt}, Full{fullinertia[6]} | `ToDiagonal()` (eigen-decomp, as `mjuu_fullInertia`) |
| `TextureSource` | Builtin{...}, File{...}, CubeFiles{...}, Buffer{bytes} | n/a (true union) |
| `CameraIntrinsics` | Fovy, Focal{...}/Sensor{...} pixel/length forms | `ToFovy(resolution)` |
| `Limited` (tri-state) | False, True, Auto | `Resolve(autolimits, has_range)` |

Angles are stored in **radians always**; `compiler.degree` is consumed at read time and recorded
only as a write-style preference. Euler orientation keeps its values (converted to radians) and
its form. Downstream code that wants "just a quaternion" calls the resolver; nothing downstream
ever branches on encoding.

**DR-4: ProtoSpec is an authoring-level model, not a compiled model.** It represents what MJCF can
say (like mjSpec pre-compile), including `Auto` states, default classes, unresolved refs-by-name,
and geoms without masses. Compile-time transformations (inertiafromgeom, fusestatic,
discardvisual, mesh fitting, autolimits resolution) are MuJoCo's job, reached through the bridge.
ProtoSpec provides *queries* that predict compile-relevant facts (sizes, effective defaults,
resolved orientations) but never mutates the tree to apply them.

**DR-5: mjModel bridge = canonical MJCF string + mjVFS.** `Compile(const Model&) -> mjModel*`
serializes ProtoSpec to a canonical MJCF string, registers in-memory assets (mesh/texture/hfield
buffers) in an `mjVFS`, and calls `mj_loadXML`. Rationale: XML is MuJoCo's most stable and most
tested interface; the `mjs_*` C API churns across versions (we bump MuJoCo often) and drags in the
tri-storage semantics we are quarantining. The bridge is one file; if profiling ever shows the XML
hop matters (it will not for scene loading), a direct `mjs_*` walker can be added behind the same
function signature. The reverse path (`mjSpec -> ProtoSpec` importer) is explicitly a non-goal for
v1; MJCF files are the interchange format.

**DR-6: Own MJCF reader, verified differentially.** ProtoSpec parses MJCF itself (tinyxml2). This
is required for form preservation (DR-3) and provenance, and it removes mjSpec from the read path.
The risk (semantic divergence from MuJoCo) is contained by the differential harness (Section 10):
every corpus model is compiled twice — `mj_loadXML(original)` vs `mj_loadXML(protospec round
trip)` — and the resulting `mjModel`s are field-diffed. That harness, not code review, is what
makes an independent reader safe.

**DR-7: Macro handling — only `include` is a read-time pre-pass; everything else is first-class.**
- `include`: expanded by ProtoSpec with MuJoCo's exact rules, provenance recorded per element.
  It is the one construct that cannot pass through (it is file plumbing, not model content).
- `frame`: first-class ProtoSpec element (it persists in mjSpec too).
- `replicate` and `attach`: **first-class ProtoSpec elements**, passed through verbatim.
  Originally slated for read-time expansion, but pass-through is strictly better: the bridge
  emits the tags into the canonical MJCF and MuJoCo's own reader performs the cloning and
  prefix/suffix namespacing during compile, so ProtoSpec never reimplements either; the compact
  authored form survives edits and saves; and the treatment is uniform with composite/flexcomp.
  Binding reaches the generated copies via MuJoCo's deterministic suffix/prefix name patterns,
  and `Expand()` (below) materializes them on demand.
- `composite` / `flexcomp`: **first-class ProtoSpec elements**, and their expansion is NOT
  reimplemented. Both are ordinary rows in the schema table with attributes and typed children
  (`joint`/`geom`/`site`/`skin`/`pin`/`edge`/`elasticity`/`contact`/`plugin`), so they generate
  like any other element: read from XML, mutated programmatically, written back out verbatim.
  The expansion geometry code (`src/user/user_composite.cc`, `user_flexcomp.cc`) stays MuJoCo's
  job: the bridge serializes the `<composite>`/`<flexcomp>` tags into the canonical MJCF and
  MuJoCo expands them during `mj_loadXML`. This fits DR-4 (authoring-level model) exactly — the
  compact form IS the model; the expanded bodies are compiled artifacts, reachable through the
  binding (DR-10) via MuJoCo's deterministic generated names (e.g. cable bodies are
  `prefix + "B_%d"`, `user_composite.cc:353`; the full name-pattern catalog is captured during
  milestone 3). For users who want to edit the expanded result, an explicit `Expand(model)`
  helper round-trips through MuJoCo (`mj_parseXML` → `mj_saveXMLString` → `Read`) and returns a
  new ProtoSpec tree with macros materialized — an opt-in transformation, never automatic.

**DR-8: References are typed.** The IDL has a `ref<Mesh>` field kind. Storage is the name string
(that is what MJCF is), but the generator knows the target element type, so: validation resolves
every ref and reports dangling names with provenance; the SDK offers `Resolve(ref)` /
`FindReferrers(elem)` / `Rename(elem, newname)` that updates all referrers. No pointers are stored
in the tree (keeps Clone/serialize trivial and avoids mjSpec's compile-time-resolution trap).

**DR-9: Provenance is a first-class field.** Every element carries `SourceLoc {file, line}`
(mjSpec's `info` string, but structured). Populated by the reader through include expansion;
empty for programmatically built elements. All validation errors cite it.

**DR-10: Binding by name, not by index.** This replaces mjSpec's `spec.geom("xyz").id` workflow,
which we lose by bypassing the `mjs_*` compile path. The investigation result:

- `mjModel` carries every element name plus a hash map; `mj_name2id(m, mjOBJ_GEOM, "xyz")` and
  `mj_id2name` are public, O(1) API (`src/engine/engine_name.c:240`). Name lookup is the
  supported, stable way to recover ids from a compiled model — mjSpec's stored `.id` is just a
  cache of the same assignment.
- Index prediction (counting elements in document order) is NOT safe: `discardvisual` deletes
  visual geoms/meshes and *compacts the surviving ids* (`user_model.cc:2021-2057`, id fixup at
  `:1780-1830`), `fusestatic` removes bodies, and pairs/excludes are re-sorted by body signature
  with ids reassigned (`user_model.cc:5143-5146`). Name lookup is immune to all of this;
  deletion simply surfaces as "not found", which is informative.

So `Compile()` returns `Compiled { mjModel*, Binding }` where `Binding` is built immediately
after compile by resolving every named ProtoSpec element through `mj_name2id`:

- `binding.Id(elem) -> opt<int>` — element pointer to model id; unbound (compiled away) elements
  return empty, with a diagnostic naming the responsible compiler flag when one is on.
- Typed sugar over the id: `binding.QposAdr(joint)`, `binding.DofAdr(joint)` (`jnt_qposadr`/
  `jnt_dofadr`), `binding.SensorAdr(sensor)`, `binding.ActId(actuator)`, body geom ranges, etc.
- **Unnamed elements**: the bridge auto-names them at serialization with deterministic reserved
  names (`_ps:geom:17`, where 17 is a creation serial that is stable across tree edits — see
  DR-11), so *every* element is bindable by default. The generated names are visible in the
  compiled model's name tables; an opt-out exists for users who care, at the cost of unnamed
  elements being unbindable.
- **Macro-generated elements** (composite/flexcomp/replicate/attach expansions): bindable through
  the deterministic name patterns MuJoCo/ProtoSpec produce; `Binding` offers prefix/pattern
  queries (`binding.Find(mjOBJ_BODY, "cable*")`).
- **Invalidation**: any structural edit to the tree invalidates the binding (cheap edit counter
  on the model root — the explicit version of mjSpec's hidden `signature`). Recompiling returns
  a fresh binding; stale use asserts.

**DR-11: Recompile = full compile + state migration (and that is also what mjSpec does).**
Investigated: `mj_recompile` (`src/user/user_api.cc:262-288`) is not incremental. It stashes
per-joint qpos/qvel, per-actuator act/ctrl, and per-mocap-body pos/quat onto the persistent
spec objects (`mjCModel::SaveState`, `user_model.cc:4098`), re-runs the **entire** compile
pipeline, allocates a fresh `mjData`, and writes the stashed state back at the new addresses
(`RestoreState`, `:4152`) — surviving elements keep their state, deleted elements drop theirs,
new elements get `qpos0`/zeros. There is no incremental compile to lose by going through XML.

ProtoSpec therefore supports recompile as a first-class bridge operation with identical
semantics: `Recompile(model, prevCompiled, mjData* d) -> Compiled`:
1. Using the *previous* binding, read `d->qpos/qvel/act/ctrl/mocap_pos/mocap_quat` slices into a
   state map keyed by ProtoSpec element pointer (stable under DR-2, exactly like mjSpec's C++
   object identity).
2. Compile the edited tree through the normal bridge (DR-5) and build the new binding.
3. Allocate fresh `mjData`; for each stashed element still present in the new binding, copy its
   slice to the new address; absent entries fall back to `qpos0`/zeros; restore `d->time`.

Cost: one full MuJoCo compile plus the XML serialize/parse hop — the same order as what
`mj_recompile` pays, fine for interactive edit loops. If a profile ever says otherwise, the
`mjs_*` walker escape hatch in DR-5 applies.

Correctness prerequisite (feeds DR-10): auto-generated names must be **stable across tree
edits**, or an unnamed joint's state would migrate to the wrong element after an insertion.
Auto-names are therefore derived from a monotonic creation serial stamped on each element at
construction (`_ps:joint:42` where 42 never changes or reuses), not from document position.

**DR-12: Schema version pinning + drift gate.** `mujoco.spec` declares the MuJoCo version it
covers. A CI test parses the vendored `MJCF[]` table, keyword maps, and `mjspec.h` and diffs the
attribute/element/enum surface against our schema AST. Bumping MuJoCo then *fails loudly with a
list of new/removed attributes* instead of silently missing features. (The vendored tables are
used as a checker, never as a generator input after bootstrap — the IDL stays the single source
of truth.)

---

## 5. The IDL

### Grammar (deliberately tiny)

```
schema     := header (enumdef | mixindef | elemdef | structdef)*
header     := "mujoco_version" STRING
enumdef    := "enum" NAME "{" (NAME "=" STRING)+ "}"          # value = XML keyword
mixindef   := "mixin" NAME "{" field* "}"
structdef  := "struct" NAME "{" field* "}"                    # POD cluster, no identity
elemdef    := "element" NAME annots? "{" ("use" NAME)* field* child* "}"
child      := "children" NAME ":" NAME cardinality            # tree containment
field      := NAME ":" type annots? default? comment?
type       := prim | prim "[" INT "]" | prim "[" INT ".." INT "]"   # fixed / variable arity
            | "ref<" NAME ">" | "variant" NAME | NAME (enum/struct) | prim "[]"
annots     := "(" (key "=" value),* ")"
```

Field annotations (closed set): `xml="tagname"` (name divergence), `unit=angle` (deg→rad at IO),
`required`, `variant_group=G, variant_tag=quat` (routes multiple XML attributes into one variant
field), `element_text` (data carried as element text, not attribute).

**Format annotations do not multiply.** `xml=` exists only because MJCF is a *foreign* format
whose names we do not control, and it is divergence-only: the default binding is the lowercased
IDL name, so the annotation appears on the handful of fields where MJCF disagrees with the name
we want (`dclass` vs `class`, `orient` vs the five orientation attributes) — not on every field.
Formats we own (JSON debug dump, any future binary) are defined *by* the IDL names and need no
annotations, ever. There will never be a `json=`/`binary=` axis. Child-list naming is likewise
free: `children bodies : Body *` names the C++ field `bodies` for semantic clarity, while the XML
tag for each child comes from the child element's own declaration (`element Body (xml="body")`) —
the list name never appears in XML, so IR naming and wire naming are fully decoupled.

Everything not `required` is presence-tracked (`opt<T>`) automatically. Defaults in the IDL are
*documentation + SDK convenience values* (what MuJoCo's compiler uses when nothing is authored,
extracted from `mjs_default*`); they are never silently written into models.

### Example (illustrative subset)

```
enum Integrator { euler="Euler" implicit="implicit" implicitfast="implicitfast" rk4="RK4" }

mixin Posed {
  pos    : double[3] = {0,0,0}
  orient : variant Orientation        # quat | euler | axisangle | xyaxes | zaxis
}

element Geom {                        # XML tag defaults to lowercase: <geom>
  use Posed
  name     : string
  dclass   : ref<Default>  (xml="class")   # genuine divergence: `class` is reserved
  type     : GeomType = sphere
  size     : double[0..3]
  shape    : variant GeomShape        # explicit pose+size | fromto
  friction : double[1..3] = {1, 0.005, 0.0001}
  material : ref<Material>
  mesh     : ref<Mesh>
  rgba     : float[4] = {0.5, 0.5, 0.5, 1}
  user     : double[]
}

element Body {
  use Posed
  name       : string
  childclass : ref<Default>
  inertial   : variant InertiaSpec
  children geoms   : Geom   *        # field name is IR-side only; XML tag comes
  children joints  : Joint  *        # from the child element (<geom>, <joint>, ...)
  children bodies  : Body   *        # recursion
  children frames  : Frame  *
}
```

### Bootstrap

A one-time script drafts `mujoco.spec` mechanically:
- elements + nesting + attribute names from `MJCF[]` (`xml_native_reader.cc:169-596`),
- enum keyword tables from the `mjMap` tables (`:600-1039`),
- field types/arities from `mjspec.h` cross-referenced with the reader's `ReadAttr` calls,
- default values by linking a tiny probe against MuJoCo and dumping every `mjs_default*` struct.

The draft is then hand-curated once (variant groups, mixins, ref targets, unit flags are human
judgments) and the bootstrap script is retired in favor of the drift gate (DR-12).

---

## 6. Coverage inventory (the "no blind spots" checklist)

Element families, all in scope for v1 unless marked:

- **Tree**: worldbody/body (recursive), frame, inertial variant, joint, freejoint (sugar for
  joint type=free with restricted attrs), geom, site, camera (intrinsics variant), light.
- **Assets**: mesh (incl. user vertex data + inline plugin), hfield (incl. elevation buffer),
  texture (4-way source variant), material (texture role list → `textures[]` by role), skin
  (bindpos/bindquat/vertid/vertweight nested arrays), nested `<model>` asset for attach.
- **Defaults**: default class tree (recursive), one optional sub-element per defaultable type
  (mesh, material, joint, geom, site, camera, light, pair, equality, tendon, and the ten actuator
  spellings). Note mjSpec has no body/frame defaults; MJCF's `childclass` only routes classes.
- **Contact**: pair (with solreffriction), exclude (body pair, despite the header's wrong comment).
- **Equality**: connect, weld, joint, tendon, flex, flexvert, flexstrain — one element with
  `mjtEq`-like type + `data[mjNEQDATA]` is the mjSpec shape; ProtoSpec instead gives each XML
  spelling its named parameters, exactly as MJCF does (Q-EQ; same pattern as actuators, Q-ACT).
- **Tendon**: spatial and fixed; the path is `std::vector<PathItem>` where
  `PathItem = variant{ Site(ref), Geom(ref, opt sidesite), Pulley(divisor), Joint(ref, coef) }` —
  this replaces mjSpec's stub `mjsWrap` + hidden `mjCWrap` accessors with real data.
- **Actuator**: common transmission/limit fields (target refs, gear, three tri-state limit
  groups) + a typed variant per spelling: `Motor`, `Position{kp, kv, dampratio, timeconst,
  inheritrange}`, `Velocity{kv}`, `IntVelocity`, `Damper{kv}`, `Cylinder`, `Muscle`, `Adhesion`,
  `DCMotor`, `General{dyntype, gaintype, biastype, dynprm, gainprm, biasprm, actdim}`, `Plugin`.
  You load a `<position>`, you get a `Position` and mutate `kp` — never prm arrays. Lowering to
  the general form happens only at the compile boundary (Q-ACT).
- **Sensor**: type + typed object/ref slots + intprm (Q-SENS).
- **Custom**: numeric (size + padded data), text, tuple (parallel objtype/objname/prm rows).
- **Keyframe**: time + six `opt<std::vector<double>>` (Q-KEY).
- **Extension/plugins**: `<extension><plugin><instance><config>` with the ordering rule; per-element
  plugin attachment (geom/body/mesh/actuator/sensor) as `{plugin_name | instance ref, config
  key-value list}` — configs stored as ordered string pairs, never `void*`.
- **Blocks**: compiler (all flags incl. eulerseq, dirs, LROpt), option (+ flag bitmasks), size
  (nuser_*, nkey, memory with K/M/G suffix parsing), visual (global/quality/headlight/map/scale/
  rgba sub-blocks), statistic (all presence-tracked; MuJoCo's NaN sentinels become absent fields).
- **Ordering (union child lists).** MJCF assigns ids — and for sensors, `sensor_adr` data
  addresses — in interleaved document order across different tags within a section. The IDL
  therefore has `union` types, and the actuator, sensor, equality, and tendon sections plus the
  spatial-tendon path are each a single ordered union child list (`union ActuatorAny = Motor |
  Position | ...`; `union PathItemAny = SpatialSite | SpatialGeom | Pulley`), not per-type lists
  (which silently lose the interleave — site-geom-site is unrepresentable). Emitters and IO must
  preserve list order to reproduce MuJoCo's id assignment. Contact (pair/exclude) is deliberately
  NOT a union: MuJoCo re-sorts pairs/excludes by body signature at compile, so pre-compile
  document order there is not authoritative. Unions also serve as reference targets:
  `ref<TendonAny>` makes fixed tendons valid targets of actuator transmissions, sensor slots, and
  tendon-equality constraints.
- **Procedural**: composite (type, count, spacing, per-kind sub-defaults for joint/geom/site,
  skin, plugin), flexcomp (type, count/spacing/point/element data, pin list, edge/elasticity/
  contact blocks, plugin), replicate (count/offset/euler/sep + body-context children), and
  attach (model ref/body/prefix) — all first-class pass-through elements (DR-7).
- **Explicit non-goals v1**: URDF reading (MuJoCo's `xml_urdf.cc` path; convert externally),
  mjSpec→ProtoSpec import, binary .mjb.

---

## 7. Defaults: first-class layered classes

- `Default` is an element holding `opt<>`-everything partial specs (one per defaultable family)
  and child classes (tree). The root class is `main` (enforced named or unnamed, as MJCF does).
- Elements carry `class : ref<Default>`; bodies/frames carry `childclass : ref<Default>`.
- **Resolution is a query, never a mutation**: `Effective(elem)` walks element → its class (or
  nearest enclosing childclass) → ancestors → `main` → IDL global defaults, returning the first
  authored value per field. Generated per-family merge functions make this mechanical.
- The writer emits classes verbatim (they are real data, DR-1), so class structure round-trips
  exactly — unlike MuJoCo's writer, which reconstructs classes by value-diffing and drops
  authored-equal-to-default attributes.
- SDK extras enabled by this: `FlattenDefaults(model)` (bake effective values, drop classes) and
  `ExtractClass(elems, name)` (factor shared authored values into a new class) — both pure tree
  transforms under DR-1/DR-2.

---

## 8. Quirk register

Every known MJCF/mjSpec quirk and its single owner in ProtoSpec. Each entry becomes tests.

| ID | Quirk (source) | ProtoSpec handling |
|---|---|---|
| Q-ORIENT | 5 orientation encodings; `quat` field + `alt` override; MuJoCo resolves at compile (`user_objects.cc:241-330`) | One `Orientation` variant per posed element (DR-3); resolver takes `eulerseq`; reader rejects multiple specifiers (as `ReadAlternative` does) |
| Q-ANGLE | `compiler.degree` + `eulerseq` deferred to compile; conversion is PER-CONSUMER (joint range only for limited hinge/ball, ref/springref only for hinge — `user_objects.cc:3207-3282`) | AMENDED (wave 1): form preservation. Angles are stored exactly as authored and the `angle` unit round-trips verbatim; MuJoCo converts at compile. Read-time conversion was proven wrong by `auto_limits.xml`: one defaults class is consumed by both a hinge (converted) and a free joint (not), so no single pre-converted value is correct. Consumers wanting radians use a resolver that takes the compiler block |
| Q-FROMTO | `fromto` vs pos/quat (+ site variant) sentinel-guarded | `GeomShape` variant; resolver implements the capsule/cylinder/box axis math |
| Q-INERTIA | `fullinertia[6]` vs `inertia+iquat` vs `ialt`, NaN sentinels, mutual exclusions (`user_objects.cc:2705-2716`) | `InertiaSpec` variant; exclusivity is structural; eigen-decomposition in resolver |
| Q-AUTO | `limited`/`actlimited`/`ctrllimited`/`forcelimited`/`inertiafromgeom`/`align` tri-states | `TriState` enum {False,True,Auto}; never resolved in the tree; `Resolve(autolimits, ...)` helper mirrors `checklimited` (`user_objects.cc:175-187`) |
| Q-ARITY | Variable-arity attrs (`size` 0–3, `friction` 1–3, `gear` ≤6, polynomials `mjNPOLY+1`) with type-dependent meaning | `T[min..max]` IDL arity → `InlineVec<T,N>` with filled count; geom-type-dependent size interpretation lives in the `GeomShape` resolver |
| Q-ACT | Ten actuator spellings lower to general via `mjs_setTo*` (`user_api.cc`); MuJoCo writer only emits `<general>`; our old biastype/gaintype round-trip bug | Typed variant per spelling is the stored form (see Section 6); the writer emits the authored tag with its named params, so round-trip is exact by construction — and since the bridge also emits MJCF, **MuJoCo's own reader performs the lowering at compile**; we never reimplement it on the compile path. A small lowering module replicating `mjs_setTo*` exists only for queries (`Lower(Position) -> General` for inspection, `Raise(General) -> opt<Position/...>` for typed editing of foreign files — implemented but dormant, explicit opt-in per actuator, never on a load path; dyntype knowledge for size computation) and is differentially tested against `mjs_setTo*` |
| Q-SENS | ~50 sensor tags → (type, objtype, objname, reftype, refname, intprm) with per-tag attr routing (`xml_native_reader.cc:4181-4638`) | Table in the IDL: each sensor element declares its slot routing; generated binding handles all regular cases; hand code only for rangefinder (site-xor-camera + data bitmask), contact (multi-source slots + reduce/num), distance/normal/fromto (geomN-xor-bodyN) |
| Q-KEY | Keyframe vector lengths depend on nq/nv/na/nu/nmocap, known only post-compile in MuJoCo | Vectors stored as authored; `core/sizes.cc` computes sizes from the tree (nq/nv per joint type: free 7/6, ball 4/3, slide/hinge 1/1; nu = actuators; na = Σ act dims; nmocap = mocap bodies); validation checks lengths; helper pads/truncates on request |
| Q-INC | `<include>`: once per file globally, anywhere in tree, resolved model-dir-then-includer-dir, eager assetdir capture (`xml.cc:101-240`) | Reader pre-pass with identical rules + provenance; `assetdir` sets mesh+texture dirs, explicit `meshdir`/`texturedir` override (precedence as `xml_native_reader.cc:1215-1221`) |
| Q-MACRO | `replicate`/`attach`/`composite`/`flexcomp` parse-time only in MuJoCo; `frame` persists; `worldbody/frame/replicate` share the `body` schema row | DR-7: all five first-class ProtoSpec elements; replicate/attach/composite/flexcomp pass through verbatim and MuJoCo expands them at compile; `Expand()` materializes on demand |
| Q-REFS | All cross-refs are name strings resolved at compile; dangling names fail late | DR-8 typed refs; validation tier 2 resolves everything with provenance |
| Q-TEX | Four texture load methods in one struct; material references textures by role array | `TextureSource` variant; material holds `opt<ref<Texture>>` per role |
| Q-PLUGIN | Opaque `void*` configs; explicit-instance-before-implicit ordering rule (`xml_native_reader.cc:3148-3151`) | Ordered `(key,value)` string pairs; ordering rule = validation lint, not parse failure |
| Q-EQ | Equality = type enum + opaque `data[11]` (mjSpec shape) | Named per-type parameters in the IDL matching the XML spellings (`<connect>`, `<weld>`, ...); no data-vector handling on the XML paths. AMENDED: the native compile path (plan_native_compiler.md NDR-6) owns a small typed→data[11] packer, golden-tested against MuJoCo-parsed specs, because that packing lives only in MuJoCo's XML reader |
| Q-NAMES | Names unique per element type, empties allowed; top default class must be `main` | Validation tier 2; enforced exactly as MuJoCo does |
| Q-NUM | `inf`/`nan` parsing, C-locale forcing, `memory` K/M/G suffixes | One numeric IO module mirroring `xml_util.cc` behavior (NaN warning included) |
| Q-WRITE | MuJoCo writer drops attrs equal to class defaults; conditional inertial emission | Not inherited: ProtoSpec writes authored fields, period (DR-1). Bridge serialization for compile may emit fully-resolved canonical form; user-facing save preserves form |

---

## 9. Validation (three tiers, all provenance-cited)

1. **Structural** (fully generated from the IDL): required fields present, enum values legal,
   arity within bounds, variant groups well-formed, children cardinality.
2. **Referential**: every `ref<T>` resolves; name uniqueness per element type; class references
   acyclic; attach/namespace collision detection.
3. **Semantic lint** (handwritten, small): keyframe lengths vs computed sizes; plugin instance
   ordering; orientation-specifier conflicts already impossible structurally; the handful of
   MuJoCo compile errors worth pre-empting with better messages (range without limited when
   autolimits=false; hinge/slide joints needing axis; mocap bodies must be static children of
   world; nuser_* vs `user` array lengths).

Validation never mutates. `Compile` (bridge) runs tiers 1–2 mandatorily.

---

## 10. Testing strategy

1. **Differential compile harness (the backbone).** For every corpus model M:
   `mjModel A = mj_loadXML(M)`; `mjModel B = Compile(Read(M))` (i.e. our reader → our writer →
   MuJoCo). Diff A vs B field-by-field (sizes exactly; floats with tolerance; qpos0 spot-simulated
   a few steps). Corpus: MuJoCo's vendored `model/` + test models, our robot models (G1, arms),
   and every MJCF in `Scripts`/test fixtures. Any divergence is a bug in reader, writer, or a
   deliberate documented normalization.
2. **Round-trip fixpoint.** `Read(Write(Read(M)))` must equal `Read(M)` under generated deep
   equality (guarantees write→read closure), and `Write(Read(Write(Read(M))))` must be
   byte-identical to `Write(Read(M))` (deterministic output).
3. **Lowering equivalence.** Actuator shortcut tables and equality-constraint lowering tested
   directly against `mjs_setTo*` / mjSpec values on generated cases.
4. **Generator golden tests.** Schema AST snapshots; emitted-code golden files; IDL parser
   negative cases.
5. **Drift gate (DR-12).** Runs against the vendored MuJoCo on every bump.
6. **Binding tests.** Every corpus element with a name binds to the id MuJoCo assigned
   (cross-checked via `mj_id2name`); discardvisual/fusestatic corpus cases verify unbound
   elements are reported, not mis-bound; composite models verify pattern binding.
7. **Recompile equivalence.** Apply the same structural edit through `mjs_*` + `mj_recompile`
   and through ProtoSpec + `Recompile`; the resulting `mjData` state (qpos/qvel/act/ctrl/mocap,
   time) must match. Cases: add/delete joint, add actuator, delete body, edits touching unnamed
   elements (exercises serial-stable auto-names).
8. **Fuzz (late).** Attribute-level mutation fuzzing of corpus files through Read+validate; must
   never crash, only report.

---

## 11. Milestones

Each has an exit criterion; no milestone starts until the previous criterion is green.

1. **Schema bootstrap + IDL parser.** Bootstrap script drafts `mujoco.spec` from `MJCF[]` /
   `mjMap` / `mjspec.h` / `mjs_default*` probe; hand-curate; parser → JSON AST with structural
   validation. *Exit: AST covers 100% of `MJCF[]` elements/attributes (drift gate passes); parser
   golden + negative tests.*
2. **Core emitters.** types / visit / reflect / keywords / defaults; generated equality, clone,
   diff. *Exit: generated code compiles standalone (no MuJoCo); reflection walk enumerates every
   field; clone+equality property tests.*
3. **MJCF IO, family by family.** Reader+writer land together per family, ordered:
   (a) compiler/option/size/statistic/visual, (b) body/geom/joint/site/frame + Q-ORIENT/Q-ANGLE/
   Q-FROMTO/Q-INERTIA/Q-ARITY, (c) defaults + Q-AUTO, (d) assets + Q-TEX + include (Q-INC),
   (e) contact/equality/tendon (Q-EQ, path items), (f) actuators (Q-ACT), (g) sensors (Q-SENS),
   (h) custom/keyframe/extension (Q-KEY, Q-PLUGIN), (i) replicate/attach/composite/flexcomp as
   first-class pass-through elements (Q-MACRO), incl. cataloging MuJoCo's generated-name
   patterns for binding.
   *Exit per family: differential harness green for corpus models exercising it; fixpoint test
   green. Final exit: full corpus green, composite/flexcomp models included (pass-through).*
4. **Validation + provenance.** Tiers 1–3, SourceLoc through includes. *Exit: curated
   broken-model suite produces the expected structured errors, each with file:line.*
5. **mjModel bridge + binding + recompile.** XML-string + mjVFS compile with in-memory asset
   registration; auto-naming of unnamed elements (stable creation serials); `Binding`
   construction, typed address helpers, pattern queries, invalidation; `Recompile` with state
   migration (DR-11); `Expand()` helper (macro materialization via MuJoCo round-trip).
   *Exit: differential harness runs through `Compile()`; binding tests (Section 10.6) green;
   a corpus robot compiles, binds every joint/actuator/sensor, steps, survives a mid-simulation
   structural edit via `Recompile` with state preserved.*
6. **SDK.** Typed builders with IDL defaults, find/traverse, recursive delete (refs checked),
   attach with prefixing, rename-with-referrers, FlattenDefaults/ExtractClass. *Exit: build a
   non-trivial robot programmatically, compile it, simulate; delete/rename property tests.*
7. **pybind11.** Generated bindings over reflect tables; Read/Write/Validate/Compile exposed.
   *Exit: Python round-trip of corpus; usable from the existing `uv` bridge env for validation
   scripts.*
8. **Adoption prep (out of scope to execute).** Short design note mapping reflect tables to the
   plugin's import path; deletion plan for the old codegen once the plugin consumes ProtoSpec.

---

## 12. Execution notes (agents and models)

Model policy: **all code-writing agents run on Opus.** Research/survey/read-only agents may use
lighter models. Design decisions, plan changes, and review synthesis stay in the main session.

What parallelizes cleanly across subagents, and what must not:

- **Serial, single-owner (no fan-out):** the IDL grammar + parser (milestone 1) and the core
  emitters (milestone 2). These define the contracts everything else consumes; splitting them
  invites drift. One Opus agent per component, sequentially.
- **Milestone 1 side-tasks (parallel, read-only):** the bootstrap extractions are independent —
  one agent over `MJCF[]` + `mjMap` tables, one over `mjspec.h` field types, one building the
  `mjs_default*` probe, one cataloging XMLreference semantics for hand-curation notes.
- **Milestone 3 (the big fan-out):** IO families (a)–(j) are designed to be independent work
  units once (b) body/geom/joint core has landed and fixed the reader/writer idioms. Family (b)
  is the pathfinder and is done first by a single Opus agent; families (c)–(j) then fan out in
  parallel, each agent owning reader + writer + corpus tests for its family, each gated by the
  differential harness. The quirk register rows (Section 8) double as the per-agent spec sheets.
- **Milestone 4 validators:** tier 1 is generated (belongs to the emitter owner); tier 2/3
  checks are small independent units, good for parallel agents.
- **Milestone 5 bridge:** single owner (compile, binding, recompile, and auto-naming interact
  too tightly to split), with the binding/recompile equivalence *tests* farmed out in parallel.
- **Milestones 6–7 (SDK, pybind):** parallel per feature once the core API is frozen.
- **Cross-cutting rule:** test harnesses (differential, fixpoint, drift gate) are built by a
  different agent than the code they test, and every fan-out ends with an adversarial review
  pass over the merged result before the milestone exit criterion is declared met.

## 13. Resolved decisions


1. **Actuators are typed, not lowered.** A loaded `<position>` is a `Position{kp, kv, ...}` you
   mutate in those terms; `General` is just one variant among eleven. Lowering exists only as a
   query helper and inside no critical path (Q-ACT).
2. **Standalone repo** while iterating; the plugin adopts later as a dependency.
3. **All macros except `include` are first-class**, via pass-through: composite, flexcomp,
   replicate, and attach are represented in their compact authored form, MuJoCo expands at
   compile, binding reaches generated elements by name pattern, `Expand()` materializes on
   demand (DR-7).
4. **Binding without the mjs compiler** is solved by name: `mj_name2id` over the compiled
   `mjModel`, auto-naming for unnamed elements, `Binding` object returned by `Compile()` (DR-10).

5. **Auto-naming is on by default** (reserved `_ps:` prefix); every element is bindable.
   Opt-out flag exists for users who care about pristine name tables.
6. **`Raise(General)` is implemented but dormant.** It ships behind an explicit opt-in and is
   never called from any load/read path — misidentifying a hand-tuned `<general>` as a shortcut
   is the risk, so raising only ever happens when the user asks for it on a specific actuator.
   Pattern scope (position/velocity/damper first) decided from real files during milestone 3(f).
7. **Recompile is in scope and loses nothing.** mjSpec's `mj_recompile` is a full recompile
   wrapped in state save/restore, not an incremental compile; ProtoSpec replicates it as
   `Recompile` with element-keyed state migration over the XML bridge (DR-11).

No remaining open questions block milestone 1.
