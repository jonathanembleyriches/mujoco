# ProtoSpec canonicalization: redundant encodings vs typed semantics

Companion to `docs/plan.md`. Owner mandate: **ProtoSpec is a minimal representation** — multiple
authored spellings of the same underlying value are fluff to be resolved at read time; genuinely
distinct semantics stay typed. This partially reverses the M3 form-preservation posture of DR-3
and the Q-ORIENT/Q-FROMTO/Q-INERTIA rows (Q-ANGLE's form preservation survives — see the boundary
cases). All MuJoCo citations are into the vendored tree at
`third_party/MuJoCo/src` (UnrealRoboticsLab plugin), MuJoCo 3.10.0.

## 1. The criterion

Multiple authored forms are an **ENCODING** (canonicalize at read; store one form; writer emits
the canonical form) iff they resolve to the same compiled representation with **no
per-consumer/contextual branching**. They are **SEMANTICS** (keep typed) iff they map to distinct
`mjModel` state, change compile behavior, or resolution is consumer-dependent.

Two refinements the audit forced, both direct applications of the criterion:

- **"Contextual" includes the defaults layer.** A spelling that is uniformly resolvable on a
  concrete element can still be per-consumer when authored in a `<default>` class, because the
  resolution context (e.g. the geom `type` that decides `fromto` size packing, or the camera
  `resolution` that converts pixel intrinsics) lives on the *consuming* element, not the authoring
  site. MJCF's default section admits orientation/fromto/intrinsics spellings on classes
  (`xml_native_reader.cc:232-247`), so every ruling below was checked at both authoring levels.
- **"Same compiled representation" is gated empirically**, not by argument: the differential
  harness (359/387 identical) recompiles every roundtrip through MuJoCo, and the roundtrip now
  carries the canonical form. If the canonical form compiled differently in any corpus file, the
  harness turns red. That harness is THE safety net for this whole plan.

Canonical-form eligibility has one hard precondition: **the canonical form must itself be a legal
MJCF spelling** (the bridge and the user-facing writer emit it back onto the wire). `quat`,
`diaginertia`, `type=`, `springlength="a b"`, `<layer role="rgb">` all qualify. This is also why
`replicate.euler` cannot be canonicalized even in principle — MJCF accepts no other spelling there.

## 2. Inventory — every ruling

| # | Candidate | Verdict | Reason (one line) |
|---|---|---|---|
| 1 | Orientation `quat`/`euler`/`axisangle`/`xyaxes`/`zaxis` (Body/Geom/Site/Camera/Frame, Inertial `ialt`) | **ENCODING → store quat** (owner-resolved) | Uniform resolution against global compiler degree/eulerseq only (`user_objects.cc:241-330`); all five → the same `*_quat` mjModel field |
| 2 | Flexcomp orientation attrs (`quat`/`axisangle`/`xyaxes`/`zaxis`/`euler` plain fields) | **ENCODING → join #1** | Same resolver, same context (`user_flexcomp.cc:93,194` → `mjs_resolveOrientation`); flexcomp is not defaultable, so no class-layer hazard |
| 3 | Inertial `fullinertia[6]` vs `diaginertia+iquat` vs `ialt` | **ENCODING → diaginertia + iquat** | Deterministic eigendecomposition `mjuu_fullInertia` (`user_objects.cc:2713-2716`); MuJoCo's own writer already only emits diaginertia (`xml_native_writer.cc:1731-1737`); inertial is not defaultable |
| 4 | Light `directional` bool vs `type` enum | **ENCODING → type** | Reader maps both onto `light->type` with mutual exclusion (`xml_native_reader.cc:2123-2131`); mjModel stores only `light_type` (`mjmodel.h:513`) |
| 5 | Tendon `springlength` scalar vs pair | **ENCODING → pair** | Reader duplicates the scalar into `[1]` before storing (`xml_native_reader.cc:2372-2374`); one `tendon_lengthspring[2]` |
| 6 | Cylinder actuator `diameter` vs `area` | **ENCODING → area** | Both fill `gainprm[0]`; `area = π/4·d²` from the element's own value (`user_api.cc:1236-1248`, `mjs_setToCylinder`) |
| 7 | Material `texture` attr vs `<layer role="rgb">` | **ENCODING → layers** | `texture=` is literally `textures[mjTEXROLE_RGB]` (`xml_native_reader.cc:1844`), layers set the same slots; reader forbids mixing (`:1849-1852`); one `mat_texid[]` per role |
| 8 | Numeric `size` > data length (zero padding) | **ENCODING → materialized data** | Reader pads data to `size` (`xml_native_reader.cc:3200-3217`); identical `numeric_adr/size/data` |
| 9 | Keyword-set attrs: Camera `output`, Rangefinder `data` (order/duplicates); SensorContact `data` | **ENCODING → enum-order, deduped** (contact: already single-spelling) | `MapValues` ORs keywords into one bitmask (`xml_native_reader.cc:2078-2084`, rangefinder branch) — order-insensitive; the contact sensor *rejects* out-of-order data ("data attributes must be in order", `:4517-4530`), so enum order is the only legal spelling there and our reader/writer must enforce/emit exactly it |
| 10 | Size `memory` K/M/G suffixes | **ENCODING — already canonical bytes** (record) | ProtoSpec reader already parses the suffix and stores the byte count (`cpp/io/mjcf_reader.cc:776-796`); MuJoCo parity at `xml_native_reader.cc:1365-1395` |
| 11 | `fromto` vs pos/quat/size (geom + site) | **KEEP — blocker found** (reverses the owner's lean; §5.2) | Class-authored fromto is corpus-real and its size packing is geom-**type**-dependent (`user_objects.cc:3992-4007`) → per-consumer through the defaults layer; element-level canonicalization bakes class-inherited `type`/`size[0]` and breaks inheritance |
| 12 | Camera `fovy` vs `focal`/`sensorsize` physical family | **KEEP** | Distinct mjModel state that renderers branch on: `cam_fovy` AND `cam_intrinsic`/`cam_sensorsize`/`cam_resolution` coexist (`mjmodel.h:501-506`); sensorsize presence changes `intrinsic` and recomputes fovy (`user_objects.cc:4419-4456`) |
| 13 | Camera `focalpixel`/`principalpixel` vs `focal`/`principal` (length) | **KEEP** | Pixel→length conversion needs the *consumer's* `resolution`+`sensorsize` (`user_objects.cc:4438-4449`); MJCF allows class-level `focalpixel` (`xml_native_reader.cc:244-246`) → per-consumer, the criterion's SEMANTICS branch verbatim |
| 14 | Joint angle FIELDS (`range`/`ref`/`springref`) + compiler `angle` attr | **KEEP** (resolved; the boundary case) | Conversion is per-consumer: `auto_limits.xml` — one class range feeds a hinge (converted) and a free joint (not), so no single pre-converted value is correct (`user_objects.cc:3207-3282`); Q-ANGLE stands |
| 15 | Position/IntVelocity `dampratio` vs `kv` | **KEEP** | Sign-encoded *different runtime models* in `biasprm[2]`: `-kv` = constant damping, `+dampratio` = per-step critically-scaled damping (`user_api.cc:1147-1164`; `engine_forward.c:596` vs `:702`); mutually exclusive, not interchangeable |
| 16 | `inheritrange` vs `ctrlrange`/`actrange` | **KEEP** | Resolved at compile from the *target* joint/tendon's resolved range (`user_objects.cc:7139-7180`) — cross-element, changes when the target is edited |
| 17 | Typed actuator spellings (`position`/`motor`/… vs `general`) | **KEEP** (owner explicit; Q-ACT) | Named-parameter forms are the authoring model; lowering `mjs_setTo*` is compile-boundary only |
| 18 | Typed equality spellings; Connect/Weld body-form vs site-form | **KEEP** | body vs site is distinct mjModel state: `eq_objtype` (`mjmodel.h:722`) set from `site1/site2` presence (`xml_native_reader.cc:2210-2285`); runtime constraint assembly branches on it — not a packing of the same thing into `eq_data` |
| 19 | Typed sensor spellings (~50 tags) | **KEEP** (owner explicit; Q-SENS) | Each tag is a distinct `(type, objtype, slots, intprm)` tuple |
| 20 | `Limited`/`inertiafromgeom`/`align` tri-states | **KEEP** (resolved; Q-AUTO) | `auto` changes compile behavior via `autolimits`+range (`islimited`, `user_objects.cc:175-187`) |
| 21 | TextureSource `builtin` vs `file` vs cube `file*`/`gridsize+gridlayout` | **KEEP** | Distinct asset inputs (procedural, one image, six images, one composite grid image — grid decode at `user_objects.cc:5437-5463`); not resolvable without loading payloads |
| 22 | Mesh sources: `file` vs inline `vertex/face` vs `builtin+params` | **KEEP** | Distinct sources, same class as #21 |
| 23 | Mesh `refpos`/`refquat`/`scale` | **KEEP** | Transform parameters applied to the (usually file-borne) vertex payload at compile (`user_mesh.cc:1258-1272`); read-time baking would require rewriting external assets and contradicts DR-4 |
| 24 | Hfield sources: `file` vs `nrow/ncol+elevation` vs procedural (size only) | **KEEP** | Distinct sources; the `[0,1]` re-normalization of elevation (`user_objects.cc:4853-4867`) is compile *value* semantics, not a spelling — rejected as a canonicalization target (would rewrite user data) |
| 25 | Keyframe `qpos` vs `mpos`/`mquat` | **KEEP** | Different state, different arrays: `key_qpos` vs `key_mpos`/`key_mquat`; not alternative spellings at all |
| 26 | Skin `bindpos`/`bindquat` | **KEEP** | Per-bone bind data, single spelling; nothing to canonicalize |
| 27 | FreeJoint element vs Joint `type="free"` | **KEEP** | Different compile behavior: `<freejoint>` is created *without* defaults (`mjs_addFreeJoint`, `xml_native_reader.cc:3701-3712`), `<joint type="free">` gets the full class merge (damping, armature, …) |
| 28 | Replicate `euler` | **KEEP** | Only spelling (wire constraint) AND a family parameterization, not one rotation: copy *i* re-resolves `i·euler` (`xml_native_reader.cc:3845-3858`, mirrored in `cpp/compile/build.cc:1251-1257`); `resolve(i·euler) ≠ resolve(euler)^i` for non-commuting sequences |
| 29 | Composite `quat` | **KEEP** | Single spelling in 3.10's reduced composite; pass-through macro anyway |
| 30 | Geom `mass` vs `density` | **KEEP** | Different parameterizations coupled through compile-time volume (`mjCGeom::SetInertia`, `user_objects.cc:3493+`; body accumulation `:2469-2514`); baking density→mass freezes the size dependence |
| 31 | Macros (`replicate`/`attach`/`composite`/`flexcomp`) | **KEEP** (resolved; DR-7) | Generative, not redundant encodings |
| 32 | Variable-arity prefixes (`size`, `friction`, `gear`, `solimp`, …) | **KEEP** (Q-ARITY) | Presence, not redundancy: unauthored trailing components inherit class/global defaults component-wise; materializing them would bake inherited values. `springlength` (#5) is the lone true duplication (reader copies the value) and is the exception that proves the rule |
| 33 | VisualGlobal `orthographic`/`fovy` | **KEEP** | Free-camera settings; a different object from `Camera.projection`, no aliasing |

Schema-hygiene riders discovered during the audit (in scope for the wave, not encodings):

- **R1 — dissolve the `CameraIntrinsics` variant.** `fovy` XOR `focal` is not MJCF's actual
  structure: the reader's exclusion is `fovy` vs `sensorsize` on the *same element*
  (`xml_native_reader.cc:2086-2090`), and class/element splits (class fovy + element sensorsize)
  are legal and resolved at compile. With #13 KEEP, the honest shape is six plain presence-tracked
  fields (`fovy, focal, focalpixel, principal, principalpixel, sensorsize`) plus the reader-level
  same-element `fovy+sensorsize` error and a tier-3 lint for `focal/principal without sensorsize`
  (compile error at `user_objects.cc:4426-4435`).
- **R2 — drop `(unit=angle)` from `Camera.fovy`.** fovy is never subject to `compiler.angle`; it
  is degrees always — and a *length* for orthographic cameras (`mjmodel.h:501` "(ortho ? len :
  deg)"; no conversion anywhere in the reader). `cpp/io/mjcf_reader.cc:742` already documents
  this; the schema annotation is wrong metadata.
- **R3 — unify springlength arity** (`TendonDefault double[2]` vs `Spatial/Fixed double[0..2]`):
  after #5 all three store `double[2]`; the wire accepts 1-or-2 values.

## 3. Resolution mechanics (shared by all ENCODING rulings)

**Where.** One canonicalization pass at **parse end**, after the whole document (includes already
flattened by the Q-INC pre-pass) is in the tree. Never during element parse: MJCF allows
`<compiler>` blocks anywhere, including after `<worldbody>`, and MuJoCo defers all
degree/eulerseq-dependent resolution to compile — resolving mid-parse would make the result
depend on document order. **Document-order independence** is the invariant: the pass computes the
effective compiler context by folding `Model.compilers` in document order (later authored
attributes override, matching MuJoCo's accumulate-into-one-spec behavior), then resolves every
collected raw form against that single context. Defaults: `angle="degree"`, `eulerseq="xyz"` when
unauthored. `<attach>`/`ModelAsset` sub-models are separate documents that ProtoSpec never parses
(pass-through; MuJoCo loads them at compile with their own compiler blocks) — no cross-document
context leakage is possible.

**With what math.** The resolvers are the lifted originals, registered in the existing lift
registry (provenance + drift gate, as NC0 did for the native compiler): `ResolveOrientation`
(`user_objects.cc:241-330` — already lifted as `ResolveQuat`, `cpp/compile/build.cc:264-267`;
relocate to `core/resolve.cc` so reader and native compiler share it) and `mjuu_fullInertia`
(eigendecomposition). Bit-identical resolution is what makes the differential harness a valid
gate: the canonical quat we write is the quat MuJoCo would have computed from the authored form.

**Reader surface (coverage unchanged).** The reader keeps accepting every authored attribute —
`euler`, `axisangle`, `xyaxes`, `zaxis`, `fullinertia`, `directional`, `diameter`, scalar
`springlength`, `texture=`, `size=` on numeric. The IDL loses the variant structs and gains a
binding annotation (new codegen rule shape) marking "these wire attributes feed this canonical
field through resolver X"; `xml_binding.cc`'s current variant-arm tables
(`cpp/generated/xml_binding.cc:208,1018,1063,…`) become resolver-routing tables. Corpus attribute
coverage (`tools/corpus_study.py`, `tests/test_corpus_coverage.py`) stays 100% because coverage
counts accepted input attributes; the study's attr→field mapping gains the wire-only rows
(euler→orient, fullinertia→{diaginertia,iquat}, directional→type, …).

**Error parity.** Reader-level errors in MuJoCo remain reader-level errors in ProtoSpec:
"multiple orientation specifiers are not allowed" (`xml_base.cc:73-75`), "type and directional
cannot both be defined" (`xml_native_reader.cc:2128-2130`), "fovy and sensorsize"
(`:2086-2090`), material texture-attr-plus-layers (`:1849-1852`). Compile-level exclusivity
errors (fullinertia vs diaginertia vs ialt, `user_objects.cc:2705-2709`) stay tier-3 validation
lints on the authored forms, checked before canonicalization erases the evidence.

**Presence interplay (defaults).** Canonicalization happens *per authored site*, classes
included: a class authoring `euler` stores `orient = quat` with presence set; inheritance is
unchanged because the orientation field is atomic in both worlds (a nested class overriding its
parent's euler — corpus witness `planks.xml` plank→joist — resolves identically). One deliberate
divergence in a pathological case: in MuJoCo, an element's authored `quat` does NOT clear a
class-inherited alt form (`ReadAlternative` counts `quat` but only the alt spellings set
`alt.type`, `xml_base.cc:54-77`), so a class euler silently *beats* an element quat at compile.
Canonical ProtoSpec is atomic element-wins. Corpus scan: the five files with class-authored alt
orientations (`cube_3x3x3.xml`, `planks.xml`, `tendon_wrap.xml` ×2, `makemesh.xml`) have no
element-quat-under-alt-class usage, so the harness stays identical; the divergence is documented
as fixing a MuJoCo wart, gated by the harness on every future corpus bump.

**Writer.** Both the user-facing writer and the bridge serialization emit only the canonical
spelling (quat, diaginertia+quat, `type=`, two-value springlength, `area`, `<layer role="rgb">`,
materialized numeric data, byte-count memory, enum-ordered keyword sets). This *narrows* the gap
between user-facing save and bridge-canonical MJCF. Q-WRITE's "user-facing save preserves form"
is amended: presence is still preserved exactly (DR-1); *spelling* within an encoding group is
canonical.

## 4. Per-ruling migration notes (ENCODING rows)

**#1+#2 Orientation → quat.**
- Schema: `variant Orientation` + `AxisAngle/XYAxes/ZAxis/Euler` structs deleted; `Posed.orient :
  Quat`; `Inertial.iorient` → `iquat : Quat`; Flexcomp's five plain fields collapse into
  `use Posed`-style `orient` (Flexcomp keeps `pos`, gains `orient`; `origin` untouched).
- Reader: parse-end resolver (§3). Writer: quat only.
- Native compiler: `build.cc` `ResolveQuat` call sites collapse to a copy (orientation is
  pre-resolved); the function itself moves to `core/resolve.cc` for the reader.
  `ReplicateEulerQuat` (`build.cc:1251-1257`) is untouched (#28 KEEP).
- Studio: `transform_math.cc` `OrientationToQuat` (:118-230) and its per-consumer calls collapse;
  the Details orientation dual-widget (quat row + authored-form arm) becomes a plain quat row;
  plan_studio_editor §4 bullet and §11 resolution 1 ("gizmo rotations materialize quat, euler
  stays editable") become vacuous — gizmos and Details now edit the same single form. DR-S6's
  delta rule is unaffected (it never inverted authored forms).
- Tests: fixpoint goldens with euler/axisangle/xyaxes/zaxis re-emit as quat; differential harness
  and native ratchet must not move.

**#3 Inertia → diaginertia + iquat.**
- Schema: `InertiaSpec` variant + `DiagInertia/FullInertia` structs deleted; `Inertial{pos, mass,
  iquat : Quat, diaginertia : double[3]}`.
- Reader: `fullinertia` → lifted `mjuu_fullInertia` eigendecomp at parse end (it writes iquat
  directly; exclusivity with authored iquat/ialt is MuJoCo's own rule, `user_objects.cc:2705-2709`,
  kept as read error matching `xml_native_reader.cc:3682-3684`).
- `saveinertial`/`explicitinertial` roundtrip: unchanged in kind — MuJoCo's writer already never
  emits fullinertia (`xml_native_writer.cc:1731-1737` emits `diaginertia` unconditionally), so the
  canonical form is exactly what MuJoCo itself round-trips; the `explicitinertial` bit remains
  "the Inertial child exists" (`xml_native_reader.cc:3676`).
- Studio Details: InertiaSpec variant widget → two plain rows.

**#4 Light type.** Schema: drop `directional : bool`; keep `type : LightType`. Reader maps
`directional` → `type` (spot/directional) with the existing mutual-exclusion error. Class layer is
safe: both spellings already write the same `light->type` field inside MuJoCo's reader, so merge
granularity is identical. Corpus goldens with `directional="true"` re-emit as `type="directional"`.

**#5 Springlength pair.** Schema: `double[2]` everywhere (R3). Reader duplicates a single authored
value exactly as `xml_native_reader.cc:2372-2374`. Writer emits both values (emitting `"v v"`
where `"v"` was authored — MuJoCo reads them identically by construction).

**#6 Cylinder area.** Schema: drop `diameter`; keep `area`. Reader: `diameter` → `area = π/4·d²`.
Known narrowing (same shape as the ReadAlternative wart): in MuJoCo a class-authored `diameter`
beats an element-authored `area` because the diameter override runs after the merge
(`user_api.cc:1241-1243`); canonical ProtoSpec is element-wins. No corpus witness (cylinder
actuators are rare); harness gates.

**#7 Material layers.** Schema: drop `Material.texture`; `layers` child list is the single form.
Reader: `texture="foo"` → prepend `layer{texture=foo, role=rgb}`. Writer emits the layer form.
Defaults note for implementers: MuJoCo merges material textures per-role slot; ProtoSpec's
Effective/inspector treatment of layer children must mirror per-role merge (the wire behavior is
preserved regardless, since classes are emitted verbatim and MuJoCo does the merging).

**#8 Numeric data.** Reader materializes `data` to length `size` (zero-padded) and drops the
authored `size` (field becomes reader-computed; schema keeps `size` only if the "size without
data" spelling must stay cheap — it need not: `size="5"` alone canonicalizes to
`data="0 0 0 0 0"`). Reader keeps MuJoCo's 1..500 bound check (`xml_native_reader.cc:3212-3215`).

**#9 Keyword sets.** Camera `output` / rangefinder `data`: reader sorts + dedupes into enum order
(bitmask OR is order/dup-insensitive); writer emits enum order. Contact-sensor `data`: MuJoCo's
reader already enforces strict enum order (`xml_native_reader.cc:4517-4530`) — ProtoSpec keeps
the same read error and the writer's enum-order emission is the only legal output anyway. Pure
value normalization; no schema change.

**#10 Memory.** Already done (`cpp/io/mjcf_reader.cc:776-796`). Record in the quirk register;
optional follow-up: type the field `uint64` instead of a stringified byte count (deferred, not
load-bearing).

## 5. Boundary cases (the audit's two reversals of intuition)

### 5.1 Joint angle fields — the documented KEEP (unchanged)
`compiler.angle` conversion is per-consumer (`auto_limits.xml`: one class `range` feeds a hinge,
converted, and a free joint, not converted — `user_objects.cc:3207-3282` converts only for
rotational consumers). Canonicalizing stored angle values is WRONG; Q-ANGLE's form preservation
stands. The orientation ruling does not contradict it: orientation resolution consumes
degree/eulerseq *uniformly* (same result for every consumer of a class-authored euler), which is
precisely the ENCODING side of the criterion.

### 5.2 fromto — recommended KEEP, reversing the owner's lean (flagged, §7)
The single-element resolution is beautifully uniform (`user_objects.cc:3977-4016`, site twin at
`:4237-4276`: midpoint pos, `mjuu_z2quat` orientation, half-length into the size vector). The
blockers are all in the defaults layer:

1. **Class-authored fromto is per-consumer.** The size packing branches on geom `type`
   (capsule/cylinder: `size[1]=len/2`; box/ellipsoid: `size[2]=len/2, size[1]=size[0]` —
   `user_objects.cc:3992-4007`), and `type` belongs to the consumer. It is corpus-real:
   `test/engine/testdata/actuation/actuator_group_disable.xml` authors
   `<default class="decor"><geom type="cylinder" size=".03" fromto="0 -.03 0 0 .03 0"/></default>`
   (several other test files similarly). A class fromto cannot be resolved at its authoring site,
   so the FromTo representation would have to survive in classes — defeating "store one form".
2. **Element-level canonicalization bakes inherited context.** `humanoid.xml` is the common
   pattern: class authors `type="capsule"`, elements author `fromto` (+ sometimes only
   `size[0]`). Resolving at read requires `Effective(type)` and produces a fully-authored
   pos/quat/size — freezing class-inherited `type`/`size[0]` into the element, so editing the
   class no longer propagates (breaks the Studio defaults-editing contract and `Effective()`
   semantics).
3. **Presence-granularity divergences.** `fromto` is one attribute that atomically implies
   pos+quat+size-tail; the canonical form is three separately-inheritable fields. Class fromto +
   element pos (MuJoCo: compile error `user_objects.cc:3987-3990`; canonical: silently compiles),
   class pos + element fromto (MuJoCo: error; canonical: compiles), class fromto + element
   orientation (MuJoCo: fromto wins; canonical: element wins).

Consequence of KEEP: the `GeomShape` variant stays; the fromto gizmo special-casing in
`apps/studio/editor/transform_math.cc:346-505` (endpoint translate/rotate) stays — the promised
deletion does not happen. If the owner still wants element-level-only canonicalization, it must
carry the class exception + an `Effective(type)` parse-end lookup + two tier-3 lints to preserve
MuJoCo's error surface — recorded as the rejected halfway option (non-uniform, keeps both forms
in the schema).

### 5.3 Camera pixel intrinsics — KEEP by the class-layer test
`focalpixel → focal` is a same-element unit conversion (`pixel · sensorsize / resolution`,
`user_objects.cc:4438-4449`) — but a class may author `focalpixel` (`xml_native_reader.cc:244-246`)
while its consumers author different `resolution`/`sensorsize`, giving a different length per
consumer. Per-consumer ⇒ SEMANTICS. Keep all four fields; R1 (§2) fixes the variant shape instead.

## 6. The plan: two waves

**Wave A — the Orientation family (one shared resolver).**
Rows #1, #2, #3 (+ R2). One mechanism: parse-end resolution pass, effective-compiler fold,
`core/resolve.cc` with lifted `ResolveOrientation`/`mjuu_fullInertia` (lift-registry entries +
drift gate), the new resolver-routing binding annotation, writer emits quat/diag+iquat. Schema:
variants deleted, `draft_schema.py` curation data updated in the same commit (both `--check`
regeneration gates green). Downstream: `build.cc` consumes pre-resolved quats;
`transform_math.cc` + Details simplify; fixpoint goldens updated.

**Wave B — scalar/slot canonicalizations + hygiene. LANDED.**
Rows #4-#10 (light type, springlength, cylinder area, material layers, numeric data, keyword-set
order, memory recorded) + R1 (CameraIntrinsics tier-3 lint) + R3 (arity unify) + documentation
amendments. No shared machinery; each is a local reader/writer/schema change gated by the same
invariants.

STATUS (Wave B): DONE. Per-row landing:
- **#4 Light `directional` -> `type`** — `directional` bool dropped; `type` gains
  `aliases="directional" resolver=lighttype`. Reader `ResolveLightType` maps directional
  true/false -> directional/spot with the `type and directional cannot both be defined` error
  (xml_native_reader.cc:2123-2131). Native `LightCompile` reads the canonical `type`. Writer emits
  `type=` only.
- **#5 Tendon `springlength` scalar -> pair (with R3)** — Spatial/Fixed unified from `double[0..2]`
  to `double[2]` (R3); reader `ResolveSpringlength` (SpringlengthField trait over Spatial/Fixed/
  TendonDefault) duplicates a lone value into the second slot (xml_native_reader.cc:2372-2374).
  build.cc `TendonAuthored.springlength` retyped to `std::array<double,2>`.
- **#6 Cylinder `diameter` -> `area`** — `diameter` dropped; `area` gains `aliases="diameter"
  resolver=cylinderarea`. Reader folds `area = pi/4 d^2` (a non-negative diameter wins,
  element-wins-atomic; owner-approved Section 7). Native passes diameter=-1 to the lifted
  SetToCylinder. No corpus witness; gated by the harness.
- **#7 Material `texture` -> `<layer role="rgb">`** — `Material.texture` field dropped; `texture=`
  is an **element-level input alias** (emit.py `ELEMENT_INPUT_ALIASES`, the sole case where the
  alias folds into a child list, not a field, so it cannot hang off a field annotation). Reader
  `MaterialTextureFixup` (post-children) prepends the RGB layer and rejects mixing texture-attr with
  authored `<layer>` (xml_native_reader.cc:1849-1852). corpus_study + schema-coverage gates teach
  the same table.
- **#8 Numeric `size` -> materialized data** — `size` field dropped; `data` gains `aliases="size"
  resolver=numericdata`. Reader materializes `data` to the authored size (zero-pad/truncate), keeps
  the 1..500 bound (xml_native_reader.cc:3200-3217). Native `NumericCompile` simplified (size ==
  data length). The `size`-only reserve nuance is erased (owner-approved).
- **#9 Keyword-set order** — reader sorts every enum keyword set into enum-declaration order (=
  MapValues bitmask, order-insensitive) so camera `output` / rangefinder `data` store canonically;
  the contact sensor keeps its strict in-order read error (xml_native_reader.cc:4517-4530), checked
  on the raw input before the sort. No schema change. Corpus is already enum-ordered, so the
  differential/roundtrip do not move.
- **#10 Memory** — already canonical bytes (mjcf_reader.cc MemoryFixup); recorded, no change.
- **R1 CameraIntrinsics tier-3 lint** — `focal/principal require a positive sensorsize` (a MuJoCo
  compile error, user_objects.cc:4426-4435) added to cpp/validate tier 3 on the authored form
  (the same-element `fovy+sensorsize` reader error landed in Wave A).

Baselines held: pytest 1625/120, differential 359/387 identical (0 differ, 28 skips), native
ratchet 201, corpus coverage 100% (all axes), cpp ctest 6/6, studio ctest 5/5, emit --check + lift
registry green. Pre-existing reds unchanged: `draft_schema.py --check` (schema hand-maintained past
the generator), test_python_bindings skips (venv/pyd version mismatch).

**Exit criteria — the invariants, per wave (no wave merges until all hold):**
1. Differential harness: **359/387 identical, 0 differ, 28 honest skips** — unchanged. The
   roundtrip now carries canonical forms; MuJoCo compiling them identically is the proof of every
   ENCODING verdict.
2. Native ratchet stays **201** (`tests/native_ratchet.json`); `ps_native_diff` green.
3. Corpus attribute coverage stays **100%** (input surface unchanged; corpus-study mapping
   extended with wire-attr→canonical-field rows).
4. Fixpoint suite green: `Read(Write(Read(M))) == Read(M)` (canonicalization is idempotent by
   construction — canonical forms resolve to themselves) and byte-deterministic double-write;
   goldens updated in a reviewed, deliberate diff.
5. Drift gate + `draft_schema.py --check` + `protospec_gen.emit --check` green.
6. Docs amended: DR-3 and the Q-ORIENT/Q-FROMTO/Q-INERTIA/Q-ANGLE quirk rows restate the policy
   as **"typed semantics preserved; redundant encodings canonicalized at read"**, with Q-ORIENT/
   Q-INERTIA marked canonicalized, Q-FROMTO marked KEEP-with-blocker (this file), Q-ANGLE
   unchanged; Q-WRITE amended per §3; plan_studio_editor §4/§11.1 riders noted.

## 7. Flagged for owner

**All three flags OWNER-APPROVED 2026-07-13** (resolutions inline below).

1. **fromto reversal (§5.2).** The owner leaned "include unless blocker"; the audit found one
   (class-authored fromto is per-consumer via type-dependent size packing; corpus-real). Sign-off
   requested on KEEP — or on the non-uniform element-only variant with its documented caveats.
   **APPROVED 2026-07-13: fromto stays KEEP** (the `GeomShape` variant and the fromto gizmo
   special-casing are retained; Wave A does not touch them).
2. **Pathological-precedence divergences** introduced deliberately: element-quat vs class-alt
   (§3, ReadAlternative wart) and element-area vs class-diameter (#6). No corpus witnesses; both
   make ProtoSpec's merge element-wins-atomic where MuJoCo's is spelling-priority. Confirm this is
   acceptable as a documented fix-of-wart rather than something to emulate.
   **APPROVED 2026-07-13: both divergences accepted** as documented fix-of-wart. The element-quat
   vs class-euler case is realized in Wave A (element-wins-atomic; reader header documents it);
   the class-diameter case is Wave B (cylinder area).
3. **Numeric `size` deletion (#8):** materializing `size="5"` (no data) as five zeros erases the
   "reserve without initializing" authoring nuance (compiled result identical). Cheap to keep the
   field instead; default is delete per the minimality mandate.
   **APPROVED 2026-07-13: numeric padding accepted.** This is Wave B (scalar/slot canonicalizations),
   not Wave A — no action here.
4. ~~Keyword-set ordering (#9)~~ — resolved during the audit: camera `output`/rangefinder `data`
   are pure OR bitmasks (`xml_native_reader.cc:2078-2084`); the contact sensor already rejects
   out-of-order `data` (`:4517-4530`, output slots keyed by `mjCONDATA_*` bits in
   `engine_sensor.c:455-473`), so enum order is the sole legal spelling there. No open check.
