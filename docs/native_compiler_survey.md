# Native Compiler Survey: ProtoSpec -> mjSpec -> mj_compile

Reuse audit for replacing DR-5's XML hop (ProtoSpec -> canonical MJCF -> mjVFS -> `mj_loadXML`)
with a direct walk: ProtoSpec tree -> `mjs_*` C API -> `mj_compile`. The walker is the only new
code; every compiler pass (mesh/qhull, BVH/octree, inertia, defaults, lengthrange, fusestatic,
keyframes) is inherited unchanged, and the XML path remains as the differential oracle.

All file references are into the vendored tree at
`C:\Users\jonat\Documents\Unreal Projects\url_proj\Plugins\UnrealRoboticsLab\third_party\MuJoCo\src`
(cited below as paths relative to that root, e.g. `src/user/user_api.cc:386`). The pin is
main-HEAD-ish `mjVERSION_HEADER 3010000` (3.10.0 dev); its changelog "Upcoming" section ends at
the `mju_threadpool`/logging entries (`doc/changelog.rst:5-57`), i.e. **before** the upstream
attach-policy work. Sections marked **UPSTREAM** are sourced from the web with URLs.

---

## 1. Attach: vendored semantics and the new upstream policy flags

### 1.1 Vendored pin: `mjs_attach` (no policy argument)

```c
MJAPI mjsElement* mjs_attach(mjsElement* parent, const mjsElement* child,
                             const char* prefix, const char* suffix);   // mujoco.h:1609-1610
```

Definitively: **no policy/flag argument exists in this pin.** Case-insensitive searches for
`AttachPolicy | attach_policy | mjsAttach | mjtAttach | attachpolicy` across `include/mujoco` and
`src/user` return nothing; the only "policy" in `mjspec.h` is the unrelated `mjtSleepPolicy sleep`
(`include/mujoco/mjspec.h:275`). Behavior is controlled solely by the global per-spec deep-copy
flag `mjs_setDeepCopy` (`mujoco.h:315`).

Dispatch (`src/user/user_api.cc:386-461`) — legal parent/child combinations:

| parent | child | mechanism | citation |
|---|---|---|---|
| frame | body | `mjCFrame::operator+=(mjCBody)` via `attachBody` | user_api.cc:415-418, 305-319 |
| frame | frame | `attachFrame` onto the frame's parent body, then `mjs_setFrame` | user_api.cc:419-432 |
| body | frame | `attachFrame` (`mjCBody::operator+=(mjCFrame)`) | user_api.cc:437-440 |
| site | body | synthesizes a frame at the site pose (alt resolved via `mjs_resolveOrientation` with `compiler.degree`/`eulerseq`), then `attachBody` | user_api.cc:343-359, 445-448 |
| site | frame | same synthesis, then `attachFrame` + `mjs_setFrame` | user_api.cc:363-383, 449-451 |
| — | **whole spec** (`mjOBJ_MODEL`) | child's `world` body contents (body/site/frame/joint/geom/light/camera) are wrapped in a new frame; attach proceeds as frame child. Error if child has no world body. | user_api.cc:397-413 |

Anything else errors ("parent element is not a frame, body or site", user_api.cc:456-458).
Failures set the parent model's error and return NULL; retrieve via `mjs_getError`
(user_api.cc:466-473).

**Namespacing.** `prefix`/`suffix` are stashed on the child (`user_api.cc:308-309, 326-327`) and
applied as `name = prefix + name + suffix` (`mjCBase::NameSpace`, `src/user/user_objects.cc:1490-1496`)
to: the recursive body subtree incl. joints/geoms/sites/cameras/lights/frames and plugin instance
names (`mjCBody::NameSpace_`, user_objects.cc:1955-2000); geom asset refs material/hfield/mesh
(`mjCGeom::NameSpace`, user_objects.cc:3396-3409); the child's whole default-class tree
(`mjCDef::NameSpace`, user_objects.cc:1381-1388); keyframes (`prefix + name + suffix` in
`StoreKeyframes`, `src/user/user_model.cc:4229`); copied assets and all cross-tree referencing
elements — pairs, excludes, tendons, equalities, actuators, sensors, tuples, flexes — during
`mjCModel::operator+=` -> `CopyList` (user_model.cc:271, 462-480).

**Name conflicts.** Attach itself does **not** check collisions — the merge runs
`ProcessLists(/*checkrepeat=*/false)` (user_model.cc:456). Duplicates surface only at compile
time via `CheckRepeat` (sort + adjacent_find, throws `repeated name '<name>' in <objtype>`,
user_model.cc:4592-4618; invoked from `TryCompile` at :5057). Separately, `CopyList` **silently
skips** referencing elements whose targets don't resolve in the merged model (user_model.cc:266-282
— "if not present, skip the element"): a dangling sensor/actuator/tendon is *dropped*, not
diagnosed. ProtoSpec's tier-2 referential validation should therefore run before native attach.

**Deep vs shallow.** `mjs_setDeepCopy(s, flag)` sets `mjCModel::deepcopy_`
(user_api.cc:529-533; `src/user/user_model.h:328`). Fresh `mj_makeSpec()` defaults to **shallow**
(user_model.cc:171); the XML reader temporarily forces deep while parsing `<worldbody>`
(`src/xml/xml_native_reader.cc:1175-1184`), so XML `<attach>`/`<replicate>` are deep copies.
Shallow attach **mutates the source**: the original elements are re-parented and their names
prefixed in place (user_objects.cc:2955-2967, 1706-1743, 1794-1809), the source spec is refcounted
alive (`AppendSpec` + `AddRef`, user_objects.cc:1685-1688, 2941-2944) and marked attached
(`SetAttached`, user_model.h:331), after which **compiling the source throws** "cannot compile
child spec if attached by reference to a parent spec" (user_model.cc:4671-4673). Deep copy
allocates new objects and leaves the source untouched and compilable (user_objects.cc:1751, 2955-2957).

**Merging.** Child defaults are namespaced and hung under the parent's `main` when orphaned
(user_model.cc:644-659); **all** child assets are copied whether referenced or not (user_model.cc:462-466,
documented known issue), skipped for self-attach (user_model.cc:459-461). Keyframes become
*pending* (`StoreKeyframes`, user_model.cc:4213-4237) and are materialized/resized only at compile
(user_model.cc:5048-5054, 5197-5202); attaching twice without an intervening compile loses the
first attachment's keyframes (`doc/programming/modeledit.rst:170-172`).

**`<replicate>` is literally a loop of self-attach**: template subtree + frame, then `count` calls
of `mjs_attach(body, frame, "", zero_padded_suffix)` with accumulated pos/quat offsets, then
delete the template (`src/xml/xml_native_reader.cc:3806-3874`, attach call at :3865). `<attach>`
maps to `mjs_attach(frame, child, prefix, "")` (:3928-3960, call at :3952). This is the exact
recipe for ProtoSpec's native Replicate/Attach decode and for `Expand()`.

### 1.2 UPSTREAM: attach conflict policy (post-pin; the user's tip)

Sources:
- changelog: https://raw.githubusercontent.com/google-deepmind/mujoco/main/doc/changelog.rst
- header: https://raw.githubusercontent.com/google-deepmind/mujoco/main/include/mujoco/mjspec.h
- docs: https://mujoco.readthedocs.io/en/latest/programming/modeledit.html

The upstream "Upcoming version" adds, beyond our pin:

1. **`compiler/conflict` attribute + `mjtConflict` enum** — "Added the compiler/conflict attribute
   for controlling how conflicting global attributes are resolved during attachment." Upstream
   `mjspec.h` (verbatim):

   ```c
   typedef enum mjtConflict {         // conflict resolution for attach
     mjCONFLICT_WARNING = 0,          // keep parent, warn on conflict
     mjCONFLICT_MERGE,                // merge: min/max/error per field
     mjCONFLICT_ERROR,                // error on any conflict
   } mjtConflict;
   ```

   carried as a new `mjtConflict conflict;` field in `mjsCompiler` (between `alignfree` and
   `LRopt`). Semantics per docs: a conflict = both parent and child have *authored* differing
   values for the same global field (option/visual/size); `warning` keeps parent values and warns
   (default, pre-existing behavior); `merge` applies per-field strategies — **min** for
   timestep/tolerances/znear/realtime, **max** for iteration counts and size fields
   (memory, nkey, nuser_*), **OR** for disableflags/enableflags/disableactuator, **error** for
   gravity/wind/magnetic/density/viscosity/integrator/cone/jacobian/solver/impratio/o_* — and
   child-only authored values are adopted; `error` fails on any conflict. This is exactly what the
   vendored pin's `mjsAuthored` bitmasks (`include/mujoco/mjspec.h:181-192`) were laid in to support.
2. **Attach authoring flags on the XML element**: `<attach>` supports **self-attachment** (omit
   `model`) and a new **`frame` attribute** (mutually exclusive with `body`).
3. `mjtByte -> mjtBool` migration across `mjsCompiler` and boolean API surface (signature churn
   for the drift gate to watch).

**Plan impact:** ProtoSpec's first-class Attach element should carry `conflict : ConflictPolicy`
(opt, tri-state) and `frame` alongside `body` **now**, even though the vendored pin ignores it —
authored bitmask fidelity (Section 2.1) is the prerequisite for `merge`/`error` doing anything
useful after the next MuJoCo bump. The C-API `mjs_attach` signature itself is unchanged upstream;
policy rides on the parent spec's compiler block, which fits our compiler-block decode.

---

## 2. The mjs_* decode surface, by ProtoSpec element family

General mechanics that apply everywhere:

- Every `mjsXxx` is a plain struct of value fields plus `mjString*`/`mj*Vec*` handles
  (`include/mujoco/mjspec.h:35-55`); scalars/arrays are written directly, var-length fields go
  through `mjs_setString/StringVec/InStringVec/appendString/setInt/appendIntVec/setFloat/
  appendFloatVec/setDouble/setBuffer` (`mujoco.h:1851-1884`).
- **Ids follow add order.** `ProcessLists_` assigns `id = position in list`
  (`src/user/user_model.cc:4570-4581`, comment "id equals position in array" at :4576-4577), and
  the per-type lists are appended in `mjs_add*` call order. Decoding ProtoSpec's ordered union
  lists (actuators, sensors, equalities, tendons, path items) in list order therefore reproduces
  MuJoCo's id and `sensor_adr` assignment exactly — same guarantee the XML reader relies on.
- **Names**: `mjs_setName` checks duplicates immediately via `CheckRepeat` and returns -1 with the
  error on the spec (user_api.cc:2151-2166). Note `CheckRepeat` sorts the whole type list per call
  (user_model.cc:4592-4618) — O(n log n) per setName; fine at robot scale (10^3), measurable at
  10^5. Auto-names (DR-10 `_ps:` serials) are set the same way.
- **Error surface**: element `info` strings are "message appended to compiler errors" (every
  `mjsXxx.info`, e.g. mjspec.h:279) — the XML reader stores line numbers there; **ProtoSpec writes
  its `SourceLoc` rendering into `info`** and MuJoCo's own diagnostics become file:line-cited for
  free. Spec-level: `mjs_getError` / `mjs_isWarning` / `mjs_getTimer` (mujoco.h:1013-1019).
  Compile failure deletes the partial model, `Clear()`s, stores `errInfo`, returns NULL — the
  spec **remains usable for retry** (user_model.cc:4646-4712, error path :4688-4706).

Family-by-family map:

| ProtoSpec family | creation | setters / notes | impedance |
|---|---|---|---|
| Model root / blocks | `mj_makeSpec` (mujoco.h:303), `mjs_defaultSpec` (:1946) | `spec->option`, `spec->visual`, `spec->stat` are embedded engine structs written directly (mjspec.h:204-206); sizes `memory/nuserdata/nuser_*/nkey/...` direct (mjspec.h:209-224); `modelname/comment` via `mjs_setString` | **Authored bitmasks**: presence-tracked fields must also set `spec->authored.*` bits (mjspec.h:181-192) and `compiler.authored` (mjspec.h:177) — the XML reader writes them directly (`src/xml/xml_native_reader.cc:1307-1347`), so can we. Statistic unset = NaN sentinel (leave default). Needed for faithful `mj_saveXML` and the upstream conflict policy. |
| Compiler block | part of spec | all flags direct on `spec->compiler` (mjspec.h:158-178) incl. `eulerseq[3]`, `LRopt`, `meshdir/texturedir` strings | **Set `compiler.degree = 0`.** See Section 4 unit notes — `mj_makeSpec` defaults to degrees. |
| Body tree | `mjs_addBody(body, def)` (mujoco.h:1617) | pos/quat direct; orientation alt via `mjsOrientation alt` (mjspec.h:237-243, 262); inertial: mass/ipos/iquat/inertia (+`ialt`) or `fullinertia[6]` — NaN sentinels mean unset (plan Section 2) | ProtoSpec `Orientation`/`InertiaSpec` variants lower by *tagging* `alt.type` + writing the matching array, or writing `quat` for the quat form — compile resolves via `ResolveOrientation` at user_objects.cc:2698/2721. Exclusivity is ours by construction. |
| Frames | `mjs_addFrame(body, parentframe)` (mujoco.h:1643); membership via `mjs_setFrame(elem, frame)` (:1915) | pos/quat/alt/childclass as body | ProtoSpec Frame is a *container* with children; mjs frames are flat per-body objects + per-element frame assignment. Decode: create frame, create children on the owning body **in document order**, `mjs_setFrame` each. `mjs_bodyToFrame` (:1922) exists for SDK use. |
| joint/geom/site/camera/light | `mjs_addJoint/Geom/Site/Camera/Light(body, def)` (mujoco.h:1621-1640), `mjs_addFreeJoint` (:1628) | all fields direct in mjsJoint/Geom/Site/Camera/Light (mjspec.h:293-456); `fromto[6]` NaN-sentinel alternative; camera's six intrinsic arrays direct; `userdata` via `mjs_setDouble` | GeomShape/CameraIntrinsics variants lower to field writes. Hinge/ball `range`, `ref`, `springref` are angle-converted at compile per `compiler.degree` (user_objects.cc:3216-3220, 3277-3281) — radians-in requires degree=0. |
| Assets: mesh | `mjs_addMesh(s, def)` (mujoco.h:1760); builtin primitives `mjs_makeMesh(mesh, mjtMeshBuiltin, params, n)` (:1776) | file/content_type strings; `uservert/usernormal/usertexcoord/userface/...` via `mjs_setFloat`/`mjs_setInt` (mjspec.h:521-526); `maxhullvert`, `inertia` enum, `octree_maxdepth` direct | In-memory mesh buffers fully supported — no VFS needed for user-vertex meshes; file meshes resolve through the `mjVFS*` passed to `mj_compile` (mujoco.h:156), same as today. |
| Assets: hfield | `mjs_addHField` (:1763) | `userdata` elevation buffer is `mjFloatVec*` via `mjs_setFloat` + nrow/ncol (mjspec.h:534-543) | No gap — user elevation data is first-class. |
| Assets: texture | `mjs_addTexture` (:1769) | 4-way source: builtin fields; single file; `cubefiles` via `mjs_appendString`; **raw bytes** `data : mjByteVec*` via `mjs_setBuffer` (mjspec.h:572-606, mujoco.h:1854) | Buffer path: `nchannel/width/height` + bytes — our TextureSource::Buffer maps 1:1. |
| Assets: material | `mjs_addMaterial(s, def)` (:1773) | `textures` is a role-indexed `mjStringVec*`: use `mjs_setInStringVec(vec, role, name)` (mujoco.h:1863) | Vector must be pre-sized (mjs_setInStringVec `mju_error`s on OOB, user_api.cc:2189-2196 — bounds-check first; default material has it sized mjNTEXROLE). |
| Assets: skin | `mjs_addSkin` (:1766) | vert/texcoord/face via set-vec; `vertid : mjIntVecVec*` via `mjs_appendIntVec`, `vertweight : mjFloatVecVec*` via `mjs_appendFloatVec` (mjspec.h:547-569, mujoco.h:1872-1878) | Append-per-body order defines binding order — preserve list order. |
| Defaults tree | `mjs_addDefault(s, classname, parent)` (mujoco.h:1714); root via `mjs_getSpecDefault` (:1818); lookup `mjs_findDefault` (:1815) | mutate the returned `mjsDefault`'s per-family pointers directly (`def->geom->type = ...`) — they alias the class's internal dummy specs (mjsDefault struct mjspec.h:820-834; wiring user_objects.cc:1414-1440). New class starts as a copy of its parent (`AddDefault` -> `CopyWithoutChildren`, user_model.cc:1533-1559). | **Ordering constraint**: class values are applied *eagerly at element construction* — `mjs_addGeom(body, def)` copies the whole family struct then we overwrite authored fields (`*this = _def->Geom()`, user_objects.cc:3054-3070 etc.). `mjs_setDefault(elem, def)` after creation sets **only the class label**, no values (user_api.cc:1792-1795). Decode defaults first, fully, then elements. Matches ProtoSpec `Effective()` semantics exactly. No C-API enumeration of a class's children/parent exists — irrelevant for decode (we own the tree), blocks only mjSpec->ProtoSpec import (explicit non-goal). |
| Contact | `mjs_addPair(s, def)` (:1672), `mjs_addExclude` (:1675) | geomname1/2, bodyname1/2 strings; params direct (mjspec.h:625-647) | None. (Pairs/excludes are re-sorted by signature at compile, user_model.cc:5142-5146 — why plan keeps them non-union.) |
| Equality | `mjs_addEquality(s, def)` (:1679) | type + `data[mjNEQDATA]` + objtype + name1/name2 (mjspec.h:650-661) | **ProtoSpec's typed spellings must be lowered into `data[]` by us** — the packing lives in the XML reader, not the API (`src/xml/xml_native_reader.cc:2192-2311`): connect anchor->data[0..2]; weld anchor->data[0..2], relpose(7)->data[3..9], torquescale->data[10]; joint/tendon polycoef(5)->data[0..4]; flexstrain cell->data[0..2]; site-vs-body semantics select `objtype` mjOBJ_SITE/mjOBJ_BODY. Small, table-driven, needs golden tests. This revises Q-EQ's "no data-vector handling anywhere" for the native path. |
| Tendons + path | `mjs_addTendon(s, def)` (:1683); path items appended **in call order** via `mjs_wrapSite/Geom/Joint/Pulley` (mujoco.h:1686-1695 -> `mjCTendon::Wrap*` push_back, user_objects.cc:6470-6520) | read-back exists too: `mjs_getWrapNum/getWrap/getWrapTarget/getWrapSideSite/getWrapDivisor/getWrapCoef` (mujoco.h:1900-1906, 1837-1846) | ProtoSpec PathItem union list decodes by appending in order — construction surface is complete; there is no reorder/insert API, but a decoder never needs one. |
| Actuators | `mjs_addActuator(s, def)` (:1653) | typed variants -> `mjs_setToMotor/Position/IntVelocity/Velocity/Damper/Cylinder/Muscle/Adhesion/DCMotor` (mujoco.h:1720-1753); General -> write gaintype/biastype/dyntype/prm arrays directly (mjspec.h:704-747); transmission/limits direct | **Zero lowering risk**: MuJoCo's own XML reader calls the very same `mjs_setTo*` functions (`src/xml/xml_native_reader.cc:2500-2640`), so ProtoSpec's typed-variant lowering is byte-identical to the XML path by construction. This also retires most of Q-ACT's differential-testing burden for the native path. |
| Sensors | `mjs_addSensor` (:1656) | direct tuple fields: type/objtype/objname/reftype/refname/intprm + datatype/needstage/dim + cutoff/noise/nsample/interp/delay/interval (mjspec.h:750-780); `mjs_sensorDim` (:1941) for validation | The Q-SENS per-tag attr routing (ours) produces these tuples; identical to what the reader computes. |
| Keyframes | `mjs_addKey` (:1707) | time + six `mjDoubleVec*` via `mjs_setDouble` (mjspec.h:807-817) | Compile validates sizes **strictly** — wrong-size qpos/qvel/act/ctrl/mpos/mquat throws with a clear message (user_model.cc:4238-4259); attach paths pad short vectors from qpos0 (`ExpandKeyframe`, :4903-4936). Our `core/sizes.cc` predictions + tier-3 lint stay the front line. |
| Custom | `mjs_addNumeric/Text/Tuple` (:1698-1704) | numeric data+size; tuple parallel `objtype:mjIntVec`/`objname:mjStringVec`/`objprm:mjDoubleVec` (mjspec.h:783-804) | None. |
| Plugins / extension | `mjs_addPlugin` (:1710) + `mjs_activatePlugin(s, name)` (:312); per-element `mjsPlugin` sub-struct (name=instance, plugin_name, active — mjspec.h:246-252) | config key-values via `mjs_setPluginAttributes(plugin, void*)` (:1884) | The `void*` **must be a `std::map<std::string,std::string,std::less<>>*`** — that is what the reader passes (`src/xml/xml_native_reader.cc:129-151`) and what the impl moves out of (user_api.cc:2298-2302). A C++-shape contract behind a void*: works for us (we link from source), but pin-fragile; isolate in one function + drift-gate it. Ordering rule (explicit instances before implicit) is ours to enforce, as today (`hasImplicitPluginElem`, mjspec.h:230). |
| Flex / flexcomp | `mjs_addFlex` (:1659); **`mjs_makeFlex`** — the C-API flexcomp (mujoco.h:1663-1668, impl calls `mjCFlexcomp::Make`, user_api.cc:752) | full mjsFlex field set (mjspec.h:459-507) | `mjs_makeFlex` covers the common flexcomp params (type/dof/count/spacing/scale/radius/mass/pos/quat/file...) but not the full XML element (pin lists, edge/elasticity/contact sub-blocks) — for full fidelity link `mjCFlexcomp` internally or route flexcomp models to hybrid fallback in early phases. |
| Composite | **no public API** — the only compiler-side entry is internal `mjCComposite::Make(spec, body, error, sz)` (`src/user/user_composite.h:57`, called from the reader at xml_native_reader.cc:2681-2843) | | Gap. Options: (a) link `mjCComposite` (internal, moderate churn risk), (b) hybrid fallback: models containing composite compile via the existing XML path. Recommend (b) first, (a) in the macro phase. |
| Attach / Replicate elements | `mjs_attach` (Section 1); Replicate = loop of self-attach exactly as the reader does (xml_native_reader.cc:3846-3868: zero-padded suffix, accumulated pos/quat per iteration, deepcopy on, delete template) | | Set `mjs_setDeepCopy(s, 1)` during decode so ProtoSpec-owned sub-specs are never mutated/locked (Section 1 shallow-attach hazards). |
| Enumeration/query (SDK, Expand) | `mjs_firstChild/nextChild` (recurse flag), `mjs_firstElement/nextElement`, `mjs_findBody/Element/Child/Frame`, `mjs_getParent/getFrame/getId`, `mjs_as*` casts (mujoco.h:1781-1846, 2024-2093) | | Sufficient to walk a compiled-from-attach spec for native `Expand()`; the one hole is default-tree topology (above). |

---

## 3. Reusable utility inventory

Classification: **public** = `MJAPI` in `include/mujoco/mujoco.h` (stable C API); **internal** =
C++ under `src/user/` — linkable since we build MuJoCo from source, but no ABI/semantic stability
promise; treat each internal dependency as a drift-gate item (DR-12 extension).

### 3.1 Free via `mj_compile` (no code to write, no linkage risk)

The whole point of the native path — everything below runs inside `mjCModel::Compile`
(`src/user/user_model.cc:4646` -> `TryCompile` :4976-5388) on a spec we built via `mjs_*`:

| machinery | entry point | citation |
|---|---|---|
| Mesh load (OBJ/STL/MSH), dedup, processing | `mjCMesh::Compile/Process` | user_mesh.cc:674, 1350 |
| **Convex hull via qhull** (`"qhull Qt"`, maxhullvert -> `Q9 TA<n>`) | `mjCMesh::MakeGraph` | user_mesh.cc:1726, 1731, 1820-1833 |
| Mesh volume/CoM/inertia (volume, shell, exact, legacy) + principal-axis reorient | `ComputeVolume/ComputeSurfaceArea/ComputeInertia` + `mjuu_eig3` | user_mesh.cc:1192, 1230, 1575 |
| Primitive fitting to mesh (`fitscale`, fitaabb) | `mjCMesh::FitGeom` | user_mesh.cc:944 |
| BVH per mesh/body/flex | `mjCBoundingVolumeHierarchy::CreateBVH`; `mjCBody::ComputeBVH` | user_objects.cc:399, 2623 |
| Octree + SDF coeffs (needsdf, octree_maxdepth) | `mjCOctree::CreateOctree/ComputeSdfCoeffs` | user_objects.cc:616, 831 |
| Builtin textures (gradient/checker/flat, 2D + cube) | `mjCTexture::Builtin2D/BuiltinCube` | user_objects.cc:5040, 5113 |
| Body inertia from geoms, boundmass/balanceinertia | `mjCBody::InertiaFromGeom` | user_objects.cc:2458 |
| fusestatic / discardvisual with id compaction | `FuseStatic`; `IndexAssets(discardvisual)` | user_model.cc:4364/5109; 5060/5148 |
| autolimits tri-state resolution | `checklimited` | user_objects.cc:175 (joints :3203, tendons :6709, actuators :7186-7194) |
| **Lengthrange pipeline** (opt. threaded) | `mjCModel::LengthRange` -> `mj_setLengthRange` per actuator | user_model.cc:2444-2510 (threaded :2477+), invoked :5306 |
| Keyframe sizing/padding/normalization | `ExpandAllKeyframes`/`ResolveKeyframes` + quat normalize | user_model.cc:5048, 5197, 5294-5297 |
| Statistics (extent/meaninertia), arena sizing, sparsity, validation step | TryCompile stages | user_model.cc:5308-5320, 5219-5280, 5322-5365 |
| Compile timing diagnostics | `mjs_getTimer` (public) + `mjtCTimer` | mujoco.h:1016; mjspec.h:132-147 |

### 3.2 Public API worth calling directly

| function | header:line | role for us |
|---|---|---|
| `mjs_resolveOrientation(quat, degree, seq, orient)` | mujoco.h:1918 | ProtoSpec `Orientation::ToQuat` resolver — same code compile uses (wrapper of `ResolveOrientation`, user_objects.cc:241); returns error string or NULL |
| `mju_euler2Quat / mju_axisAngle2Quat / mju_mulQuat / mju_quat2Mat / mju_rotVecQuat / mju_quatZ2Vec` ... | mujoco.h:1224-1275 | quaternion math for SDK/resolvers |
| `mju_muscleGain/Bias/Dynamics` | mujoco.h:1363-1371 | muscle queries (Lower/inspect helpers) |
| `mj_setLengthRange` + `mj_defaultLROpt` | mujoco.h:299, 213 | on-demand LR outside compile |
| `mj_name2id / mj_id2name`, `mju_type2Str/str2Type` | mujoco.h:599, 1398-1401 | DR-10 binding (unchanged) |
| `mj_recompile(s, vfs, m, d)` | mujoco.h:163, impl user_api.cc:262-288 | see Section 4 recompile discussion |
| `mj_saveXMLString / mj_saveXML` | mujoco.h:176-180 | golden snapshots of built specs; native `Expand()` |
| `mj_copyBack(s, m)` | mujoco.h:159 | pull real-valued arrays from model back into spec (post-LR ranges etc.) |
| `mjs_getId` | mujoco.h:1821 | post-compile id cross-check against name binding |
| `mjuu_eigendecompose` (the one MJAPI mjuu) | user_util.h:166 | n-x-n symmetric eigendecomp (fullinertia resolver) |

### 3.3 Internal C++ we might link (each = maintenance-risk line item)

| symbol | header:line | why we'd want it | risk |
|---|---|---|---|
| `mjuu_fullInertia` | user_util.h:172 | fullinertia[6] -> (quat, diag[3]) for ProtoSpec's `InertiaSpec::ToDiagonal()` query | low churn; could reimplement on `mjuu_eigendecompose` instead |
| `mjuu_frameaccum` family, local/global frame math | user_util.h:132-147 | SDK frame-composition helpers | replaceable by public mju_* |
| `mjCComposite::Make` | user_composite.h:57 | native composite decode | **highest churn area**; prefer XML fallback first |
| `mjCFlexcomp::Make` | via user_api.cc:752 | full-fidelity flexcomp beyond `mjs_makeFlex` | medium |
| `mjCModel::SaveState/RestoreState` | user_model.cc:4098/4152 | only if we wanted spec-keyed state migration; DR-11's binding-keyed design avoids the need | avoid |
| `mujoco::user::FilePath`, `VectorToString/StringToVector` | user_util.h:177, 262-268 | path + string-vec plumbing | trivial to own instead |

Verdict: with equality-packing and (optionally) fullinertia reimplemented on public primitives,
the native compiler needs **zero mandatory internal linkage** except for composite (deferrable).

---

## 4. What we actually have to write (the gap list)

1. **The tree-walk decoder** (`bridge/mjs_walker.cc`, one visit per family, driven by the same
   generated reflect/visit tables the XML writer uses). Per-family effort is dominated by the
   table in Section 2; the only families with real logic are equality (data[] packer), defaults
   (two-pass ordering), frames (flatten + setFrame), and macros.
2. **Unit handling — one line, but the top silent-failure risk:** `spec->compiler.degree`
   **defaults to 1 (degrees) even for a bare `mj_makeSpec()`** (`src/user/user_init.c:38`), and
   euler/axisangle/joint-range/ref conversion happens *at compile* against that flag
   (user_objects.cc:2698, 3216-3220, 3277-3281). ProtoSpec is radians-always, so the walker sets
   `degree = 0` first and copies `eulerseq` verbatim. A dedicated regression test (euler-authored
   model, native vs XML) guards it forever.
3. **Variant lowering.** Orientation: write `alt.type` + the authored array and let compile
   resolve (identical code path to XML; `mjs_resolveOrientation` remains available for our
   `ToQuat` query API). Inertia: diagonal -> ipos/iquat/inertia, full -> fullinertia, absent ->
   leave NaN sentinels from `mjs_defaultBody`. Actuators: call `mjs_setTo*` (reader-identical,
   Section 2). Equality: **new small lowering module** replicating xml_native_reader.cc:2192-2311
   packing (plan Q-EQ note gets a native-path amendment).
4. **Presence -> authored bits.** For option/visual/compiler/size fields, presence-tracked decode
   also sets `spec->authored` bitmasks (Section 2 row 1) — mechanical, generated from the same
   field tables.
5. **Names / DR-10.** Unchanged: serial auto-names assigned before decode; `mjs_setName` return
   codes surface duplicate names *at decode time* with SourceLoc (earlier than the XML path's
   compile-time detection — strictly better diagnostics). Binding still via `mj_name2id`;
   optionally assert against `mjs_getId` post-compile as a cross-check.
6. **Error mapping.** Populate every element's `info` string with our SourceLoc rendering;
   wrap `mj_compile` NULL-return + `mjs_getError` into ProtoSpec diagnostics; bounds-check before
   any `mjs_setInStringVec`-class call that `mju_error`s on misuse (user_api.cc:2189-2196).
7. **Recompile (DR-11) — design unchanged, faster.** The native walker rebuilds a fresh mjSpec
   per compile, so mjSpec identity is not persistent and `mj_recompile`'s spec-keyed
   SaveState/RestoreState (user_api.cc:262-288) does not apply; DR-11's binding-keyed state
   migration stays as designed, minus the XML serialize/parse cost it was budgeted for.
8. **Native `Expand()` and macros.** Attach -> `mjs_attach` with `mjs_setDeepCopy(1)`; Replicate
   -> the reader's loop recipe (Section 1); flexcomp -> `mjs_makeFlex` (full-fidelity path later);
   composite -> hybrid XML fallback initially. `Expand()` becomes: decode -> (macros execute
   during decode) -> `mj_saveXMLString` -> `Read()`, dropping the current parse-XML round trip's
   first half.
9. **Drift-gate extension (DR-12).** Add the `mjs_*` signature surface + `mjsCompiler` layout +
   the plugin-attributes map contract + upstream watchlist (`mjtConflict`, `mjtBool`, attach
   `frame` attr) to the existing gate.

---

## 5. Golden / verification strategy

Existing assets: `cpp/harness/mj_model_diff.cc` (mjxmacro-driven full mjModel comparison —
MJMODEL_SIZES + MJMODEL_POINTERS + name tables + fk invariants; exit 0/2/1),
`tests/test_differential.py` (corpus-wide XML-path differential with supported-tag gating via
`cpp/io/supported.json`), and a 387-file MJCF corpus (`snapshots/corpus_report.json`:
total_files=387, all parseable).

1. **The backbone golden**: for every corpus model M, `mjModel A = mj_loadXML(M)` vs
   `mjModel B = mj_compile(walk(Read(M)))` — diff with the mj_model_diff core. Mechanical change:
   factor mj_model_diff's comparison into a small library target and add a `ps_native_diff in.xml`
   harness binary that produces both models in-process (avoids writing temp XML and inherits the
   asset-dir strategy trivially since both loads share the model dir). Same tolerance policy
   (sizes exact, floats rtol/atol, fk spot-check). Expectation: **bit-identical for most fields**
   since both paths execute the same compiler on near-identical specs; genuine tolerance should
   only appear where we resolve something MuJoCo also resolves (none planned — we let compile
   resolve).
2. **Three-way diff during bring-up**: A = XML direct, B = ProtoSpec->canonical-XML->load (existing
   DR-5 path), C = native. A==B green already isolates any A!=C failure to the walker, not the
   reader — keep B in the harness permanently as the arbiter.
3. **Unit-level goldens per family**: build a minimal ProtoSpec fragment -> walk -> 
   `mj_saveXMLString(spec)` (mujoco.h:176; works pre-compile on the spec) -> snapshot. Catches
   decode drift in human-reviewable form before it becomes an mjModel diff. Plus targeted
   equivalence tests for the two hand-written lowerings: equality data[] packing vs
   reader-produced spec (parse tiny XML with `mj_parseXMLString`, compare `mjsEquality.data`),
   and `mjs_setTo*` variants (already reader-identical; one smoke case each).
4. **Property/fuzz**: reuse the planned attribute-mutation fuzzer, but run mutants through
   *both* paths and require identical accept/reject + identical models on accept; add
   attach/replicate-specific properties (idempotent template deletion, suffix numbering,
   keyframe pending-loss reproduction).
5. **Incremental landing behind `Compile()`** (DR-5's promised swap point):
   `Compile()` keeps its signature; internally a per-model gate — walk the ProtoSpec tree, and if
   any element family is outside the native-supported set, fall back to the XML path (mirror of
   `supported.json` gating in test_differential.py). The differential suite runs the native path
   exactly on the models it claims, so scope grows family-by-family with the corpus as the
   ratchet. Composite-containing models are the designed long-tail fallback.
6. **Recompile equivalence** (plan 10.7) gains a third leg: mjs-native edit + `mj_recompile`
   vs ProtoSpec edit + native `Recompile` — state vectors must match.

---

## 6. Risk register

| # | risk | severity | mitigation |
|---|---|---|---|
| R1 | `compiler.degree` defaults to degrees in `mj_makeSpec` -> silent 57.3x orientation/range corruption | high, trivial to fix, easy to reintroduce | set degree=0 first in walker ctor; dedicated euler regression; assert in walker |
| R2 | Defaults interplay: class values copied eagerly at `mjs_add*(…, def)`; later class edits don't propagate; `mjs_setDefault` is label-only (user_api.cc:1792-1795) | high (the classic past-pain area) | strict two-pass decode (defaults fully, then elements); per-family defaulted-field goldens; never mutate a class mid-decode |
| R3 | Attach namespacing: shallow attach mutates + locks the source; double-attach keyframe loss; `CopyList` silently drops dangling referencers (user_model.cc:266-282) | high for Expand/SDK attach | deepcopy=1 always in walker/Expand; ProtoSpec tier-2 validation pre-attach; test the silent-drop case explicitly |
| R4 | Keyframe sizing: strict compile errors on wrong lengths; padding semantics differ between authored and attach-pending keys | medium | keep sizes.cc prediction + tier-3 lint; corpus keyframe models in phase 1 of family rollout; error-message mapping test |
| R5 | Composite has no public API | medium | hybrid XML fallback (designed, not a hack); decide link-vs-fallback in macro phase with usage data from corpus (corpus_report usage counts) |
| R6 | Plugin config `void*` = C++ map shape (user_api.cc:2298-2302) | low-medium | one isolated adapter fn; drift-gate the contract |
| R7 | mjs API churn on bumps (mjtBool migration, `mjsCompiler.conflict` insertion changes struct layout) | medium, chronic | DR-12 gate extended to mjs surface; we already rebuild MuJoCo from source per bump |
| R8 | `mjs_setName` O(n log n)/call duplicate check | low (robot scale) | measure in phase 1; if hot, name-set locally and batch-verify once |
| R9 | Authored-bitmask infidelity -> wrong `mj_saveXML` of built specs + wrong upstream conflict resolution | low today, rises with upstream bump | generate bitmask writes from field tables (gap item 4) |
| R10 | Silent behavioral deltas vs XML path in un-exercised corners | — | the 387-corpus three-way differential is the answer; corpus gaps (flex? composite variants?) get synthetic fixtures |

---

## 7. Phased plan skeleton

Sized against the existing milestone style; each phase lands behind the stable `Compile()`
signature with the per-model fallback gate (Section 5.5), so the XML path is never broken.

- **P0 — Harness first (small).** Factor mj_model_diff into a lib; `ps_native_diff` three-way
  harness; walker skeleton that decodes *only* the compiler/option/size/statistic/visual blocks
  (incl. degree=0 and authored bitmasks) and falls back for everything else.
  *Exit: harness runs whole corpus in fallback mode; block-only synthetic models bit-identical.*
- **P1 — Pathfinder families (the bulk of value).** Defaults tree (two-pass), body tree with
  frames, joints/geoms/sites/cameras/lights, meshes/hfields/textures/materials (incl. buffer
  paths). This unlocks every plain robot model.
  *Exit: all corpus models whose tags are within this set compile natively bit-identical
  (expected: the large majority of the 387); R1/R2 regressions in place.*
- **P2 — Constraint & drive families.** Contact, equality (data[] packer + equivalence tests),
  tendons + wrap paths, actuators (`mjs_setTo*`), sensors, keyframes, custom, plugins, skin, flex.
  *Exit: full corpus native except macro-containing models; recompile-equivalence leg green.*
- **P3 — Macros native.** Attach + Replicate via `mjs_attach` recipes; flexcomp via
  `mjs_makeFlex` (fidelity audit vs XML on flexcomp corpus); composite decision (link
  `mjCComposite` vs permanent fallback); native `Expand()`.
  *Exit: full corpus native; fallback list empty or exactly = documented composite set.*
- **P4 — Switch + upstream readiness.** Native becomes `Compile()`'s default (XML path retained
  as `--oracle` and for Expand's writer half); DR-11 Recompile rewired; drift-gate upstream
  watchlist (mjtConflict/mjtBool/attach-frame) armed for the next MuJoCo bump; ProtoSpec Attach
  element grows `conflict`/`frame` fields (dormant until bump).
  *Exit: differential suite runs native-primary; perf note (compile-time delta vs XML hop,
  using `mjs_getTimer`).*
