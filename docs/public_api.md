# ProtoSpec public C++ API

ProtoSpec is the product; the studio editor is one example integration. This
document defines the **supported public surface** — the exact set of headers and
symbols a consumer (the studio editor today, the UE plugin tomorrow) is meant to
include and call. Everything not listed here is internal and may change without
notice.

## The public include set

A consumer adds `protospec/lib/include` and the relevant subsystem include roots to its
include path (all four existing consumers already do) and includes **only** the
`<protospec/...>` umbrella headers below. Each umbrella re-exports the supported
types from the implementation headers; the implementation headers are internal.

| Public header | Gives you | Link |
| --- | --- | --- |
| `protospec/core.h` | Runtime primitives: `ps::opt`, `ps::Ref`, `ps::InlineVec`, `ps::SourceLoc` | `protospec` |
| `protospec/model.h` | The object model (`ps::mjcf::*` element/enum/union/variant types) + value ops `Clone`, `ApplyDefault`, `ToMjcf`/`FromMjcf`. Pulls `core.h`. | `protospec` |
| `protospec/reflect.h` | Runtime reflection (`ps::mjcf::reflect::*`) + the generic `ps::mjcf::Visit` hook, for schema-driven tooling | `protospec` |
| `protospec/io.h` | MJCF read/write: `ps::mjcf::io::{ParseMjcfFile, ParseMjcfString, WriteMjcf, ParseResult, Diagnostic}` | `protospec_io` |
| `protospec/validate.h` | `ps::mjcf::validate::{Validate, Diagnostic, Tier, Severity, TierMask, kAllTiers, ...}` | `protospec_validate` |
| `protospec/compile.h` | MuJoCo compile: `ps::mjcf::{Compile, Recompile, CompileToXml, Compiled, Binding, CompileOptions, VfsAsset, CompileReport, CompilePath}` + the pose-patch API `ps::mjcf::{RigidPose, PosePatch, ApplyPosePatch, Compose, Invert}` | `protospec_bridge` (+ MuJoCo) |
| `protospec/sdk.h` | Ergonomic authoring (`ps::sdk::*`): builders, traversal, refs, default classes, attach | `protospec_sdk` |
| `protospec/save.h` | Model persistence: `ps::sdk::{Save, SaveAs, ExternalizeAssets, ModelAssetDir, InMemoryAsset}` | `protospec_sdk_io` |
| `protospec/protospec.h` | All of the above in one include (links the whole library + MuJoCo) | all |

`protospec/sdk.h` and `protospec/save.h` physically live under `protospec/lib/sdk/protospec/`
(the SDK's own include root); the rest live under `protospec/lib/include/protospec/`. Both
resolve as `protospec/<name>.h`.

### Namespace scheme (final)

| Namespace | Contents |
| --- | --- |
| `ps` | Schema-independent runtime core: `opt`, `Ref`, `InlineVec`, `SourceLoc`, container helpers |
| `ps::mjcf` | The generated object model — every element type, enum, union wrapper, variant alias, `ElementType`, `element_type_of`, the value ops `Clone` / `ApplyDefault` / `ToMjcf` / `FromMjcf` / `Visit` — **and** the MuJoCo compile surface (`Compile`, `Recompile`, `Compiled`, `Binding`, `CompileOptions`, `CompileReport`, `PosePatch`, `ApplyPosePatch`, ...). One namespace for the whole model + compile product |
| `ps::mjcf::reflect` | Runtime schema reflection (descriptors, `Describe`, field/child/union metadata) |
| `ps::mjcf::io` | MJCF parse / write |
| `ps::mjcf::validate` | Three-tier validation |
| `ps::sdk` | The ergonomic authoring layer over `ps::mjcf` |

The compile bridge is **not** a consumer-facing namespace: it is written as
`ps::mjcf::Compile` / `ps::mjcf::Binding`, never `ps::mjcf::bridge::...`. `protospec/lib/compile`
and `attic/compile` are the implementation directories behind it, not a namespace a
consumer names. Nested `detail` namespaces (`ps::mjcf::detail`, `ps::sdk::detail`,
...), the SDK/compiler shared core `ps::sdk::internal`, and the internal headers
below are **not** public.

## Hello world (public headers only)

Load → edit → validate → compile → step → save, touching nothing internal. This
is exactly what `protospec/lib/test/test_public_api.cc` compiles and runs.

```cpp
#include <mujoco/mujoco.h>          // the simulation engine YOU bring

#include "protospec/io.h"
#include "protospec/sdk.h"
#include "protospec/save.h"
#include "protospec/validate.h"
#include "protospec/compile.h"

namespace io = ps::mjcf::io;
namespace sdk = ps::sdk;
namespace mj = ps::mjcf;   // Compile / Binding / Compiled live here too

int main() {
  // 1. LOAD
  auto parsed = io::ParseMjcfFile("scene.xml");
  if (!parsed.ok()) return 1;
  mj::Model& model = *parsed.model;

  // 2. EDIT — SDK verbs only, never the raw generated types
  mj::Body& box = sdk::AddBody(sdk::World(model), "box");
  box.pos = std::array<double, 3>{0, 0, 1};
  sdk::AddFreeJoint(box, "box_free");
  mj::Geom& g = sdk::AddPrimitive(box, mj::GeomType::box, "box_geom");

  mj::Material& mat = sdk::AddMaterial(model, "red");
  mat.rgba = std::array<float, 4>{1, 0, 0, 1};
  sdk::SetRef(g.material, mat);     // typed reference by target element

  // 3. VALIDATE
  for (auto& d : ps::mjcf::validate::Validate(model))
    if (d.severity == ps::mjcf::validate::Severity::Error) return 2;

  // 4. COMPILE
  mj::Compiled compiled = mj::Compile(model);
  if (!compiled.ok()) return 3;

  // 5. STEP (your engine, your mjData)
  mjData* d = mj_makeData(compiled.model.get());
  for (int i = 0; i < 100; ++i) mj_step(compiled.model.get(), d);
  mj_deleteData(d);

  // 6. SAVE
  sdk::Save(model, "scene_edited.xml");
  return 0;
}
```

## What is internal (do not include)

The include graph makes the boundary unambiguous: if it is not a
`<protospec/...>` umbrella header, it is internal.

- **Generated headers** — `protospec/lib/generated/{types,keywords,defaults,reflect,visit,xml_binding}.h`.
  The object model and its value/reflection ops. Reach them through
  `protospec/model.h` and `protospec/reflect.h`, never directly. (These are
  emitted by `protospec_gen`; the umbrellas are the stable façade over them.)
- **IO internals** — `protospec/lib/io/{numeric,include,defaults}.h`.
- **Validate internals** — `protospec/lib/validate/sizes.h`.
- **Compile internals** — `attic/compile/**` (native compiler, build, lifted MuJoCo
  code) and `protospec/lib/core/resolve.h`. The supported compile surface is
  `protospec/compile.h` only.
- **SDK internals** — `protospec/lib/sdk/protospec/detail.h` (`ps::sdk::detail::*`):
  genuinely SDK-private reflection helpers (type-erased element `Handle`, the
  union/ref-accepts descriptors, MuJoCo name-category folding, the dynamic-keyword
  table and its drift guard). No in-tree consumer outside `protospec/lib/sdk` refers to a
  symbol that stays private here. Its broadly-useful primitives are re-exported as
  public `ps::sdk` verbs (see below).
- **SDK / compiler shared core** — `protospec/lib/sdk/protospec/model_core.h`
  (`ps::sdk::internal::*`): the generic tree machinery the SDK's public verbs AND
  the in-tree native compiler (`attic/compile/{native,build}.cc`) both program
  against — the whole-tree walk, name access, reference-shape traits, per-field
  probes, ref-target sets, the reference prefixer, plus the `<default>` class
  index / class-name resolution (the last defined in `classes.h`, over
  `ParentMap`). It is a **named internal seam, not a public surface**: not
  versioned, not exported, and any change must update both consumers in the same
  change. The editor's reflection-driven inspector (`details_panel`) is the only
  other in-tree consumer, for the per-field probes and ref-target sets that have
  no public verb.
- **Bridge internals** — `binding.h`'s `detail::BindingBuilder`, `report.h`'s
  `detail::`. `Binding`, `Compiled`, `CompileReport`, `Diagnostic`, `PosePatch`,
  and `ApplyPosePatch` are public (in `ps::mjcf`) via `protospec/compile.h`. The
  `bridge`/`compile` names are directory paths, not a consumer namespace.

## SDK affordances (closing the "reach past the SDK" gaps)

The editor integration is the usability test: any place it had to reach past
`ps::sdk` into raw generated types or `ps::sdk::detail` for common authoring is a
ProtoSpec defect. These verbs were added so a consumer never needs the raw layer
for them:

- **Element identity** (`protospec/sdk.h`, traversal): `sdk::Name(e)`,
  `sdk::SetName(e, name)`, `sdk::TypeOf(e)`, `sdk::WalkSubtree(e, fn)`,
  `sdk::WalkModel(model, fn)` — the everyday name/type/walk operations, replacing
  direct `ps::sdk::detail::{NameOf, SetName, WalkTree, WalkModelAll}` use.
- **Reference assignment** (refs): `sdk::SetRef(field, name)`,
  `sdk::SetRef(field, targetElement)`, `sdk::ClearRef(field)` — set a typed
  `opt<Ref<T>>` without spelling `ps::Ref<T>` or knowing the storage shape. The
  target-element overload is compile-time checked: the target's element type
  must be a valid target of the field's `Ref<T>` (itself for a concrete ref, a
  union member for a union ref such as `Ref<ActuatorAny>`), so a mismatched
  target (`SetRef(materialRef, someBody)`) is a build error, not a silent no-op.
- **Find by serial** (traversal): `sdk::FindBySerial(model, serial) -> void*`
  and `sdk::FindBySerialTyped(model, serial) -> Located{ptr, type}` — resolve the
  process-unique element serial (the identity a UI holds across edits, since the
  pointer moves on mutation) back to a live element with one generic walk,
  replacing the hand-rolled "walk and match `e.serial`" at each call site.
- **Prune by predicate** (refs): `sdk::PruneSubtrees(model, pred)` — remove every
  element for which `pred(const auto& element)` is true, subtrees included. A raw
  structural prune (no referrer bookkeeping, unlike the Delete verbs): for
  whole-partition drops such as compile-input filtering or layer pruning.
- **Scan references** (refs): `sdk::ScanRefs(element, on)` — invoke `on(field_id,
  field_name, std::string& ref_name, target_types)` for every authored typed
  reference of a single element (scalar `Ref<T>`, each `ref<T>[]` entry, and the
  schema's dynamic `target_from` refs). Reflection-driven, so it tracks the schema
  with no per-element code; `ref_name` is the live mutable string (edit it to
  rewrite the reference, or just read it). Backs the editor's cross-layer
  dependency graph; promoted from `ps::sdk::detail`.
- **Primitive sizing** (builders): `sdk::SeedPrimitiveSize(geom)` and
  `sdk::AddPrimitive(parent, type, name)` — a bare `AddGeom` authors only what
  you pass (DR-1), so a primitive has no size and will not compile;
  `AddPrimitive` returns one ready to simulate.
- **Appearance** (builders): `sdk::AddMaterialLayer(mat, role, texture)`,
  `sdk::SetLayerTexture`, `sdk::SetLayerRole`, `sdk::RemoveMaterialLayer`,
  `sdk::SetTextureFile(tex, path)`, `sdk::SetTextureBuiltin(tex, builtin)` — the
  material/texture/layer authoring the editor previously kept as its own private
  mutator library.
- **Runtime-typed structural edits** (refs / attach): `sdk::Rename(model, elem*,
  name)`, `sdk::DeleteSubtree(model, elem*, cascade)`, `sdk::Duplicate(model,
  elem*)`, `sdk::Reparent(model, elem*, newParent*)` — keyed on a runtime element
  pointer (not the `<E>` template), so a consumer that resolves elements
  dynamically (pick / serial / path) never pays the ~140x template-instantiation
  compile cost or reimplements the referrer bookkeeping. See the section below.
- **Generic model helpers** promoted from the editor (traversal): `sdk::ForEachElement(model, fn)`
  (every element except the Model root — the guarded whole-tree walk the editor
  spelled out at six call sites), `sdk::SerialOf(model, ptr)` (the inverse of
  `FindBySerial`: a raw element pointer back to its stable creation serial),
  `sdk::FindBySerialAs<T>(model, serial)` (the typed slice of the serial walk),
  `sdk::NameOfSerial(model, serial)` (an element's authored name from its serial),
  `sdk::UniqueName(model, type, base)` (a name unique within `type`'s MuJoCo
  name-category — folds joint spellings and each actuator/sensor/tendon/equality
  union), and `sdk::CloneModelWithSerials(model)` (a deep clone whose every
  element keeps its source serial — the undo-snapshot primitive; it **asserts**
  the source/clone walks are a bijection rather than silently truncating a
  mismatch).

Already present and sufficient (the editor simply had not adopted them):
`sdk::ForEachOfType<T>(model, fn)` covers "enumerate every material / element of
a type"; `sdk::Find<T>`, `sdk::ParentMap`, `sdk::Resolve`/`ResolveTo`,
`sdk::FindReferrers`, `sdk::Attach`/`AttachModel`, `sdk::Effective`.

## Runtime-typed structural edits (`ps::sdk`)

The five edits below are keyed on a **runtime element pointer** (any element you
hold in the tree you compiled), not on the element's static type. The `<E>`
templates (`Rename<E>`, `DeleteRecursive<E>`) still exist, but instantiating them
across all ~140 families — each embedding a whole-model walk — is a ~140x140
compile blow-up, so the runtime-pointer forms are the supported API for anything
that resolves an element dynamically (pick / serial / path). One model walk
recovers the element's `ElementType` and drives the same referrer bookkeeping.

### Error convention — one result-object style

The structural verbs report through a small **result object**, uniformly: a
truthy `explicit operator bool` (`if (auto r = sdk::Reparent(...)) ...`) and,
where a failure has a cause, a `std::string reason` (empty on success). This
replaces the former mix of `void*`, `int` sentinels, and ad-hoc structs the deep
audit flagged. The two verbs whose scalar returns changed — `Rename` (`int` → a
`RenameResult`) and `Duplicate` (`void*` → a `DuplicateResult`) — keep the old
return shape as a **deprecated one-release conversion** on the result object, so
existing call sites keep compiling for a transition period; new code should read
`.ok` / `.updated` / `.clone`. The `int`/`void*` conversions are removed in the
next minor release.

| Verb | Result object | Fields |
| --- | --- | --- |
| `Rename` | `RenameResult` | `ok`, `updated` (referrer fields rewritten), `reason`; deprecated `operator int` (updated, or `-1` on reject) |
| `Duplicate` | `DuplicateResult` | `ok`, `clone` (`void*`; `.As<T>()` convenience), `reason` |
| `Reparent` | `ReparentResult` | `ok`, `reason` |
| `DeleteSubtree` / `DeleteRecursive` | `DeleteReport` | `removed` (== `bool`), `dangling`, `cascaded` |
| `Attach` / `AttachModel` | `AttachResult` | `ok`, `attached`, `collisions` |

- `RenameResult sdk::Rename(Model&, const void* elem, const std::string& newname)`
  — rename + rewrite every typed referrer atomically. `ok` with `updated`
  referrers on success (0 for an accepted no-op or a nameless element gaining its
  first name); `ok == false` with a `reason` (model untouched) when `elem` is not
  in the model or `newname` is rejected (empty, reserved `_ps:` prefix, or already
  held by a different element of the same name-category).
- `DeleteReport sdk::DeleteSubtree(Model&, const void* elem, bool cascade=false)`
  — remove the subtree; `report.dangling` lists every reference left pointing at
  nothing (with its path). `cascade` clears them; otherwise the caller resolves.
  `removed` (also the object's `bool`) is false when `elem` is not found.
- `DuplicateResult sdk::Duplicate(Model&, const void* elem)` — deep-clone the
  subtree as the next sibling with fresh serials (generated `Clone`), re-unique
  the clone's names, remap refs **internal** to the clone to the new names, and
  preserve refs pointing outside. `clone` is the clone's root (its `ElementType`
  matches `elem`); `.As<Body>()` casts it. `ok == false` when `elem` is unfound.
- `ReparentResult sdk::Reparent(Model&, const void* elem, void* newParent)` —
  pure-tree move of a body-context child to a new container (Body / Frame /
  `nullptr` = world). Rejects cycles and non-container targets (`ok == false`,
  `reason` set). **No pose fixup**: the element keeps its authored local pose (its
  world pose changes with the new parent). Pose-preserving reparent is a
  compile-aware concern that stays with the bridge / editor — the SDK is a pure
  tree library and does not compute a compiled parent world pose.

## Reference assignment — `sdk::SetRef` (load-bearing)

`SetRef` is the one way to write a typed cross-reference (`opt<Ref<T>>`) without
spelling `ps::Ref<T>` or knowing the opt/Ref storage shape. It is load-bearing in
the generated `mjs_binding` and in every consumer that authors references. Two
overloads:

```cpp
// (1) By name — stores the name verbatim; no lookup (resolution is a separate
//     concern). An EMPTY name clears the field (unauthored).
sdk::SetRef(geom.material, std::string_view("steel"));   // geom.material = "steel"
sdk::SetRef(geom.material, std::string_view{});          // clears it
sdk::ClearRef(geom.material);                            // equivalent explicit clear

// (2) By target element — sets the ref to the target's authored name. Returns
//     false (field untouched) when the target has no usable name.
sdk::Material& steel = *sdk::Find<sdk::Material>(model, "steel");
bool ok = sdk::SetRef(geom.material, steel);             // true; geom.material = "steel"
```

**Negatives (what SetRef does and does not catch):**

- **Mismatched target type is a *compile* error, not a silent no-op.** The
  by-target overload `static_assert`s that the target's element type is a valid
  target of the field's `Ref<T>` (itself for a concrete ref; a union member for a
  union ref such as `Ref<ActuatorAny>`). `sdk::SetRef(geom.material, someBody)`
  does not build — a `Body` is not a valid `Ref<Material>` target.
- **An unnamed target returns `false`** and leaves the field untouched — an
  unnamed element cannot be referred to in MJCF.
- **A name that resolves to nothing is *not* an error here.** SetRef does no
  lookup; a dangling name is caught downstream by `sdk::Resolve`/`ResolveTo`
  (returns null) and by the referential validator, not by SetRef. This keep-it-
  local split is deliberate: authoring order need not create the target first.

These behaviors are covered by `test_sdk.cc` (`TestSetRefNegatives`).

## Moving a compiled element without recompiling (`PosePatch`)

A gizmo drag should not re-run the compiler every frame (a single mesh geom
already costs ~18 ms; large models 30–600 ms — see
`docs/drag_perf_investigation.md`). The compiled pose field of every spatial
element factors as `field = A ∘ L_authored ∘ B`, where the baked frames `A`
(enclosing `<frame>` chain, and for an align-free free joint the inverse-inertial
frame) and `B` (mesh/fit recentering; a body's free-joint inertial fold) are
**constant** across a pose-only edit. So a pose drag captures `A` and `B` once and
writes `A ∘ L_new ∘ B` into `mjModel` — no recompile:

```cpp
// mjData d is at qpos0 (mj_resetData). `elem` is a tree Body/Geom/Site/Camera/Light.
if (auto pp = compiled.binding.PosePatchFor(elem)) {   // std::optional<PosePatch>
  mj::RigidPose L_new = /* the new authored local pose (pos[3], quat[4]) */;
  mj::ApplyPosePatch(compiled.model.get(), *pp, L_new); // writes A ∘ L_new ∘ B
  mj_kinematics(compiled.model.get(), d);               // (mj_resetData first for a
                                                        //  free/ball body: it reseeds
                                                        //  qpos0). No recompile.
} else {
  /* unpatchable element: fall back to a full Recompile */
}
```

`PosePatchFor` is compile-path agnostic: it reconstructs `A` from the tree, reads
`L` via `sdk::Effective` (so a class-inherited pose resolves exactly as the
compiler saw it), and captures `B` as the residual `(A ∘ L)^-1 ∘ field` read back
from `mjModel` — a single inverse done **once** at capture. `ApplyPosePatch` only
ever composes `A ∘ L_new ∘ B` forward; it never inverts `B`, which is precisely
why the read-back-and-invert approach that "behaves weirdly for mesh geoms"
(DR-S6) is avoided. `ApplyPosePatch` also clears the element's `sameframe`
optimisation flag (so `mj_kinematics` recomposes from the patched field) and, for
a free/ball-jointed body, reseeds `model->qpos0` (whose rest pose drives the body,
not `body_pos`). `mj::RigidPose` + `mj::Compose` / `mj::Invert` are public so a
caller can build `L_new`. On drag release, do one real `Compile` to reconcile any
second-order effects.

**Live paths / remaining sharp edge.** `A`/`B` capture is live via tree + `mjModel`
reconstruction, which covers both the native and XML compile paths (the current
bridge is XML-path only). The one case not reconstructed is the align-free
free-joint **inverse-inertial prefix** on a child geom (`A` omits it); such
align-free-body child geoms fall outside the exact-`A` capture and should recompile
until that term is added. Light **orientation** patching (`light_dir`) is likewise
deferred — `PosePatch` moves a light's position; its direction is left as authored.

## Compile warning capture — the host-owned `mju_user_warning` handler

MuJoCo surfaces load/compile warnings through `mju_user_warning`, a single
**process-global** function pointer. ProtoSpec runs inside a host process (Studio)
that may install its own handler, so `Compile` / `Recompile` do **not** own that
pointer — they **borrow** it. For the duration of one `mj_loadXML` / `mj_compile`
call, and only that call, `Compile` saves the host's handler, installs its own
collector (routing captured text into the `CompileReport`'s warnings), then
restores the host's handler. The borrow window is held under a **process-global
mutex that serializes ProtoSpec compiles**, so two ProtoSpec compiles never
overlap their borrow windows and the save/restore can never interleave.

The documented limitation: a *host* compile (or any host code that raises a MuJoCo
warning) running **concurrently** with a ProtoSpec compile will briefly see
ProtoSpec's collector instead of the host handler, for the borrow window only.
This is accepted — the real fix is an upstream context-local warning callback,
which is on the upstream ask list. Callers who install their own
`mju_user_warning` should not rely on it being active across a concurrent
`Compile`. Warning *capture itself* is unchanged: every warning MuJoCo raises
during the compile still lands in `Compiled::report.warnings`.

## Python surface (`protospec`)

The `protospec` extension mirrors this contract for Python authors. `load` /
`loads` / `write` / `save` / `validate` / `compile` / `recompile` are the
module functions; typed element handles expose every schema field as a property
(`opt<T>` → a `None`-able attribute, enums → their MJCF keyword string, refs →
the referenced name string). The authoring surface below is **generated from the
schema** (`protospec_gen/emit_py.py` → `lib/python/generated/py_builders.cc`), so
a new upstream field, union member, or asset family reaches Python at the next
`emit` with no hand edit.

**Constructors take keyword fields.** Every element class default-constructs and
accepts field keywords through the builders: `ps.Material(...)` values are set the
same way `add_material(...)` sets them. The builders are the entry points:

```python
import protospec as ps

m = ps.Model()
cart = m.worldbody.add_body(name="cart", pos=[0, 0, 0.2])       # body-context
cart.add_joint(name="slide_x", type="slide", axis=[1, 0, 0])    # builders
cart.add_geom(name="box", type="box", size=[0.15, 0.1, 0.05], material="shiny")

m.add_material(name="shiny", rgba=[0.2, 0.6, 0.9, 1], specular=0.8)  # asset family
m.add_actuator("position", name="servo", joint="slide_x", kp=8.0)   # union family
m.add_sensor("framepos", name="cart_pos", objtype="body", objname="cart")
```

Builder families:

| Method | Family | Kind argument |
|---|---|---|
| `body.add_body/geom/joint/freejoint/site/camera/light/frame/inertial(**kw)` | body-context | — |
| `model.add_body(**kw)` | top-level body | — |
| `model.add_actuator(kind="motor", **kw)` | actuator union | MJCF keyword (`motor`, `position`, `general`, …) |
| `model.add_sensor(kind, **kw)` | sensor union | MJCF keyword (`framepos`, `touch`, `jointpos`, …) |
| `model.add_tendon(kind, **kw)` | tendon union | `spatial` / `fixed` |
| `model.add_equality(kind, **kw)` | equality union | `connect` / `weld` / `joint` / `tendon` / `flex` / … |
| `model.add_material/mesh/texture/hfield/skin(**kw)` | asset section | — |
| `model.add_pair/exclude(**kw)` | contact section | — |
| `model.add_key(**kw)` | keyframe section | — |
| `model.add_numeric/text/tuple(**kw)` | custom section | — |

An unknown union `kind` raises `ValueError` naming the accepted keywords.

**Structural edit verbs** live on the `Model` and mirror the runtime-typed
`ps::sdk` verbs above. **Error convention — one style, uniform:** a verb *raises*
on failure and *returns its natural success value*.

- `model.rename(elem, name) -> int` — rename an element **and rewrite every
  referrer**. Returns the number of referrer fields updated (`0` for a no-op or a
  previously-nameless element). Raises `ValueError` if the name is empty,
  reserved, already held by another element, or `elem` is not in this model. This
  is the safe rename.
- `model.delete(elem, cascade=False) -> DeleteResult` — remove `elem` and its
  subtree. The result carries `.removed`, `.cascaded`, and `.dangling` (a list of
  `{field, name, path}` dicts for references left pointing at a deleted name). A
  non-cascade delete that leaves danglers is a **success** — the danglers are
  reported, not raised, for the caller to resolve; `cascade=True` clears them.
  Raises `ValueError` only if `elem` is not in this model.
- `model.duplicate(elem) -> handle` — deep-clone the subtree as the next sibling
  (fresh serials, re-uniqued names, internal refs remapped); returns the clone as
  the same element type. Raises `ValueError` if `elem` is not in this model.
- `model.reparent(elem, new_parent=None) -> None` — pure-tree move of a
  body-context child under a Body/Frame (or `None` = world). No pose fixup (local
  pose kept; world pose changes). Raises `ValueError` on a bad target or a cycle.

**Setting `.name` directly warns.** The `.name` property setter has no handle on
the owning model, so it *cannot* rewrite referrers. Renaming an already-named
element via `elem.name = "..."` therefore emits a `UserWarning` and leaves
referrers dangling; use `model.rename(elem, ...)` to keep references in sync.
Naming a previously-nameless element (or a no-op set) is silent — nothing could
refer to it yet.

## Consumers and build wiring

Four independent builds consume this surface; all stay green:

- `protospec/lib/CMakeLists.txt` — the library + tests (`ctest`), including the new
  `protospec_public_api_tests` that includes only the umbrella headers.
- `protospec/lib/python/` — the pybind module (compiles the sources directly).
- `studio/` — the standalone editor host (compiles the sources directly).
- The MuJoCo Studio fork — compiles the editor + core live from this repo via
  `PROTOSPEC_ROOT` (see `docs/studio_build.md`).

The umbrella headers reach their implementation headers by **repo-relative path**
(`../../generated/...`, `../../io/...`, etc.), so they work identically under
every one of these builds without new `-I` entries.

The compile surface was hoisted from `ps::mjcf::bridge` up to `ps::mjcf` as a real
rename (no compatibility aliases): the definitions in `protospec/lib/compile` and every caller
across `protospec/lib/**`, `protospec/lib/python`, `studio`, and the fork were updated in lockstep,
so `ps::mjcf::bridge` no longer exists anywhere. `ps::mjcf::Compile` is the name.

## Stability levels (per header)

Every public header carries one of three stability levels. **Stable** surfaces
change only under the semver policy below. **Provisional** surfaces are public and
supported but may still change shape before the 1.0 freeze (they carry a
migration note when they do). **Internal** is everything else — no guarantee, may
change in any commit; the include graph is the boundary (if it is not a
`<protospec/...>` umbrella, it is internal).

| Header | Level | Notes |
| --- | --- | --- |
| `protospec/core.h` | **stable** | runtime primitives |
| `protospec/model.h` | **stable** | object model + value ops |
| `protospec/reflect.h` | **provisional** | descriptor *shapes* may grow as the schema does |
| `protospec/io.h` | **stable** | MJCF parse/write |
| `protospec/validate.h` | **stable** | three-tier validation |
| `protospec/compile.h` | **stable (pin-versioned)** | byte-exactness is versioned against the MuJoCo pin — see below |
| `protospec/sdk.h` | **stable** | builders, traversal, refs, classes, attach. *Provisional within it:* the deprecated `int`/`void*` conversions on `RenameResult`/`DuplicateResult` (removed next minor); the promoted editor helpers (`ForEachElement`, `SerialOf`, `FindBySerialAs`, `NameOfSerial`, `UniqueName`, `CloneModelWithSerials`) are stable |
| `protospec/save.h` | **stable** | model persistence |
| `protospec/protospec.h` | **stable** | the aggregate umbrella |

Internal (never include, no stability): everything under `protospec/lib/{io,validate,core,compile,generated}`
internals, `protospec/lib/sdk/protospec/{detail,model_core}.h` (the `ps::sdk::detail` /
`ps::sdk::internal` seams), the `harness/`, and the editor plugin ABI
(`studio/editor/plugin_abi.h`) — the last deliberately tracks upstream's
experimental plugin surface and carries no promise.

## Versioned byte-exactness

Byte-exactness is only meaningful against a specific MuJoCo compiler, so the claim
is **always versioned** and never stated bare. The single source of the current
claim is [`docs/SYNC_STATE.md`](SYNC_STATE.md); as of that pin:

> ProtoSpec is byte-exact vs **MuJoCo 3.10.1+3990305**, except the named entries
> in [`docs/EXCEPTIONS.md`](EXCEPTIONS.md).

Any README/doc "exact" claim resolves to this versioned form. A ProtoSpec release
names (a) the frozen API surface (this document), (b) the MuJoCo pin it is
byte-exact against (`SYNC_STATE.md`), and (c) the exception ledger at that pin. A
re-pin happens only via a full sync-ritual run (`docs/mujoco_bump.md`).

## Semver policy (stub)

ProtoSpec-as-a-library versions with **semver from `v1.0.0`** at the Wave-3
surface freeze:

- **MAJOR** — a breaking change to any *stable* surface above (removed/renamed
  symbol, changed signature or semantics, changed schema file format).
- **MINOR** — additive changes (new verbs, new stable-ified provisional surface,
  new schema attributes flowing from a MuJoCo bump) and the **removal of a
  deprecated shim** that shipped deprecated in the prior minor (e.g. the
  `int`/`void*` conversions on `RenameResult`/`DuplicateResult`). Deprecations
  ship for **one minor release** before removal.
- **PATCH** — bug fixes with no surface change. A patch **may re-pin MuJoCo** (and
  thus move the byte-exactness version string) only via a full ritual run; the
  pin and exception ledger are named in the release notes.

Provisional surfaces are exempt from MAJOR until they are declared stable at 1.0.

## SDK 1.0 surface-freeze candidate

What 1.0 commits to as **frozen stable** (declared at release, not before):

- the nine umbrella headers — `core.h model.h reflect.h io.h validate.h compile.h
  sdk.h save.h protospec.h` — at the stability levels tabled above;
- the Python module surface (`protospec`) — the section above;
- the CLI contracts of `ps_compile` / `ps_roundtrip` / `ps_validate`;
- the schema file format (`protospec/schema/mujoco.spec`).

**Explicitly NOT frozen:** anything under `protospec/lib/*` internals, the editor
plugin ABI (tracks upstream experimental), `harness/`, and generated-file
internals. The freeze is declared at the Wave-6 release once the audit's
ergonomics and robustness items have re-scored A.
