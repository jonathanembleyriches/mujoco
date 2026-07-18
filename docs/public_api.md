# ProtoSpec public C++ API

ProtoSpec is the product; the studio editor is one example integration. This
document defines the **supported public surface** — the exact set of headers and
symbols a consumer (the studio editor today, the UE plugin tomorrow) is meant to
include and call. Everything not listed here is internal and may change without
notice.

## The public include set

A consumer adds `cpp/include` and the relevant subsystem include roots to its
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

`protospec/sdk.h` and `protospec/save.h` physically live under `cpp/sdk/protospec/`
(the SDK's own include root); the rest live under `cpp/include/protospec/`. Both
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
`ps::mjcf::Compile` / `ps::mjcf::Binding`, never `ps::mjcf::bridge::...`. `cpp/bridge`
and `cpp/compile` are the implementation directories behind it, not a namespace a
consumer names. Nested `detail` namespaces (`ps::mjcf::detail`, `ps::sdk::detail`,
...) and the internal headers below are **not** public.

## Hello world (public headers only)

Load → edit → validate → compile → step → save, touching nothing internal. This
is exactly what `cpp/test/test_public_api.cc` compiles and runs.

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

- **Generated headers** — `cpp/generated/{types,keywords,defaults,reflect,visit,xml_binding}.h`.
  The object model and its value/reflection ops. Reach them through
  `protospec/model.h` and `protospec/reflect.h`, never directly. (These are
  emitted by `protospec_gen`; the umbrellas are the stable façade over them.)
- **IO internals** — `cpp/io/{numeric,include,defaults}.h`.
- **Validate internals** — `cpp/validate/sizes.h`.
- **Compile internals** — `cpp/compile/**` (native compiler, build, lifted MuJoCo
  code) and `cpp/core/resolve.h`. The supported compile surface is
  `protospec/compile.h` only.
- **SDK internals** — `cpp/sdk/protospec/detail.h` (`ps::sdk::detail::*`: the
  whole-tree walk, name access, ref scan, handles). Its useful primitives are
  now re-exported as public `ps::sdk` verbs (see below).
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

Already present and sufficient (the editor simply had not adopted them):
`sdk::ForEachOfType<T>(model, fn)` covers "enumerate every material / element of
a type"; `sdk::Find<T>`, `sdk::ParentMap`, `sdk::Resolve`/`ResolveTo`,
`sdk::FindReferrers`, `sdk::Attach`/`AttachModel`, `sdk::Effective`.

## Runtime-typed structural edits (`ps::sdk`)

The four edits below are keyed on a **runtime element pointer** (any element you
hold in the tree you compiled), not on the element's static type. The `<E>`
templates (`Rename<E>`, `DeleteRecursive<E>`) still exist, but instantiating them
across all ~140 families — each embedding a whole-model walk — is a ~140x140
compile blow-up, so the runtime-pointer forms are the supported API for anything
that resolves an element dynamically (pick / serial / path). One model walk
recovers the element's `ElementType` and drives the same referrer bookkeeping.

- `int sdk::Rename(Model&, const void* elem, const std::string& newname)` —
  rename + rewrite every typed referrer atomically. Returns referrers updated, or
  `-1` if `elem` is not in the model.
- `DeleteReport sdk::DeleteSubtree(Model&, const void* elem, bool cascade=false)`
  — remove the subtree; `report.dangling` lists every reference left pointing at
  nothing (with its path). `cascade` clears them; otherwise the caller resolves.
- `void* sdk::Duplicate(Model&, const void* elem)` — deep-clone the subtree as
  the next sibling with fresh serials (generated `Clone`), re-unique the clone's
  names, remap refs **internal** to the clone to the new names, and preserve refs
  pointing outside. Returns the clone's root (its `ElementType` matches `elem`).
- `ReparentResult sdk::Reparent(Model&, const void* elem, void* newParent)` —
  pure-tree move of a body-context child to a new container (Body / Frame /
  `nullptr` = world). Rejects cycles and non-container targets. **No pose fixup**:
  the element keeps its authored local pose (its world pose changes with the new
  parent). Pose-preserving reparent is a compile-aware concern that stays with
  the bridge / editor — the SDK is a pure tree library and does not compute a
  compiled parent world pose.

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

## Consumers and build wiring

Four independent builds consume this surface; all stay green:

- `cpp/CMakeLists.txt` — the library + tests (`ctest`), including the new
  `protospec_public_api_tests` that includes only the umbrella headers.
- `cpp/python/` — the pybind module (compiles the sources directly).
- `apps/studio/` — the standalone editor host (compiles the sources directly).
- The MuJoCo Studio fork — compiles the editor + core live from this repo via
  `PROTOSPEC_ROOT` (see `docs/studio_build.md`).

The umbrella headers reach their implementation headers by **repo-relative path**
(`../../generated/...`, `../../io/...`, etc.), so they work identically under
every one of these builds without new `-I` entries.

The compile surface was hoisted from `ps::mjcf::bridge` up to `ps::mjcf` as a real
rename (no compatibility aliases): the definitions in `cpp/bridge` and every caller
across `cpp/**`, `cpp/python`, `apps/studio`, and the fork were updated in lockstep,
so `ps::mjcf::bridge` no longer exists anywhere. `ps::mjcf::Compile` is the name.
