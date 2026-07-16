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
| `protospec/compile.h` | MuJoCo compile bridge: `ps::mjcf::bridge::{Compile, Recompile, CompileToXml, Compiled, Binding, CompileOptions, VfsAsset, CompileReport, CompilePath}` | `protospec_bridge` (+ MuJoCo) |
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
| `ps::mjcf` | The generated object model — every element type, enum, union wrapper, variant alias, `ElementType`, `element_type_of`, and the value ops `Clone` / `ApplyDefault` / `ToMjcf` / `FromMjcf` / `Visit` |
| `ps::mjcf::reflect` | Runtime schema reflection (descriptors, `Describe`, field/child/union metadata) |
| `ps::mjcf::io` | MJCF parse / write |
| `ps::mjcf::validate` | Three-tier validation |
| `ps::mjcf::bridge` | Compile to MuJoCo (`mjModel` + `Binding`) |
| `ps::sdk` | The ergonomic authoring layer over `ps::mjcf` |

Nested `detail` namespaces (`ps::mjcf::bridge::detail`, `ps::sdk::detail`, ...)
and the internal headers below are **not** public.

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
namespace bridge = ps::mjcf::bridge;
namespace sdk = ps::sdk;
namespace mj = ps::mjcf;

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
  bridge::Compiled compiled = bridge::Compile(model);
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
  `detail::`. `Binding`, `Compiled`, `CompileReport`, and `Diagnostic` are
  public via `protospec/compile.h`.

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
  `opt<Ref<T>>` without spelling `ps::Ref<T>` or knowing the storage shape.
- **Primitive sizing** (builders): `sdk::SeedPrimitiveSize(geom)` and
  `sdk::AddPrimitive(parent, type, name)` — a bare `AddGeom` authors only what
  you pass (DR-1), so a primitive has no size and will not compile;
  `AddPrimitive` returns one ready to simulate.
- **Appearance** (builders): `sdk::AddMaterialLayer(mat, role, texture)`,
  `sdk::SetLayerTexture`, `sdk::SetLayerRole`, `sdk::RemoveMaterialLayer`,
  `sdk::SetTextureFile(tex, path)`, `sdk::SetTextureBuiltin(tex, builtin)` — the
  material/texture/layer authoring the editor previously kept as its own private
  mutator library.

Already present and sufficient (the editor simply had not adopted them):
`sdk::ForEachOfType<T>(model, fn)` covers "enumerate every material / element of
a type"; `sdk::Find<T>`, `sdk::ParentMap`, `sdk::Resolve`/`ResolveTo`,
`sdk::FindReferrers`, `sdk::Attach`/`AttachModel`, `sdk::Effective`.

## Remaining sharp edges (follow-up)

These consumer needs still lack a clean public verb and are flagged for a later
pass:

1. **Runtime-typed rename / delete.** `sdk::Rename<E>` and
   `sdk::DeleteRecursive<E>` exist and work, but are per-element-type templates;
   instantiating them across all ~140 families is a heavy compile cost, so the
   editor reimplements them from `ps::sdk::detail`. Fix: add non-template
   `Rename` / `DeleteSubtree` variants keyed on `ElementType` + element pointer.
2. **Duplicate.** No `sdk::Duplicate(model, elem)` (deep-clone next to the
   original, re-unique names, remap internal refs). `attach.h` does the
   namespaced-splice half; a same-model duplicate verb should join it.
3. **Reparent / move.** No `sdk::MoveSubtree(model, elem, newParent)` for the
   pure-tree relink (unlink from the owning union list, splice under a new body).
   Keep-world-pose recomputation stays an application concern.
4. **Binding id → world pose helpers.** Consumers map a tree element to its
   `mjModel` id via `Binding` then read `mjData`/`mjModel` arrays by hand; a thin
   pose-accessor layer over `Binding` would remove the last raw-MuJoCo reach in
   gizmo/overlay code.

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
every one of these builds without new `-I` entries. No implementation header was
moved or renamed; the surface is additive, so every existing consumer keeps
building unchanged.
