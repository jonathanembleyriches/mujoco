# ProtoSpec

ProtoSpec is an IDL-driven redesign of the MJCF/mjSpec authoring layer. One
schema file describes the model format; a generator emits the C++ object
model, serialization, reflection, validation tables, and Python bindings from
it. Compilation to `mjModel` goes through MuJoCo itself — ProtoSpec adds no
second compiler — and correctness is defined as byte-exact agreement with
stock MuJoCo, enforced by differential tests.

## How it works

```
schema/mujoco.spec          the IDL: every element, field, type, default,
        │                   union, and reference relationship, stated once
        ▼
protospec_gen/              the generator (pure Python, no deps)
        │   emit.py         → lib/generated/   C++ types, XML binding tables,
        │                     reflection, keywords, defaults   (13 files)
        │   emit_py.py      → lib/python/generated/  pybind11 bindings
        │   emit_mjs.py     → mjSpec builder verbs for the compile bridge
        ▼
lib/                        handwritten library around the generated core
```

Generated code is **checked in** and byte-gated: `uv run python -m
protospec_gen.emit --check` fails if `lib/generated/` disagrees with the
schema by a single byte, so drift between schema and code cannot exist.

The object model is deliberately plain: generated structs of owned values,
every optional field presence-tracked (`std::optional`), references stored by
name with typed wrappers. No hidden compiler state, no pointers into a graph
— a `Model` is a value you can copy, diff, and serialize.

The layers on top:

- **`lib/io/`** — MJCF reader/writer (vendored tinyxml2), table-driven by the
  generated XML binding, with handwritten quirk handlers for the format's
  irregular corners. MuJoCo-free.
- **`lib/core/`** — canonicalization resolvers (orientation and inertia
  spellings fold into canonical quat/diaginertia at parse end). MuJoCo-free.
- **`lib/validate/`** — three-tier structural / referential / semantic
  validation over a const `Model`, driven by the generated tables.
- **`lib/sdk/`** — the ergonomic authoring layer (below).
- **`lib/compile/`** — the bridge to `mjModel` (below).
- **`lib/python/`** — pybind11 module `protospec`, generated from the same
  schema, mirroring the C++ surface.

## The SDK

`lib/sdk/protospec/sdk.h` is a pure tree library over the generated types —
it is written once against the reflection/visit hooks and never needs
regenerating when the schema grows:

- `builders.h` — typed `Add*` verbs that insert into the right child list
  (`AddBody`, `AddPrimitive`, `AddFreeJoint`, `AddMaterial`, …).
- `traversal.h` — `World`, `Find<T>`, `ForEachOfType<T>`, `ParentMap`,
  path-to-element.
- `refs.h` — typed reference handling: `SetRef`, `Resolve`, `FindReferrers`,
  `Rename` (referrer-safe), `DeleteRecursive`.
- `classes.h` — defaults-class queries (`Effective`) and rewrites
  (`FlattenDefaults`, `ExtractClass`).
- `attach.h` — namespaced deep-clone splice of one model into another.

A complete load → edit → validate → compile round trip
(`lib/test/test_public_api.cc` runs exactly this):

```cpp
#include "protospec/sdk.h"
namespace mj  = ps::mjcf;
namespace sdk = ps::sdk;

auto parsed = ps::mjcf::io::ParseMjcfString(xml, "hello.xml");
mj::Model& model = *parsed.model;

mj::Body& box = sdk::AddBody(sdk::World(model), "box");
box.pos = std::array<double, 3>{0, 0, 1};
sdk::AddFreeJoint(box, "box_free");
mj::Geom& g = sdk::AddPrimitive(box, mj::GeomType::box, "box_geom");

mj::Material& mat = sdk::AddMaterial(model, "grid_mat");
sdk::SetRef(g.material, mat);            // typed, name-backed reference

auto diags = ps::mjcf::validate::Validate(model);   // pre-compile checks
mj::Compiled compiled = mj::Compile(model);          // → mjModel via MuJoCo
```

The Python module exposes the same schema-generated surface, so the model you
script in Python is the same object model the C++ editor-side code sees.

## Compile paths and correctness

`mj::Compile` reaches `mjModel` by two routes, selected by `CompilePath`:

- **XmlPath** — write canonical MJCF, load with `mj_loadXML`.
- **MjsPath** — author an `mjSpec` directly through the public `mjs_*` API,
  including faithful mirrors of the `flexcomp`/`composite` expansions, then
  `mj_compile`.
- **Auto** — prefer MjsPath; a scan of the model routes the few constructs
  the public mjs API cannot express to XmlPath.

`lib/harness/ps_path_diff.cc` is the permanent correctness net: it compiles a
corpus twice and diffs the resulting `mjModel`s field-by-field (every sizes
int, name table, and pointer array) in three modes — identity (determinism),
mjs-parity (XmlPath vs MjsPath), and against-stock (ProtoSpec vs a pristine
`mj_loadXML` of the original file, catching reader/writer drift).

The claim this suite enforces: **byte-exact vs the enclosing MuJoCo checkout**
over MuJoCo's own model corpus (last verified against main at 3.11.0,
2026-07-22). The one named exception is performance, not correctness: the
O(n²) `mjs_setName` rename cost on large fully-named models.

## Building and testing

Everything runs from this `protospec/` directory. Python tooling uses
[uv](https://docs.astral.sh/uv/); the C++ library is a standalone CMake
project with a MuJoCo-free core.

```sh
# Generated code matches the schema, byte for byte.
uv run python -m protospec_gen.emit --check

# C++ core (object model, io, validate, SDK) + unit tests. No MuJoCo needed.
cmake -S lib -B lib/build && cmake --build lib/build -j && ctest --test-dir lib/build

# Python suite: schema, generator, extractors, differentials.
uv run pytest
```

Tests that need MuJoCo *source* (the bootstrap extractors under `tools/`,
which regenerate `snapshots/` ground truth) default to the enclosing checkout
— this directory lives inside the MuJoCo repo — and honor
`PROTOSPEC_MUJOCO_SRC` as an override. Tests that need *prebuilt* libraries
(the `ps_path_diff` differentials link `libmujoco.so` + `libprotospec_core.a`)
skip unless `PROTOSPEC_BUILD_PS_LIB` points at a build tree containing both,
so a plain `uv run pytest` stays green everywhere.
