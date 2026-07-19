# Attic — the native ProtoSpec compiler (parked, off the default build)

This directory holds the **native compiler** (`ProtoSpec Model -> mjModel` directly,
no XML round-trip) and everything that only exists to serve it. It is deliberately
**not built by default** and is off the project's main mental map. Nothing here is
deleted — it is parked, with full git history preserved (every file arrived via
`git mv`).

## What lives here

- `compile/` — the native compiler: `native.*`, `build.*`, `context.h`,
  `native_supported.h`, and `lifted/` (the copied-with-provenance MuJoCo asset
  pipeline: mesh/marching-cube/image/builtin-mesh + `mjuu_util`).
- `harness/ps_native_diff.cc` — the three-way XML-vs-native differential driver.
- `test/test_native.cc` — the native-compiler ctest (purity gate, feature gate,
  allocator round-trip, lifted goldens).
- `snapshots/lifted_code.json` + `snapshots/lifted_upstream/` — provenance
  snapshots for every lifted symbol (the CDR-3 drift ledger).
- `tools/lift_registry.py` — maintains that ledger.
- `tests/test_native_differential.py`, `tests/test_lift_registry.py`,
  `tests/native_ratchet.json` — the native pytest suites.
- `docs/` — the native-compiler survey + design/impl plans.

## Why it is parked

The native compiler is scaffolding today: its feature gate (`native_supported.h`)
admits nothing, so every model reports `UnsupportedNatively` and the compile
bridge falls back to the XML oracle path. Carrying its ~17k LOC + 141 provenance
snapshots in the main line of sight made the product look far larger and more
tangled than it is. Parking it here leaves the shipping product — the library
(`protospec/lib/`) + generator + studio — as the default surface.

## How the default build excludes it

The compile bridge (`protospec/lib/compile/compile.cc`) guards its call into the
native compiler behind `#ifdef PROTOSPEC_NATIVE`. With the macro undefined
(the default), the bridge never includes `native.h` and never links the native
compiler; forced `NativePath` reports "not built". The attic sources are simply
not compiled. `Auto` no longer routes to the native compiler at all: it compiles
every valid model through the mjSpec path (`MjsPath`), which now reaches full
parity across all families (the XML oracle is reachable only when forced), so
building the native compiler back in changes only what forced `NativePath` does
-- the native compiler itself is unchanged by the mjSpec-first flip.

## How to opt in (build the native compiler)

Every build carries a `PROTOSPEC_BUILD_NATIVE` option (default `OFF`). Turn it on
to compile the attic sources back in and define `PROTOSPEC_NATIVE`:

    # reference library build
    cmake -S protospec/lib -B protospec/lib/build -DPROTOSPEC_BUILD_NATIVE=ON -DMUJOCO_ROOT=...
    cmake --build protospec/lib/build

    # studio fork build (build_ps)
    cmake -S <studio> -B build_ps -DMUJOCO_STUDIO_PROTOSPEC=ON \
          -DPROTOSPEC_ROOT=<repo> -DPROTOSPEC_BUILD_NATIVE=ON

With the option ON the build re-adds `attic/compile/**` to the sources and the
`-I` include path (the attic's headers and the library's `compile/` bridge headers
do not collide — disjoint basenames), and defines `PROTOSPEC_NATIVE` so the bridge
dispatches into `compile::NativeCompile` again.
