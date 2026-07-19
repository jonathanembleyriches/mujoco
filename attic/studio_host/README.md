# Attic — the standalone Studio host (parked, pre-plugin)

This is the **old standalone host application** that once drove the ProtoSpec
editor as a first-class app: a vendored thin shell (SDL2 window + Dear ImGui +
the classic `mjr` renderer + step control + pick math) plus a plugin host that
registered the editor cluster directly. It predates the current arrangement, in
which the editor is *purely a plugin* loaded by the real MuJoCo Studio host — so
this host is now dead weight and is parked here with full git history (every file
arrived via `git mv`).

## What lives here

- `host/app.{cc,h}` — the standalone application: window/loop/menus, the load
  slot, and the Play/Stop + File/Edit wiring the standalone app owned.
- `platform/` — the vendored thin shell: `hal/` (SDL2 window + classic `mjr`
  renderer), `sim/step_control`, and `ux/` (the `ps::studio` plugin API —
  `plugin.h`, `ps_plugin_ext.*`, `registry.*` — plus the `interaction` pick math).
- `test/` — the editor-behaviour test batteries (~4.7k lines): `test_host.cc`,
  `test_hostui.cc`, `test_studio.cc`, `test_gizmo.cc`, `test_movability.cc`,
  `test_authoring.cc`, `test_details.cc`, `test_se5_assets.cc`.
- `CMakeLists.txt`, `main.cc` — the standalone build + entry point.

## Why it is parked (and stale)

The host is **stale against the 4-type retarget**: the editor plugins now bind to
the four upstream MuJoCo Studio plugin types, and the old `ps::studio` seam types
this host defines (in `platform/ux/`) no longer match what the editor cluster is
compiled against in the plugin build. The `ps_plugin_ext` / `registry` shims here
reference seam surfaces that were retargeted, so this host will not build as-is
without repointing.

## The tests are worth reviving someday

`test/` carries **~4.7k lines of editor-behaviour tests** — the G1–G9
certification batteries (Play/Stop discard, Save-As externalization,
delete-confirm referrer flow, the key-routing gate, pick/gizmo priority, the
movability class-catch, generated-Details, authoring, SE5 assets, and the gizmo
delta-math battery). They exercise the editor's *own* logic (which is still live
at `studio/editor/`), not the dead host, so they are worth **re-pointing at the
plugin shape** — driving the four upstream plugin vtables over a headless ImGui
context — rather than being rewritten from scratch.

## State of the build files

`CMakeLists.txt` and `main.cc` are parked **as-is**: their source paths
(`editor/…`, `platform/…`) are relative to the former `studio/` root. The editor
cluster still lives at `studio/editor/` (the single source, compiled live by the
plugin fork); reviving this host would mean repointing those paths and updating
the `ps::studio` seam types to the 4-type plugin API. It is off the default build
and not covered by the layering boundary gate (see `protospec/tests/test_boundaries.py`,
which skips `attic/studio_host/`).
