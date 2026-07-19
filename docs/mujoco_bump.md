# MuJoCo version-bump ritual

The repeatable procedure for re-pinning ProtoSpec + the Studio fork to a newer
upstream MuJoCo. Run it monthly, before any release or exactness claim, and
opportunistically when upstream lands something we asked for. "Synced properly"
means every step below is green and `docs/SYNC_STATE.md` is current — anything
less is *partially synced* and blocks releases.

First executed 2026-07-19 (67a1ea6d -> 3990305, 234 commits, MuJoCo 3.10.0-dev
-> 3.10.1). Lessons from that run are folded in as NOTES.

## Steps

1. **Recon.** In the fork: `git fetch upstream main`. Record the candidate pin
   (latest green main SHA), `git rev-list --count HEAD..upstream/main`, and the
   per-path churn on the contact surfaces: `include/mujoco/mjspec.h`,
   `mjspecmacro.h`, `src/user/user_api.cc`, `src/xml/xml_native_reader.cc`,
   `src/experimental/platform/ux/*`, `src/experimental/studio/*`, engine fields
   walked by `mj_model_diff`, filament.
2. **Rebase** the fork's curated patch series onto the pin, on a working branch
   `studio-sync-<shortsha>` (see `git log studio` — after the plugin-side moves
   the whole delta is three files: `.gitignore`, the CMake mount block, and the
   keyhandlers-first hunk in `app.cc`; the dock layout and screenshot machinery
   live in this repo as editor plugins). Drop any patch upstream obsoleted —
   re-check the keyhandler hunk especially (it retires the day upstream
   dispatches plugin KeyHandlers before its built-in chords). The delta stat
   must come out **no larger** than the previous sync's — growth is a red flag.
   NOTE: stale `MUJOCO_DEP_VERSION_*` entries in `build_ps/CMakeCache.txt`
   override the pin's dep SHAs and break dep patches — unset them and clear the
   affected `_deps` dirs (filament can usually stay).
3. **Rebuild `build_ps`** (`-DMUJOCO_USE_FILAMENT=ON -DMUJOCO_BUILD_STUDIO=ON
   -DMUJOCO_STUDIO_PROTOSPEC=ON`) **plus the plugin targets** (`elasticity
   sdf_plugin sensor actuator` — not attached to the app target; reconfigures
   drop them). A `studio/editor/plugin_abi.h` static_assert failure is the guard
   WORKING: reconcile struct-by-struct against the new `plugin.h`; never loosen
   a guard to pass.
4. **Snapshot refresh** (`docs/snapshot_refresh.md`): `PROTOSPEC_MUJOCO_SRC=<pin
   checkout>` (a detached `git worktree` of the fork works); run the three
   extractors; **read every JSON diff by hand** — each changed line is upstream
   telling you what moved.
5. **Schema catch-up**: edit `protospec/schema/mujoco.spec` until
   `test_schema_coverage.py` + the enum-pin tests pass with no new un-reasoned
   waivers; check semantic reader changes against our reader mirror and the mjs
   builder (upstream reader FIXES need mirroring — e.g. the damper-`kv`
   inheritance fix); regenerate; `emit --check` clean; the emit_mjs coverage
   gate green (new mjs fields need alias/ENUM_MJT rows or reasoned waivers).
6. **Differentials — the blocking gate, in this order** (NOTE: run only after
   steps 2-5; the harness links `build_ps` artifacts, and stale binaries
   manifest as mass SIGSEGV noise, not real failures): full
   `uv run pytest -q` including `test_path_diff.py` three modes over fixtures
   AND the corpus gates (the pin's `model/` corpus GROWS — new corpus models
   are new coverage; expect them to exercise the pin's new features first).
7. **Named-exception re-test**: run every entry in `docs/EXCEPTIONS.md` against
   the pin; retire what upstream fixed (record the fixing pin), keep the rest.
   Verify by repro, never by commit message.
8. **Host-assumption probes + editor verification**: the windowless plugin
   tests, then a headless smoke (screenshot capture is env-driven:
   `MUJOCO_SCREENSHOT_DIR/_AFTER/_EXIT`, dir must exist) — the editor UI up,
   the model compiling `path=mjs`, viewport non-black. One manual live-window
   pass of the certification checklist when UI behavior churned.
9. **Docs + claim**: update `docs/SYNC_STATE.md` (pin, date, delta stat, waiver
   review, exceptions retired/kept, the versioned exactness claim) — the claim
   always names the pin, never an unversioned "exact".
10. **Tag** both repos `synced-<shortsha>`; fast-forward `studio` to the sync
    branch only after every step above is green.

## Weekly 2-minute drift check (not a ritual run)

`git -C <fork> fetch upstream main && git rev-list --count studio..upstream/main`
plus a glance at `SYNC_STATE.md`'s stale-after date.
