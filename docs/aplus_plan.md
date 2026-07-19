# ProtoSpec + Studio Editor — A+ Master Plan

Date: 2026-07-19. Repos: `/home/buzz/Documents/proto/mujoco` (branch `fix/linux-build-and-mode-ui`, HEAD `ed954ef1`) and fork `/home/buzz/Documents/proto/mujoco-studio` (branch `studio`, merge-base `67a1ea6d`).
Inherits findings from: the deep audit (grades below), `docs/studio_plugin_min_ask.md`, `docs/studio_plugin_feasibility.md`, `docs/snapshot_refresh.md`, `docs/editor_certification.md`, and the audit deliverables committed at `ed954ef1`.

---

## EXECUTIVE SUMMARY (one page)

**Grades: current → target**

| Axis | Now | Target | Path |
|---|---|---|---|
| Exactness enforcement | D → fixing (bridge gaps in flight) | **A+ (versioned)** | Wave 1 sync + Wave 2; claim becomes "byte-exact vs MuJoCo \<pin\>", never unversioned |
| Ergonomics C++ | C+ | **A** | Wave 3: SDK promotion, error alignment, contract doc |
| Ergonomics Python | D | **A** | Wave 3: edit verbs + builders + kwargs ctors (the big move) |
| Robustness | B | **A** | Wave 4: six scoped hardening items |
| Cleanliness | A− | **A** | Falls out of Waves 3–4 (rename, duplicate deletion) |
| Plugin fragility | B+ | **A− (cap)** | Wave 1 re-verify assumptions + Wave 5 upstream asks; A+ unreachable until upstream grants ViewportStatePlugin |

**The headline finding (upstream recon, verified remotely today):** we are **234 commits behind upstream main**. Our merge-base `67a1ea6d` is dated **2026-06-13**; **MuJoCo 3.10.0 was released 2026-06-22** — nine days after our snapshot — and main has moved five further weeks. This confirms the user's "not synced properly" diagnosis and explains the cylinder-bias discrepancy. Critically, upstream churn *changes our plugin strategy*:

- **`ScenePlugin` landed upstream 2026-06-23** ("plugin for adding mjvGeoms to the scene", `5637f743`) — the depth-occluded selection outline, our *worst* accepted degradation and one of the min-ask doc's three "irreducible" gaps, is now **zero-ask**.
- **`LaunchStudio()` launcher landed 2026-06-29** (`1e04c568`, `studio/launcher.h`) — a downstream `main()` can register plugins and launch stock Studio. The **registration-hook ask is essentially satisfied**; the fork's `main.cc` patch dies.
- **`ModelPlugin` grew `pre_step`/`post_step`/path-loading** (2026-06-23) — strengthens the commit-on-release MJB route.
- **`SpecEditorPlugin::pre_compile` still dispatches per-frame with `&camera_`** (verified at main `app.cc:504-506`) — the camera-abuse mechanism survives, but it remains an abuse; ViewportStatePlugin stays ask #1.
- **Transport semantics changed** ("Viscous posing mode" replaced Viscous Pause, `StepControl` refactored, timeline scrubber added) — the highest-risk churn against our `do_update` freeze trick and three-state Space transport.
- **`mjspec.h`/`mjs_*` churned** ("Migrate types in mjs structs" `11f1da0c`, style modernization `0fd7bea9`) and **`xml_native_reader.cc` took 9 commits** (new: geom `surfacevel`, `body/simple`, attach `frame` attr, self-attach, global-attribute conflict policies; fixed: damper `kv` default inheritance) — direct hits on our generated bindings, schema, and reader mirror.
- The **cylinder-bias fix is not identifiable by message** in the 234 public commits; the MIMO actuator restructure (`d507e921`, 2026-07-15) rewrote the relevant counts/params code. The ritual verifies by *re-running our repro*, not by trusting a commit ID.

**Critical path:** Wave 0 (land in-flight bridge work, freeze green baseline) → **Wave 1 (upstream sync + install the repeatable sync ritual — everything else is graded against a stale ruler until this is done)** → Wave 2 (exactness A+, versioned) → Waves 3+4 in parallel (ergonomics / robustness) → Wave 6 (freeze + release). Wave 5 (upstream conversations + bug reports) starts the day Wave 1's recon is confirmed and runs on its own calendar.

**Honest caps:** plugin fragility tops out at A− until upstream accepts ViewportStatePlugin (we sit on an experimental API with no stability promise — feasibility doc risk #1); manual live-window checks stay manual (no input injection headless); the C++ harness stays Windows-first with Linux builds local-only (user deprioritized CI); exactness carries named exceptions for upstream bugs by design; web-viewer readiness is unanswerable until Studio authors answer the 7 questions in the feasibility doc §4.

---

## WAVE 0 — Preflight: land in-flight work, freeze the baseline

**Goal:** a tagged, all-green baseline on the *old* base so every Wave-1 diff is attributable to the sync, not to concurrent work.
**Effort:** S. **Owner-shape:** main agent only (coordinates with the in-flight mjs-bridge agent; do not parallelize over the same files).

Checklist:
1. Let the in-flight mjs-bridge-gap work (protospec/lib/compile + tests, currently dirty in git status) land and commit.
2. Full local gate run on the old base: `pytest -q` (~1660), `ctest` 6/6 in `protospec/lib/build`, the 8 windowless editor batteries (per `docs/editor_certification.md`), `emit --check`, `lift_registry check`, three `test_path_diff.py` modes green.
3. Commit stray root files (`mug_coacd.xml`, `test.xml`) or delete them; clean `git status` on both repos.
4. Tag both repos: `pre-sync-baseline` (mujoco repo) and `studio-pre-sync` (fork).

**Gate to Wave 1:** everything in step 2 green; both tags pushed.
**Rollback point:** the tags themselves — this is the rollback anchor for the entire plan.

---

## WAVE 1 — Upstream sync + the Sync Ritual (first-class)

**Goal:** both repos re-based on a pinned, current upstream; every gate green against the new ruler; the ritual documented so this never rots again.
**Effort:** L (the largest single wave — five weeks of upstream churn hit schema, generated bindings, reader mirror, and the Studio host at once).
**Owner-shape:** partially subagent-able — see the parallel split below.

### 1a. Sync state (established facts — do not re-derive)

- Upstream HEAD today: `3990305` (main). Latest release: **3.10.0** (2026-06-22). Merge-base `67a1ea6d` = 2026-06-13, **234 commits behind, 0 ahead** (GitHub compare API).
- Churn on our contact surfaces since the base: `mjspec.h` 5 commits / ~130 lines (type migration + style modernization + `surfacevel` + `body/simple` + attach policies); `user_api.cc` 7 commits (incl. `mj_encode` 64-bit, resource writing); `xml_native_reader.cc` 9 commits; `platform/ux/plugin.h` 3 commits (**ScenePlugin, PreStep/PostStep, path-loading**); `studio/app.cc` **22 commits** (viscous posing, StepControl refactor, timeline scrubber, launcher, `mjv_moveCamera`); `platform/ux/gui.cc` 13 commits. Engine: `mjData.qM` **removed** (2026-07-04), midpoint integrator removed, MIMO actuator count split (`nu`/`nactuator`/`nout`) — the last two matter to `mj_model_diff` field-walking and the MJB adopt route.
- `include/mujoco/mjrfilament.h` is now a **public 380-line C API** — the natural home for the Filament fast-path ask (no flush/reset-material-cache in it yet — ask still open, but reframed).

### 1b. Decisions (made, not enumerated)

- **Sync target: a pinned upstream `main` SHA, not the 3.10.0 tag.** Justification: Studio and the plugin surface only meaningfully exist on main (ScenePlugin, launcher, viscous posing are all post-3.10.0); 3.10.0 is only 9 days past our stale base and would leave us behind on the exact surfaces we care about. Record the pin in a new `docs/SYNC_STATE.md`. Exactness claims cite the pin: "byte-exact vs MuJoCo 3.10.0+\<shortsha\>".
- **Strategy: rebase the `studio` branch as a curated patch series, not merge.** Justification from the delta shape: the fork delta is now only **521 insertions / 46 deletions across 14 files**, deliberately minimized by the recent seam-stripping commits (`1285b49e`, `b2ec3fa1`, `306f1b2b`). A merge would preserve 15+ intermediate commits whose whole history is churn *toward* minimalism and would compound conflicts every sync. Instead, squash the delta into **~4 thematic patches** rebased onto the pin: (1) `protospec/` subtree + CMake registration, (2) `app.cc` keyhandler-ordering + screenshot/headless flags, (3) `gui.cc` dock layout (RootV3), (4) `.gitignore`/misc. Expected shrinkage during this very rebase: the `main.cc` hunk (+32) is replaced by our own `main()` over upstream's new `LaunchStudio()`; screenshot hunks may shrink if upstream's `update_live.yml`/headless work overlaps. Each sync = `git rebase --onto <pin>` of 4 patches; conflicts are small and local by construction.
- **Cadence:** run the full ritual **monthly**, plus **mandatorily before any release or exactness claim**, plus opportunistically when upstream lands something we asked for. A 2-minute `git ls-remote` + compare-API check is part of a weekly habit, not a ritual run.

### 1c. THE SYNC RITUAL (the repeatable procedure — becomes `docs/mujoco_bump.md`)

1. **Recon.** Add/refresh `upstream` remote (`google-deepmind/mujoco`) on the fork; fetch; record candidate pin (latest green main SHA), `ahead_by`, and the per-path churn on: `mjspec.h`, `mjspecmacro.h`, `user_api.cc`, `xml_native_reader.cc`, `platform/ux/*`, `studio/*`, `engine/*` (fields walked by `mj_model_diff`), filament.
2. **Rebase** the 4-patch series onto the pin; resolve; drop any patch upstream obsoleted. Re-verify the fork delta stat is ≤ its previous size (a growing delta is a red flag to investigate).
3. **Rebuild `build_ps`** (`-DMUJOCO_USE_FILAMENT=ON -DMUJOCO_BUILD_STUDIO=ON -DMUJOCO_STUDIO_PROTOSPEC=ON`, per `docs/studio_build.md`). A compile failure in `studio/editor/plugin_abi.h`'s `static_assert`s is the guard working — reconcile struct-by-struct against upstream's `plugin.h`, never loosen a guard to pass.
4. **Snapshot refresh** (`docs/snapshot_refresh.md`): point `PROTOSPEC_MUJOCO_SRC` at the pinned checkout; run the three extractors (`extract_mjcf_schema.py`, `extract_mjspec_fields.py`, `extract_spec_defaults.py`); **review the JSON diffs by hand** — every changed line is upstream telling you what moved.
5. **Schema catch-up:** update `protospec/schema/mujoco.spec` until `test_schema_coverage.py` and `test_corpus_coverage.py` pass without new un-reasoned waivers (this bump: `surfacevel`, `body/simple`, attach `frame`, self-attach, conflict policies, damper-`kv` inheritance semantics). Then `emit --check` after regeneration; commit generated files.
6. **Corpus differentials — the blocking ritual:** all three `test_path_diff.py` modes green over `fixtures/pathdiff/` — *identity* (determinism), *against-stock* (reader/writer drift vs the pin's `mj_loadXML`), *mjs* (MjsPath vs XmlPath parity) — plus `test_differential.py` round-trip, `test_bridge_corpus.py`, `test_validate_corpus.py` against the pin's vendored corpus (which itself grows — new corpus models are new coverage, not noise).
7. **Named-exception re-test:** run every repro in `docs/EXCEPTIONS.md` (Wave 2 artifact) against the pin; retire fixed entries (expected first retiree: **cylinder-actuator bias stack-overrun** — verify with our repro, note the fixing commit range in the retirement entry).
8. **Host-assumption probes + editor tests:** windowless batteries (8/8) + `test_plugin_windowless.py` + the new assumption-probe tests (below) + `test_studio_smoke.py` + one manual live-window pass of the certification checklist.
9. **Docs + claim:** update `SYNC_STATE.md` (pin, date, delta stat, waiver review, exceptions retired/kept) and the versioned exactness string everywhere it appears.
10. **Tag** both repos `synced-<pin-shortsha>`.

**"Synced properly" =** all ten steps complete, `SYNC_STATE.md` current, zero un-reasoned waivers, exceptions re-tested, delta ≤ previous. Anything less is "partially synced" and blocks releases.

### 1d. The 8 host-behavior assumptions (min-ask doc risk register) — churn assessment for THIS bump

Ranked by likelihood the current bump broke them; each gets a cheap probe test added to the windowless/smoke tier so future breaks are loud:

1. **`do_update`-returns-true skips `StepControl::Advance` (the freeze trick)** — **HIGHEST RISK**: StepControl refactor (`58d6910a`) + viscous-posing-mode replaced pause semantics (3 commits, 07/11–07/14). Our three-state Space transport (`5511fd27`) must be revalidated and possibly redesigned against the new transport model. Budget real time here.
2. **Full-window PassthruCentralNode + `DisplaySize` picking math** — HIGH: `gui.cc` took 13 commits (toolbar scrubber, UI restyle); verify pick-ray normalization still matches.
3. **`get_model_to_load` MJB + `post_model_loaded` fires same-frame inside `OnModelLoaded`** — MEDIUM: ModelPlugin extended (path-loading, pre/post_step); mechanism likely intact, verify the same-frame camera-restore ordering.
4. **`camera_` pointer stability** — MEDIUM: `mjv_moveCamera` migration (`b2106db5`) touched camera handling.
5. **`pre_compile` dispatched per-frame, unconditionally, with `&camera_`** — LOW-MEDIUM: verified present at main; confirm dispatch is still unconditional (the "disable spec editor when no spec" commit `c665f0b5` must be checked for dispatch gating, not just menu gating).
6. **Filament full-rebuild-per-load economics** — LOW: filament churn (buffer sizes, public C API) doesn't change the rebuild; re-measure the hitch once.
7. **`WantCaptureMouse` gate at top of `HandleMouseEvents`** — LOW: mouse handling refactors possible; one grep + one probe.
8. **`SetNextFrameWantCaptureMouse` availability (vendored ImGui ≥1.92.5)** — LOWEST: ImGui rarely regresses API.

### 1e. Parallelization, gate, rollback

- **Parallel subagents:** (A) fork rebase + build_ps rebuild (steps 2–3); (B) snapshot refresh + schema catch-up + emit (steps 4–5) — B needs only a read-only checkout of the pin, so A and B run concurrently; (C) EXCEPTIONS repro harness prep (feeds step 7). Differentials (step 6) are the join point — they need both A's `build_ps` and B's schema.
- **Gate to Wave 2:** the full ritual checklist green; `SYNC_STATE.md` committed; `docs/mujoco_bump.md` written (the ritual *is* a Wave-1 deliverable, not just an execution).
- **Rollback:** `studio-pre-sync` / `pre-sync-baseline` tags; the rebase happens on a working branch (`studio-sync-<sha>`) and only fast-forwards `studio` at gate-pass.

---

## WAVE 2 — Exactness to A+ (versioned)

**Goal:** exactness is *enforced*, *complete over the schema*, and *honest* — every deviation is named, tracked, and re-tested on every bump.
**Effort:** M. **Owner-shape:** highly subagent-able (fixture authoring, exception harness, invariant test are independent).

Checklist:
1. **Fixture-family completeness 137 → 142-or-waivered.** HEAD `ed954ef1` took pathdiff coverage to 137 of the 142 schema element types across all three differential modes. Identify the remaining 5; for each either author a pathdiff fixture or add an explicit reasoned waiver (e.g. an element type that cannot appear in a compilable standalone model). Target statement: "every element type is either differentially tested or waivered with a reason" — no silent gaps.
2. **Recompile state-migration invariant test.** New windowless battery asserting the editor's adopt path preserves `qpos/qvel/act/ctrl` and time across recompile (commit-on-release route), keyed by element index/id per the min-ask probe finding (name-keying breaks on unnamed `<replicate>` elements). Also assert camera restoration invariants (anti-flicker) already covered ad hoc.
3. **Named-exception policy → `docs/EXCEPTIONS.md`.** One entry per known upstream-bug deviation: repro fixture path, expected-vs-stock behavior, upstream issue link (file the issue if none exists — Wave 5 owes three), and the "re-test on sync" hook (ritual step 7 reads this file). First entries: cylinder bias (likely retired immediately post-sync), `mjs_setName` O(n²) CheckRepeat (performance exception), MJB `LoadModelFromBuffer` null-spec crash.
4. **Differential = bump-blocking ritual.** Already made so by Wave 1 step 6; here, additionally wire `test_path_diff.py` + `test_differential.py` into the pre-release checklist and (optional, CI deprioritized) a manual workflow-dispatch CI job.
5. **Versioned byte-exactness claims — YES, adopt.** Byte-exactness is only meaningful against a specific compiler; all claims in README/docs/public_api become "byte-exact vs MuJoCo 3.10.0+\<pin\>". Unversioned "exact" claims are removed. `SYNC_STATE.md` is the single source of the current claim.

**Gate to Wave 3/4:** 142/142 accounted; invariant test green; EXCEPTIONS.md exists with all three known entries resolved-or-linked; claims versioned.
**Rollback:** additive only (tests + docs) — revert individual commits; no rollback point needed.

---

## WAVE 3 — Ergonomics to A (the D→A Python move + C++ surface)

**Goal:** the audit's ranked ergonomics list, executed; ends with a coherent SDK 1.0 surface freeze candidate.
**Effort:** L (Python builders dominate). **Owner-shape:** strongly subagent-able — the Python tracks split cleanly by family; C++ items are independent of Python.

Checklist (inheriting the audit's ranking):
1. **Python edit verbs + asset/sensor/tendon builders + kwargs ctors** (D→A). Extend `protospec_gen/emit_py.py` so builders/ctors are *generated from the schema*, not hand-written — this keeps Wave 1's ritual cheap (new upstream attributes flow into Python for free via emit). Acceptance: the corpus's common authoring patterns writable in idiomatic Python; `test_python_bindings.py` grows a builder/verb suite; parity examples in docs.
2. **SDK promotion of the editor's 15 wrapper-gaps; delete the 2 verbatim duplicates.** Move per the audit's list into `protospec/lib/sdk/protospec/`; editor consumes the SDK copies; duplicates deleted the same commit (cleanliness A− → A). Guard: `test_sdk.cc` additions + editor batteries stay green.
3. **Error-convention alignment across the five structural verbs.** Pick ONE convention — recommendation: `absl::Status`-style result object as the SDK already trends, applied uniformly; document in `public_api.md`; migrate the outliers with deprecation shims for one release.
4. **`ps_compile.h` collision — rename, decisively.** `protospec/lib/compile/compile.h` (internal bridge) → `compile/bridge.h`; the public `include/protospec/compile.h` keeps its name (it is the contract). Fix `tools/ps_compile.cc`'s include-order-dependent `#include "compile.h"`. Removes both the CPython-shadow hazard and the include-path fragility found in recon.
5. **SetRef: ADOPT (not remove).** The in-flight work already landed target typing (`1ae03d94`, by-name + by-target overloads with `static_assert`); it is now load-bearing in `mjs_binding`. Decision: keep, promote to `public_api.md` with examples, add negative-case tests. Removal would orphan the generated binding's reference plumbing.
6. **`docs/public_api.md` as a true contract.** Recon verdict: it already *is* a 285-line contract doc enforced by `protospec_public_api_tests`. Remaining work: add the Python surface, per-header stability levels (stable / provisional / internal), the versioned exactness statement, and semver policy.
7. **SDK 1.0 surface-freeze milestone — what we commit to as stable:** the nine umbrella headers (`core.h model.h reflect.h io.h validate.h compile.h sdk.h save.h protospec.h`) + the Python module surface + the CLI contracts of `ps_compile`/`ps_roundtrip`/`ps_validate` + the schema file format. Explicitly NOT frozen: anything under `lib/*` internals, the editor plugin ABI (tracks upstream experimental), `harness/`, generated-file internals. Freeze is declared at Wave 6 release, not before.

**Gate to Wave 6:** audit ergonomics re-score A on both languages; `protospec_public_api_tests` covers the frozen surface; zero verbatim duplicates.
**Rollback:** per-item revert; item 3's deprecation shims are themselves the rollback for callers.

---

## WAVE 4 — Robustness to A (runs parallel with Wave 3)

**Goal:** close the audit's six named robustness gaps. All are small; the value is picking policies and enforcing them with tests.
**Effort:** S–M total. **Owner-shape:** one subagent can take all six; they are independent, ordered by silent-failure severity.

1. **Adoption dim-assert (the silent-corruption self-check).** Assert `mjModel` dimension/count invariants at every editor adopt boundary (and post-MJB-roundtrip); mismatch = loud diagnostic + refuse adopt, never render corrupted state. Note the MIMO `nu/nactuator/nout` split lands in exactly this code — write the assert against the post-sync field set.
2. **`CloneWithSerials` assert** (`studio/editor/undo.cc`): assert serial-map bijectivity on clone; corrupted undo stacks currently fail far from the cause.
3. **Nesting-depth cap in `mjcf_reader.cc`.** Recon confirms recursion is unbounded. Policy: cap at **200** body-tree levels (far beyond any real model; stack-safe on default 8 MB stacks) with a clear diagnostic naming the element and depth. Test with a generated 201-deep fixture.
4. **Duplicate-name parse diagnostic.** Today only the referential validator tier catches it (`test_validate.cc:183`). Add a parse-time warning-tier diagnostic at first sight of the duplicate (with both source locations), keeping the validator as the authoritative error. Matches stock MuJoCo's early failure shape.
5. **`mju_user_warning` strategy — decided:** we live in someone else's process, so the handler is not ours to own. (a) Compile-scoped capture: save the previous handler, install ours, restore in RAII — all under a process-global mutex serializing ProtoSpec compiles; (b) document loudly in `public_api.md` that the handler is host-owned and briefly borrowed, and that host compiles concurrent with a ProtoSpec compile will see our handler (accepted, documented limitation); (c) add "context-local warning callback" to the upstream question list (Wave 5) — the real fix is upstream's to make.
6. **Include-path traversal — decided:** resolve `<include>` relative to the *including file*; **reject** resolved paths escaping the root model's directory tree by default with a clear diagnostic; explicit `allow_external_includes` opt-in (API flag + CLI flag) for trusted workflows. Rationale: the editor opens untrusted files from disk inside a GUI host; silent traversal is an exfiltration-shaped bug.

**Gate:** each item lands with its test in the appropriate tier; audit robustness re-score A.
**Rollback:** per-item revert; item 6 ships flag-guarded so the default flip is a one-line revert.

---

## WAVE 5 — Plugin / Upstream track (calendar-parallel from end of Wave 1)

**Goal:** convert fork-carried fragility into upstream-granted stability; pay our bug-report debts. Externally gated — sequence conversations so each ask unblocks a named quality item.
**Effort:** M for us (RFCs, repros, probes); calendar-long externally. **Owner-shape:** repro/RFC prep subagent-able; the conversations are the user's.

**Re-scoped ask list after today's recon (supersedes the counts in both session docs):**

| # | Item | Status after recon | Unblocks |
|---|---|---|---|
| 0 | ScenePlugin (was OverlayPlugin ask) | **LANDED upstream** (`5637f743`) — adopt it in Wave 1; delete our screen-space-outline compromise | Kills degradation #1 (non-occluded selection outline), zero ask |
| 0 | Registration hook | **EFFECTIVELY LANDED** via `LaunchStudio()` (`1e04c568`) — our own `main()` registers + launches; fork's `main.cc` patch deleted in Wave 1 rebase | Fork moves most of the way to "build recipe" |
| 1 | **ViewportStatePlugin** (~15 lines: per-frame `{const mjvCamera*, float aspect, bool paused}`) | Still the one ask worth making (min-ask doc §"THE minimal ask list") — the `pre_compile` camera abuse survives at main but remains an abuse | **Kills the worst remaining fragility** (assumptions #4/#5 in §1d); lifts plugin grade toward A |
| 2 | Filament fast-path | Reframed: propose as additions to the now-public `include/mujoco/mjrfilament.h` (flush + material-cache reset) | Cosmetic (per-commit reload hitch) — pitch #1 in feasibility doc §5b still quotable |
| 3 | Transport-intent + dock-layout hooks | Cosmetic; **hold** until the viscous-posing dust settles (assumption #1 churn) — asking against a moving transport design wastes goodwill | Host Play/Stop mirroring; curated layout without the `gui.cc` patch |

**Sequencing of the user's upstream conversations:**
1. **First meeting: bug reports (give before asking).** (a) cylinder-actuator bias overrun — *verify against the pin first*; if fixed, open with "confirmed fixed at \<pin\>, here's our regression fixture if you want it"; (b) O(n²) `mjs_setName` CheckRepeat (with timing repro — editors rename in bulk); (c) MJB `LoadModelFromBuffer` null-spec crash from R1 (with minimal repro). All three get entries/retirements in `EXCEPTIONS.md`.
2. **Second: the ViewportStatePlugin RFC** (one page, seeded from min-ask doc; note we already run degraded without it — the ask is de-hacking, not unblocking — maintainers accept that framing more readily).
3. **Third: the web-viewer question list** (feasibility doc §4, 7 questions, verbatim) + `mjrfilament.h` fast-path. The web answers are a **hard dependency for any remoting decision** — no remoting work is planned or scoped until answered.
4. Add to the question list: context-local `mju_user_warning` (Wave 4 item 5c) and the plugin-API stability commitment (feasibility question 6).

**Gate:** none we control — track asks in `SYNC_STATE.md`; each granted ask triggers an opportunistic ritual run to adopt it.
**Rollback:** n/a (all upstream-side); our adoption commits are individually revertable.

---

## WAVE 6 — Quality infrastructure: staying at A+

**Goal:** the machinery that keeps grades from decaying. Mostly docs + wiring; ends with a release.
**Effort:** S–M. **Owner-shape:** subagent-able per artifact.

1. **`docs/mujoco_bump.md`** — the Wave-1 ritual, written as an operator checklist (largely a Wave 1 deliverable; finalize here with lessons from the first execution).
2. **Test-tier map** (section in HANDOFF.md or `docs/testing.md`): **unit** (ctest 6/6 + pytest generator/IDL) → **fixture** (pathdiff 142-accounted, three modes) → **corpus** (bridge/validate/coverage over vendored corpus) → **windowless** (8 editor batteries + plugin/assumption probes) → **smoke** (`test_studio_smoke.py` + manual certification list). For each tier: trigger (every commit / pre-push / ritual / release), runtime, and env prerequisites (`PROTOSPEC_MUJOCO_SRC`, `PROTOSPEC_BUILD_PS_LIB`, corpus root).
3. **Gates matrix — local-first, CI optional (respecting the user's deprioritization).** Local: unit+fixture on every change; corpus+windowless before push; full ritual monthly/pre-release. CI (`.github/workflows/ci.yml`) stays Python-only (drift gate, boundary gate, pytest) as today; *optional appendix*: a manual `workflow_dispatch` Linux C++ job for the differential — documented, not required.
4. **Release/versioning story for ProtoSpec-as-a-library:** semver from `v1.0.0` at the Wave 3 surface freeze; git tags + `CHANGELOG.md`; every release names (a) the frozen API surface per `public_api.md`, (b) the MuJoCo pin it is byte-exact against, (c) the EXCEPTIONS list at that pin. Consumers get one sentence: *"ProtoSpec vX.Y.Z is byte-exact against MuJoCo A.B.C+\<sha\> except the N named exceptions."* Patch releases may re-pin MuJoCo only via a full ritual run.
5. **Drift alarms:** the weekly 2-minute upstream check (ls-remote + compare-API `ahead_by`) noted in HANDOFF; `SYNC_STATE.md` gets a "stale after" date (pin date + 6 weeks) that the ritual doc tells you to treat as a blocking TODO.

**Gate (plan completion):** v1.0.0 tagged; all six waves' checklists closed or explicitly carried as EXCEPTIONS/asks.

---

## WHAT CANNOT REACH A+ (and why — honest register)

1. **Plugin fragility beyond A−** — gated on upstream accepting ViewportStatePlugin and on `src/experimental` gaining any stability promise (feasibility doc risk #1). Until then we run on verified-but-unpromised host behaviors; the mitigation (ABI static_asserts + assumption probes + the ritual) makes breaks *loud and cheap*, not absent.
2. **Automated live-window verification** — input injection doesn't reach the window headless (editor_certification.md); the manual checklist stays manual. Screenshot-sequence capture partially compensates.
3. **Windows/Linux harness parity in CI** — the C++ harness is Windows-locked and the user deprioritized CI; Linux C++ verification stays a local ritual step. Accepted.
4. **Unversioned byte-exactness** — impossible by construction against a moving compiler; A+ here *means* versioned exactness with a named-exception ledger, which Wave 2 delivers.
5. **Exceptions themselves** — where stock MuJoCo has a bug we refuse to reproduce, we are deliberately non-identical until upstream fixes it; the ledger + re-test-on-sync is the A+ form of that stance.
6. **Web/remoting readiness** — unanswerable until the Studio authors answer the 7 questions; explicitly out of scope until then (Wave 5 dependency).

## WAVE MAP (sequence + effort at a glance)

```
W0 Preflight            S   ──gate: baseline green+tagged──▶
W1 Upstream sync        L   ──gate: ritual 10/10 green────▶ ┬──▶ W2 Exactness  M ──▶ ┐
   (3 parallel tracks)                                      ├──▶ W3 Ergonomics L ──▶ ├─▶ W6 Freeze+release S/M
                                                            ├──▶ W4 Robustness S/M ─▶ ┘
                                                            └──▶ W5 Upstream track M (calendar-parallel, no internal gate)
```
