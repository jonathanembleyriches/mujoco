# Named-Exceptions Ledger

**Policy.** ProtoSpec is byte-exact against a *pinned* MuJoCo (`docs/SYNC_STATE.md`)
**except** for the deliberate deviations named in this file. We do **not** replicate
upstream bugs: where stock MuJoCo is wrong-to-spec, ProtoSpec stays faithful-to-spec
and is intentionally non-identical until upstream fixes it. Every such deviation — and
every performance/robustness exception we knowingly carry — is named here, with a repro,
a root cause pinned to an upstream `file:line`, an upstream issue link (or a
`TODO-file-issue` draft to paste), an exact re-test command, and explicit retirement
criteria.

**Sync ritual step 7** (`docs/mujoco_bump.md` / `docs/aplus_plan.md` Wave 1 §1c.7)
re-runs *every* active entry against the new pin on each sync. An entry retires only when
its re-test shows the divergence gone; retirement records the **fixing pin** and moves the
entry to the *Retired* section. A retired entry is never deleted — it is the regression
record that proves we would notice a re-introduction.

- Merge-base for all "root cause" line numbers below: **`67a1ea6d`** (2026-06-13).
- Current sync pin referenced by the verdicts: **`3990305373b8`** ("Add docstrings to
  mjCPair", 2026-07-19). Verdicts marked **[pin-verified]** were read directly out of the
  pinned source tree, not inferred from commit messages.

---

## Active exceptions

### EXC-1 — Cylinder-actuator 3-valued `bias` stack overrun (stock bug; ProtoSpec faithful)

- **id:** `EXC-1`
- **title:** Stock `OneActuator` reads a cylinder's 3-valued `bias` into a scalar,
  overrunning it and clobbering the adjacent `gainprm` "area"; ProtoSpec's mjs leg is
  faithful-to-spec and keeps `area`.
- **repro fixture (in-repo):**
  `protospec/tests/fixtures/pathdiff_gated/cylinder_bias_upstream.xml`
  (a `<cylinder>` actuator with `area="0.02" bias="0 -1 0"`).
- **test (in-repo):**
  `protospec/tests/test_path_diff.py::test_cylinder_bias_upstream_documented`.
- **expected ProtoSpec vs stock behavior:**
  - *ProtoSpec MjsPath:* faithful — `actuator_gainprm[0]` ("area") = `0.02` as authored.
  - *Stock (and ProtoSpec XmlPath, which mirrors stock byte-for-byte):* the reader parses
    the 3-valued `bias` into a scalar slot and the overrun zeroes `actuator_gainprm[0]`
    ("area" → `0`).
  - The two legs therefore diverge on `actuator_gainprm` **by design**; both legs compile
    successfully, and XmlPath still matches stock `mj_loadXML` (the bug lives in stock).
- **root cause (upstream `file:line` @ `67a1ea6d`):**
  `src/xml/xml_native_reader.cc`, `mjXReader::OneActuator`, cylinder branch (lines
  **2554–2564**):
  - 2557 `double bias = actuator->biasprm[0];`  ← scalar
  - 2558 `double area = actuator->gainprm[0];`  ← adjacent local
  - 2561 `ReadAttr(elem, "bias", 3, &bias, text);`  ← writes **3** doubles into the
    address of the single scalar `bias`, overrunning into the adjacent local(s); the
    compiled model's `gainprm` "area" comes out `0`.
- **upstream issue:** `TODO-file-issue` — none exists. Draft in "Issue drafts" §I-1 below.
- **re-test procedure (exact command, from repo root):**
  ```
  PROTOSPEC_MUJOCO_SRC=<pinned checkout> \
  PROTOSPEC_BUILD_PS_LIB=<studio build_ps/lib> \
  PROTOSPEC_MJ_INCLUDE=<pinned include dir> \
  pytest protospec/tests/test_path_diff.py::test_cylinder_bias_upstream_documented -q
  ```
  Green = the documented divergence still holds (nonzero-exit `actuator_gainprm` diff on
  `--path-a XmlPath --path-b MjsPath`, and `--against-stock` still PASS). The test itself
  is the sync-ritual probe.
- **retirement criteria:** retire when, at the new pin, `--against-stock` on the fixture
  shows stock keeping `area=0.02` (i.e. XmlPath and MjsPath converge). At that point the
  `test_cylinder_bias_upstream_documented` assertions invert; convert them to a plain
  parity assertion and move this entry to *Retired* with the fixing pin recorded (template
  below).
- **PIN VERDICT [pin-verified] — NOT fixed at `3990305373b8`.**
  The user reported this bug as *patched upstream* and the A+ plan lists EXC-1 as the
  "expected first retiree". **Reading the pinned source contradicts that:** at
  `3990305373b8`, `src/xml/xml_native_reader.cc:2573–2582` is byte-identical in
  substance to the merge-base —
  `double bias = actuator->biasprm[0];` (2575), `double area = actuator->gainprm[0];`
  (2576), `ReadAttr(elem, "bias", 3, &bias, text);` (2579). The overrun is **still
  present**. EXC-1 therefore **stays Active** at this sync; do **not** retire it. The
  "confirmed fixed at <pin>, here's our regression fixture" framing planned for the first
  upstream meeting is premature — the correct first-meeting framing is a live bug report
  (§I-1), because the bug reproduces at today's `main`.

> **Retirement note template (for whenever EXC-1 actually goes away):**
> ```
> Retired at pin <shortsha> (<date>). Fixing commit(s): <sha…range or "not identifiable
> by message; located by re-running the fixture">. Verified: --against-stock on
> cylinder_bias_upstream.xml now PASSes with area=0.02 on both legs; the OneActuator
> cylinder branch no longer reads bias(3) into a scalar (see <file:line at fixing pin>).
> test_cylinder_bias_upstream_documented converted to a parity assertion in commit <sha>.
> ```

---

### EXC-2 — O(n²) `mjs_setName` / `mjCModel::CheckRepeat` (performance exception)

- **id:** `EXC-2`
- **title:** Per-element linear (sort-based) name-uniqueness scan makes large
  fully-named models quadratic to build via the mjs API.
- **kind:** **performance** exception (not a correctness deviation — output is correct;
  the cost is the problem). ProtoSpec does not "fix" it locally; it is inherited from
  upstream's `mjs_*` API and is named here so the sync ritual keeps measuring it.
- **repro fixture (in-repo):**
  `protospec/tests/fixtures/pathdiff/benchmark_many.xml` — a flat worldbody of **800**
  independent bodies, each with one joint and one geom (800 bodies / 800 joints / 800
  geoms), regenerable via `protospec/tests/fixtures/pathdiff/gen_benchmark.py`.
- **expected ProtoSpec vs stock behavior:**
  Both the mjs build path and stock's own XML parse path pay the same super-linear name
  cost — this is *not* a ProtoSpec-only regression. Measured: an 800-body fully-named
  scene spends **~300 ms** dominated by the repeated name scan, on both paths.
- **root cause (upstream `file:line` @ `67a1ea6d`):**
  - `src/user/user_api.cc:2151–2166` — `mjs_setName` calls
    `baseC->model->CheckRepeat(element->elemtype)` (line **2160**) on **every** name set.
  - `src/user/user_model.cc:4593–4618` — `mjCModel::CheckRepeat` builds a fresh
    `vector<string>` of **all** names of that object type, `std::sort`s it, and
    `adjacent_find`s — **O(k log k)** per call. Setting names on `n` elements of a type is
    therefore Σ O(k log k) = **O(n² log n)** overall.
  - The same `CheckRepeat` is invoked from the compile path (`user_model.cc:4586`), so
    stock's XML parse is equally penalized on large fully-named models.
- **upstream issue:** `TODO-file-issue` — none exists. Draft in §I-2 below (suggests a
  per-type hash-set of live names for O(1) amortized uniqueness checks).
- **re-test procedure (exact command):**
  The bench is a manual harness run (no pytest wrapper). Build `ps_path_diff` the way the
  test fixture does, then run `--bench`:
  ```
  # build (same recipe as test_path_diff.py::ps_path_diff)
  g++ -std=c++20 -O2 \
    protospec/lib/harness/ps_path_diff.cc \
    protospec/lib/harness/model_diff_lib.cc \
    protospec/lib/harness/plugin_registry.cc \
    -Iprotospec/lib/include -Iprotospec/lib/generated -Iprotospec/lib/compile \
    -Iprotospec/lib/io -Iprotospec/lib/harness -I"$PROTOSPEC_MJ_INCLUDE" \
    "$PROTOSPEC_BUILD_PS_LIB/libprotospec_core.a" \
    "$PROTOSPEC_BUILD_PS_LIB/libmujoco.so" \
    -Wl,-rpath,"$PROTOSPEC_BUILD_PS_LIB" -o /tmp/ps_path_diff

  # measure (N-run average of write/parse/compile stages)
  /tmp/ps_path_diff --bench 20 protospec/tests/fixtures/pathdiff/benchmark_many.xml
  ```
  Record the per-stage average against the prior sync's number; a jump is the signal
  upstream changed the name machinery (in either direction).
- **retirement criteria:** retire when upstream replaces the sort-based scan (e.g. a live
  name hash-set) and the 800-body bench drops from ~O(n²) to ~linear — i.e. the compile
  stage no longer dominates and scaling `--n` 400→800→1600 (regenerate via
  `gen_benchmark.py --n`) shows ~2× per doubling rather than ~4×. Record the fixing pin
  and the new baseline timing in *Retired*.
- **PIN VERDICT [pin-verified] — NOT fixed at `3990305373b8`.**
  `mjs_setName` at the pin still calls `model->CheckRepeat(element->elemtype)` per set;
  the mechanism is unchanged. **Active.**

---

### EXC-3 — Studio `LoadModelFromBuffer` null-spec crash on `application/mjb` load

- **id:** `EXC-3`
- **title:** Studio host's `App::LoadModelFromBuffer` unconditionally calls
  `spec_editor_.Reset(*spec())` after a buffer load; a compiled `.mjb` (`application/mjb`)
  buffer yields a null `spec()`, so `*spec()` → `mj_copySpec(nullptr)` → crash.
- **kind:** **host (Studio) robustness** bug, surfaced during the plugin retarget. This is
  a fork/host-side deviation, not a ProtoSpec corpus divergence.
- **repro fixture (in-repo):** *none applicable.* This is not reproducible from a corpus
  XML — it requires the Studio host loading an `application/mjb` buffer (e.g. a
  `ModelPlugin` whose `get_model_to_load` returns compiled `.mjb` bytes). Minimal repro is
  described inline in §I-3, not carried as a checked-in file (no fabricated path).
- **expected ProtoSpec/host vs stock behavior:**
  - *Correct:* an `.mjb` load has no editable `mjSpec`; the host should reset the spec
    editor to an empty/no-spec state and continue.
  - *Buggy (merge-base):* the host dereferences the null spec and crashes on first `.mjb`
    load.
- **root cause (upstream `file:line` @ `67a1ea6d`):**
  `src/experimental/studio/app.cc`, `App::LoadModelFromBuffer` (lines **170–181**):
  line **180** is an unconditional `spec_editor_.Reset(*spec());`. `spec()` returns
  `model_holder_->spec()` (`app.h`), which is **null** for a compiled-`.mjb` load, so
  `Reset(*spec())` dereferences null and reaches `mj_copySpec(nullptr)`.
- **upstream issue:** `TODO-file-issue` — none exists. Draft in §I-3 below. Because the
  bug is **already fixed at the pin** (verdict below), the issue is best filed as an
  acknowledgement + offer of a regression test rather than a live report.
- **re-test procedure (exact command):**
  No windowless harness drives the Studio buffer-load path today; the check is a
  **source-level probe** run during the ritual against the pinned fork tree:
  ```
  git -C <fork> show <pin>:src/experimental/studio/app.cc \
    | sed -n '/void App::LoadModelFromBuffer/,/^}/p'
  ```
  Confirm the `spec_editor_.Reset(*spec())` call is guarded by an `if (spec())` (or
  `has_spec()`) test. (When the Wave-2 recompile-invariant windowless battery grows an
  `.mjb`-adopt case, wire this into it and cite the test id here.)
- **retirement criteria:** retire once the pinned host source guards the reset against a
  null spec (see verdict). Record the fixing pin.
- **PIN VERDICT [pin-verified] — FIXED at `3990305373b8`.**
  At the pin, `App::LoadModelFromBuffer` (`src/experimental/studio/app.cc:195–210`) now
  guards the reset:
  ```cpp
  if (spec()) {
    spec_editor_.Reset(*spec());
  } else {
    spec_editor_.Reset();
  }
  ```
  (`app.h` also adds `bool has_spec() const { return model_holder_ && model_holder_->spec(); }`.)
  The null deref is gone. **EXC-3 retires at this sync** — see *Retired* below.

---

## Retired exceptions

### EXC-3 (retired) — Studio `LoadModelFromBuffer` null-spec crash

Retired at pin **`3990305373b8`** (2026-07-19). Fixing change: the host's
`App::LoadModelFromBuffer` now guards `spec_editor_.Reset(*spec())` with `if (spec()) …
else Reset();` (`src/experimental/studio/app.cc:205–208` at the pin; `has_spec()` helper
in `app.h`). Verified by reading the pinned fork source (see EXC-3 pin verdict). Kept as a
regression record: if a future sync re-introduces an unconditional `Reset(*spec())` on the
buffer-load path, this entry is the trail. Upstream issue (§I-3) to be filed as
"confirmed fixed at <pin>, here is a regression test if you want it".

> Note: EXC-1 was projected by the A+ plan to be the first retiree, but pinned-source
> verification (EXC-1 pin verdict) shows it is **not** fixed at `3990305373b8`; the actual
> first retiree at this sync is **EXC-3**.

---

## Issue drafts (paste-ready)

### §I-1 — Cylinder actuator: 3-valued `bias` overruns a scalar and zeroes `gainprm` area

> **Title:** `<cylinder>` actuator: reading a 3-valued `bias` overruns a scalar local and
> clobbers `gainprm` "area"
>
> **Component:** XML native reader (`src/xml/xml_native_reader.cc`, `mjXReader::OneActuator`)
>
> **Version:** reproduces on `main` at `3990305373b8` (and at `67a1ea6d`).
>
> **Description:** In `OneActuator`'s `cylinder` branch, `bias` is declared as a scalar
> `double bias = actuator->biasprm[0];` but is then filled with
> `ReadAttr(elem, "bias", 3, &bias, text);` — three doubles written into the address of a
> single `double`. The overrun writes past `bias` into the adjacent local(s); in practice
> the compiled model's `actuator_gainprm[0]` ("area") is zeroed. A cylinder actuator that
> specifies both `area` and a 3-valued `bias` silently loses its area.
>
> **Minimal repro (XML):**
> ```xml
> <mujoco>
>   <worldbody>
>     <body pos="0 0 1">
>       <joint name="wrist" type="hinge" axis="0 1 0"/>
>       <geom type="capsule" size="0.03" fromto="0 0 0 0.3 0 0"/>
>     </body>
>   </worldbody>
>   <actuator>
>     <cylinder name="a" joint="wrist" area="0.02" bias="0 -1 0"/>
>   </actuator>
> </mujoco>
> ```
> After `mj_loadXML`, `actuator_gainprm[0]` is `0` instead of `0.02`.
>
> **Suggested fix:** read `bias` into a 3-element buffer (`double bias3[3]`) matching the
> attribute arity, or read a single value if `bias` is defined as scalar; either way stop
> passing count `3` with a scalar destination.
>
> **Note:** we carry a regression fixture (`cylinder_bias_upstream.xml`) and a
> faithful-to-spec reader; happy to contribute the fixture as a test.

### §I-2 — `mjs_setName` is O(n²) via per-set `CheckRepeat` on large fully-named models

> **Title:** `mjs_setName` / `mjCModel::CheckRepeat` is O(n² log n) when naming many
> elements (mjs API and XML parse both affected)
>
> **Component:** `src/user/user_api.cc` (`mjs_setName`), `src/user/user_model.cc`
> (`mjCModel::CheckRepeat`)
>
> **Description:** `mjs_setName` calls `model->CheckRepeat(elemtype)` on every name
> assignment. `CheckRepeat` rebuilds a `std::vector` of *all* names of that object type,
> sorts it, and `adjacent_find`s — O(k log k) per call. Building a fully-named model with
> `n` elements of a type therefore costs Σ O(k log k) ≈ O(n² log n). Measured: an 800-body
> scene (800 bodies/joints/geoms, all named) spends ~300 ms dominated by this scan. The
> same `CheckRepeat` runs on the XML compile path (`user_model.cc:4586`), so stock's own
> parse of large fully-named models pays it too — editors that rename in bulk hit it hard.
>
> **Suggested fix:** maintain a per-object-type hash-set (`absl::flat_hash_set<string>` /
> `unordered_set`) of currently-live names on `mjCModel`, updated on set/clear/remove, so
> uniqueness is an O(1) amortized membership check instead of a full sort per set. Falls
> back to the existing sort only for the final whole-model validation if desired.
>
> **Repro:** flat worldbody of N named bodies/joints/geoms; time `mjs_setName` across N =
> 200/400/800 and observe ~4× per doubling. (We have a generator + `--bench` harness we can
> share.)

### §I-3 — Studio `LoadModelFromBuffer` null-spec crash on `application/mjb` (already fixed on main)

> **Title:** (fixed on `main`) `App::LoadModelFromBuffer` crashed on `application/mjb`
> loads via null-spec `Reset(*spec())`
>
> **Component:** `src/experimental/studio/app.cc` (`App::LoadModelFromBuffer`)
>
> **Description:** On older revisions (through `67a1ea6d`), `LoadModelFromBuffer`
> unconditionally ran `spec_editor_.Reset(*spec());`. For a compiled-`.mjb`
> (`application/mjb`) buffer, `spec()` is null, so `*spec()` dereferences null and
> `mj_copySpec(nullptr)` crashes. Triggered by a `ModelPlugin` whose `get_model_to_load`
> returns compiled `.mjb` bytes (content type `application/mjb`), or any host path feeding
> `.mjb` bytes to `LoadModelFromBuffer`.
>
> **Status:** confirmed **fixed** at `3990305373b8` — the reset is now guarded with
> `if (spec()) … else spec_editor_.Reset();`. Filing for the record and to offer a
> regression test for the buffer-load-with-null-spec case if useful.
>
> **Minimal repro (pre-fix):** register a `ModelPlugin` returning `application/mjb` bytes
> from `get_model_to_load`; on load, Studio crashes in `Reset(*spec())` before first frame.
