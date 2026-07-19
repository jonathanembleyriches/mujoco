# SYNC_STATE — ProtoSpec ↔ MuJoCo pin

Single source of truth for *which* upstream MuJoCo ProtoSpec is currently synced to,
what the last sync changed, and what remains open. Updated by every run of the sync
ritual (`docs/aplus_plan.md` Wave 1 §1c; future `docs/mujoco_bump.md`).

## Current pin

| Field | Value |
|---|---|
| Upstream repo | `google-deepmind/mujoco` `main` |
| Pin SHA | `3990305373b81df7ed8af52e549ac02146bae400` (`3990305`) |
| Pin subject | "Add docstrings to mjCPair." |
| Pin date | 2026-07-19 |
| Engine version header | `mjVERSION_HEADER 3010001` → reported as **3.10.1** |
| Latest tagged release upstream | 3.10.0 (2026-06-22) |
| Merge-base of prior sync | `67a1ea6d` (2026-06-13) |
| **Ahead-count at this sync** | **234 commits ahead, 0 behind** (merge-base → pin) |
| Contact-surface churn | `include/mujoco/mjspec.h` 5 commits · `src/xml/xml_native_reader.cc` 9 commits |

**Versioned exactness claim (current):**

> ProtoSpec is byte-exact vs **MuJoCo 3.10.1+3990305**, except the named entries in
> `docs/EXCEPTIONS.md`.

Status: **CONFIRMED at the join gate (2026-07-19).** All legs green at this pin:
schema/snapshot/generator (Track B), the rebased fork build (Track A, all eight host
assumptions held), and the full `build_ps` differential harness — `test_path_diff.py`
identity + mjs-parity + against-stock over the 79-model pin corpus (grown from 77;
the new `surfacevel/carousel.xml` exposed the pin's self-attach feature, now gated
`mjs.attach_frame` pending Wave-2 mjs support), plus the damper-`kv` inheritance fix
mirrored into the mjs builder. Suite: 301 passed / 30 skipped.

**Stale-after:** 2026-08-30 (pin date + 6 weeks). Treat as a blocking TODO past this date.

## What this sync changed

Snapshots refreshed via the three extractors (`tools/bootstrap/extract_mjcf_schema.py`,
`extract_mjspec_fields.py`, `extract_spec_defaults.py`) pointed at the pin worktree; JSON
diffs reviewed by hand. Schema header bumped `3.10.0` → `3.10.1` to match the engine
header the snapshots were lifted from.

### `mjcf_schema.json` (MJCF surface — 5 new attributes, 2 new keyword maps)

- **geom `surfacevel`** — new `double[6]` attribute (surface velocity: linear+angular);
  appears on `<geom>` in worldbody/body, in `<default>`, and on composite `<geom>`.
  (upstream `4787c809` "Add geom surfacevel").
- **body `simple`** — new attribute controlling simple-body optimization; keyword map
  **`FAuto_map`** (`false`=0 / `auto`=1), default `auto`. (upstream `5618666a`).
- **compiler `conflict`** — new global-attribute conflict-resolution policy; keyword map
  **`conflict_map`** (`warning` / `merge` / `error`). (upstream `410c7316`).
- **actuator `actlimited`** — now listed on `<intvelocity>` (general already had it).
- **attach `frame`** — new `<attach frame="...">` attribute (upstream `c6c3ec31`); plus
  self-attach permitted in MJCF (`2d283b5e`).

### `mjspec_fields.json` (mjs struct surface — type migration + new fields/enum)

- **Type migration ("Migrate types in mjs structs", modernization).** ~35 boolean fields
  changed ctype `mjtByte` → `mjtBool`; several `int` fields became their proper enum type:
  `inertiafromgeom`→`mjtInertiaFromGeom`, joint/tendon/actuator `limited`/`actfrclimited`/
  `ctrllimited`/`forcelimited`/`actlimited`→`mjtLimited`, `align`→`mjtAlignFree`,
  `selfcollide`→`mjtFlexSelf`, texture `builtin`/`mark`→`mjtBuiltin`/`mjtMark`, compiler
  `alignfree`→`mjtBool`. These flow through the generated `static_cast<…>` in
  `mjs_binding.cc` for free on regen.
- **Enum tag normalization.** All `mjt*_` enum *tags* lost the trailing underscore
  (`mjtGeomInertia_`→`mjtGeomInertia`, …) — cosmetic in the snapshot; the generator keys
  on enum *name*, not tag, so no generated-code effect.
- **New enum `mjtConflict`** (`mjCONFLICT_WARNING`/`MERGE`/`ERROR`).
- **New fields:** `mjsCompiler.conflict` (`mjtConflict`), `mjsBody.simple` (`mjtByte`),
  `mjsGeom.surfacevel` (`double[6]`).
- Comment typo fix `errorsx`→`errors` (`mjsCompiler.info`).

### `spec_defaults.json`

- `mjsBody.simple` default **1** (auto); `mjsGeom.surfacevel` unset-by-memset (zeros).

## Schema edits (`protospec/schema/mujoco.spec`)

- New enums `Conflict {warning,merge,error}` and `SimpleMode {false_,auto}`.
- `Compiler.conflict : Conflict = warning`.
- `Body.simple : SimpleMode = auto`.
- `Geom.surfacevel : double[6]` and `CompositeGeom.surfacevel : double[6]`.
- `IntVelocity.actlimited : TriState`.
- `Attach.frame : string` (object-model round-trip fidelity; `<attach>` is a whole-element
  emit_mjs waiver — no mjs binding).
- Header `mujoco_version "3.10.1"`.

## Generator edits (`protospec/protospec_gen/emit_mjs.py`)

- `ENUM_MJT += Conflict → mjtConflict` and `SimpleMode → mjtByte` (raw 0/1, no dedicated
  mjt enum). Regenerated; `emit --check` clean.

## Waiver review

- **Schema-coverage waivers** (`test_schema_coverage.py`): unchanged — only the standing
  `attach` element waiver (read-time expansion). **No new un-reasoned waivers.**
- **mjs-binding coverage** (`test_mjs_binding.py`): green with **no new** `FIELD_WAIVERS`
  or `ELEMENT_WAIVERS`. The three new mapped fields (`conflict`, `simple`, `surfacevel`)
  and `intvelocity/actlimited` all bind exactly / via the two new `ENUM_MJT` rows.
  `Attach` remains a whole-element waiver (Wave-4 `mjs_attach`); its new `frame` attribute
  therefore needs no mjs entry.

## Exception ledger review (`docs/EXCEPTIONS.md`, ritual step 7)

- **EXC-1 cylinder-actuator `bias` stack overrun — NOT RETIRED at this pin.** The bug the
  plan expected to retire *survives*: `src/xml/xml_native_reader.cc` `OneActuator` cylinder
  branch (pin lines **2575–2582**) still declares scalar `double bias = actuator->biasprm[0]`
  then `ReadAttr(elem, "bias", 3, &bias, text)` — a 3-into-1 overrun clobbering the adjacent
  `area = gainprm[0]`. Source-verified read; the runtime re-test (`test_path_diff.py::
  test_cylinder_bias_upstream_documented`) is a build_ps join item. **Entry stays active.**
- Other EXC entries (mjs_setName O(n²), MJB null-spec) not re-tested here (build_ps).

## Open join items (need `build_ps` + the pinned engine — deferred to the differentials join)

1. **Damper `kv` default-class inheritance (upstream `faf0dabc`).** Upstream fixed the
   classic reader so an un-authored damper `kv` inherits `-gainprm[2]` when the inherited
   gaintype is affine. Our **MjsPath** builder has the *pre-fix* shape:
   `protospec/lib/compile/mjs_builder.cc:1295-1298` uses `double kv = 0` and never reads the
   inherited `gainprm[2]` (contrast the `Velocity` branch just above, which *does* read
   `a->gainprm[0]`). To match the fixed pin, the Damper branch should become
   `double kv = (a->gaintype == mjGAIN_AFFINE) ? -a->gainprm[2] : 0;` before applying the
   authored override. **Not changed here:** `lib/compile/` is outside this track's file set
   and the fix is only verifiable via the mjs-vs-stock differential against the pin engine.
   XmlPath uses the pin's own (fixed) `mj_loadXML`, so only the MjsPath leg is at risk.
2. **`build_ps` differential suite at the pin** — `test_path_diff.py` (identity /
   against-stock / mjs), `test_differential.py`, `test_bridge_corpus.py`,
   `test_validate_corpus.py`. These link `libprotospec_core.a` + `libmujoco.so` from the
   studio `build_ps` tree (owned/churned by the concurrent fork agent) and were **not run**
   in this track. They confirm the *byte-exact* half of the claim, the new `surfacevel` /
   `simple` / `conflict` write-order, and item 1's damper fix. `test_path_diff.py` also
   probes for `libmujoco.so.3.10.0` by name — with the pin at 3.10.1 the versioned filename
   moves; the plain `libmujoco.so` fallback still matches.
3. **Conflict-policy & attach-frame / self-attach compile semantics.** The MJCF surface is
   modelled (schema + round-trip). The *behavior* (how attach merges global attributes
   under `warning`/`merge`/`error`, self-attach) is engine-side and, for our stack, part of
   the deferred Wave-4 `mjs_attach` work; no ProtoSpec compile action is taken here.

## Verification run in this track (MuJoCo-free)

- Three extractors ran clean against the pin (detected 3.10.1).
- `emit` + `emit --check`: clean (13 generated files + python bindings).
- Pin-gated extractor integration tests (`PROTOSPEC_MUJOCO_SRC=<pin>`): pass.
- Coverage / binding / generator gates green: `test_schema_coverage`, `test_mjs_binding`,
  `test_corpus_coverage`, `test_emit`, `test_idl`, `test_schema_parses`,
  `test_handtable_keys`, `test_python_bindings`.
