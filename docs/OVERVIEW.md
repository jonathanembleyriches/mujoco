# ProtoSpec + ProtoSpec Studio — overview, run, update

The one-page map of the project: what it is, how to build and run it, how to keep
it synced with MuJoCo, and how it works under the hood. Written for a demo and for
the MuJoCo-team integration conversation. Everything here links to the deeper doc
for each area.

Current state (2026-07-20): ProtoSpec repo HEAD `d7397007`; fork (`mujoco-studio`
branch `studio`) HEAD `7654ddeb`, tag `synced-3990305`; byte-exact vs **MuJoCo
3.10.1** (`docs/SYNC_STATE.md`).

---

## 1. What it is (in three sentences)

**ProtoSpec** is a redesign of MuJoCo's `mjSpec`: one IDL schema
(`schema/mujoco.spec`) generates the C++ object model, its MJCF serialization,
reflection, and the mjSpec bindings — so the model representation, the reader/
writer, validation, and both compile paths are all derived from a single source
of truth. **ProtoSpec Studio** is a Unity/Unreal-style model editor built on top
of it, which runs as a **plugin of MuJoCo's own Studio app** (not a fork of it).
The whole thing is **byte-exact against a pinned MuJoCo version** — proven by a
differential harness, not asserted.

Two repos, one GitHub project, checked out as side-by-side working trees:

| Path | Branch | What lives here |
|---|---|---|
| `~/Documents/proto/mujoco` | `fix/linux-build-and-mode-ui` | ProtoSpec (schema, generator, library, Python, tests) **and** the editor plugin (`studio/editor/`, `studio/glue/`) |
| `~/Documents/proto/mujoco-studio` | `studio` | The MuJoCo Studio host — **stock upstream `3990305` + a 3-file fork** that mounts our plugin |

---

## 2. How to run it

The editor is compiled into the Studio app via one CMake option. The fork's
`build_ps/` is already configured; a rebuild is just:

```
cd ~/Documents/proto/mujoco-studio
cmake --build build_ps --target mujoco_studio        # + elasticity sdf_plugin sensor actuator for plugin models
```

Launch (this machine renders on llvmpipe — the NVIDIA driver isn't built for the
kernel — so use the **classic** renderer; the Filament backend crashes on
software GL, unrelated to the editor):

```
env -u WAYLAND_DISPLAY SDL_VIDEODRIVER=x11 XDG_SESSION_TYPE=x11 \
  ./build_ps/bin/mujoco_studio --gfx=classic \
  --model_file=/home/buzz/Documents/proto/coacd_assets/mug/mug_coacd.xml
```

Once the NVIDIA driver is fixed (`sudo dkms autoinstall && reboot`), drop
`--gfx=classic` and the default Filament path works.

**Using the editor:** the **Edit** toolbar button toggles authoring — Edit ON
freezes the sim at `qpos0` and enables gizmos; toggling off (or `Space`) hands the
sim to Studio's own transport (`Space` then pauses/resumes mid-flight). Panels:
**Hierarchy** (left), **Details** (right), **Layers** (right-bottom), and one
combined **ProtoSpec** panel (`File / + Add / Assets / Diagnostics` tabs, bottom).
Gizmo-drag a geom → release commits a recompile; the model is re-adopted by the
host with sim state and camera preserved.

Headless screenshot (for demos/CI): set `MUJOCO_SCREENSHOT_DIR` (must exist),
`MUJOCO_SCREENSHOT_AFTER=<frames>`, `MUJOCO_SCREENSHOT_EXIT=1`.

From-scratch configure and platform caveats: **`docs/studio_build.md`**.

## 2b. Using ProtoSpec as a library (no editor)

- **Python:** `uv run python` → `import protospec as ps` (build the extension per
  `protospec/tests/test_python_bindings.py` header). Author natively:
  `m = ps.Model(); b = m.add_body(...); m.add_material(...); m.rename(joint, "hinge")`
  then `m.compile()` / `m.save(path)`. Full surface + conventions:
  **`docs/public_api.md`** (Python section).
- **C++:** include `protospec/sdk.h` (authoring verbs), `protospec/compile.h`
  (`ps::mjcf::Compile` → `mjModel` + `Binding`), `protospec/io.h` (MJCF read/write).
  Contract + stability tiers: **`docs/public_api.md`**.
- **CLI:** `protospec/lib/tools/ps_compile.cc` (compile + JSON report),
  `ps_validate`, `ps_roundtrip`.

---

## 3. How it works (the pipeline)

```
schema/mujoco.spec                 ← single source of truth (the IDL)
      │  protospec_gen/ (emit.py, emit_py.py, emit_mjs.py)
      ▼
protospec/lib/generated/           ← C++ types, MJCF bindings, reflection,
                                      defaults, mjSpec bindings, Python bindings
      │
      ├── io/         MJCF reader/writer (the wire format)
      ├── sdk/        the public behavior layer (authoring verbs, refs, classes)
      ├── validate/   enforcement tiers (structural / referential / semantic)
      └── compile/    Model → mjModel, TWO ways:
             • XmlPath : WriteMjcf → mj_loadXML  (the oracle)
             • MjsPath : build an mjSpec directly via mjs_* → mj_compile
             CompilePath::Auto prefers MjsPath, falls back to XmlPath for the
             few families it can't yet author (loud, recorded).
```

**Key design decisions** (rationale in `docs/plan.md`, `docs/refs_design.md`):

- **The schema declares; the SDK behaves; validate enforces; storage is always the
  MJCF wire form.** A `ref<T>` is a name string + a phantom type — typed references
  drive rename/delete fixups, validation, and the editor's ref pickers generically.
- **Element identity is a creation serial**, not a name or a pointer — stable
  across edits and recompiles, which is what makes undo, live-state migration, and
  the editor's binding survive structural changes.
- **The mjSpec path is bit-exact or it doesn't ship.** Every macro (composite,
  flexcomp, cable) and every element family is either authored directly through the
  public `mjs_*` API or a faithful mirror of MuJoCo's own expansion code — **no
  grafting, no XML round-trips inside the mjs path.** The `ps_path_diff` differential
  (`XmlPath` vs `MjsPath` vs stock `mj_loadXML`) is the permanent net.

**The editor as a plugin** (`docs/studio_plugin_min_ask.md`): it drives Studio
through only the **four stock plugin types** (`GuiPlugin`, `ModelPlugin`,
`KeyHandlerPlugin`, `SpecEditorPlugin`) plus the newly-landed `ScenePlugin`. It owns
a ProtoSpec tree and its own compiler; on each recompile it hands the host
compile-XML bytes through `ModelPlugin::get_model_to_load`, and the host adopts them
through its normal load path — so **the host never sees a ProtoSpec type**. The dock
layout and screenshot capture are themselves plugins. The entire fork of the Studio
tree is **3 files, +45/−8**: a ~15-line CMake mount block, one keyhandler-ordering
hunk in `app.cc`, and `.gitignore`.

---

## 4. How to update it (MuJoCo version bumps)

This is the load-bearing part for long-term integration. The full procedure is
**`docs/mujoco_bump.md`**; the current pin and last-sync summary is
**`docs/SYNC_STATE.md`**. In brief, a bump is:

1. **Recon** — `git fetch upstream main` in the fork; note the new SHA and the
   churn on the contact surfaces (`mjspec.h`, `xml_native_reader.cc`, the Studio
   plugin API).
2. **Rebase** the fork's small curated patch series onto the new pin (delta must
   not grow); rebuild `build_ps`. A `plugin_abi.h` static-assert failure is the
   guard *working* — reconcile struct-by-struct against the new `plugin.h`.
3. **Refresh snapshots** (`docs/snapshot_refresh.md`) with `PROTOSPEC_MUJOCO_SRC`
   pointed at the pin; **read every JSON diff** — it enumerates exactly what moved.
4. **Catch up the schema** until coverage/enum gates pass with no new waivers;
   regenerate; `emit --check` clean.
5. **Run the differentials** (the blocking gate): `uv run pytest -q` including the
   three `ps_path_diff` modes over fixtures and the corpus.
6. **Re-test the exceptions ledger** (`docs/EXCEPTIONS.md`) — retire what upstream
   fixed, *verify by repro not by commit message*.
7. **Update `SYNC_STATE.md`** (the versioned exactness claim always names the pin)
   and tag `synced-<sha>`.

The first run of this ritual absorbed **234 commits (five weeks of MuJoCo)** in
about a day, with exactly two real code fixes — both flagged by the gates, neither
a surprise. That is the whole point of the schema-driven design: a bump is *reading
diffs*, not archaeology.

**"Synced properly"** = every step green + `SYNC_STATE.md` current. A weekly
2-minute `git rev-list --count studio..upstream/main` check keeps drift visible.

---

## 5. Verification & guarantees

| Gate | What it protects | Command |
|---|---|---|
| `emit --check` | generated code matches the schema, byte-identical | `uv run python -m protospec_gen.emit --check` |
| Full suite | everything (301 core + Python + differentials) | `uv run pytest -q` |
| `ps_path_diff` (3 modes) | mjs-path bit-exactness, reader/writer round-trip, vs stock | in the suite / `protospec/lib/harness/` |
| Corpus differential | the real 79-model MuJoCo corpus, Auto path | env-gated in the suite |
| Boundary test | the editor only uses the public library surface | in the suite |
| Windowless plugin tests | the editor's host-behavior assumptions | in the suite |

Exactness is a **versioned invariant**: "byte-exact vs MuJoCo 3.10.1+3990305,
except the named entries in `docs/EXCEPTIONS.md`." Those entries are upstream bugs
we deliberately don't replicate (each with a paste-ready issue draft).

---

## 6. For the MuJoCo team conversation

- **`docs/upstream_asks.md`** — the ask document: two small generic plugin-API
  additions (KeyHandler-first dispatch, which retires our last fork hunk; and a
  `ViewportStatePlugin` that de-hacks our one remaining seam), one optional CMake
  mount hook, and three bug reports (cylinder-bias overrun, O(n²) `mjs_setName`,
  the MJB null-spec crash — the last already fixed on main).
- **`docs/studio_plugin_feasibility.md`** — the full plugin-integration analysis,
  including the 7 web/networked-viewer questions that gate any remoting work.
- Headline for the pitch: *the editor already runs on your existing plugin surface;
  our entire fork is three files, and one of them disappears the day you take the
  KeyHandler ordering PR.*

---

## 7. Importing into a new repo (repackaging checklist)

Everything below is what actually matters when moving this work into a fresh
repository (internal hosting, monorepo, wherever). The project is TWO logical
pieces sharing one git history today; they separate cleanly.

### Piece 1 — the ProtoSpec repo content (branch `protospec`)

Import the whole working tree of the `protospec` branch. Per-directory role:

| Dir | Role | Import? |
|---|---|---|
| `protospec/` | the product: schema, generator, lib (incl. committed `generated/` — gated by `emit --check`, import as-is), snapshots (drift-gate ground truth), python, tests | **yes, all** |
| `studio/` | the editor plugin (`editor/`) + the fork-facing build glue (`glue/`) | **yes, all** |
| `docs/` | contracts + rituals (`SYNC_STATE.md`, `mujoco_bump.md`, `EXCEPTIONS.md`, `public_api.md` are load-bearing) | **yes** |
| `pyproject.toml`, `uv.lock` | the python env; package = `protospec/protospec_gen`, testpaths = `protospec/tests` + `attic/tests` | **yes** (trim `attic/tests` from testpaths if attic is dropped) |
| `attic/` | parked native compiler + old standalone host; not built | optional (history keeps it) |
| `.github/` | minimal CI (drift gate + pytest) | optional |
| root `mug_coacd.xml`, `test.xml` | untracked scratch | no |

Git **tags do not transfer** — the pin and rollback anchors live as SHAs inside
`docs/SYNC_STATE.md`; keep that file authoritative. No absolute paths are baked
into the build; test conveniences that reference sibling checkouts
(`../mujoco-studio`, menagerie, coacd assets) degrade to skips when absent.

### Piece 2 — the Studio fork (branch `studio`)

Do NOT import the fork's history. The fork is exactly:
**pinned upstream MuJoCo (`3990305373b8`, records in `SYNC_STATE.md`) + a
3-file delta (+45/−8)** — `.gitignore`, the `studio/CMakeLists.txt` mount
block, the `app.cc` keyhandler-first hunk. Regenerate it anywhere with:

```
git diff 3990305373b8..studio > protospec-studio-fork.patch   # from this repo
# new location: clone upstream mujoco at the pin, git apply the patch
```

Consider committing that patch file into the new repo (e.g.
`studio/fork/PIN` + `studio/fork/fork.patch`) so the fork is reproducible from
upstream + the new repo alone. Future MuJoCo bumps re-derive it via the ritual
(`docs/mujoco_bump.md`).

### The one cross-piece coupling

The fork's mount block consumes a single CMake cache var:
`PROTOSPEC_ROOT` → must point at the imported ProtoSpec tree root (it mounts
`${PROTOSPEC_ROOT}/studio/glue`, which locates `studio/editor/` and
`protospec/lib/` relative to itself). That variable is the ENTIRE contract
between the two pieces.

Build configure (from `docs/studio_build.md`):
`-DMUJOCO_USE_FILAMENT=ON -DMUJOCO_BUILD_STUDIO=ON -DMUJOCO_STUDIO_PROTOSPEC=ON
-DPROTOSPEC_ROOT=<protospec-tree>`; build targets `mujoco_studio` **plus**
`elasticity sdf_plugin sensor actuator` (plugin `.so`s are not attached to the
app target).

### Environment variables the gates/tools understand (all optional)

`PROTOSPEC_MUJOCO_SRC` (MuJoCo source for snapshot extractors),
`PROTOSPEC_CORPUS` (big-corpus differential), `PROTOSPEC_BUILD_PS_LIB` /
`PROTOSPEC_MJ_INCLUDE` (harness lib/header override), `PROTOSPEC_PLUGIN_DIR`
(first-party plugin `.so` dir), `PROTOSPEC_PYD` (prebuilt python extension),
`PROTOSPEC_STUDIO_EXE` / `PROTOSPEC_STUDIO_SRC` (smoke/attic tests),
`PROTOSPEC_PATHDIFF_SKIP`, `PROTOSPEC_RUN_ASAN`, `PROTOSPEC_NATIVE`.
Runtime: `MUJOCO_SCREENSHOT_DIR/_AFTER/_COUNT/_EXIT` (headless capture).

### Post-import acceptance (15 minutes)

1. `uv sync` → `uv run python -m protospec_gen.emit --check` (clean) →
   `uv run pytest -q` (313+ passed; build_ps-linked tests skip until step 3).
2. Clone upstream MuJoCo at the pin, apply the fork patch.
3. Configure + build `build_ps` (flags above) → rerun `uv run pytest -q`
   (differential + windowless suites now run).
4. Headless smoke with a jointed model (`--gfx=classic` + the screenshot env
   vars) → viewport renders, Diagnostics shows `compiled ok path=mjs`.

## 8. Doc index

- `docs/plan.md` — the canonical design record (DRs).
- `docs/public_api.md` — the SDK contract, stability tiers, Python surface, semver.
- `docs/refs_design.md` — typed references.
- `docs/studio_build.md` — build the fork + editor, platform caveats.
- `docs/mujoco_bump.md` + `docs/SYNC_STATE.md` — the update ritual + current pin.
- `docs/EXCEPTIONS.md` — named deviations from stock + issue drafts.
- `docs/studio_plugin_min_ask.md` / `studio_plugin_feasibility.md` — plugin analysis.
- `docs/upstream_asks.md` — the MuJoCo-team ask document.
- `docs/aplus_plan.md` — the quality program that produced the current state.
- `docs/editor_certification.md` — the manual editor checklist (live-window pass).
- `attic/` — the parked native compiler and the pre-plugin standalone host (not built).
