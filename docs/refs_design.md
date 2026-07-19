# Reference typing: audit + design

Audit date: 2026-07-17 (schema at `main`). Companion to DR-8 (typed references)
in `docs/plan.md`; this record extends DR-8 from "how refs are stored" to "which
fields are refs, and where each concern lives."

## Principle

**The schema declares semantics; storage is always the MJCF wire form; the SDK
owns behavior; validate owns enforcement.**

A `ref<T>` costs nothing at the wire level -- `ps::Ref<T>` is a name string with
a phantom target type, and the tree never holds pointers. So "typed vs string"
is not a storage question; it is a question of where the knowledge "this field
names a Body" lives:

- in the **schema**, generators derive everything downstream mechanically:
  reflect metadata (Details ref combos), `refs.h` referrer scan (rename /
  delete fixups), validate's per-target referential checks;
- in **SDK code or side tables**, every consumer needs a hand-maintained list,
  and the lists drift. The inconsistency this audit found (`Subtreecom.body :
  string` vs `Adhesion.body : ref<Body>`) is exactly that drift: ref targets
  were decided by a hand-curated `REFS` dict in the bootstrap generator, not by
  the schema.

So the answer to "strings in schema, resolve in the SDK?" is **no**: declare in
the schema, store as strings (already true), resolve in the SDK. The
declaration is what lets the SDK resolve *generically* instead of by table.

## Audit

157 `ref<T>` fields across 14 target types (Site 54, Joint 31, Body 20,
TendonAny 18, Geom 12, Material 6, ...). The remainder classifies as:

### B. Plain misses -- should be `ref<T>`, are `string` (12 fields)

| Field | Fix |
| --- | --- |
| `CompositeSkin/CompositeGeom/CompositeSite/Flexcomp/Flex .material` | `ref<Material>` |
| `Actuatorpos/Actuatorvel/Actuatorfrc .actuator` | `ref<ActuatorAny>` |
| `Subtreecom/Subtreelinvel/Subtreeangmom/SkinBone .body` | `ref<Body>` |

No wire change; no generator change; identical writer output. Verified by the
emit-parity suite.

### C. Ref lists (2 fields): `Flex.body`, `Flex.node`

Space-separated body-name lists on the wire (`mjs_setStringVec` upstream). The
IDL has no `ref<T>[]` -- the grammar's ref branch takes no arity. Typing these
means a small generator feature: grammar, C++ type (`std::vector<ps::Ref<Body>>`),
XML join/split, reflect arity, python binding, and a list case in `refs.h`'s
`RefScan` (today it visits `opt<Ref<T>>` only) so rename/delete fixups cover
list entries. Deliberately a follow-up: a generator feature deserves its own
verified pass.

### D. Dynamic-typed pairs (13 `objtype`+`objname`, 8 `reftype`+`refname`)

`TupleElement`, `Insidesite`, `SensorUser`, `SensorPlugin`, and the nine frame
sensors (`Framepos` ... `Frameangacc`; seven of them -- not `Framelinacc`/
`Frameangacc` -- also carry `reftype`/`refname` for the reference frame, and
SensorPlugin carries both pairs). The target type is *runtime data in a sibling field*; a
phantom-typed `Ref<T>` structurally cannot express it. Options:

1. keep strings, SDK helper resolves via the sibling (status quo, knowledge in
   code);
2. an IDL **annotation** `(target_from=objtype)` on the name field -- no storage
   or type change, only reflect metadata, letting `refs.h` / Details / validate
   handle the pair generically.

Decision: option 2 is the clean endgame; defer until a feature consumes it
(the layer dependency graph over sensors is the first candidate).

### E. Cross-model (1 field): `Attach.body`

Names a body **inside the referenced `ModelAsset`'s file**, not this tree.
`ref<Body>` would resolve against the local model and be wrong. Stays `string`
with this documented scope; attach-aware SDK code resolves it inside the
asset's parsed model.

### F. Correctly strings (not references)

`Model.model` (the model's own name), `prefix` on Attach/Composite/Replicate
(name *transformers*), file paths (a file reference is a different concept from
an element reference), key/user text, and the four `plugin` fields
(PluginDef/PluginRef/ActuatorPlugin/SensorPlugin) -- reference-shaped, but they
name entries in the process-global plugin registry, not model elements.

## Sequencing

1. **DONE**: the 12 category-B fixes, schema-edited directly + `REFS` kept in
   sync.
2. **DONE**: `ref<T>[]` (category C) -- IDL grammar, reader/writer, RefScan /
   ClearRefs / RefPrefixer / native collector / validate / Details panel;
   Flex.body and Flex.node typed. Ref<ActuatorAny> union expansion landed with
   it (RefTargetTypes + RefNs).
3. **When consumed**: `(target_from=...)` annotation (category D). First
   candidate consumer: the layer dependency graph over sensors.
4. **Never**: typing `Attach.body` as a local ref (category E).

## Housekeeping

`tools/bootstrap/draft_schema.py` is a **bootstrap drafter, not a regenerator**:
running it over the current snapshots discards hand-refinements the committed
schema carries (the `BodyChildAny` union, resolvers, aliases). The README's
"Regeneration" section listing it as `--check`-reproducible is wrong and should
be corrected. `protospec/schema/mujoco.spec` is the source of truth; edit it directly and
mirror ref-target knowledge into `REFS` so future drafts start closer.
