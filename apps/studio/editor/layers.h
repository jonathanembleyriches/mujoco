// ProtoSpec Studio: layers as provenance tags (ps::studio, ours).
//
// There is ONE authored tree (ctx.tree). A layer is not a container: it is a
// distinct `loc.file` value (the per-element provenance the reader stamps,
// through <include> expansion -- DR-9) plus {display name, enabled} flags. The
// consequences, in order:
//
//   * Split on load: after a parse, the distinct loc.file values partition the
//     tree, one Layer per file in first-appearance order. A single-file model
//     is the degenerate one-layer case with zero special-casing. A file
//     spliced INSIDE another file's <body> (the nested-include case) is fully
//     represented: the spliced elements carry their own file while sitting in
//     the foreign subtree.
//   * Active layer = edit scope. Everything stays visible/selectable, but the
//     editor may only author elements whose loc.file equals the active key.
//     New elements are stamped with the active key at the compile seam
//     (BuildCompileModel), which runs right after every CommitEdit -- the one
//     reliable choke point every authoring path already funnels through.
//   * Enabled/disabled = pruned from the compile input only. All layers
//     enabled compiles ctx.tree directly (no clone); otherwise a
//     serial-preserving clone with the disabled layers' elements removed
//     (removing a body removes its subtree -- correct and expected). Dangling
//     references caused by disabling a depended-on layer surface as honest
//     compile/validate errors.
//   * Export via <include>: one file per enabled layer plus a root document of
//     <include> lines. Elements nested inside ANOTHER layer's element export
//     as an <include> at that point inside the owning layer's file, with the
//     fragment (root <mujocoinclude>) in the nested layer's file.

#ifndef PS_STUDIO_EDITOR_LAYERS_H_
#define PS_STUDIO_EDITOR_LAYERS_H_

#include <cstdint>
#include <memory>
#include <string>

#include "editor/editor_context.h"
#include "types.h"

namespace ps::studio {

// --- Partition / lookup ---------------------------------------------------- //

// Rebuild ctx.layers by partitioning ctx.tree on distinct loc.file values, in
// first-appearance (document) order. Elements with an empty loc.file are
// stamped `root_key` first (the loaded file's path, or a "layer://" key for an
// in-editor model), so every element belongs to a layer. The active layer
// becomes the root_key's layer when present, else the first. All enabled.
void SplitLayersFromTree(EditorContext& ctx, const std::string& root_key,
                         const std::string& root_name);

// Append Layer rows for any loc.file key present in the tree but missing from
// ctx.layers (first-appearance order). Never removes rows (an empty authored
// layer is legitimate). Called at the compile seam and after grafts.
void ReconcileLayers(EditorContext& ctx);

// The layer index whose key equals `key`, or -1.
int LayerIndexForKey(const EditorContext& ctx, const std::string& key);

// The layer index owning the element with `serial` (via its loc.file), or -1
// when the serial does not resolve or its key matches no layer.
int LayerOfSerial(EditorContext& ctx, std::uint64_t serial);

// The edit-scope gate: true when the element's layer is the active layer.
// Fails OPEN (true) when layers are absent, the serial does not resolve, or
// the element's key is unknown -- the gate exists to scope authoring, not to
// brick the editor on a bookkeeping gap.
bool SerialInActiveLayer(EditorContext& ctx, std::uint64_t serial);

// Number of elements tagged with layer `index`'s key.
int CountLayerElements(EditorContext& ctx, int index);

// --- Compile seam ---------------------------------------------------------- //

// The model the next compile should run on. Also the housekeeping seam: stamps
// empty-loc.file elements with the active key, reconciles layer rows, and
// recomputes ctx.layer_graph (so the graph refreshes per recompile, not per
// frame). Every layer enabled: returns ctx.tree (no clone) and leaves
// *out_pruned null. Otherwise: *out_pruned receives a serial-preserving clone
// with every disabled layer's elements removed, and the return points at it.
// The caller adopts *out_pruned into ctx.compile_tree only on compile success,
// so a failed compile keeps the last good artifact's source model alive.
ps::mjcf::Model* BuildCompileModel(EditorContext& ctx,
                                   std::unique_ptr<ps::mjcf::Model>* out_pruned);

// Recompute ctx.layer_graph from the tree's cross-layer references (typed
// scalar refs, ref<T>[] lists, and dynamic target_from refs, via
// ps::sdk::detail::ScanRefs).
void RecomputeLayerGraph(EditorContext& ctx);

// The enable-toggle lock for layer `index`: locked while any ENABLED layer
// depends on it. `dependents` names them; `example` is one concrete reference.
struct LayerLock {
  bool locked = false;
  std::string dependents;
  std::string example;
};
LayerLock LayerLockInfo(const EditorContext& ctx, int index);

// --- Layer list ops -------------------------------------------------------- //

// Append an empty authored layer (key "layer://<slug>") and make it active.
void AddEmptyLayer(EditorContext& ctx, const std::string& name);

// Parse `path` and graft its top-level content into ctx.tree (document order
// appended per section). The grafted elements keep their parse provenance, so
// they form the new layer (plus one layer per nested include the file pulls
// in). One undo step. The file's own layer becomes active. Returns false with
// `error` set on a parse failure (tree untouched).
bool AddLayerFromFile(EditorContext& ctx, const std::string& path,
                      std::string* error);

// Remove layer row `index`. When `delete_elements`, every element tagged with
// its key is deleted from the tree (one undo step); otherwise the row must be
// empty (no tagged elements) or the call is a no-op returning false.
bool RemoveLayer(EditorContext& ctx, int index, bool delete_elements);

// Reorder the display list (order carries no composition semantics -- there is
// one tree -- but it is the export/include order).
void MoveLayer(EditorContext& ctx, int index, int delta);

// --- Export ---------------------------------------------------------------- //

// Write the enabled layers as an <include> stack rooted at `root_path`:
//   <root>.xml            <mujoco model="..."> <include file="<layer>.xml"/> ...
//   <layer-name>.xml      <mujocoinclude> ...that layer's elements... </>
// A layer's elements nested inside another layer's element are exported as an
// <include> at that point inside the owning layer's file, with the fragment in
// the nested layer's file -- so exporting and reloading reproduces both the
// composed model and the layer partition. Disabled layers are pruned (not
// written). Limitation (documented, detected, not silent): a layer whose
// elements form MULTIPLE disjoint nested regions needs one fragment file per
// region ("<name>.2.xml", ...), which a reload necessarily partitions as
// separate layers. Returns false with `error` set; nothing is written then.
bool ExportLayeredMjcf(EditorContext& ctx, const std::string& root_path,
                       std::string* error);

}  // namespace ps::studio

#endif  // PS_STUDIO_EDITOR_LAYERS_H_
