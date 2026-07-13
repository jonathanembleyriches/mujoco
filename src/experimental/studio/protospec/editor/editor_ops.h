// ProtoSpec Studio editor operations (ps::studio, ours): the pure, windowless
// core of the editor cluster. No SDL/ImGui/plugin-registry dependencies, so the
// ctest target links this directly and drives the ProtoSpec loop end to end.

#ifndef PS_STUDIO_EDITOR_EDITOR_OPS_H_
#define PS_STUDIO_EDITOR_EDITOR_OPS_H_

#include <cstdint>
#include <string>
#include <vector>

#include "editor/editor_context.h"

namespace ps::studio {

// Loads a model through ProtoSpec (DR-S1 load path):
//   ParseMjcf(file) -> Validate(tiers 1-2, logged) -> Compile(Auto).
// On success, populates ctx.tree + ctx.compiled, sets ctx.model_ready, records
// the source path/base_dir, clears undo history and dirty state, and logs the
// validation/compile summary. On failure, logs the errors, leaves any prior good
// compiled artifact untouched, and returns false.
bool LoadModel(EditorContext& ctx, const std::string& path);

// Recompiles the current authored tree in place (the deferred-recompile target of
// RequestRecompile). Swaps ctx.compiled on success; on failure leaves the last
// good artifact running and logs the errors. Returns true when a fresh artifact
// is ready for the host to adopt.
bool RecompileTree(EditorContext& ctx);

// The mode-machine gate on the debounced recompile: Edit mode services queued
// recompiles (the drag preview); Play defers them so mid-play edits take effect
// only on the next compile (DR-S2), unless the one-shot `apply_edits` is set (the
// ▶ "compile if dirty then run" transition). The ModelSource poll calls this.
bool ShouldServiceRecompile(const EditorContext& ctx);

// Serializes the authored tree to `path` via WriteMjcf (authored forms / classes
// / macros preserved). Clears the dirty flag and updates the source path on
// success. Returns false and logs on I/O failure.
bool SaveModel(EditorContext& ctx, const std::string& path);

// The element a pick resolved to, via Binding reverse lookup.
struct PickResolution {
  bool hit = false;
  std::string type;         // element type name ("Geom", "Body", ...)
  std::string name;         // authored name, or "<unnamed>"
  std::uint64_t serial = 0;
};

// Resolves a compiled geom/body id to a ProtoSpec element through ctx.compiled's
// Binding (geom preferred, else body), updates ctx.selected_*, and logs the hit
// to the Diagnostics panel. Pass -1 for a missing id. Proves the ProtoSpec loop
// (compiled id -> Binding -> tree element) end to end.
PickResolution ResolvePick(EditorContext& ctx, int geom_id, int body_id);

// --- Serial-addressed element access -------------------------------------- //

// Any element (any section, defaults included) by creation serial, as a
// type-erased ElementRef, or a null ref when no element carries that serial.
ElementRef FindBySerial(EditorContext& ctx, std::uint64_t serial);

// Set the current selection to `serial`, refreshing selected_desc. Returns false
// (and clears the selection) when the serial no longer resolves -- the hook every
// selection source (Hierarchy click, pick logger, post-undo re-resolution) funnels
// through so the two-way sync stays consistent.
bool SelectBySerial(EditorContext& ctx, std::uint64_t serial);

// Rename the element with `serial` and rewrite every typed referrer
// (sdk::Rename). Returns the number of referrer fields updated, or -1 when the
// serial does not resolve. Caller wraps this in BeginEdit/CommitEdit.
int RenameBySerial(EditorContext& ctx, std::uint64_t serial,
                   const std::string& new_name);

// Outcome of a delete, with dangling referrers rendered for the confirm modal.
struct DeleteResult {
  bool found = false;                 // the serial resolved to an element
  bool removed = false;               // the element was unlinked from the tree
  std::vector<std::string> dangling;  // "path.field -> 'name'" for each dangler
};

// Remove the element with `serial` and its subtree (sdk::DeleteRecursive). With
// `cascade`, references left dangling are cleared; otherwise they are reported in
// the result. Caller wraps this in BeginEdit/CommitEdit.
DeleteResult DeleteBySerial(EditorContext& ctx, std::uint64_t serial,
                            bool cascade);

// Referrers that deleting `serial` would leave dangling, rendered as
// "path.field -> 'name'" lines. Read-only: computed on a serial-preserving clone,
// so it drives the confirm modal (§6) without mutating the tree, recompiling, or
// touching undo history.
std::vector<std::string> PreviewDeleteReferrers(EditorContext& ctx,
                                                std::uint64_t serial);

// --- Undo / redo ---------------------------------------------------------- //

// Restore the previous / next authored state, re-resolve the selection by serial,
// mark dirty, and schedule a recompile. Returns false when the corresponding
// history deque is empty.
bool Undo(EditorContext& ctx);
bool Redo(EditorContext& ctx);

}  // namespace ps::studio

#endif  // PS_STUDIO_EDITOR_EDITOR_OPS_H_
