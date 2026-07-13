// ProtoSpec Studio undo/redo (ps::studio, ours): snapshot history over the
// authored ps::Model.
//
// MECHANISM (the one subtle correctness point of the editor core).
//
// The object model stamps every element with a process-unique creation serial
// (core.h, DR-10/11). Selection identity, the compile Binding, and the compile
// bridge's state migration all key on that serial: "surviving elements must keep
// their serials" (bridge/compile.h Recompile contract). The generated deep
// Clone(), however, MINTS FRESH serials -- `std::make_unique<T>()` runs the
// `serial = next_serial()` initializer and Clone copies every field EXCEPT the
// serial (generated/types.cc). A naive snapshot-clone would therefore return a
// tree whose elements no longer match `selected_serial`, and whose unnamed
// elements would auto-bind under different `_ps:<family>:<serial>` names after a
// recompile -- selection and unnamed-element state would both break on undo.
//
// We solve it at the source: `CloneWithSerials` performs the generated deep
// clone and then, in one lockstep dual walk of source and clone (identical
// document order by construction -- Clone preserves structure), copies each
// source element's serial onto its clone. A snapshot is thus serial-IDENTICAL to
// the tree it was taken from. Because a snapshot only ever REPLACES the live tree
// wholesale (never coexists with it), duplicating serials across the two is safe
// -- the concern Clone's fresh-serial policy guards against (splicing a clone
// back into the same live model) never arises here. With serial-faithful
// snapshots, undo/redo is a plain owned-tree shuttle between two deques and the
// live slot; selection re-resolves by serial with no path heuristics, and the
// recompiled Binding reproduces the exact prior name table.

#ifndef PS_STUDIO_EDITOR_UNDO_H_
#define PS_STUDIO_EDITOR_UNDO_H_

#include <cstddef>
#include <deque>
#include <memory>
#include <optional>
#include <string>

#include "types.h"  // ps::mjcf::Model, ps::Clone

namespace ps::studio {

// A deep clone of `src` whose every element carries the SAME serial as its
// source counterpart (see the mechanism note above). The returned tree is
// structurally identical and safe to install as the live model.
std::unique_ptr<ps::mjcf::Model> CloneWithSerials(const ps::mjcf::Model& src);

// Bounded snapshot history. Owns the pre-edit snapshot captured by BeginEdit and
// the undo/redo deques. Trees are shuttled by ownership between the deques and
// the caller's live slot; only BeginEdit clones (once per edit).
class UndoStack {
 public:
  static constexpr std::size_t kMaxDepth = 128;

  // Capture the pre-edit state. Overwrites any uncommitted pending snapshot.
  void BeginEdit(const ps::mjcf::Model& live);

  // Finalize the edit: the pending pre-edit snapshot becomes the newest undo
  // entry and the redo history is dropped. No-op without a pending snapshot.
  void CommitEdit(std::string label);

  // Abandon the pending snapshot without recording it. Returns the snapshot so
  // the caller can restore the live tree from it (revert), or null if none.
  std::unique_ptr<ps::mjcf::Model> TakePending();

  bool has_pending() const { return pending_.has_value(); }
  bool can_undo() const { return !undo_.empty(); }
  bool can_redo() const { return !redo_.empty(); }
  std::size_t undo_depth() const { return undo_.size(); }
  std::size_t redo_depth() const { return redo_.size(); }

  // Undo one step: stash `current` (the live tree) on the redo deque and return
  // the restored pre-edit tree. Returns `current` unchanged when there is
  // nothing to undo. `out_label` (optional) receives the edit's label.
  std::unique_ptr<ps::mjcf::Model> Undo(std::unique_ptr<ps::mjcf::Model> current,
                                        std::string* out_label = nullptr);

  // Redo one step: the inverse shuttle.
  std::unique_ptr<ps::mjcf::Model> Redo(std::unique_ptr<ps::mjcf::Model> current,
                                        std::string* out_label = nullptr);

  void Clear();

 private:
  struct Entry {
    std::unique_ptr<ps::mjcf::Model> model;
    std::string label;
  };

  std::optional<Entry> pending_;
  std::deque<Entry> undo_;
  std::deque<Entry> redo_;
};

}  // namespace ps::studio

#endif  // PS_STUDIO_EDITOR_UNDO_H_
