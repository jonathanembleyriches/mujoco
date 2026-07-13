// ProtoSpec Studio editor cluster: shared state (ps::studio, ours).
//
// The ProtoSpec model authority (DR-S1) lives here, NOT in the host App. One
// EditorContext is shared (via each plugin's `data` pointer) across the editor
// plugin cluster: the ModelSource plugin that owns the load/compile pipeline,
// the Gui panels that display the tree/diagnostics, and the Viewport pick
// logger. The host only ever sees the compiled mjModel*/mjData*.

#ifndef PS_STUDIO_EDITOR_EDITOR_CONTEXT_H_
#define PS_STUDIO_EDITOR_EDITOR_CONTEXT_H_

#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <string_view>

#include "compile.h"  // ps::mjcf::bridge::Compiled
#include "editor/undo.h"
#include "types.h"    // ps::Model

namespace ps::studio {

// A type-erased reference to a tree element: the mutable element pointer plus its
// runtime type and creation serial. This is the FROZEN return shape of
// FindBySerial (editor_ops.h) that the Details panel (SE1b) codes against.
//
// Recovering typed access: `reflect::Describe(type)` gives the field table and
// the void*-based present/clear hooks; a `Visit` dispatch on `type` (a switch to
// the concrete `Visit(*static_cast<T*>(ptr), v)`) yields per-field value access.
// The pointer is valid until the next structural edit / recompile, which rebuilds
// the tree; re-resolve from `serial` after any RequestRecompile.
struct ElementRef {
  void* ptr = nullptr;
  ps::mjcf::ElementType type = ps::mjcf::ElementType::Model;
  std::uint64_t serial = 0;

  explicit operator bool() const { return ptr != nullptr; }
};

// The deferred model-load slot (Studio's pending-load pattern): a requested path
// is stashed and consumed on a later frame. Pure state machine so it can be
// unit-tested without a window.
class PendingLoad {
 public:
  void Request(std::string path) {
    path_ = std::move(path);
    pending_ = true;
  }
  bool pending() const { return pending_; }

  // Consumes the pending request, returning the path. pending() is false after.
  std::string Take() {
    pending_ = false;
    std::string out;
    out.swap(path_);
    return out;
  }

 private:
  std::string path_;
  bool pending_ = false;
};

struct EditorContext {
  // The authored ProtoSpec tree (the single source of truth) and its last good
  // compiled artifact (owns mjModel + Binding + report).
  std::unique_ptr<ps::mjcf::Model> tree;
  ps::mjcf::bridge::Compiled compiled;

  // Deferred load slot, fed by the host (CLI arg / drag-drop).
  PendingLoad pending;
  std::string source_name;   // display name of the loaded model
  std::string source_path;   // last loaded/saved file path (for Save)
  std::string base_dir;      // model dir for on-disk asset resolution on recompile
  bool model_ready = false;  // a compiled model is available for the host

  // Diagnostics panel log (validation tiers 1-2, compile report, pick events).
  std::deque<std::string> diagnostics;
  std::string status_line;   // compile path taken + timing + dirty state

  // Selection (creation serial; survives recompiles by construction, DR §5).
  std::uint64_t selected_serial = 0;
  std::string selected_desc;

  // A double-click focus request the viewport (SE2) will consume for framing;
  // the host ignores it for now. Non-zero == a fresh request for that serial.
  std::uint64_t focus_request_serial = 0;

  // Editor state.
  UndoStack history;
  bool dirty = false;                 // unsaved authored edits pending
  bool recompile_requested = false;   // a deferred recompile is queued (§ DR-S3)

  void Log(std::string line) {
    diagnostics.push_back(std::move(line));
    while (diagnostics.size() > 500) {
      diagnostics.pop_front();
    }
  }

  // --- Editor commit contract (SE1b codes against these exact signatures) --- //
  //
  // BeginEdit snapshots the current tree; the caller then mutates `tree` in
  // place; CommitEdit records the snapshot as an undo step and schedules a
  // recompile; CancelEdit reverts the in-progress mutation from the snapshot.
  // RequestRecompile coalesces to at most one compile on the next frame.

  void BeginEdit() {
    if (tree) {
      history.BeginEdit(*tree);
    }
  }

  void CommitEdit(std::string_view label) {
    history.CommitEdit(std::string(label));
    dirty = true;
    RequestRecompile();
  }

  void CancelEdit() {
    if (std::unique_ptr<ps::mjcf::Model> snapshot = history.TakePending()) {
      if (tree) {
        tree = std::move(snapshot);
        RequestRecompile();
      }
    }
  }

  void RequestRecompile() { recompile_requested = true; }
};

}  // namespace ps::studio

#endif  // PS_STUDIO_EDITOR_EDITOR_CONTEXT_H_
