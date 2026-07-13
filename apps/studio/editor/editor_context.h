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

#include "compile.h"  // ps::mjcf::bridge::Compiled
#include "types.h"    // ps::Model

namespace ps::studio {

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
  bool model_ready = false;  // a compiled model is available for the host

  // Diagnostics panel log (validation tiers 1-2, compile report, pick events).
  std::deque<std::string> diagnostics;
  std::string status_line;   // compile path taken + timing + dirty state

  // Selection (creation serial; survives recompiles by construction, DR §5).
  std::uint64_t selected_serial = 0;
  std::string selected_desc;

  void Log(std::string line) {
    diagnostics.push_back(std::move(line));
    while (diagnostics.size() > 500) {
      diagnostics.pop_front();
    }
  }
};

}  // namespace ps::studio

#endif  // PS_STUDIO_EDITOR_EDITOR_CONTEXT_H_
