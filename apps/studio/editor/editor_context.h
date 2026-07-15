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
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "compile.h"  // ps::mjcf::bridge::Compiled, ps::mjcf::bridge::VfsAsset
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

// Unity-style mode machine (DR-S2). Edit is the default: physics paused, mjData
// at qpos0, gizmos live. Play runs the simulation; Stop discards sim state and
// returns to Edit.
enum class EditorMode { Edit, Play };

// A structured Diagnostics-panel entry. Beyond the rendered message it carries
// what a producer knew when it logged: the acting element's creation `serial`
// (so a click can SelectBySerial back to it) and/or a `loc` (so the row can show
// file:line). Both are optional -- a plain log line (`Log(std::string)`) is an
// Info entry with neither, so legacy call sites keep working unchanged.
struct DiagEntry {
  enum class Severity { Info, Warning, Error };

  Severity severity = Severity::Info;
  std::string message;
  std::optional<std::uint64_t> serial;  // click -> SelectBySerial(*serial)
  std::optional<ps::SourceLoc> loc;     // file:line origin when known
};

// The active transform tool (Q/W/E/R). Select shows no gizmo.
enum class GizmoTool { Select, Translate, Rotate, Scale };

// Gizmo interaction settings surfaced on the toolbar (§ deliverable 2).
struct GizmoSettings {
  GizmoTool tool = GizmoTool::Select;
  bool world_space = true;   // Local/World toggle: true == world axes
  bool snap = false;
  double snap_translate = 0.1;    // metres
  double snap_rotate_deg = 15.0;  // degrees
  double snap_scale = 0.05;       // scale-factor increment
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

  // In-memory assets (mesh/texture bytes) for models with no file on disk yet:
  // injected into every compile via CompileOptions.vfs_assets, and externalized
  // to disk on Save (asset_import.*). Cleared once written out.
  std::vector<ps::mjcf::bridge::VfsAsset> vfs_assets;

  bool model_ready = false;  // a compiled model is available for the host
  bool fresh_load = false;   // one-shot: the last adopt was a file load, not a
                            // drag recompile (host resets the free camera)

  // Diagnostics panel log (validation tiers 1-2, compile report, pick events).
  std::deque<DiagEntry> diagnostics;
  std::string status_line;   // compile path taken + timing + dirty state

  // Selection (creation serial; survives recompiles by construction, DR §5).
  std::uint64_t selected_serial = 0;
  std::string selected_desc;

  // A double-click focus request the viewport consumes for framing (F). Non-zero
  // == a fresh request for that serial.
  std::uint64_t focus_request_serial = 0;

  // A delete request from the viewport (Del key). The Hierarchy panel services it
  // through the SE1a referrer-confirm flow (preview -> cascade/cancel modal).
  std::uint64_t delete_request_serial = 0;

  // Editor state.
  UndoStack history;
  bool dirty = false;                 // unsaved authored edits pending
  bool recompile_requested = false;   // an Edit-mode debounced recompile is queued
  bool apply_edits = false;           // one-shot: compile now regardless of mode
                                      // (the ▶ "compile if dirty then run" path)

  // Mode machine + gizmo interaction settings (SE2). The host toolbar drives the
  // mode; the viewport/gizmo plugins read it (gizmos render only in Edit mode).
  EditorMode mode = EditorMode::Edit;
  bool play_paused = false;           // ⏸ within Play mode
  GizmoSettings gizmo;
  bool gizmo_active = false;           // a gizmo drag is in progress (host reads
                                      // this so Esc cancels the drag, not exit)
  bool show_all_joints = false;        // View toggle: draw joints for every body,
                                      // not just the selected body (deliverable 3)
  std::string transient_status;       // gizmo notes (euler->quat, mesh-scale warn)

  // Ring cap: the Diagnostics panel is a bounded scrollback, never a leak.
  static constexpr std::size_t kMaxDiagnostics = 500;

  // Legacy plain-string append: an Info entry with no serial / loc.
  void Log(std::string line) {
    Diagnose(DiagEntry{DiagEntry::Severity::Info, std::move(line), {}, {}});
  }

  // Structured append: producers wire the severity / serial / loc they know.
  void Diagnose(DiagEntry entry) {
    diagnostics.push_back(std::move(entry));
    while (diagnostics.size() > kMaxDiagnostics) {
      diagnostics.pop_front();
    }
  }

  void ClearDiagnostics() { diagnostics.clear(); }

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
