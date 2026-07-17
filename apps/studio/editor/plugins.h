// ProtoSpec Studio editor plugin cluster registration (ps::studio, ours).
//
// Registers the ProtoSpec-native editor as a set of Studio plugins sharing one
// EditorContext: a ModelSource plugin (load/compile pipeline), the Gui panels
// (Hierarchy / Details / Assets / Diagnostics), and a Viewport pick logger.
// Called once by main() after the host is constructed.

#ifndef PS_STUDIO_EDITOR_PLUGINS_H_
#define PS_STUDIO_EDITOR_PLUGINS_H_

#include "editor/editor_context.h"

namespace ps::studio {

// The ModelSource plugin that owns the ProtoSpec load/compile pipeline.
void RegisterProtoSpecEditorPlugin(EditorContext& ctx);

// The authored-tree Hierarchy panel (hierarchy_panel.cc).
void RegisterHierarchyPanel(EditorContext& ctx);

// The Assets / Diagnostics / File panels + the undo/redo/save key handlers
// (panels.cc).
void RegisterEditorPanels(EditorContext& ctx);

// The Viewport editor plugin cluster (SE2): pick/select, the transform gizmo
// (mouse + draw hooks), the selection outline, and the QWER/Del key handlers.
void RegisterViewportEditor(EditorContext& ctx);

// Model-level creation menu items (actuator/sensor spellings; tendon, equality,
// contact pair/exclude, keyframe). Defined in panels.cc, shared with the
// Hierarchy's per-section context menus so an EMPTY family can be bootstrapped
// from its section row. `target` wires an actuator/sensor to the selection when
// compatible (0 = none). Call inside an open ImGui menu/popup.
void DrawAddActuatorItems(EditorContext& ctx, std::uint64_t target);
void DrawAddSensorItems(EditorContext& ctx, std::uint64_t target);

// The generated Details panel, owned by the concurrent SE1b cluster
// (details_panel.cc, namespace ps::studio::details).
namespace details {
void RegisterDetailsPanel(EditorContext& ctx);
}  // namespace details

// Registers the whole cluster.
inline void RegisterEditorPlugins(EditorContext& ctx) {
  RegisterProtoSpecEditorPlugin(ctx);
  RegisterHierarchyPanel(ctx);
  details::RegisterDetailsPanel(ctx);
  RegisterEditorPanels(ctx);
  RegisterViewportEditor(ctx);
}

}  // namespace ps::studio

#endif  // PS_STUDIO_EDITOR_PLUGINS_H_
