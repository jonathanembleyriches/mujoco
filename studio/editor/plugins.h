// ProtoSpec Studio editor plugin cluster registration (ps::studio, ours).
//
// Registers the ProtoSpec-native editor as a set of Studio plugins sharing one
// EditorContext: a ModelSource plugin (load/compile pipeline), the Gui panels
// (Hierarchy / Details / Layers, plus the folded Assets / Diagnostics / File),
// and the Viewport editor cluster (pick/select, transform gizmo, selection
// overlay). Called once by main() after the host is constructed.

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

// Host-service plugins. Live-host only (they drive ImGui/SDL, not the model),
// so they take no EditorContext; windowless harnesses never call them.
// Curated default dock layout — pre-empts the host's stock first-run layout
// under the same "Root" dockspace id (dock_layout.cc).
void RegisterDockLayoutService();
// Env-driven composited-framebuffer self-screenshot + F12 capture, classic
// backend (screenshot_service.cc).
void RegisterScreenshotService();

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

// Registers the whole cluster. The screenshot service registers LAST so its
// foreground-draw-list capture callback is appended after the gizmo's own
// foreground draws within a frame (ModelPlugins dispatch in registration
// order).
inline void RegisterEditorPlugins(EditorContext& ctx) {
  RegisterProtoSpecEditorPlugin(ctx);
  RegisterHierarchyPanel(ctx);
  details::RegisterDetailsPanel(ctx);
  RegisterEditorPanels(ctx);
  RegisterViewportEditor(ctx);
  RegisterDockLayoutService();
  RegisterScreenshotService();
}

}  // namespace ps::studio

#endif  // PS_STUDIO_EDITOR_PLUGINS_H_
