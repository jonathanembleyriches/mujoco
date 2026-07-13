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

// The Gui panel plugins (Hierarchy / Details / Assets / Diagnostics).
void RegisterEditorPanels(EditorContext& ctx);

// The Viewport plugin that ray-picks and logs the resolved element.
void RegisterPickLogger(EditorContext& ctx);

// Registers the whole cluster.
inline void RegisterEditorPlugins(EditorContext& ctx) {
  RegisterProtoSpecEditorPlugin(ctx);
  RegisterEditorPanels(ctx);
  RegisterPickLogger(ctx);
}

}  // namespace ps::studio

#endif  // PS_STUDIO_EDITOR_PLUGINS_H_
