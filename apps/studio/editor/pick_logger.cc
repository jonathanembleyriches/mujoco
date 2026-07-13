// ProtoSpec Studio: the Viewport pick-logger plugin (ps::studio, ours). On a
// left double-click it casts a pick ray, then resolves the hit compiled id back
// to a ProtoSpec element through the Binding and logs it -- the end-to-end proof
// of the ProtoSpec loop for SE0. SE2 grows this into gizmo interaction.

#include "editor/editor_ops.h"
#include "editor/plugins.h"
#include "platform/ux/interaction.h"
#include "platform/ux/ps_plugin_ext.h"

namespace ps::studio {

void RegisterPickLogger(EditorContext& ctx) {
  ViewportPlugin plugin;
  plugin.name = "ProtoSpec Pick";
  plugin.data = &ctx;

  plugin.on_mouse = [](ViewportPlugin* self, const ViewportInput& in) -> bool {
    if (!in.left_double || in.model == nullptr || in.data == nullptr) {
      return false;
    }
    EditorContext* c = static_cast<EditorContext*>(self->data);
    if (c->compiled.model.get() != in.model) {
      return false;  // pick only against the model we hold the Binding for
    }
    const PickResult pick = Pick(in.model, in.data, in.camera, in.x, in.y,
                                 in.aspect_ratio, in.vis_option);
    ResolvePick(*c, pick.geom, pick.body);
    return true;
  };

  RegisterPlugin<ViewportPlugin>(plugin);
}

}  // namespace ps::studio
