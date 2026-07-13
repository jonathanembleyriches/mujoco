// ProtoSpec Studio: the ModelSource plugin that owns the ProtoSpec load/compile
// pipeline (ps::studio, ours). This is the DR-S1 authority as a plugin: the host
// pushes a requested path via submit_load and adopts the compiled mjModel* the
// plugin publishes via poll_compiled. The host never touches ps::Model.

#include "editor/editor_ops.h"
#include "editor/plugins.h"
#include "platform/ux/ps_plugin_ext.h"

namespace ps::studio {

static EditorContext* Ctx(void* data) {
  return static_cast<EditorContext*>(data);
}

void RegisterProtoSpecEditorPlugin(EditorContext& ctx) {
  ModelSourcePlugin plugin;
  plugin.name = "ProtoSpec Model";
  plugin.data = &ctx;

  plugin.submit_load = [](ModelSourcePlugin* self, const char* path) {
    Ctx(self->data)->pending.Request(path ? path : "");
  };

  plugin.poll_compiled = [](ModelSourcePlugin* self, CompiledModel* out) -> bool {
    EditorContext* c = Ctx(self->data);
    if (!c->pending.pending()) {
      return false;
    }
    const std::string path = c->pending.Take();
    if (path.empty()) {
      return false;
    }
    if (!LoadModel(*c, path)) {
      return false;  // errors already logged to Diagnostics
    }
    out->model = c->compiled.model.get();  // host borrows; plugin retains
    return true;
  };

  RegisterPlugin<ModelSourcePlugin>(plugin);
}

}  // namespace ps::studio
