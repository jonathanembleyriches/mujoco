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

    // A pending file load takes priority (fresh tree + compile).
    if (c->pending.pending()) {
      const std::string path = c->pending.Take();
      if (!path.empty() && LoadModel(*c, path)) {
        c->recompile_requested = false;  // the load already produced an artifact
        out->model = c->compiled.model.get();  // host borrows; plugin retains
        return true;
      }
      return false;  // errors already logged to Diagnostics
    }

    // Otherwise service a debounced recompile request (edits coalesce to at most
    // one compile per frame, DR-S3). A failed recompile keeps the last good
    // artifact running and does not re-adopt. In Play mode, recompiles are
    // deferred (edits take effect on the next compile) unless `apply_edits` is
    // set -- the one-shot the ▶ "compile if dirty then run" path raises.
    if (ShouldServiceRecompile(*c)) {
      c->recompile_requested = false;
      c->apply_edits = false;
      if (RecompileTree(*c)) {
        out->model = c->compiled.model.get();
        return true;
      }
    }
    return false;
  };

  RegisterPlugin<ModelSourcePlugin>(plugin);
}

}  // namespace ps::studio
