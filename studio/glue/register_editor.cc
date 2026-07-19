// ProtoSpec Studio editor registration (host glue).
//
// The host fork's main.cc is kept 100% stock upstream: it calls
// mujoco::studio::LaunchStudio, which has no plugin-registration hook. Instead
// the editor registers itself the launcher-idiomatic way -- the same
// mjPLUGIN_LIB_INIT startup-constructor mechanism upstream uses for its own
// bundled plugins (object_launcher_plugin). The constructor runs before
// main(), so the plugins are in the global registry before LaunchStudio builds
// the App and enters its loop. The host fork compiles this TU directly into
// the mujoco_studio executable (not the static editor library), so the linker
// cannot drop the constructor.
//
// The single shared EditorContext must outlive the App loop; a function-local
// static gives it static storage duration without a global-init-order hazard
// (it is constructed on first entry to the constructor, before any plugin
// callback can fire).

#include <mujoco/mjplugin.h>

#include "editor/plugins.h"

mjPLUGIN_LIB_INIT(protospec_editor) {
  static ps::studio::EditorContext ctx;
  ps::studio::RegisterEditorPlugins(ctx);
}
