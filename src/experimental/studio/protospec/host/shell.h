// ProtoSpec Studio host glue: the editor shell (ps::studio, fork-side).
//
// The shell contributes File / Edit menus, the transform toolbar, and the
// Play/Stop mode bridge to the real MuJoCo Studio menu bar and toolbar. It binds
// to host plugin types (MainMenuPlugin / ToolbarPlugin / EditorShellPlugin /
// FileDialogPlugin) that only the Filament Studio host provides, so it lives with
// the host rather than in the shared editor cluster. The standalone apps/studio
// host has no menu bar / toolbar and does not build it.

#ifndef PS_STUDIO_HOST_SHELL_H_
#define PS_STUDIO_HOST_SHELL_H_

#include "editor/editor_context.h"

namespace ps::studio {

// Registers the SE4 editor shell against the host menu bar / toolbar plugins,
// sharing the same EditorContext as the editor cluster. Call after
// RegisterEditorPlugins.
void RegisterEditorShell(EditorContext& ctx);

}  // namespace ps::studio

#endif  // PS_STUDIO_HOST_SHELL_H_
