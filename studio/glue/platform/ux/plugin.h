// ProtoSpec Studio compatibility header. The editor sources were written
// against a host whose plugin API mirrors the MuJoCo Studio platform plugin
// API in the ps::studio namespace; this header maps those names onto the real
// platform registry so the sources compile unchanged. It is the ONLY
// "platform/ux/plugin.h" on the editor's include path (the glue directory is
// searched first; the pre-plugin standalone host's copy is parked in attic/
// and not built).

#ifndef PS_STUDIO_GLUE_PLATFORM_UX_PLUGIN_H_
#define PS_STUDIO_GLUE_PLATFORM_UX_PLUGIN_H_

#include "experimental/platform/ux/plugin.h"

namespace ps::studio {

using mujoco::platform::ForEachPlugin;
using mujoco::platform::GuiPlugin;
using mujoco::platform::KeyHandlerPlugin;
using mujoco::platform::ModelPlugin;
using mujoco::platform::RegisterPlugin;
using mujoco::platform::ScenePlugin;
using mujoco::platform::SpecEditorPlugin;

}  // namespace ps::studio

#endif  // PS_STUDIO_GLUE_PLATFORM_UX_PLUGIN_H_
