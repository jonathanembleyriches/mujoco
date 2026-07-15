// ProtoSpec Studio compatibility header. The extension plugin types the
// editor was developed against (viewport input, viewport overlays, scene
// overlays, and the compiled-model source) are now part of the platform
// plugin API; this header maps the ps::studio names onto them so the editor
// sources compile unchanged.

#ifndef MUJOCO_SRC_EXPERIMENTAL_STUDIO_PROTOSPEC_PLATFORM_UX_PS_PLUGIN_EXT_H_
#define MUJOCO_SRC_EXPERIMENTAL_STUDIO_PROTOSPEC_PLATFORM_UX_PS_PLUGIN_EXT_H_

#include "experimental/platform/ux/plugin.h"
#include "platform/ux/plugin.h"

namespace ps::studio {

using mujoco::platform::CompiledModel;
using mujoco::platform::EditorShellPlugin;
using mujoco::platform::FileDialogPlugin;
using mujoco::platform::MainMenuPlugin;
using mujoco::platform::ModelSourcePlugin;
using mujoco::platform::OverlayPlugin;
using mujoco::platform::ToolbarPlugin;
using mujoco::platform::ViewportGuiPlugin;
using mujoco::platform::ViewportInput;
using mujoco::platform::ViewportPlugin;

}  // namespace ps::studio

#endif  // MUJOCO_SRC_EXPERIMENTAL_STUDIO_PROTOSPEC_PLATFORM_UX_PS_PLUGIN_EXT_H_
