// ProtoSpec Studio compatibility header. The editor sources were written
// against a standalone host whose plugin API mirrors the MuJoCo Studio
// platform plugin API in the ps::studio namespace; this header maps those
// names onto the real platform registry so the sources compile unchanged.

#ifndef MUJOCO_SRC_EXPERIMENTAL_STUDIO_PROTOSPEC_PLATFORM_UX_PLUGIN_H_
#define MUJOCO_SRC_EXPERIMENTAL_STUDIO_PROTOSPEC_PLATFORM_UX_PLUGIN_H_

#include "experimental/platform/ux/plugin.h"

namespace ps::studio {

using mujoco::platform::ForEachPlugin;
using mujoco::platform::GuiPlugin;
using mujoco::platform::KeyHandlerPlugin;
using mujoco::platform::ModelPlugin;
using mujoco::platform::RegisterPlugin;
using mujoco::platform::SpecEditorPlugin;

}  // namespace ps::studio

#endif  // MUJOCO_SRC_EXPERIMENTAL_STUDIO_PROTOSPEC_PLATFORM_UX_PLUGIN_H_
