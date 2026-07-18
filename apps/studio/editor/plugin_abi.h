// ProtoSpec Studio: plugin C-ABI helpers + layout guards (ps::studio, ours).
//
// The editor talks to the host through POD plugin structs (function pointers +
// a `void* data` back-pointer to the owning object). Two hazards this header
// addresses:
//   * every callback recovers its object with static_cast<T*>(self->data) --
//     ctx_cast<T> is the one typed spelling of that, and
//   * the ext plugin structs are HAND-SYNCED against the host's
//     mujoco::platform:: definitions (platform/ux/plugin.h keeps them VERBATIM
//     so a plugin loads into stock Studio). The static_asserts below pin the
//     offsets the editor actually reads, so a field reorder in either copy is a
//     compile error, not a silent C-ABI mismatch. Compiled against whichever
//     copy is on the include path (ours, or the host shim in build_ps), so the
//     guard tracks the layout the editor is really built against.

#ifndef PS_STUDIO_EDITOR_PLUGIN_ABI_H_
#define PS_STUDIO_EDITOR_PLUGIN_ABI_H_

#include <cstddef>
#include <type_traits>

#include "platform/ux/plugin.h"
#include "platform/ux/ps_plugin_ext.h"

namespace ps::studio {

// The object behind a plugin callback's `self`: static_cast<Ctx*>(self->data).
// `self` is any plugin POD carrying a `void* data` (GuiPlugin, KeyHandlerPlugin,
// ViewportPlugin, ViewportGuiPlugin, OverlayPlugin, ModelSourcePlugin, ...).
template <class Ctx, class Self>
Ctx* ctx_cast(Self* self) {
  return static_cast<Ctx*>(self->data);
}

// --- Layout guards -------------------------------------------------------- //
// pv = one native pointer. The ext structs are name-first, then their callback
// pointers, then the `data` back-pointer the editor recovers via ctx_cast.
namespace abi_check {
inline constexpr std::size_t pv = sizeof(void*);

// ViewportInput: the on_mouse callbacks read model/data/camera/vis_option and
// the mouse/modifier block; guard the pointer quartet and the float block start.
static_assert(std::is_standard_layout_v<ViewportInput>);
static_assert(offsetof(ViewportInput, model) == 0 * pv);
static_assert(offsetof(ViewportInput, data) == 1 * pv);
static_assert(offsetof(ViewportInput, camera) == 2 * pv);
static_assert(offsetof(ViewportInput, vis_option) == 3 * pv);
static_assert(offsetof(ViewportInput, x) == 4 * pv);

// ViewportPlugin / OverlayPlugin / ViewportGuiPlugin: name, one callback, data.
static_assert(std::is_same_v<decltype(ViewportPlugin::data), void*>);
static_assert(offsetof(ViewportPlugin, name) == 0 * pv);
static_assert(offsetof(ViewportPlugin, data) == 2 * pv);
static_assert(std::is_same_v<decltype(OverlayPlugin::data), void*>);
static_assert(offsetof(OverlayPlugin, name) == 0 * pv);
static_assert(offsetof(OverlayPlugin, data) == 2 * pv);
static_assert(std::is_same_v<decltype(ViewportGuiPlugin::data), void*>);
static_assert(offsetof(ViewportGuiPlugin, name) == 0 * pv);
static_assert(offsetof(ViewportGuiPlugin, data) == 2 * pv);

// ViewportGuiPlugin::Context: the draw callback reads all five fields.
static_assert(offsetof(ViewportGuiPlugin::Context, model) == 0 * pv);
static_assert(offsetof(ViewportGuiPlugin::Context, data) == 1 * pv);
static_assert(offsetof(ViewportGuiPlugin::Context, camera) == 2 * pv);

// ModelSourcePlugin: name, two callbacks, data.
static_assert(std::is_same_v<decltype(ModelSourcePlugin::data), void*>);
static_assert(offsetof(ModelSourcePlugin, name) == 0 * pv);
static_assert(offsetof(ModelSourcePlugin, data) == 3 * pv);

// CompiledModel: the handed-off artifact is a single mjModel*.
static_assert(std::is_same_v<decltype(CompiledModel::model), mjModel*>);
static_assert(offsetof(CompiledModel, model) == 0);
static_assert(sizeof(CompiledModel) == pv);
}  // namespace abi_check

}  // namespace ps::studio

#endif  // PS_STUDIO_EDITOR_PLUGIN_ABI_H_
