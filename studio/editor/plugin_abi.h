// ProtoSpec Studio: plugin C-ABI helpers + layout guards (ps::studio, ours).
//
// R1: the editor operates through the FOUR upstream Studio plugin types only
// (GuiPlugin / ModelPlugin / KeyHandlerPlugin / SpecEditorPlugin). Every callback
// recovers its owning object with static_cast<T*>(self->data) -- ctx_cast<T> is
// the one typed spelling of that. The struct layouts are HAND-SYNCED against the
// host's mujoco::platform:: definitions (platform/ux/plugin.h keeps them VERBATIM
// so a plugin loads into stock Studio); the static_asserts below pin the offsets
// the editor relies on so a field reorder in either copy is a compile error, not
// a silent C-ABI mismatch. Compiled against whichever copy is on the include path
// (ours, or the host shim in build_ps), so the guard tracks the real layout.

#ifndef PS_STUDIO_EDITOR_PLUGIN_ABI_H_
#define PS_STUDIO_EDITOR_PLUGIN_ABI_H_

#include <cstddef>
#include <type_traits>

#include "platform/ux/plugin.h"

namespace ps::studio {

// The object behind a plugin callback's `self`: static_cast<Ctx*>(self->data).
// `self` is any plugin POD carrying a `void* data` back-pointer.
template <class Ctx, class Self>
Ctx* ctx_cast(Self* self) {
  return static_cast<Ctx*>(self->data);
}

// --- Layout guards (the four upstream types) ------------------------------ //
// pv = one native pointer. Each POD is name-first (GuiPlugin's leading `active`
// bool aside), then its callback pointers, then the `data` back-pointer.
namespace abi_check {
inline constexpr std::size_t pv = sizeof(void*);

static_assert(std::is_standard_layout_v<GuiPlugin>);
static_assert(offsetof(GuiPlugin, name) == 1 * pv);
static_assert(offsetof(GuiPlugin, data) == 3 * pv);
static_assert(std::is_same_v<decltype(GuiPlugin::data), void*>);

static_assert(std::is_standard_layout_v<ModelPlugin>);
static_assert(offsetof(ModelPlugin, name) == 0 * pv);
static_assert(offsetof(ModelPlugin, data) == 4 * pv);
static_assert(std::is_same_v<decltype(ModelPlugin::data), void*>);

static_assert(std::is_standard_layout_v<KeyHandlerPlugin>);
static_assert(offsetof(KeyHandlerPlugin, name) == 0 * pv);
static_assert(offsetof(KeyHandlerPlugin, data) == 3 * pv);
static_assert(std::is_same_v<decltype(KeyHandlerPlugin::data), void*>);

static_assert(std::is_standard_layout_v<SpecEditorPlugin>);
static_assert(offsetof(SpecEditorPlugin, name) == 0 * pv);
static_assert(offsetof(SpecEditorPlugin, data) == 3 * pv);
static_assert(std::is_same_v<decltype(SpecEditorPlugin::data), void*>);
}  // namespace abi_check

}  // namespace ps::studio

#endif  // PS_STUDIO_EDITOR_PLUGIN_ABI_H_
