// ProtoSpec Studio: plugin C-ABI helpers + layout guards (ps::studio, ours).
//
// R1: the editor operates through the FOUR upstream Studio plugin types only
// (GuiPlugin / ModelPlugin / KeyHandlerPlugin / SpecEditorPlugin). Every callback
// recovers its owning object with static_cast<T*>(self->data) -- ctx_cast<T> is
// the one typed spelling of that. The four PODs are the upstream
// mujoco::platform:: types themselves, re-exported into ps::studio by the host
// shim (platform/ux/plugin.h); the editor never hand-copies them. The
// static_asserts below pin the offset of every field the editor reads or
// writes, so an upstream field reorder/retype is a compile error here rather
// than a silent C-ABI mismatch when a stale editor .so meets a newer host.

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
// bool aside), then its callback pointers, then the `data` back-pointer. Every
// field the editor touches is pinned, not just name/data.
namespace abi_check {
inline constexpr std::size_t pv = sizeof(void*);

static_assert(std::is_standard_layout_v<GuiPlugin>);
static_assert(offsetof(GuiPlugin, active) == 0);
static_assert(offsetof(GuiPlugin, name) == 1 * pv);
static_assert(offsetof(GuiPlugin, update) == 2 * pv);
static_assert(offsetof(GuiPlugin, data) == 3 * pv);
static_assert(std::is_same_v<decltype(GuiPlugin::active), bool>);
static_assert(std::is_same_v<decltype(GuiPlugin::data), void*>);

static_assert(std::is_standard_layout_v<ModelPlugin>);
static_assert(offsetof(ModelPlugin, name) == 0 * pv);
static_assert(offsetof(ModelPlugin, get_model_to_load) == 1 * pv);
static_assert(offsetof(ModelPlugin, post_model_loaded) == 2 * pv);
static_assert(offsetof(ModelPlugin, do_update) == 3 * pv);
static_assert(offsetof(ModelPlugin, data) == 4 * pv);
static_assert(std::is_same_v<decltype(ModelPlugin::data), void*>);

static_assert(std::is_standard_layout_v<KeyHandlerPlugin>);
static_assert(offsetof(KeyHandlerPlugin, name) == 0 * pv);
static_assert(offsetof(KeyHandlerPlugin, key_chord) == 1 * pv);
static_assert(offsetof(KeyHandlerPlugin, on_key_pressed) == 2 * pv);
static_assert(offsetof(KeyHandlerPlugin, data) == 3 * pv);
static_assert(std::is_same_v<decltype(KeyHandlerPlugin::key_chord), int>);
static_assert(std::is_same_v<decltype(KeyHandlerPlugin::data), void*>);

static_assert(std::is_standard_layout_v<SpecEditorPlugin>);
static_assert(offsetof(SpecEditorPlugin, name) == 0 * pv);
static_assert(offsetof(SpecEditorPlugin, pre_compile) == 1 * pv);
static_assert(offsetof(SpecEditorPlugin, post_compile) == 2 * pv);
static_assert(offsetof(SpecEditorPlugin, data) == 3 * pv);
static_assert(std::is_same_v<decltype(SpecEditorPlugin::data), void*>);
}  // namespace abi_check

}  // namespace ps::studio

#endif  // PS_STUDIO_EDITOR_PLUGIN_ABI_H_
