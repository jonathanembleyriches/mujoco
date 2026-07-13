// Vendored from MuJoCo Studio, adapted for ProtoSpec Studio (ps::studio).
//
// Upstream: src/experimental/platform/ux/plugin.h @ mujoco 67a1ea6d
// Adaptation: namespace (mujoco::platform -> ps::studio) only. The four plugin
// struct layouts and the RegisterPlugin/ForEachPlugin signatures are kept
// VERBATIM so a plugin authored against this host can be loaded into stock
// MuJoCo Studio with at most a namespace shim. The registry MECHANISM is
// reimplemented (registry.cc) with a simple per-type table instead of MuJoCo's
// internal engine_global_table.h, to avoid pulling an engine-private singleton
// into the editor; the observable contract (register by unique name, iterate)
// is identical.
//
// These are the Studio-compatible "core" interfaces. ProtoSpec's own additions
// (viewport input, scene overlays, compiled-artifact handoff) live in the
// clearly separated ps_plugin_ext.h and are candidates for upstreaming.

#ifndef PS_STUDIO_PLATFORM_UX_PLUGIN_H_
#define PS_STUDIO_PLATFORM_UX_PLUGIN_H_

#include <functional>

#include <mujoco/mujoco.h>

namespace ps::studio {

// Registers a plugin with the global registry. Plugins must have a
// case-insensitive unique name for their type. Plugins are copied by value, so
// do not use inheritance.
template <typename T>
void RegisterPlugin(T plugin);

// Executes fn for each registered plugin of type T.
template <typename T>
void ForEachPlugin(const std::function<void(T*)>& fn);

// Plugin for processing custom UI windows. Listed in the "Plugins" main menu;
// when active, an ImGui window titled with the plugin name is opened and
// `update` populates it. All callbacks receive `this` as the first argument.
struct GuiPlugin final {
  using UpdateFn = void (*)(GuiPlugin* self);
  bool active = false;
  const char* name = "";
  UpdateFn update = nullptr;
  void* data = nullptr;
};

// Plugin for loading and updating models.
struct ModelPlugin final {
  using GetModelToLoadFn = const char* (*)(ModelPlugin* self, int* size,
                                           char* content_type,
                                           int content_type_size,
                                           char* model_name,
                                           int model_name_size);
  using PostModelLoadedFn = void (*)(ModelPlugin* self, const char* model_path);
  using DoUpdateFn = bool (*)(ModelPlugin* self, mjModel* model, mjData* data);

  const char* name = "";
  GetModelToLoadFn get_model_to_load = nullptr;
  PostModelLoadedFn post_model_loaded = nullptr;
  DoUpdateFn do_update = nullptr;
  void* data = nullptr;
};

// Plugin for handling custom keyboard events.
struct KeyHandlerPlugin final {
  using OnKeyPressedFn = void (*)(KeyHandlerPlugin* self);
  const char* name = "";
  int key_chord = 0;
  OnKeyPressedFn on_key_pressed = nullptr;
  void* data = nullptr;
};

// Plugin for editing the mjSpec. Kept VERBATIM (mjSpec* in the signatures) for
// stock-Studio source compatibility; ProtoSpec's editor does not use mjSpec and
// leaves these callbacks null, driving edits through the ext interfaces instead.
struct SpecEditorPlugin final {
  using PreCompileFn = bool (*)(SpecEditorPlugin* self, mjSpec* spec,
                                const mjModel* model, const mjData* data,
                                const mjvCamera* camera);
  using PostCompileFn = void (*)(SpecEditorPlugin* self, const mjSpec* spec,
                                 const mjModel* model, mjData* data);

  const char* name = "";
  PreCompileFn pre_compile = nullptr;
  PostCompileFn post_compile = nullptr;
  void* data = nullptr;
};

}  // namespace ps::studio

#endif  // PS_STUDIO_PLATFORM_UX_PLUGIN_H_
