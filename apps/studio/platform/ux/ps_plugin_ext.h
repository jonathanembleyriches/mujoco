// ProtoSpec Studio plugin EXTENSIONS (ps::studio, ours -- not from MuJoCo).
//
// These interfaces extend the Studio-compatible core (plugin.h) where the stock
// plugin API is insufficient for a ProtoSpec-native editor. They follow the
// same C-ABI shape (POD struct of function pointers + `self` + `data`) and the
// same RegisterPlugin/ForEachPlugin dispatch, so the core/extension split is
// purely additive. Each is a candidate for upstreaming into MuJoCo Studio.
//
//   ViewportPlugin      viewport mouse input + the model/data/camera needed for
//                       ray picking (SE0) and, later, gizmo interaction (SE2).
//                       Stock Studio hard-codes this in app.cc; we expose it.
//   OverlayPlugin       the §5 scene-overlay hook, surfaced as a plugin: append
//                       geoms to the mjvScene between mjv_updateScene and
//                       mjr_render (selection outlines now, gizmos later). The
//                       host fans OverlayPlugins out through Renderer's hook.
//   ModelSourcePlugin   a compiled-artifact handoff: the editor compiles a
//                       ProtoSpec Model through the bridge and hands the host a
//                       ready mjModel* directly. Stock Studio's ModelPlugin only
//                       accepts an XML/MJB buffer via get_model_to_load (which
//                       would re-run MuJoCo's loader and discard our Binding);
//                       this ext lets the host adopt the artifact we already
//                       have. A stock-Studio port would fall back to returning
//                       CompileToXml(model) through get_model_to_load.

#ifndef PS_STUDIO_PLATFORM_UX_PS_PLUGIN_EXT_H_
#define PS_STUDIO_PLATFORM_UX_PS_PLUGIN_EXT_H_

#include <mujoco/mujoco.h>

#include "platform/ux/plugin.h"  // RegisterPlugin / ForEachPlugin declarations

namespace ps::studio {

// Snapshot of viewport mouse state for a frame. Positions are normalized to the
// viewport in [0, 1] with y measured from the top; deltas are in the same units.
struct ViewportInput {
  const mjModel* model = nullptr;
  const mjData* data = nullptr;
  const mjvCamera* camera = nullptr;
  const mjvOption* vis_option = nullptr;
  float x = 0, y = 0;
  float dx = 0, dy = 0;
  float scroll = 0;
  float aspect_ratio = 1.0f;
  bool left_down = false, right_down = false, middle_down = false;
  bool left_double = false, right_double = false;
  bool ctrl = false, shift = false, alt = false;
};

// Receives viewport mouse events. Returns true if the event was consumed (the
// host may use this to suppress default camera handling; SE0 does not rely on
// it). Only invoked when ImGui is not capturing the mouse.
struct ViewportPlugin final {
  using OnMouseFn = bool (*)(ViewportPlugin* self, const ViewportInput& input);
  const char* name = "";
  OnMouseFn on_mouse = nullptr;
  void* data = nullptr;
};

// Appends editor overlays to the live mjvScene before it is drawn.
struct OverlayPlugin final {
  using AddOverlayFn = void (*)(OverlayPlugin* self, const mjModel* model,
                                const mjData* data, mjvScene* scene);
  const char* name = "";
  AddOverlayFn add_overlay = nullptr;
  void* data = nullptr;
};

// A compiled model ready for the host to adopt. The host renders/steps `model`
// but does NOT own it; the producing plugin retains ownership and must keep it
// alive until it publishes a replacement.
struct CompiledModel {
  mjModel* model = nullptr;
};

// Polled by the host each frame for a freshly compiled model to adopt. The host
// also pushes load requests (CLI arg / drag-drop) in through `submit_load`; the
// plugin does the ParseMjcf -> Validate -> Compile work and publishes the result
// via `poll_compiled`. This keeps the ProtoSpec pipeline entirely plugin-side --
// the host never touches ps::Model.
struct ModelSourcePlugin final {
  // The host pushes a requested model file path (deferred load slot). The plugin
  // stashes it and compiles on a later frame.
  using SubmitLoadFn = void (*)(ModelSourcePlugin* self, const char* path);

  // Returns true and fills `out` when a new compiled model is ready; the host
  // then re-inits the renderer and allocates a fresh mjData. Returns false when
  // nothing new is pending.
  using PollFn = bool (*)(ModelSourcePlugin* self, CompiledModel* out);

  const char* name = "";
  SubmitLoadFn submit_load = nullptr;
  PollFn poll_compiled = nullptr;
  void* data = nullptr;
};

}  // namespace ps::studio

#endif  // PS_STUDIO_PLATFORM_UX_PS_PLUGIN_EXT_H_
