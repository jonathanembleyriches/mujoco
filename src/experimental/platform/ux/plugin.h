// Copyright 2026 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MUJOCO_SRC_EXPERIMENTAL_PLATFORM_UX_PLUGIN_H_
#define MUJOCO_SRC_EXPERIMENTAL_PLATFORM_UX_PLUGIN_H_

#include <functional>

#include <mujoco/mujoco.h>

namespace mujoco::platform {

// Registers plugins with the global registry. The plugins must have a
// case-insensitive unique name for the plugin type. Note that plugins are
// copied by value, so do not use inheritance.
template <typename T>
void RegisterPlugin(T plugin);

// Executes the given function for each registered plugin of type T.
template <typename T>
void ForEachPlugin(const std::function<void(T*)>& fn);

// Plugin for processing custom UI windows. The plugin will be listed in the
// "Plugins" main menu and, when selected, an ImGui window will be opened with
// the name of the plugin as the title. The `update` function can then be used
// to process the GUI. All functions will be called by passing `this` as the
// first argument.
struct GuiPlugin final {
  using UpdateFn = void (*)(GuiPlugin* self);

  // Whether or not to display the plugin window.
  bool active = false;

  // The name of the plugin; must be unique.
  const char* name = "";

  // The function that will update the plugin's window. Plugin GUI updates
  // happen when the window is `active` and after all other Studio GUI updates.
  UpdateFn update = nullptr;

  // Optional data pointer.
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

  // The name of the plugin; must be unique.
  const char* name = "";

  // Callback for when the plugin wants to load a new model. This function will
  // return a buffer containing the model data as well as the content type of
  // the buffer. Returns nullptr if no model needs to be loaded.s
  GetModelToLoadFn get_model_to_load = nullptr;

  // Callback when a new model is loaded.
  PostModelLoadedFn post_model_loaded = nullptr;

  // Callback when the physics simulation is updated. Returns true if the
  // simulation should be stepped.
  DoUpdateFn do_update = nullptr;

  // Optional data pointer.
  void* data = nullptr;
};

// Plugin for handling custom keyboard events.
struct KeyHandlerPlugin final {
  using OnKeyPressedFn = void (*)(KeyHandlerPlugin* self);

  // The name of the plugin; must be unique.
  const char* name = "";

  // The ImGui key codes for the key combination that triggers the plugin.
  int key_chord = 0;

  // The function to be called when the above key combination is pressed.
  OnKeyPressedFn on_key_pressed = nullptr;

  // Optional data pointer.
  void* data = nullptr;
};

// Plugin for editing the mjSpec.
struct SpecEditorPlugin final {
  using PreCompileFn = bool (*)(SpecEditorPlugin* self, mjSpec* spec,
                                const mjModel* model, const mjData* data,
                                const mjvCamera* camera);
  using PostCompileFn = void (*)(SpecEditorPlugin* self, const mjSpec* spec,
                                 const mjModel* model, mjData* data);

  // The name of the plugin; must be unique.
  const char* name = "";

  // Callback that edits the spec. If it returns true, then the spec will be
  // recompiled and `post_compile` will be called with the result.
  PreCompileFn pre_compile = nullptr;

  // Callback that is called after the spec has been recompiled.
  PostCompileFn post_compile = nullptr;

  // Optional data pointer.
  void* data = nullptr;
};

// Snapshot of viewport mouse state for a frame. Positions are normalized to
// the viewport in [0, 1] with y measured from the top; deltas are in the same
// units.
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

// Plugin that receives viewport mouse events. Returns true if the event was
// consumed; the app then suppresses its default camera/perturb handling for
// that frame. Only invoked when ImGui is not capturing the mouse.
struct ViewportPlugin final {
  using OnMouseFn = bool (*)(ViewportPlugin* self, const ViewportInput& input);

  // The name of the plugin; must be unique.
  const char* name = "";

  // Callback invoked with the viewport mouse state each frame.
  OnMouseFn on_mouse = nullptr;

  // Optional data pointer.
  void* data = nullptr;
};

// Plugin contributing menus to the main menu bar. Invoked inside the host's
// BeginMainMenuBar()/EndMainMenuBar() pair, so the callback may open top-level
// menus (e.g. an external editor's File / Edit menus). When any such plugin is
// registered the host suppresses its own stock File menu.
struct MainMenuPlugin final {
  using DrawFn = void (*)(MainMenuPlugin* self);
  const char* name = "";
  DrawFn draw = nullptr;
  void* data = nullptr;
};

// Plugin contributing controls to the top toolbar (e.g. an editor's transform
// tool buttons, snap toggle, add-menu). Invoked inside the toolbar row after the
// host's own controls, on the same line.
struct ToolbarPlugin final {
  using DrawFn = void (*)(ToolbarPlugin* self);
  const char* name = "";
  DrawFn draw = nullptr;
  void* data = nullptr;
};

// Bridge that lets the host's play/stop toolbar drive an external editor's mode
// machine. The host owns physics run/pause (StepControl); this hook tells the
// editor to enter Play (compile pending edits) or return to Edit (discard sim
// state). `mode`: 0 == Edit/stop, 1 == Play.
struct EditorShellPlugin final {
  using SetModeFn = void (*)(EditorShellPlugin* self, int mode);
  using IsDirtyFn = bool (*)(EditorShellPlugin* self);
  const char* name = "";
  SetModeFn set_mode = nullptr;
  // True when the editor has unsaved authored edits (drives the dirty indicator).
  IsDirtyFn is_dirty = nullptr;
  void* data = nullptr;
};

// Plugin that draws into the active ImGui frame over the 3D viewport (e.g.
// screen-space transform gizmos). Invoked each frame during GUI building with
// the live camera and viewport metrics so the draw list can project against
// the rendered scene. `edit_mode` is true while the simulation is paused. The
// camera is mutable so a plugin can service a frame-selection ("F") request.
struct ViewportGuiPlugin final {
  struct Context {
    const mjModel* model = nullptr;
    const mjData* data = nullptr;
    mjvCamera* camera = nullptr;
    float aspect_ratio = 1.0f;
    bool edit_mode = true;
  };
  using DrawFn = void (*)(ViewportGuiPlugin* self, const Context& ctx);

  // The name of the plugin; must be unique.
  const char* name = "";

  // Callback that draws the overlay for the current frame.
  DrawFn draw = nullptr;

  // Optional data pointer.
  void* data = nullptr;
};

// Plugin that appends transient geoms (e.g. selection outlines) to the live
// mjvScene after mjv_updateScene and before the scene is rendered.
struct OverlayPlugin final {
  using AddOverlayFn = void (*)(OverlayPlugin* self, const mjModel* model,
                                const mjData* data, mjvScene* scene);

  // The name of the plugin; must be unique.
  const char* name = "";

  // Callback that appends geoms to the scene.
  AddOverlayFn add_overlay = nullptr;

  // Optional data pointer.
  void* data = nullptr;
};

// A compiled model ready for the app to adopt. The app renders and steps
// `model` but does not own it; the producing plugin retains ownership and must
// keep it alive until it publishes a replacement.
struct CompiledModel {
  mjModel* model = nullptr;
};

// Plugin that produces ready-built mjModel instances for the app to adopt,
// bypassing the mjSpec/XML load pipeline. The app pushes requested model file
// paths in through `submit_load` and polls `poll_compiled` each frame for a
// freshly compiled model.
struct ModelSourcePlugin final {
  // The app pushes a requested model file path (deferred load slot). The
  // plugin stashes it and compiles on a later frame.
  using SubmitLoadFn = void (*)(ModelSourcePlugin* self, const char* path);

  // Returns true and fills `out` when a new compiled model is ready. Returns
  // false when nothing new is pending.
  using PollFn = bool (*)(ModelSourcePlugin* self, CompiledModel* out);

  // The name of the plugin; must be unique.
  const char* name = "";

  // Callback receiving requested model file paths.
  SubmitLoadFn submit_load = nullptr;

  // Callback polled each frame for a freshly compiled model.
  PollFn poll_compiled = nullptr;

  // Optional data pointer.
  void* data = nullptr;
};

}  // namespace mujoco::platform

#endif  // MUJOCO_SRC_EXPERIMENTAL_PLATFORM_UX_PLUGIN_H_
