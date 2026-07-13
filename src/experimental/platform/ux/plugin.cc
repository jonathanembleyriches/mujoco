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

#include "experimental/platform/ux/plugin.h"

#include <functional>
#include <string_view>

#include <mujoco/mujoco.h>
#include "engine/engine_global_table.h"

using GuiPlugin = mujoco::platform::GuiPlugin;
using ModelPlugin = mujoco::platform::ModelPlugin;
using KeyHandlerPlugin = mujoco::platform::KeyHandlerPlugin;
using SpecEditorPlugin = mujoco::platform::SpecEditorPlugin;
using ViewportPlugin = mujoco::platform::ViewportPlugin;
using ViewportGuiPlugin = mujoco::platform::ViewportGuiPlugin;
using OverlayPlugin = mujoco::platform::OverlayPlugin;
using ModelSourcePlugin = mujoco::platform::ModelSourcePlugin;
using MainMenuPlugin = mujoco::platform::MainMenuPlugin;
using ToolbarPlugin = mujoco::platform::ToolbarPlugin;
using EditorShellPlugin = mujoco::platform::EditorShellPlugin;

namespace mujoco::platform {

template <typename T>
void RegisterPlugin(T plugin) {
  if (plugin.name == nullptr || plugin.name[0] == '\0') {
    mju_error("Plugin name must not be empty or null.");
  }
  GlobalTable<T>::GetSingleton().AppendIfUnique(plugin);
}

template <typename T>
void ForEachPlugin(const std::function<void(T*)>& fn) {
  auto& table = mujoco::GlobalTable<T>::GetSingleton();
  for (int i = 0; i < table.count(); ++i) {
    const T* plugin = table.GetAtSlot(i);
    fn(const_cast<T*>(plugin));
  }
}

}  // namespace mujoco::platform

#define MUJOCO_SPECIALIZE_PLUGIN(PLUGIN, NAME)                                 \
  template <>                                                                  \
  const char* mujoco::GlobalTable<PLUGIN>::HumanReadableTypeName() {           \
    return NAME;                                                               \
  }                                                                            \
  template <>                                                                  \
  std::string_view mujoco::GlobalTable<PLUGIN>::ObjectKey(const PLUGIN& p) {   \
    return std::string_view(p.name);                                           \
  }                                                                            \
  template <>                                                                  \
  bool mujoco::GlobalTable<PLUGIN>::ObjectEqual(const PLUGIN& p1,              \
                                                const PLUGIN& p2) {            \
    return CaseInsensitiveEqual(p1.name, p2.name);                             \
  }                                                                            \
  template <>                                                                  \
  bool mujoco::GlobalTable<PLUGIN>::CopyObject(PLUGIN& dst, const PLUGIN& src, \
                                               ErrorMessage& err) {            \
    dst = src;                                                                 \
    return true;                                                               \
  }                                                                            \
  namespace mujoco::platform {                                                 \
  template void RegisterPlugin<PLUGIN>(PLUGIN plugin);                         \
  template void ForEachPlugin<PLUGIN>(const std::function<void(PLUGIN*)>& fn); \
  }

MUJOCO_SPECIALIZE_PLUGIN(GuiPlugin, "gui plugin");
MUJOCO_SPECIALIZE_PLUGIN(ModelPlugin, "model plugin");
MUJOCO_SPECIALIZE_PLUGIN(KeyHandlerPlugin, "key handler plugin");
MUJOCO_SPECIALIZE_PLUGIN(SpecEditorPlugin, "spec editor plugin");
MUJOCO_SPECIALIZE_PLUGIN(ViewportPlugin, "viewport plugin");
MUJOCO_SPECIALIZE_PLUGIN(ViewportGuiPlugin, "viewport gui plugin");
MUJOCO_SPECIALIZE_PLUGIN(OverlayPlugin, "overlay plugin");
MUJOCO_SPECIALIZE_PLUGIN(ModelSourcePlugin, "model source plugin");
MUJOCO_SPECIALIZE_PLUGIN(MainMenuPlugin, "main menu plugin");
MUJOCO_SPECIALIZE_PLUGIN(ToolbarPlugin, "toolbar plugin");
MUJOCO_SPECIALIZE_PLUGIN(EditorShellPlugin, "editor shell plugin");
