// Copyright 2025 DeepMind Technologies Limited
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

// Main entry point for the Filament-based MuJoCo renderer.

#include <cstdlib>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <ios>
#include <iosfwd>
#include <string>
#include <string_view>
#include <vector>

#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <mujoco/mujoco.h>
#include "experimental/platform/hal/graphics_mode.h"
#include "experimental/platform/sys_utils.h"
#include "experimental/studio/app.h"

#ifdef MUJOCO_STUDIO_PROTOSPEC
#include "editor/plugins.h"
#endif

ABSL_FLAG(int, window_width, 1400, "Window width");
ABSL_FLAG(int, window_height, 720, "Window height");
ABSL_FLAG(std::string, model_file, "", "MuJoCo model file.");
ABSL_FLAG(std::string, gfx, "", "Graphics API");
ABSL_FLAG(std::string, screenshot_seq, "",
          "Directory for self-screenshots (F12 + auto mode). Classic backend "
          "only.");
ABSL_FLAG(int, screenshot_after, 0,
          "Auto-capture starting on this frame (0 disables auto mode).");
ABSL_FLAG(int, screenshot_count, 1,
          "Number of auto-capture frames to write.");
ABSL_FLAG(bool, screenshot_exit, false,
          "Quit once the auto-capture sequence is written.");

std::string Resolve(std::string_view path) {
  std::string_view subpath = path.substr(path.find(':') + 1);
  std::filesystem::path exe_dir = mujoco::platform::GetModuleDir((void*)&Resolve);
  if (exe_dir.empty()) {
    return std::string("assets/") + std::string(subpath);
  }
  return (exe_dir / "assets" / subpath).string();
}

class FileResource {
 public:
  explicit FileResource(const std::string& path)
      : file_(path, std::ios::binary | std::ios::ate) {
    if (!file_.is_open()) {
      mju_warning("Cannot open file %s", path.c_str());
      return;
    }

    size_ = file_.tellg();
    file_.seekg(0, std::ios::beg);
  }

  int Read(const void** buffer) {
    buffer_.resize(size_);
    if (!file_.read(reinterpret_cast<char*>(buffer_.data()), size_)) {
      return 0;
    }
    *buffer = buffer_.data();
    return size_;
  }

  int Size() const { return size_; }

  FileResource(const FileResource&) = delete;
  FileResource& operator=(const FileResource&) = delete;

 private:
  std::ifstream file_;
  std::vector<char> buffer_;
  int size_ = 0;
};

int main(int argc, char** argv, char** envp) {
  absl::ParseCommandLine(argc, argv);

  const char* home = std::getenv("HOME");
  const std::string ini_path = std::string(home ? home : ".") + "/.mujoco.ini";

  mjpResourceProvider resource_provider;
  mjp_defaultResourceProvider(&resource_provider);

  resource_provider.open = [](mjResource* resource) {
    const std::string resolved_path = Resolve(resource->name);
    FileResource* f = new FileResource(resolved_path);
    if (f->Size() == 0) {
      delete f;
      return 0;
    }
    resource->data = f;
    return f->Size();
  };
  resource_provider.read = [](mjResource* resource, const void** buffer) {
    FileResource* f = static_cast<FileResource*>(resource->data);
    return f->Read(buffer);
  };
  resource_provider.close = [](mjResource* resource) {
    delete static_cast<FileResource*>(resource->data);
    resource->data = nullptr;
  };

  resource_provider.prefix = "font";
  mjp_registerResourceProvider(&resource_provider);
  resource_provider.prefix = "filament";
  mjp_registerResourceProvider(&resource_provider);

  std::string gfx = absl::GetFlag(FLAGS_gfx);

  const char* session_type = std::getenv("XDG_SESSION_TYPE");
  const char* wayland_display = std::getenv("WAYLAND_DISPLAY");
  if ((session_type && std::string_view(session_type) == "wayland") ||
      wayland_display) {
    if (gfx.empty()) {
      gfx = "opengl_headless";
    } else if (gfx == "classic" || gfx == "opengl") {
      mju_error(
          "Wayland does not support '%s' graphics mode. "
          "Restart with a different graphics mode, or login using X11.",
          gfx.c_str());
    }
  }

  mujoco::platform::GraphicsMode gfx_mode =
      mujoco::platform::GraphicsModeFromString(
          gfx, mujoco::platform::GraphicsMode::FilamentOpenGl);

#ifdef MUJOCO_STUDIO_PROTOSPEC
  // The editor plugin cluster shares one context that must outlive the loop.
  // R1: the whole editor (model adoption, viewport, menus/toolbar, mode, dialogs)
  // registers through the four upstream plugin types via RegisterEditorPlugins;
  // the former host-side EditorShell seams are gone.
  static ps::studio::EditorContext protospec_editor_context;
  ps::studio::RegisterEditorPlugins(protospec_editor_context);
#endif

  const int width = absl::GetFlag(FLAGS_window_width);
  const int height = absl::GetFlag(FLAGS_window_height);
#ifdef MUJOCO_STUDIO_PROTOSPEC
  const std::string app_title = "ProtoSpec Studio";
#else
  const std::string app_title = "MuJoCo Studio";
#endif
  mujoco::studio::App app({
    .width = width,
    .height = height,
    .ini_path = ini_path,
    .gfx_mode = gfx_mode,
    .title = app_title,
    .screenshot_dir = absl::GetFlag(FLAGS_screenshot_seq),
    .screenshot_after = absl::GetFlag(FLAGS_screenshot_after),
    .screenshot_count = absl::GetFlag(FLAGS_screenshot_count),
    .screenshot_exit = absl::GetFlag(FLAGS_screenshot_exit),
  });

  // If the model file is not specified, try to load it from the first argument
  std::string model_file = absl::GetFlag(FLAGS_model_file);
  if (model_file.empty() && argc > 1 && argv[1][0] != '-') model_file = argv[1];

  if (model_file.empty()) {
    app.InitEmptyModel();
  } else {
    app.LoadModelFromFile(model_file);
  }

  while (app.Update()) {
    app.BuildGui();
    app.Render();
  }
  return 0;
}
