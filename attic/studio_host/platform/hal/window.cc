// Vendored from MuJoCo Studio, adapted for ProtoSpec Studio (ps::studio).
//
// Upstream: src/experimental/platform/hal/window.cc @ mujoco 67a1ea6d
// Adaptation: classic-OpenGL-only (DR-S5). Removed the Filament/Vulkan/WebGl
// swap-chain branches, the software-headless blit path, macOS native-handle
// plumbing, and the bundled-font resource loading (fonts are opt-in and default
// off; ImGui's built-in font is used otherwise). A single `hidden` flag creates
// an offscreen-capable window that still owns a real GL context.

#include "platform/hal/window.h"

#include <algorithm>
#include <string>
#include <string_view>

#include <SDL.h>
#include <SDL_syswm.h>
#include <backends/imgui_impl_sdl2.h>
#include <imgui.h>

namespace ps::studio {

static void InitImGui(SDL_Window* window) {
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
  io.IniFilename = nullptr;
  ImGui::StyleColorsDark();
  ImGui_ImplSDL2_InitForOpenGL(window, SDL_GL_GetCurrentContext());
}

Window::Window(std::string_view title, int width, int height, Config config)
    : width_(width), height_(height), config_(config) {
  if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS) != 0) {
    SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "SDL_Init: %s", SDL_GetError());
    return;
  }

  SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
  SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);

  Uint32 flags = SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE |
                 SDL_WINDOW_ALLOW_HIGHDPI;
  if (config_.hidden) {
    flags |= SDL_WINDOW_HIDDEN;
  }

  sdl_window_ = SDL_CreateWindow(
      std::string(title).c_str(), SDL_WINDOWPOS_CENTERED,
      SDL_WINDOWPOS_CENTERED, width_, height_, flags);
  if (!sdl_window_) {
    SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "SDL_CreateWindow: %s",
                 SDL_GetError());
    return;
  }

  gl_context_ = SDL_GL_CreateContext(sdl_window_);
  SDL_GL_MakeCurrent(sdl_window_, static_cast<SDL_GLContext>(gl_context_));
  SDL_GL_SetSwapInterval(1);

  InitImGui(sdl_window_);
  if (config_.load_fonts) {
    ImGui::GetIO().Fonts->AddFontDefault();
  }

  SDL_SysWMinfo wmi;
  SDL_VERSION(&wmi.version);
  if (SDL_GetWindowWMInfo(sdl_window_, &wmi)) {
#if defined(_WIN32)
    native_window_ = reinterpret_cast<void*>(wmi.info.win.window);
#elif defined(__linux__)
    native_window_ = reinterpret_cast<void*>(wmi.info.x11.window);
#endif
  }

  int drawable_width = width_;
  int drawable_height = height_;
  SDL_GL_GetDrawableSize(sdl_window_, &drawable_width, &drawable_height);
  scale_ = width_ > 0 ? static_cast<float>(drawable_width) / width_ : 1.0f;
}

Window::~Window() {
  if (ImGui::GetCurrentContext()) {
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();
  }
  if (gl_context_) {
    SDL_GL_DeleteContext(static_cast<SDL_GLContext>(gl_context_));
  }
  if (sdl_window_) {
    SDL_DestroyWindow(sdl_window_);
  }
  SDL_Quit();
}

void Window::SetTitle(std::string_view title) {
  if (sdl_window_) {
    SDL_SetWindowTitle(sdl_window_, std::string(title).c_str());
  }
}

void Window::Resize(int width, int height) {
  if (!sdl_window_) {
    return;
  }
  SDL_SetWindowSize(sdl_window_, width, height);
  SDL_GetWindowSize(sdl_window_, &width_, &height_);
  int drawable_width = width_;
  int drawable_height = height_;
  SDL_GL_GetDrawableSize(sdl_window_, &drawable_width, &drawable_height);
  scale_ = width_ > 0 ? static_cast<float>(drawable_width) / width_ : 1.0f;
}

std::string Window::GetDropFile() {
  std::string tmp;
  std::swap(tmp, drop_file_);
  return tmp;
}

Window::Status Window::NewFrame() {
  SDL_Event event;
  while (SDL_PollEvent(&event)) {
    ImGui_ImplSDL2_ProcessEvent(&event);
    if (event.type == SDL_QUIT) {
      should_exit_ = true;
    } else if (event.type == SDL_WINDOWEVENT) {
      if (event.window.event == SDL_WINDOWEVENT_RESIZED) {
        SDL_GetWindowSize(sdl_window_, &width_, &height_);
        int drawable_width = width_;
        int drawable_height = height_;
        SDL_GL_GetDrawableSize(sdl_window_, &drawable_width, &drawable_height);
        scale_ = width_ > 0 ? static_cast<float>(drawable_width) / width_ : 1.0f;
      }
    } else if (event.type == SDL_DROPFILE) {
      drop_file_ = event.drop.file;
      SDL_free(event.drop.file);
    }
  }

  ImGui_ImplSDL2_NewFrame();
  ImGui::NewFrame();
  return should_exit_ ? kQuitting : kRunning;
}

void Window::EndFrame() { ImGui::EndFrame(); }

void Window::Present() {
  if (sdl_window_) {
    SDL_GL_SwapWindow(sdl_window_);
  }
}

}  // namespace ps::studio
