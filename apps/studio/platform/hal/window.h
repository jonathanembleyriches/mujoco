// Vendored from MuJoCo Studio, adapted for ProtoSpec Studio (ps::studio).
//
// Upstream: src/experimental/platform/hal/window.h @ mujoco 67a1ea6d
// Adaptation: reduced to the classic OpenGL path only (DR-S5). The Filament /
// Vulkan / WebGL / software-headless graphics modes and the GraphicsMode enum
// are dropped; a single boolean `hidden` flag replaces them (a hidden window
// still owns a real GL context, which is all the offscreen smoke path needs).
//
// SDL2 window + Dear ImGui context ownership; event pump; file drop.
#ifndef PS_STUDIO_PLATFORM_HAL_WINDOW_H_
#define PS_STUDIO_PLATFORM_HAL_WINDOW_H_

#include <string>
#include <string_view>

struct SDL_Window;

namespace ps::studio {

// Platform-independent window abstraction over SDL2.
//
// Initializes SDL2 + ImGui (docking enabled), creates and owns the native
// window and its OpenGL context, and pumps window events. The classic MuJoCo
// renderer draws into the GL context this window makes current.
class Window {
 public:
  struct Config {
    bool hidden = false;      // create the window hidden (headless smoke path)
    bool load_fonts = false;  // load bundled fonts (needs a resource provider)
  };

  Window(std::string_view title, int width, int height, Config config);
  ~Window();

  Window(const Window&) = delete;
  Window& operator=(const Window&) = delete;

  enum Status { kRunning, kQuitting };

  // Pumps all pending window events and starts a new ImGui frame. Returns
  // whether the window is still running.
  Status NewFrame();

  // Finalizes the ImGui frame (input handling). Must follow NewFrame.
  void EndFrame();

  // Swaps and presents the GL back buffer.
  void Present();

  void SetTitle(std::string_view title);

  int GetWidth() const { return width_; }
  int GetHeight() const { return height_; }
  float GetScale() const { return scale_; }
  float GetAspectRatio() const {
    return height_ > 0 ? static_cast<float>(width_) / height_ : 1.0f;
  }

  // The path of a file dropped on the window since the last call, else empty.
  std::string GetDropFile();

  void Resize(int width, int height);

  // Non-owning handle to the native window (unused by the classic path, kept
  // for renderer-signature parity and future backends).
  void* GetNativeWindowHandle() { return native_window_; }

 private:
  int width_ = 0;
  int height_ = 0;
  float scale_ = 1.0f;
  Config config_;
  void* native_window_ = nullptr;
  SDL_Window* sdl_window_ = nullptr;
  void* gl_context_ = nullptr;
  bool should_exit_ = false;
  std::string drop_file_;
};

}  // namespace ps::studio

#endif  // PS_STUDIO_PLATFORM_HAL_WINDOW_H_
