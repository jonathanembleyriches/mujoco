// ProtoSpec Studio self-screenshot service (ps::studio, ours): plugin-side.
//
// Captures the composited window framebuffer (3D scene + ImGui UI) to numbered
// PNGs, configured from the environment so it works through the stock
// LaunchStudio path with ZERO host patches:
//
//   MUJOCO_SCREENSHOT_DIR    output directory (default ".")
//   MUJOCO_SCREENSHOT_AFTER  auto-capture once the frame count reaches N
//   MUJOCO_SCREENSHOT_COUNT  number of auto captures (default 1)
//   MUJOCO_SCREENSHOT_EXIT   quit the app after the auto sequence is written
//
// F12 captures on demand (read raw via ImGui::IsKeyPressed, so it works
// regardless of keyboard capture / focus; it never conflicts with a text
// field). Files are named shot_%04d.png.
//
// HOW the capture reaches the framebuffer without a host hook: the host has no
// post-render plugin callback -- the last per-frame plugin hook
// (ScenePlugin::enhance_scene) runs before the renderer draws anything. So the
// service appends an ImDrawCallback to the ImGui *foreground* draw list;
// ImGui::Render() emits that list last, and the classic backend executes user
// callbacks in submission order inside Renderer::Render()'s
// ImGui_ImplOpenGL3_RenderDrawData -- i.e. after both the scene and the UI are
// in the back buffer and before the swap. That is the same capture point a
// host-side patch would use, one plugin callback earlier in name only.
//
// Classic backend only: the GL entry points are resolved through the live SDL
// GL context (SDL_GL_GetProcAddress). Under Filament there is no SDL GL
// context (it presents through its own swapchain, and its ImGui bridge is not
// the GL3 backend), so the service disarms with a warning instead of
// submitting a callback.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <SDL.h>
#include <mujoco/mujoco.h>

#include "editor/plugins.h"
#include "imgui.h"
#include "lodepng.h"
#include "platform/ux/plugin.h"

namespace ps::studio {
namespace {

// Minimal GL surface, resolved at capture time from the live SDL GL context so
// this TU needs no GL headers or link-time GL dependency.
using GlGetIntegervFn = void (*)(unsigned int pname, int* params);
using GlPixelStoreiFn = void (*)(unsigned int pname, int param);
using GlReadPixelsFn = void (*)(int x, int y, int w, int h, unsigned int format,
                                unsigned int type, void* pixels);
constexpr unsigned int kGlViewport = 0x0BA2;
constexpr unsigned int kGlPackAlignment = 0x0D05;
constexpr unsigned int kGlRgba = 0x1908;
constexpr unsigned int kGlUnsignedByte = 0x1401;

struct ScreenshotService {
  std::string dir = ".";
  int after = 0;           // auto-capture threshold; 0 = F12 only
  int remaining = 0;       // auto captures still to take
  bool exit_when_done = false;
  int frame = 0;           // counted per ModelPlugin::do_update (once a frame)
  int index = 0;           // shot_%04d.png sequence number
};

// Runs inside ImGui_ImplOpenGL3_RenderDrawData, GL context current, back
// buffer fully composited, swap not yet performed.
void CaptureCallback(const ImDrawList*, const ImDrawCmd* cmd) {
  auto* svc = static_cast<ScreenshotService*>(cmd->UserCallbackData);

  static auto* gl_get_integerv = reinterpret_cast<GlGetIntegervFn>(
      SDL_GL_GetProcAddress("glGetIntegerv"));
  static auto* gl_pixel_storei = reinterpret_cast<GlPixelStoreiFn>(
      SDL_GL_GetProcAddress("glPixelStorei"));
  static auto* gl_read_pixels =
      reinterpret_cast<GlReadPixelsFn>(SDL_GL_GetProcAddress("glReadPixels"));
  if (!gl_get_integerv || !gl_pixel_storei || !gl_read_pixels) {
    mju_warning("Screenshot capture: GL entry points unavailable.");
    return;
  }

  // The backend's SetupRenderState sets the viewport to the full framebuffer.
  int viewport[4] = {0, 0, 0, 0};
  gl_get_integerv(kGlViewport, viewport);
  const int width = viewport[2];
  const int height = viewport[3];
  if (width <= 0 || height <= 0) {
    return;
  }

  // RGBA readback (universally supported, unlike RGB on GLES), then drop alpha
  // and flip to top-row-first for the PNG.
  std::vector<unsigned char> rgba(static_cast<size_t>(width) * height * 4);
  gl_pixel_storei(kGlPackAlignment, 1);
  gl_read_pixels(0, 0, width, height, kGlRgba, kGlUnsignedByte, rgba.data());
  std::vector<unsigned char> rgb(static_cast<size_t>(width) * height * 3);
  for (int y = 0; y < height; ++y) {
    const unsigned char* src =
        rgba.data() + static_cast<size_t>(height - 1 - y) * width * 4;
    unsigned char* dst = rgb.data() + static_cast<size_t>(y) * width * 3;
    for (int x = 0; x < width; ++x) {
      dst[0] = src[0];
      dst[1] = src[1];
      dst[2] = src[2];
      dst += 3;
      src += 4;
    }
  }

  char name[64];
  std::snprintf(name, sizeof(name), "shot_%04d.png", svc->index++);
  const std::string path = svc->dir + "/" + name;
  const unsigned err =
      lodepng::encode(path, rgb.data(), static_cast<unsigned>(width),
                      static_cast<unsigned>(height), LCT_RGB, 8);
  if (err) {
    mju_warning("Screenshot encode/write failed for %s: %s", path.c_str(),
                lodepng_error_text(err));
  } else {
    std::fprintf(stderr, "[screenshot] wrote %s (%dx%d)\n", path.c_str(), width,
                 height);
  }
}

bool ScreenshotUpdate(ModelPlugin* self, mjModel*, mjData*) {
  auto* svc = static_cast<ScreenshotService*>(self->data);
  ++svc->frame;

  bool due = false;
  if (ImGui::IsKeyPressed(ImGuiKey_F12, false)) {
    due = true;
  }
  if (svc->after > 0 && svc->frame >= svc->after && svc->remaining > 0) {
    due = true;
    --svc->remaining;
  }

  if (due) {
    if (SDL_GL_GetCurrentContext()) {
      ImGui::GetForegroundDrawList()->AddCallback(&CaptureCallback, svc);
    } else {
      mju_warning(
          "Screenshot capture is only supported for the classic backend "
          "(run with --gfx=classic).");
    }
  }

  // In auto mode, quit once the requested sequence has been written. The quit
  // event is polled at the NEXT frame's NewFrame, so the current frame still
  // renders and runs the capture callback armed above.
  if (svc->exit_when_done && svc->after > 0 && svc->frame >= svc->after &&
      svc->remaining == 0) {
    SDL_Event quit;
    std::memset(&quit, 0, sizeof(quit));
    quit.type = SDL_QUIT;
    SDL_PushEvent(&quit);
  }

  return false;  // never steps the simulation
}

}  // namespace

void RegisterScreenshotService() {
  // Leaked for the app lifetime (mirrors the plugin registry's static storage).
  auto* svc = new ScreenshotService;
  if (const char* dir = std::getenv("MUJOCO_SCREENSHOT_DIR")) {
    svc->dir = dir;
  }
  if (const char* after = std::getenv("MUJOCO_SCREENSHOT_AFTER")) {
    svc->after = std::atoi(after);
  }
  if (svc->after > 0) {
    const char* count = std::getenv("MUJOCO_SCREENSHOT_COUNT");
    svc->remaining = count ? std::max(1, std::atoi(count)) : 1;
  }
  svc->exit_when_done = std::getenv("MUJOCO_SCREENSHOT_EXIT") != nullptr;

  ModelPlugin plugin;
  plugin.name = "ProtoSpec Screenshot";
  plugin.do_update = ScreenshotUpdate;
  plugin.data = svc;
  RegisterPlugin<ModelPlugin>(plugin);
}

}  // namespace ps::studio
