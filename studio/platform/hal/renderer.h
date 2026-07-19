// Vendored from MuJoCo Studio, adapted for ProtoSpec Studio (ps::studio).
//
// Upstream: src/experimental/platform/hal/renderer.h @ mujoco 67a1ea6d
// Adaptation: reduced to the classic mjr/OpenGL backend only (DR-S5). The
// Filament backend (mjrfilament, SceneBridge, ImguiBridge), the offscreen webp
// read-back requests, and the graphics-mode switch are removed. Adds the §5
// overlay hook: a callback slot invoked between mjv_updateScene and mjr_render
// so editor overlays (selection outlines, later gizmos) can append geoms to the
// mjvScene before it is drawn (SE2 needs it; default is a no-op).

#ifndef PS_STUDIO_PLATFORM_HAL_RENDERER_H_
#define PS_STUDIO_PLATFORM_HAL_RENDERER_H_

#include <chrono>
#include <functional>

#include <mujoco/mujoco.h>

namespace ps::studio {

// Renders an mjvScene through the classic OpenGL fixed-function pipeline plus
// the ImGui draw data. Assumes the caller has made a GL context current (Window
// does this).
class Renderer {
 public:
  // Invoked between mjv_updateScene and mjr_render with the live scene, so
  // overlays can append geoms via mjv_initGeom (§5). Renders on the classic
  // backend today; the seam is kept for future backends.
  using OverlayHook = std::function<void(const mjModel*, const mjData*,
                                         mjvScene*)>;

  Renderer();
  ~Renderer();

  Renderer(const Renderer&) = delete;
  Renderer& operator=(const Renderer&) = delete;

  // (Re)initializes the render context and scene for a model. Passing nullptr
  // tears the render state down.
  void Init(const mjModel* model);

  // Draws the scene + ImGui for the current frame into the active GL context.
  void Render(const mjModel* model, mjData* data, const mjvPerturb* perturb,
              mjvCamera* camera, const mjvOption* vis_option, int width,
              int height);

  // Installs the overlay hook (see OverlayHook). An empty hook is the default.
  void SetOverlayHook(OverlayHook hook) { overlay_hook_ = std::move(hook); }

  mjtByte* GetRenderFlags() { return scene_.flags; }
  double GetFps() const { return fps_; }

 private:
  void Deinit();
  void UpdateFps();

  bool initialized_ = false;
  bool imgui_backend_ = false;
  mjrContext render_context_{};
  mjvScene scene_{};
  OverlayHook overlay_hook_;

  using Clock = std::chrono::steady_clock;
  Clock::time_point last_fps_update_ = Clock::now();
  int frames_ = 0;
  double fps_ = 0;
};

}  // namespace ps::studio

#endif  // PS_STUDIO_PLATFORM_HAL_RENDERER_H_
