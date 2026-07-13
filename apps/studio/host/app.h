// ProtoSpec Studio host application (ps::studio, ours).
//
// The App is a plugin HOST, not an editor. It owns only the render/step
// substrate -- the SDL2 window, the classic renderer, StepControl, and the live
// mjModel*/mjData* -- and dispatches the Studio plugin hook set each frame:
//   ModelSourcePlugin (ext)  compiled-artifact handoff (model swap)
//   ModelPlugin              stock buffer-injection + do_update stepping
//   ViewportPlugin (ext)     viewport mouse events (picking; later gizmos)
//   OverlayPlugin (ext)      scene overlays (fanned out via the renderer hook)
//   GuiPlugin                dockable panels + the Plugins menu
//   KeyHandlerPlugin         custom key chords
//   SpecEditorPlugin         stock pre/post-compile hook (unused by ProtoSpec)
// The toolbar (play/pause/stop) is host-side because it maps to StepControl,
// which is host infrastructure. Everything editor-flavored lives in plugins.

#ifndef PS_STUDIO_HOST_APP_H_
#define PS_STUDIO_HOST_APP_H_

#include <memory>
#include <string>

#include <mujoco/mujoco.h>

#include "platform/hal/renderer.h"
#include "platform/hal/window.h"
#include "platform/sim/step_control.h"

namespace ps::studio {

class App {
 public:
  struct Config {
    std::string title = "ProtoSpec Studio";
    int width = 1400;
    int height = 820;
    bool hidden = false;  // offscreen smoke path
  };

  explicit App(Config config);
  ~App();

  App(const App&) = delete;
  App& operator=(const App&) = delete;

  // Feeds the host's deferred load slot (CLI arg / drag-drop). Forwarded to the
  // ModelSource plugins on the next frame.
  void RequestLoad(std::string path);

  // One frame of input + plugin dispatch + physics. Returns false to quit.
  bool Update();

  // Builds the dockspace, toolbar, status bar, and GuiPlugin windows.
  void BuildGui();

  // Renders the scene + ImGui and presents.
  void Render();

  bool has_model() const { return model_ != nullptr; }
  bool has_data() const { return data_ != nullptr; }

  // Runs `frames` iterations of Update/BuildGui/Render then returns. Used by the
  // --smoke-frames CI hook (typically with a hidden window). Returns 0.
  int SmokeRun(int frames);

 private:
  void AdoptCompiledModel(mjModel* model);
  void ProcessPendingLoads();
  void UpdatePhysics();
  void HandleWindowEvents();
  void HandleInput();
  void ResetPhysics();

  void ToolBarGui();
  void StatusBarGui();

  Window::Config MakeWindowConfig(const Config& config);

  std::string title_;
  std::unique_ptr<Window> window_;
  std::unique_ptr<Renderer> renderer_;
  StepControl step_control_;

  mjModel* model_ = nullptr;  // borrowed from a ModelSource plugin
  mjData* data_ = nullptr;    // owned by the host
  mjvCamera camera_{};
  mjvOption vis_options_{};
  mjvPerturb perturb_{};
  int camera_idx_ = kFreeCameraFallback;

  std::string pending_load_;
  bool has_pending_load_ = false;
  bool should_exit_ = false;

  static constexpr int kFreeCameraFallback = -3;  // matches interaction tumble
};

}  // namespace ps::studio

#endif  // PS_STUDIO_HOST_APP_H_
