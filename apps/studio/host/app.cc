// ProtoSpec Studio host application (ps::studio, ours). See app.h.

#include "host/app.h"

#include <cstdio>
#include <functional>
#include <string>
#include <utility>

#include <imgui.h>
#include <mujoco/mujoco.h>

#include "platform/ux/interaction.h"
#include "platform/ux/plugin.h"
#include "platform/ux/ps_plugin_ext.h"

namespace ps::studio {

using PauseState = StepControl::PauseState;

Window::Config App::MakeWindowConfig(const Config& config) {
  Window::Config wc;
  wc.hidden = config.hidden;
  wc.load_fonts = false;
  return wc;
}

App::App(Config config) : title_(config.title) {
  window_ = std::make_unique<Window>(title_, config.width, config.height,
                                     MakeWindowConfig(config));
  renderer_ = std::make_unique<Renderer>();

  mjv_defaultCamera(&camera_);
  mjv_defaultOption(&vis_options_);
  mjv_defaultPerturb(&perturb_);

  // Fan the renderer overlay hook out to every OverlayPlugin (§5).
  renderer_->SetOverlayHook(
      [](const mjModel* m, const mjData* d, mjvScene* s) {
        ForEachPlugin<OverlayPlugin>([&](OverlayPlugin* p) {
          if (p->add_overlay) {
            p->add_overlay(p, m, d, s);
          }
        });
      });

  // Edit mode is the default (DR-S2): physics paused until the user hits play.
  step_control_.SetPauseState(PauseState::kNormalPaused);
}

App::~App() {
  if (data_) {
    mj_deleteData(data_);
  }
  // model_ is borrowed from a ModelSource plugin; the plugin owns it.
}

void App::RequestLoad(std::string path) {
  pending_load_ = std::move(path);
  has_pending_load_ = true;
}

void App::AdoptCompiledModel(mjModel* model) {
  if (!model) {
    return;
  }
  renderer_->Init(model);
  if (data_) {
    mj_deleteData(data_);
  }
  data_ = mj_makeData(model);
  model_ = model;

  mj_resetData(model_, data_);
  mj_forward(model_, data_);
  mjv_defaultFreeCamera(model_, &camera_);
  camera_idx_ = kFreeCameraFallback;

  // Return to Edit mode (paused, spec pose) on every swap (DR-S2).
  step_control_.SetPauseState(PauseState::kNormalPaused);
  step_control_.ForceSync();

  if (window_) {
    window_->SetTitle(title_ + " : " +
                      std::string(model_->names ? model_->names : "model"));
  }
}

void App::ResetPhysics() {
  if (has_model() && has_data()) {
    mj_resetData(model_, data_);
    mj_forward(model_, data_);
    step_control_.ForceSync();
  }
}

void App::ProcessPendingLoads() {
  // Push any host-side load request (CLI arg / drag-drop) to the editor.
  if (has_pending_load_) {
    const std::string path = pending_load_;
    has_pending_load_ = false;
    pending_load_.clear();
    ForEachPlugin<ModelSourcePlugin>([&](ModelSourcePlugin* p) {
      if (p->submit_load) {
        p->submit_load(p, path.c_str());
      }
    });
  }

  // Adopt a freshly compiled artifact from any ModelSource plugin (ext handoff).
  mjModel* adopt = nullptr;
  ForEachPlugin<ModelSourcePlugin>([&](ModelSourcePlugin* p) {
    if (adopt || !p->poll_compiled) {
      return;
    }
    CompiledModel cm;
    if (p->poll_compiled(p, &cm) && cm.model) {
      adopt = cm.model;
    }
  });
  if (adopt) {
    AdoptCompiledModel(adopt);
  }
}

void App::UpdatePhysics() {
  if (!has_model() || !has_data()) {
    return;
  }
  bool stepped = false;
  ForEachPlugin<ModelPlugin>([&](ModelPlugin* p) {
    if (p->do_update && p->do_update(p, model_, data_)) {
      stepped = true;
    }
  });
  if (!stepped) {
    step_control_.Advance(model_, data_);
  }
}

void App::HandleWindowEvents() {
  const std::string drop = window_->GetDropFile();
  if (!drop.empty()) {
    RequestLoad(drop);
  }
}

void App::HandleInput() {
  ImGuiIO& io = ImGui::GetIO();

  // Esc / window close quits.
  if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
    should_exit_ = true;
  }

  if (!io.WantCaptureKeyboard) {
    if (ImGui::IsKeyPressed(ImGuiKey_Space)) {
      step_control_.SetPauseState(
          step_control_.GetPauseState() == PauseState::kUnpaused
              ? PauseState::kNormalPaused
              : PauseState::kUnpaused);
    }
    ForEachPlugin<KeyHandlerPlugin>([&](KeyHandlerPlugin* p) {
      if (p->on_key_pressed && p->key_chord &&
          ImGui::IsKeyChordPressed(p->key_chord)) {
        p->on_key_pressed(p);
      }
    });
  }

  if (io.WantCaptureMouse || !has_model() || !has_data()) {
    return;
  }

  const float mx = io.DisplaySize.x > 0 ? io.MousePos.x / io.DisplaySize.x : 0;
  const float my = io.DisplaySize.y > 0 ? io.MousePos.y / io.DisplaySize.y : 0;
  const float dx = io.DisplaySize.x > 0 ? io.MouseDelta.x / io.DisplaySize.x : 0;
  const float dy = io.DisplaySize.y > 0 ? io.MouseDelta.y / io.DisplaySize.y : 0;
  const float scroll = io.MouseWheel / 50.0f;

  // Simple orbit / pan / dolly camera (mjv math via MoveCamera).
  const bool left = ImGui::IsMouseDown(ImGuiMouseButton_Left);
  const bool right = ImGui::IsMouseDown(ImGuiMouseButton_Right);
  const bool middle = ImGui::IsMouseDown(ImGuiMouseButton_Middle);
  const bool moving = (dx != 0.0f || dy != 0.0f);
  if (moving && !io.KeyCtrl) {
    if (left) {
      MoveCamera(model_, data_, &camera_, CameraMotion::ORBIT, dx, dy);
    } else if (middle) {
      MoveCamera(model_, data_, &camera_, CameraMotion::ZOOM, dx, dy);
    } else if (right) {
      MoveCamera(model_, data_, &camera_,
                 io.KeyShift ? CameraMotion::PLANAR_MOVE_H
                             : CameraMotion::PLANAR_MOVE_V,
                 dx, dy);
    }
  }
  if (scroll != 0.0f) {
    MoveCamera(model_, data_, &camera_, CameraMotion::ZOOM, 0, -scroll);
  }

  // Dispatch viewport mouse events to plugins (picking; later gizmos).
  ViewportInput input;
  input.model = model_;
  input.data = data_;
  input.camera = &camera_;
  input.vis_option = &vis_options_;
  input.x = mx;
  input.y = my;
  input.dx = dx;
  input.dy = dy;
  input.scroll = scroll;
  input.aspect_ratio = window_->GetAspectRatio();
  input.left_down = left;
  input.right_down = right;
  input.middle_down = middle;
  input.left_double = ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left);
  input.right_double = ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Right);
  input.ctrl = io.KeyCtrl;
  input.shift = io.KeyShift;
  input.alt = io.KeyAlt;
  ForEachPlugin<ViewportPlugin>([&](ViewportPlugin* p) {
    if (p->on_mouse) {
      p->on_mouse(p, input);
    }
  });
}

bool App::Update() {
  const Window::Status status = window_->NewFrame();
  HandleWindowEvents();
  HandleInput();
  ProcessPendingLoads();
  UpdatePhysics();
  return status == Window::kRunning && !should_exit_;
}

void App::ToolBarGui() {
  const PauseState pause = step_control_.GetPauseState();
  if (ImGui::Button("Play")) {
    step_control_.SetPauseState(PauseState::kUnpaused);
    step_control_.ForceSync();
  }
  ImGui::SameLine();
  if (ImGui::Button("Pause")) {
    step_control_.SetPauseState(PauseState::kNormalPaused);
  }
  ImGui::SameLine();
  if (ImGui::Button("Stop")) {
    // Play->Stop discards simulation state, returns to Edit mode (DR-S2).
    ResetPhysics();
    step_control_.SetPauseState(PauseState::kNormalPaused);
  }
  ImGui::SameLine();
  ImGui::TextDisabled("|  mode: %s",
                      pause == PauseState::kUnpaused ? "Play" : "Edit");
}

void App::StatusBarGui() {
  ImGui::Text("%s  |  %.0f fps  |  %s", has_model() ? "model loaded" : "no model",
              renderer_ ? renderer_->GetFps() : 0.0,
              step_control_.GetPauseState() == PauseState::kUnpaused
                  ? "running"
                  : "paused");
}

void App::BuildGui() {
  ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(),
                               ImGuiDockNodeFlags_PassthruCentralNode);

  if (ImGui::Begin("Toolbar")) {
    ToolBarGui();
  }
  ImGui::End();

  if (ImGui::Begin("Status")) {
    StatusBarGui();
  }
  ImGui::End();

  // GuiPlugin panels: the Plugins menu toggles + the active windows (Studio's
  // dispatch pattern).
  ForEachPlugin<GuiPlugin>([](GuiPlugin* plugin) {
    if (!plugin->update) {
      return;
    }
    if (ImGui::BeginMainMenuBar()) {
      if (ImGui::BeginMenu("Plugins")) {
        if (ImGui::MenuItem(plugin->name, nullptr, plugin->active)) {
          plugin->active = !plugin->active;
        }
        ImGui::EndMenu();
      }
      ImGui::EndMainMenuBar();
    }
    if (plugin->active) {
      if (ImGui::Begin(plugin->name, &plugin->active)) {
        plugin->update(plugin);
      }
      ImGui::End();
    }
  });
}

void App::Render() {
  const int w = static_cast<int>(window_->GetWidth() * window_->GetScale());
  const int h = static_cast<int>(window_->GetHeight() * window_->GetScale());
  renderer_->Render(model_, data_, &perturb_, &camera_, &vis_options_, w, h);
  window_->EndFrame();
  window_->Present();
}

int App::SmokeRun(int frames) {
  int rendered = 0;
  bool stepped_any = false;
  const double t0 = has_data() ? data_->time : 0.0;
  for (int i = 0; i < frames && !should_exit_; ++i) {
    Update();
    BuildGui();
    Render();
    ++rendered;
    // After the first frame the pending load has been processed; run physics.
    if (i == 0 && has_model()) {
      step_control_.SetPauseState(PauseState::kUnpaused);
    }
  }
  if (has_data() && data_->time > t0) {
    stepped_any = true;
  }
  std::printf("smoke: rendered %d frames  model=%s  stepped=%s\n", rendered,
              has_model() ? "loaded" : "none", stepped_any ? "yes" : "no");
  return 0;
}

}  // namespace ps::studio
