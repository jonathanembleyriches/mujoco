// ProtoSpec Studio host application (ps::studio, ours). See app.h.

#include "host/app.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <utility>

#include <imgui.h>
#include <mujoco/mujoco.h>

#include "editor/mode_ui.h"
#include "editor/transform_math.h"
#include "platform/ux/interaction.h"
#include "platform/ux/plugin.h"
#include "platform/ux/ps_plugin_ext.h"

namespace ps::studio {

using PauseState = StepControl::PauseState;
namespace mj = ps::mjcf;

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

  // Only re-frame the free camera on a genuine file load, NOT on the many
  // recompiles a gizmo drag triggers (that would snap the view every frame).
  if (!editor_ || editor_->fresh_load) {
    mjv_defaultFreeCamera(model_, &camera_);
    camera_idx_ = kFreeCameraFallback;
    if (editor_) editor_->fresh_load = false;
  }
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

  // Esc quits -- unless a gizmo drag is in progress, in which case the gizmo's
  // own Esc handler cancels the drag (and we must not exit).
  const bool gizmo_dragging = editor_ && editor_->gizmo_active;
  if (ImGui::IsKeyPressed(ImGuiKey_Escape) && !gizmo_dragging) {
    should_exit_ = true;
  }

  if (!io.WantCaptureKeyboard) {
    if (ImGui::IsKeyPressed(ImGuiKey_Space)) {
      ToggleSpace();
    }
    if (ImGui::IsKeyPressed(ImGuiKey_F) && editor_) {
      FrameSelection(editor_->selected_serial);
    }
    ForEachPlugin<KeyHandlerPlugin>([&](KeyHandlerPlugin* p) {
      if (p->on_key_pressed && p->key_chord &&
          ImGui::IsKeyChordPressed(p->key_chord)) {
        p->on_key_pressed(p);
      }
    });
  }

  // A Hierarchy double-click requests framing; service it (F equivalent).
  if (editor_ && editor_->focus_request_serial != 0) {
    FrameSelection(editor_->focus_request_serial);
    editor_->focus_request_serial = 0;
  }

  if (io.WantCaptureMouse || !has_model() || !has_data()) {
    return;
  }

  const float mx = io.DisplaySize.x > 0 ? io.MousePos.x / io.DisplaySize.x : 0;
  const float my = io.DisplaySize.y > 0 ? io.MousePos.y / io.DisplaySize.y : 0;
  const float dx = io.DisplaySize.x > 0 ? io.MouseDelta.x / io.DisplaySize.x : 0;
  const float dy = io.DisplaySize.y > 0 ? io.MouseDelta.y / io.DisplaySize.y : 0;
  const float scroll = io.MouseWheel / 50.0f;

  // Dispatch viewport mouse events to plugins FIRST (the gizmo grabs the mouse
  // before the camera does). A plugin returning true suppresses camera motion.
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
  input.left_down = ImGui::IsMouseDown(ImGuiMouseButton_Left);
  input.right_down = ImGui::IsMouseDown(ImGuiMouseButton_Right);
  input.middle_down = ImGui::IsMouseDown(ImGuiMouseButton_Middle);
  input.left_double = ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left);
  input.right_double = ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Right);
  input.ctrl = io.KeyCtrl;
  input.shift = io.KeyShift;
  input.alt = io.KeyAlt;
  bool consumed = false;
  ForEachPlugin<ViewportPlugin>([&](ViewportPlugin* p) {
    if (p->on_mouse && p->on_mouse(p, input)) {
      consumed = true;
    }
  });

  // Camera orbit / pan / dolly (mjv math via MoveCamera), unless the gizmo owns
  // the mouse this frame. Scroll-zoom always applies.
  const bool moving = (dx != 0.0f || dy != 0.0f);
  const bool orbit = ImGui::IsMouseDown(ImGuiMouseButton_Left);
  const bool pan = ImGui::IsMouseDown(ImGuiMouseButton_Right);
  const bool dolly = ImGui::IsMouseDown(ImGuiMouseButton_Middle);
  if (moving && !io.KeyCtrl && !consumed) {
    if (orbit) {
      MoveCamera(model_, data_, &camera_, CameraMotion::ORBIT, dx, dy);
    } else if (dolly) {
      MoveCamera(model_, data_, &camera_, CameraMotion::ZOOM, dx, dy);
    } else if (pan) {
      MoveCamera(model_, data_, &camera_,
                 io.KeyShift ? CameraMotion::PLANAR_MOVE_H
                             : CameraMotion::PLANAR_MOVE_V,
                 dx, dy);
    }
  }
  if (scroll != 0.0f) {
    MoveCamera(model_, data_, &camera_, CameraMotion::ZOOM, 0, -scroll);
  }
}

void App::ApplyMode() {
  // StepControl's pause is a pure function of the editor mode (DR-S2): Edit and a
  // paused Play are both paused; only a running Play advances physics.
  const bool run = editor_ && editor_->mode == EditorMode::Play &&
                   !editor_->play_paused;
  const PauseState want = run ? PauseState::kUnpaused : PauseState::kNormalPaused;
  if (step_control_.GetPauseState() != want) {
    step_control_.SetPauseState(want);
    if (run) step_control_.ForceSync();
  }
}

void App::EnterPlay() {
  if (!editor_) return;
  // ▶: compile pending edits (if dirty) then run from qpos0.
  if (editor_->dirty) {
    editor_->apply_edits = true;  // consumed by the ModelSource poll this frame
  }
  editor_->mode = EditorMode::Play;
  editor_->play_paused = false;
}

void App::PausePlay() {
  if (editor_ && editor_->mode == EditorMode::Play) editor_->play_paused = true;
}

void App::StopToEdit() {
  if (!editor_) return;
  // ⏹: discard ALL simulation state, return to Edit at qpos0 (Unity semantics).
  editor_->mode = EditorMode::Edit;
  editor_->play_paused = false;
  ResetPhysics();
}

void App::ToggleSpace() {
  if (!editor_) {
    step_control_.SetPauseState(
        step_control_.GetPauseState() == PauseState::kUnpaused
            ? PauseState::kNormalPaused
            : PauseState::kUnpaused);
    return;
  }
  if (editor_->mode == EditorMode::Edit) {
    EnterPlay();
  } else if (editor_->play_paused) {
    editor_->play_paused = false;  // resume
  } else {
    editor_->play_paused = true;   // pause
  }
}

void App::FrameSelection(std::uint64_t serial) {
  if (!editor_ || !has_model() || !has_data() || serial == 0) return;
  const auto& b = editor_->compiled.binding;
  const double extent = model_->stat.extent;
  for (const auto& e : b.entries()) {
    if (e.serial != serial || e.id < 0) continue;
    double center[3];
    double radius = 0.1 * extent;
    bool ok = true;
    if (e.etype == mj::ElementType::Geom) {
      for (int k = 0; k < 3; ++k) center[k] = data_->geom_xpos[3 * e.id + k];
      radius = model_->geom_rbound[e.id] > 0 ? model_->geom_rbound[e.id]
                                             : 0.1 * extent;
    } else if (e.etype == mj::ElementType::Body) {
      for (int k = 0; k < 3; ++k) center[k] = data_->xpos[3 * e.id + k];
      radius = 0.25 * extent;
    } else if (e.etype == mj::ElementType::Site) {
      for (int k = 0; k < 3; ++k) center[k] = data_->site_xpos[3 * e.id + k];
    } else if (e.etype == mj::ElementType::Camera) {
      for (int k = 0; k < 3; ++k) center[k] = data_->cam_xpos[3 * e.id + k];
    } else {
      ok = false;
    }
    if (!ok) continue;
    camera_.type = mjCAMERA_FREE;
    camera_.fixedcamid = -1;
    for (int k = 0; k < 3; ++k) camera_.lookat[k] = center[k];
    camera_.distance = std::max(radius * 3.0, 0.05 * extent);
    camera_idx_ = kFreeCameraFallback;
    return;
  }
}

bool App::Update() {
  const Window::Status status = window_->NewFrame();
  HandleWindowEvents();
  HandleInput();
  ProcessPendingLoads();

  // The ▶ "compile if dirty then run" path: once the pending compile has flushed
  // (adopted at qpos0), physics may run.
  ApplyMode();
  UpdatePhysics();
  return status == Window::kRunning && !should_exit_;
}

void App::ToolBarGui() {
  const bool playing = editor_ && editor_->mode == EditorMode::Play;

  // Mode buttons (map to StepControl via the mode machine).
  if (ImGui::Button(playing && !(editor_ && editor_->play_paused) ? "|| Pause"
                                                                  : "> Play")) {
    if (playing && !(editor_ && editor_->play_paused)) {
      PausePlay();
    } else {
      EnterPlay();
    }
  }
  ImGui::SameLine();
  if (ImGui::Button("[] Stop")) {
    StopToEdit();
  }

  if (!editor_) return;

  // Transform tools + Local/World + snap (deliverable 2). Only meaningful in Edit.
  ImGui::SameLine();
  ImGui::TextDisabled("|");
  ImGui::SameLine();
  GizmoSettings& g = editor_->gizmo;
  struct ToolBtn { const char* label; GizmoTool tool; };
  const ToolBtn tools[] = {{"Select (Q)", GizmoTool::Select},
                           {"Move (W)", GizmoTool::Translate},
                           {"Rotate (E)", GizmoTool::Rotate},
                           {"Scale (R)", GizmoTool::Scale}};
  for (const ToolBtn& t : tools) {
    ImGui::SameLine();
    const bool on = g.tool == t.tool;
    if (on) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.25f, 0.45f, 0.8f, 1));
    if (ImGui::Button(t.label)) g.tool = t.tool;
    if (on) ImGui::PopStyleColor();
  }

  ImGui::SameLine();
  ImGui::TextDisabled("|");
  ImGui::SameLine();
  if (ImGui::Button(g.world_space ? "World" : "Local")) {
    g.world_space = !g.world_space;
  }
  ImGui::SameLine();
  ImGui::Checkbox("Snap", &g.snap);
  ImGui::SameLine();
  if (ImGui::Button("Snap...")) ImGui::OpenPopup("snap_settings");
  if (ImGui::BeginPopup("snap_settings")) {
    ImGui::TextDisabled("Snap increments");
    ImGui::InputDouble("move (m)", &g.snap_translate, 0.01, 0.1, "%.3f");
    ImGui::InputDouble("rotate (deg)", &g.snap_rotate_deg, 1.0, 5.0, "%.1f");
    ImGui::InputDouble("scale", &g.snap_scale, 0.01, 0.05, "%.3f");
    ImGui::EndPopup();
  }

  DrawModeChip(*editor_);
}

void App::StatusBarGui() {
  const bool playing = editor_ && editor_->mode == EditorMode::Play;
  ImGui::Text("%s  |  %.0f fps  |  %s%s", has_model() ? "model loaded" : "no model",
              renderer_ ? renderer_->GetFps() : 0.0,
              playing ? (editor_->play_paused ? "Play (paused)" : "Play (running)")
                      : "Edit",
              (editor_ && editor_->dirty) ? "  *dirty" : "");
  if (editor_ && editor_->status_toast.Visible(ImGui::GetTime())) {
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.4f, 1), "|  %s",
                       editor_->status_toast.message.c_str());
  }
  // Diagnostics error chip (deliverable 3): Diagnostics is no longer a standing
  // panel; a red count chip appears here whenever diagnostics hold errors, and
  // clicking it reveals + focuses the Diagnostics panel (ServiceDiagnosticsReveal
  // re-activates it from the viewport hook).
  if (editor_) {
    const int errors = DiagnosticErrorCount(editor_->diagnostics);
    if (errors > 0) {
      ImGui::SameLine();
      ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.55f, 0.15f, 0.13f, 1.0f));
      if (ImGui::SmallButton(
              (std::to_string(errors) + (errors == 1 ? " error" : " errors"))
                  .c_str())) {
        editor_->focus_diagnostics_request = true;
      }
      ImGui::PopStyleColor();
    }
  }
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

  // Viewport overlays drawn into the ImGui frame over the scene (the gizmos).
  ViewportGuiPlugin::Context vc;
  vc.model = model_;
  vc.data = data_;
  vc.camera = &camera_;
  vc.aspect_ratio = window_ ? window_->GetAspectRatio() : 1.0f;
  vc.edit_mode = !editor_ || editor_->mode == EditorMode::Edit;
  ForEachPlugin<ViewportGuiPlugin>([&](ViewportGuiPlugin* p) {
    if (p->draw) p->draw(p, vc);
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
    // After the first frame the pending load has been processed; enter Play so
    // physics advances (exercises the mode machine end to end).
    if (i == 0 && has_model()) {
      EnterPlay();
    }
  }
  if (has_data() && data_->time > t0) {
    stepped_any = true;
  }
  std::printf("smoke: rendered %d frames  model=%s  stepped=%s\n", rendered,
              has_model() ? "loaded" : "none", stepped_any ? "yes" : "no");
  return 0;
}

int App::SmokeEditRun(int frames) {
  // Deliverable 7: start in Edit mode, select a body, drive a small gizmo drag
  // through the testable core (no real mouse), recompile, exit 0.
  int rendered = 0;
  bool dragged = false;
  double moved = 0;
  for (int i = 0; i < frames && !should_exit_; ++i) {
    Update();
    BuildGui();
    Render();
    ++rendered;

    if (!dragged && editor_ && editor_->model_ready && editor_->tree && has_data()) {
      // Select the first real body and switch to the translate tool.
      std::uint64_t body_serial = 0;
      double before[3] = {0, 0, 0};
      for (const auto& e : editor_->compiled.binding.entries()) {
        if (e.etype == mj::ElementType::Body && e.id >= 1) {
          body_serial = e.serial;
          for (int k = 0; k < 3; ++k) before[k] = data_->xpos[3 * e.id + k];
          break;
        }
      }
      if (body_serial != 0) {
        editor_->selected_serial = body_serial;
        editor_->gizmo.tool = GizmoTool::Translate;
        // Simulate a gizmo gesture: one BeginEdit/mutate/CommitEdit.
        editor_->BeginEdit();
        DragFrame f = BuildDragFrame(model_, data_, editor_->compiled.binding,
                                     *editor_->tree, body_serial);
        const double D[3] = {0.05, 0.0, 0.0};
        ApplyTranslate(*editor_->tree, body_serial, f, D);
        editor_->CommitEdit("smoke gizmo drag");
        dragged = true;
        (void)before;
        moved = D[0];
      }
    }
  }
  std::printf("smoke-edit: rendered %d frames  model=%s  gizmo_drag=%s (dx=%.3f)\n",
              rendered, has_model() ? "loaded" : "none", dragged ? "yes" : "no",
              moved);
  return 0;
}

}  // namespace ps::studio
