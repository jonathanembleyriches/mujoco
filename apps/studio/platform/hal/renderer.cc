// Vendored from MuJoCo Studio, adapted for ProtoSpec Studio (ps::studio).
//
// Upstream: src/experimental/platform/hal/renderer.cc @ mujoco 67a1ea6d
// Adaptation: classic mjr/OpenGL backend only (DR-S5). The Filament context,
// SceneBridge/ImguiBridge, offscreen read-back, and GUI-plugin bridge callback
// are removed. Adds the overlay hook between mjv_updateScene and mjr_render.

#include "platform/hal/renderer.h"

#include <chrono>

#include <backends/imgui_impl_opengl3.h>
#include <imgui.h>
#include <mujoco/mujoco.h>

namespace ps::studio {

Renderer::Renderer() {
  if (ImGui::GetCurrentContext()) {
    ImGui_ImplOpenGL3_Init("#version 130");
    imgui_backend_ = true;
  }
}

Renderer::~Renderer() {
  if (imgui_backend_ && ImGui::GetCurrentContext()) {
    ImGui_ImplOpenGL3_Shutdown();
  }
  Deinit();
}

void Renderer::Init(const mjModel* model) {
  Deinit();
  if (model) {
    mjr_defaultContext(&render_context_);
    mjr_makeContext(model, &render_context_, mjFONTSCALE_150);
    mjv_defaultScene(&scene_);
    mjv_makeScene(model, &scene_, 2000);
    initialized_ = true;
  }
}

void Renderer::Deinit() {
  if (initialized_) {
    mjv_freeScene(&scene_);
    mjr_freeContext(&render_context_);
    initialized_ = false;
  }
}

void Renderer::Render(const mjModel* model, mjData* data,
                      const mjvPerturb* perturb, mjvCamera* camera,
                      const mjvOption* vis_option, int width, int height) {
  if (!initialized_) {
    // Still present ImGui so the empty-model UI draws.
    if (imgui_backend_ && ImGui::GetCurrentContext()) {
      ImGui_ImplOpenGL3_NewFrame();
      ImGui::Render();
      ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    }
    return;
  }

  mjvCamera default_cam;
  if (camera == nullptr) {
    mjv_defaultFreeCamera(model, &default_cam);
    camera = &default_cam;
  }
  mjvOption default_opt;
  if (vis_option == nullptr) {
    mjv_defaultOption(&default_opt);
    vis_option = &default_opt;
  }

  mjv_updateScene(model, data, vis_option, perturb, camera, mjCAT_ALL, &scene_);

  // §5 overlay hook: append editor geoms after the scene is built, before draw.
  if (overlay_hook_) {
    overlay_hook_(model, data, &scene_);
  }

  const mjrRect viewport = {0, 0, width, height};
  mjr_render(viewport, &scene_, &render_context_);

  if (imgui_backend_ && ImGui::GetCurrentContext()) {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
  }

  UpdateFps();
}

void Renderer::UpdateFps() {
  const Clock::time_point now = Clock::now();
  const double interval =
      std::chrono::duration<double>(now - last_fps_update_).count();
  ++frames_;
  if (interval > 0.2) {
    fps_ = frames_ / interval;
    frames_ = 0;
    last_fps_update_ = now;
  }
}

}  // namespace ps::studio
