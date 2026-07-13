// ProtoSpec Studio: the remaining editor Gui panels + commands (ps::studio,
// ours). Hierarchy lives in hierarchy_panel.cc and the generated Details panel in
// details_panel.cc (SE1b); this TU keeps Assets / Diagnostics, the File menu
// (load / save), and the Ctrl+Z / Ctrl+Y / Ctrl+S key handlers.

#include <string>

#include <imgui.h>
#include <imgui_stdlib.h>

#include "editor/editor_context.h"
#include "editor/editor_ops.h"
#include "editor/plugins.h"
#include "platform/ux/plugin.h"

namespace ps::studio {

static EditorContext* Ctx(void* data) {
  return static_cast<EditorContext*>(data);
}

// Assets: placeholder grid (SE3 wires mesh/texture/material/hfield import).
static void AssetsUpdate(GuiPlugin* self) {
  (void)self;
  ImGui::TextDisabled("Assets (meshes / textures / materials / hfields)");
  ImGui::TextDisabled("Populated in SE3.");
}

// Diagnostics: live Validate/Compile log + status line (§3).
static void DiagnosticsUpdate(GuiPlugin* self) {
  EditorContext* c = Ctx(self->data);
  if (!c->status_line.empty()) {
    ImGui::TextWrapped("%s", c->status_line.c_str());
    ImGui::Separator();
  }
  ImGui::BeginChild("##diag_log");
  for (const std::string& line : c->diagnostics) {
    ImGui::TextUnformatted(line.c_str());
  }
  if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY()) {
    ImGui::SetScrollHereY(1.0f);
  }
  ImGui::EndChild();
}

// File menu: a simple path input for load / save (no vendored file dialog in
// SE0). Lives in the main menu bar; drag-drop and the CLI arg still feed loads.
static void FileMenuUpdate(GuiPlugin* self) {
  EditorContext* c = Ctx(self->data);
  static std::string load_path;
  static std::string save_path;

  ImGui::TextUnformatted("Load model (.xml):");
  ImGui::SetNextItemWidth(-1);
  const bool load_enter =
      ImGui::InputTextWithHint("##loadpath", "path to MJCF...", &load_path,
                               ImGuiInputTextFlags_EnterReturnsTrue);
  if ((ImGui::Button("Load") || load_enter) && !load_path.empty()) {
    c->pending.Request(load_path);  // consumed by the ModelSource poll
  }

  ImGui::Separator();
  ImGui::Text("Save %s", c->dirty ? "(unsaved changes)" : "(up to date)");
  if (save_path.empty() && !c->source_path.empty()) {
    save_path = c->source_path;
  }
  ImGui::SetNextItemWidth(-1);
  ImGui::InputTextWithHint("##savepath", "path to write...", &save_path);
  const bool can_save = c->model_ready && !save_path.empty();
  ImGui::BeginDisabled(!c->model_ready || c->source_path.empty());
  if (ImGui::Button("Save")) {
    SaveModel(*c, c->source_path);
  }
  ImGui::EndDisabled();
  ImGui::SameLine();
  ImGui::BeginDisabled(!can_save);
  if (ImGui::Button("Save As")) {
    SaveModel(*c, save_path);
  }
  ImGui::EndDisabled();
}

static void RegisterGuiPanel(const char* name, GuiPlugin::UpdateFn fn, bool active,
                             EditorContext& ctx) {
  GuiPlugin plugin;
  plugin.name = name;
  plugin.update = fn;
  plugin.active = active;
  plugin.data = &ctx;
  RegisterPlugin<GuiPlugin>(plugin);
}

static void RegisterKey(const char* name, int chord,
                        KeyHandlerPlugin::OnKeyPressedFn fn, EditorContext& ctx) {
  KeyHandlerPlugin plugin;
  plugin.name = name;
  plugin.key_chord = chord;
  plugin.on_key_pressed = fn;
  plugin.data = &ctx;
  RegisterPlugin<KeyHandlerPlugin>(plugin);
}

void RegisterEditorPanels(EditorContext& ctx) {
  RegisterGuiPanel("Assets", AssetsUpdate, true, ctx);
  RegisterGuiPanel("Diagnostics", DiagnosticsUpdate, true, ctx);
  RegisterGuiPanel("File", FileMenuUpdate, false, ctx);

  RegisterKey(
      "Undo", ImGuiMod_Ctrl | ImGuiKey_Z,
      [](KeyHandlerPlugin* self) { Undo(*Ctx(self->data)); }, ctx);
  RegisterKey(
      "Redo", ImGuiMod_Ctrl | ImGuiKey_Y,
      [](KeyHandlerPlugin* self) { Redo(*Ctx(self->data)); }, ctx);
  RegisterKey(
      "Save", ImGuiMod_Ctrl | ImGuiKey_S,
      [](KeyHandlerPlugin* self) {
        EditorContext* c = Ctx(self->data);
        if (c->model_ready && !c->source_path.empty()) {
          SaveModel(*c, c->source_path);
        }
      },
      ctx);
}

}  // namespace ps::studio
