// ProtoSpec Studio: the editor Gui panel plugins (ps::studio, ours). SE0 ships
// the §3 dock layout panels (Hierarchy / Details / Assets / Diagnostics) as
// GuiPlugins with placeholder-but-live content driven off the shared
// EditorContext; SE1 fills Hierarchy + Details with the generated tree/inspector.

#include <string>

#include <imgui.h>

#include "binding.h"
#include "editor/editor_context.h"
#include "editor/plugins.h"
#include "platform/ux/plugin.h"
#include "reflect.h"
#include "types.h"

namespace ps::studio {

namespace reflect = ps::mjcf::reflect;
namespace bridge = ps::mjcf::bridge;

static EditorContext* Ctx(void* data) {
  return static_cast<EditorContext*>(data);
}

// Hierarchy: for SE0, a flat list of the compiled bodies from the Binding (SE1
// replaces this with the authored ProtoSpec worldbody subtree + typed icons).
static void HierarchyUpdate(GuiPlugin* self) {
  EditorContext* c = Ctx(self->data);
  if (!c->model_ready) {
    ImGui::TextUnformatted("No model loaded.");
    return;
  }
  ImGui::Text("Model: %s", c->source_name.c_str());
  ImGui::Separator();
  for (const bridge::Binding::Entry& e : c->compiled.binding.entries()) {
    if (e.etype != ps::mjcf::ElementType::Body) {
      continue;
    }
    const bool selected = e.serial == c->selected_serial;
    ImGui::PushID(static_cast<int>(e.serial));
    if (ImGui::Selectable(e.name.empty() ? "<unnamed body>" : e.name.c_str(),
                          selected)) {
      c->selected_serial = e.serial;
      c->selected_desc = "Body '" + e.name + "' (serial " +
                         std::to_string(e.serial) + ")";
    }
    ImGui::PopID();
  }
}

// Details: the current selection (SE1 replaces with the generated inspector).
static void DetailsUpdate(GuiPlugin* self) {
  EditorContext* c = Ctx(self->data);
  if (c->selected_desc.empty()) {
    ImGui::TextUnformatted("No selection.");
    ImGui::TextDisabled("Double-click a geom in the viewport to select.");
    return;
  }
  ImGui::TextWrapped("%s", c->selected_desc.c_str());
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

static void RegisterGuiPanel(const char* name, GuiPlugin::UpdateFn fn,
                             EditorContext& ctx) {
  GuiPlugin plugin;
  plugin.name = name;
  plugin.update = fn;
  plugin.active = true;  // panels visible by default (SE0 dock layout)
  plugin.data = &ctx;
  RegisterPlugin<GuiPlugin>(plugin);
}

void RegisterEditorPanels(EditorContext& ctx) {
  RegisterGuiPanel("Hierarchy", HierarchyUpdate, ctx);
  RegisterGuiPanel("Details", DetailsUpdate, ctx);
  RegisterGuiPanel("Assets", AssetsUpdate, ctx);
  RegisterGuiPanel("Diagnostics", DiagnosticsUpdate, ctx);
}

}  // namespace ps::studio
