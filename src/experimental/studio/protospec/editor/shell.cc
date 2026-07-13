// ProtoSpec Studio: the editor shell (ps::studio, ours). Contributes the File /
// Edit menus to Studio's real menu bar (MainMenuPlugin), the transform-tool /
// snap / add controls to Studio's top toolbar (ToolbarPlugin), and the Play/Stop
// mode bridge (EditorShellPlugin) the host toolbar drives. All state lives in the
// shared EditorContext; the ops are the same windowless SDK ops the panels use.

#include <string>

#include <imgui.h>
#include <imgui_stdlib.h>

#include "editor/asset_import.h"
#include "editor/authoring_ops.h"
#include "editor/editor_context.h"
#include "editor/editor_ops.h"
#include "editor/plugins.h"
#include "platform/ux/plugin.h"
#include "platform/ux/ps_plugin_ext.h"

namespace ps::studio {
namespace {

namespace mj = ps::mjcf;

EditorContext* Ctx(void* data) { return static_cast<EditorContext*>(data); }

void SaveExternalize(EditorContext* c, const std::string& path) {
  if (SaveModel(*c, path)) ExternalizeVfsAssets(*c, path);
}

// --- File / Edit menus (MainMenuPlugin) ----------------------------------- //

void DrawFileMenu(EditorContext* c) {
  if (!ImGui::BeginMenu("File")) return;
  if (ImGui::MenuItem("New")) NewModelOp(*c);

  // Open / Save As / Import use inline path fields (widgets are legal in menus),
  // so no OS dialog dependency is needed from the editor library.
  if (ImGui::BeginMenu("Open...")) {
    static std::string path;
    ImGui::SetNextItemWidth(280);
    const bool enter = ImGui::InputTextWithHint(
        "##open", "path to MJCF (.xml)", &path,
        ImGuiInputTextFlags_EnterReturnsTrue);
    ImGui::SameLine();
    if ((ImGui::Button("Open") || enter) && !path.empty()) {
      c->pending.Request(path);
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndMenu();
  }
  ImGui::Separator();

  const bool can_save = c->model_ready && !c->source_path.empty();
  if (ImGui::MenuItem("Save", "Ctrl+S", false, can_save)) {
    SaveExternalize(c, c->source_path);
  }
  if (ImGui::BeginMenu("Save As...", c->model_ready)) {
    static std::string path;
    if (path.empty()) path = c->source_path;
    ImGui::SetNextItemWidth(280);
    const bool enter = ImGui::InputTextWithHint(
        "##saveas", "path to write (.xml)", &path,
        ImGuiInputTextFlags_EnterReturnsTrue);
    ImGui::SameLine();
    if ((ImGui::Button("Save") || enter) && !path.empty()) {
      SaveExternalize(c, path);
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndMenu();
  }
  ImGui::Separator();

  if (ImGui::BeginMenu("Import Mesh...", c->model_ready)) {
    static std::string path;
    static std::string status;
    ImGui::SetNextItemWidth(280);
    const bool enter = ImGui::InputTextWithHint(
        "##mesh", "path to .obj / .stl / .msh", &path,
        ImGuiInputTextFlags_EnterReturnsTrue);
    ImGui::SameLine();
    if ((ImGui::Button("Import") || enter) && !path.empty()) {
      MeshImportResult r = ImportMesh(*c, path, nullptr);
      status = r.ok ? ("imported '" + path + "'" +
                       (r.vfs ? " (externalized on save)" : ""))
                    : ("import failed: " + r.error);
    }
    if (!status.empty()) ImGui::TextWrapped("%s", status.c_str());
    ImGui::EndMenu();
  }
  ImGui::EndMenu();
}

void DrawEditMenu(EditorContext* c) {
  if (!ImGui::BeginMenu("Edit")) return;
  if (ImGui::MenuItem("Undo", "Ctrl+Z")) Undo(*c);
  if (ImGui::MenuItem("Redo", "Ctrl+Y")) Redo(*c);
  ImGui::Separator();
  const bool has_sel = c->selected_serial != 0;
  if (ImGui::MenuItem("Duplicate", "Ctrl+D", false, has_sel)) {
    DuplicateOp(*c, c->selected_serial);
  }
  if (ImGui::MenuItem("Delete", "Del", false, has_sel)) {
    c->delete_request_serial = c->selected_serial;  // referrer-confirm flow
  }
  ImGui::EndMenu();
}

void OnMainMenu(MainMenuPlugin* self) {
  EditorContext* c = Ctx(self->data);
  DrawFileMenu(c);
  DrawEditMenu(c);
}

// --- Toolbar tools (ToolbarPlugin) ---------------------------------------- //

void ToolButton(const char* label, GizmoTool tool, GizmoSettings& g,
                const char* tip) {
  const bool active = g.tool == tool;
  if (active) {
    ImGui::PushStyleColor(ImGuiCol_Button,
                          ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
  }
  if (ImGui::Button(label)) g.tool = tool;
  if (active) ImGui::PopStyleColor();
  ImGui::SetItemTooltip("%s", tip);
  ImGui::SameLine();
}

void DrawAddDropdown(EditorContext* c) {
  if (!ImGui::BeginMenu("+ Add")) return;
  const std::uint64_t sel = c->selected_serial;
  if (ImGui::BeginMenu("Body / geom (world)")) {
    if (ImGui::MenuItem("Body")) AddBodyOp(*c, 0);
    if (ImGui::MenuItem("Sphere")) AddGeomOp(*c, 0, mj::GeomType::sphere);
    if (ImGui::MenuItem("Box")) AddGeomOp(*c, 0, mj::GeomType::box);
    if (ImGui::MenuItem("Capsule")) AddGeomOp(*c, 0, mj::GeomType::capsule);
    if (ImGui::MenuItem("Cylinder")) AddGeomOp(*c, 0, mj::GeomType::cylinder);
    ImGui::EndMenu();
  }
  if (ImGui::BeginMenu("Actuator (for selected joint)")) {
    if (ImGui::MenuItem("Motor")) AddActuatorOp(*c, ActuatorSpelling::Motor, sel);
    if (ImGui::MenuItem("Position"))
      AddActuatorOp(*c, ActuatorSpelling::Position, sel);
    if (ImGui::MenuItem("Velocity"))
      AddActuatorOp(*c, ActuatorSpelling::Velocity, sel);
    ImGui::EndMenu();
  }
  if (ImGui::BeginMenu("Sensor")) {
    if (ImGui::MenuItem("Jointpos"))
      AddSensorOp(*c, SensorSpelling::Jointpos, sel);
    if (ImGui::MenuItem("Framepos"))
      AddSensorOp(*c, SensorSpelling::Framepos, sel);
    if (ImGui::MenuItem("Gyro")) AddSensorOp(*c, SensorSpelling::Gyro, sel);
    ImGui::EndMenu();
  }
  if (ImGui::MenuItem("Keyframe")) AddKeyframeOp(*c);
  ImGui::EndMenu();
}

void OnToolbar(ToolbarPlugin* self) {
  EditorContext* c = Ctx(self->data);
  GizmoSettings& g = c->gizmo;

  ImGui::BeginDisabled(!c->model_ready);
  ToolButton("Q", GizmoTool::Select, g, "Select (Q)");
  ToolButton("W", GizmoTool::Translate, g, "Move (W)");
  ToolButton("E", GizmoTool::Rotate, g, "Rotate (E) / joint: reorient axis");
  ToolButton("R", GizmoTool::Scale, g, "Scale (R)");

  // Local / World.
  if (ImGui::Button(g.world_space ? "World" : "Local")) {
    g.world_space = !g.world_space;
  }
  ImGui::SetItemTooltip("%s", "Transform space (Local / World)");
  ImGui::SameLine();

  // Snap toggle + increments popup.
  if (g.snap) {
    ImGui::PushStyleColor(ImGuiCol_Button,
                          ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
  }
  if (ImGui::Button("Snap")) g.snap = !g.snap;
  if (g.snap) ImGui::PopStyleColor();
  if (ImGui::BeginPopupContextItem("##snapcfg",
                                   ImGuiPopupFlags_MouseButtonRight)) {
    ImGui::InputDouble("move (m)", &g.snap_translate, 0.01, 0.1, "%.3f");
    ImGui::InputDouble("rotate (deg)", &g.snap_rotate_deg, 1.0, 5.0, "%.1f");
    ImGui::InputDouble("scale", &g.snap_scale, 0.01, 0.1, "%.3f");
    ImGui::EndPopup();
  }
  ImGui::SetItemTooltip("%s", "Snap on/off (right-click: increments)");
  ImGui::SameLine();

  // Joint overlay: show joints for all bodies (deliverable 3 View toggle).
  if (c->show_all_joints) {
    ImGui::PushStyleColor(ImGuiCol_Button,
                          ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
  }
  if (ImGui::Button("Joints")) c->show_all_joints = !c->show_all_joints;
  if (c->show_all_joints) ImGui::PopStyleColor();
  ImGui::SetItemTooltip("%s", "Show joint overlays for all bodies");
  ImGui::SameLine();

  DrawAddDropdown(c);
  ImGui::EndDisabled();
}

// --- Mode bridge (EditorShellPlugin) -------------------------------------- //

void OnSetMode(EditorShellPlugin* self, int mode) {
  EditorContext* c = Ctx(self->data);
  if (mode == 1) {  // Play: compile pending edits, then the host unpauses.
    c->mode = EditorMode::Play;
    c->play_paused = false;
    if (c->dirty) {
      c->apply_edits = true;
      c->RequestRecompile();
    }
  } else {  // Edit / stop: the host resets physics to qpos0.
    c->mode = EditorMode::Edit;
  }
}

}  // namespace

void RegisterEditorShell(EditorContext& ctx) {
  MainMenuPlugin menu;
  menu.name = "ProtoSpec Menus";
  menu.draw = OnMainMenu;
  menu.data = &ctx;
  RegisterPlugin<MainMenuPlugin>(menu);

  ToolbarPlugin bar;
  bar.name = "ProtoSpec Tools";
  bar.draw = OnToolbar;
  bar.data = &ctx;
  RegisterPlugin<ToolbarPlugin>(bar);

  EditorShellPlugin shell;
  shell.name = "ProtoSpec Mode";
  shell.set_mode = OnSetMode;
  shell.is_dirty = [](EditorShellPlugin* self) { return Ctx(self->data)->dirty; };
  shell.data = &ctx;
  RegisterPlugin<EditorShellPlugin>(shell);
}

}  // namespace ps::studio
