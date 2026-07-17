// ProtoSpec Studio: the editor shell (ps::studio, ours). Contributes the File /
// Edit menus to Studio's real menu bar (MainMenuPlugin), the transform-tool /
// snap / add controls to Studio's top toolbar (ToolbarPlugin), and the Play/Stop
// mode bridge (EditorShellPlugin) the host toolbar drives. All state lives in the
// shared EditorContext; the ops are the same windowless SDK ops the panels use.

#include <cstdio>
#include <string>
#include <vector>

#include <imgui.h>
#include <imgui_stdlib.h>

#include "editor/asset_import.h"
#include "editor/authoring_ops.h"
#include "editor/editor_context.h"
#include "editor/editor_ops.h"
#include "editor/mode_ui.h"
#include "host/shell.h"
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

// A path field usable inside a submenu as the fallback when no native dialog is
// available (or a user prefers typing). Returns true with `out` on submit.
bool InlinePathField(const char* id, const char* hint, std::string* out) {
  ImGui::SetNextItemWidth(280);
  const bool enter = ImGui::InputTextWithHint(
      id, hint, out, ImGuiInputTextFlags_EnterReturnsTrue);
  ImGui::SameLine();
  return (ImGui::Button("Go") || enter) && !out->empty();
}

void DrawFileMenu(EditorContext* c) {
  if (!ImGui::BeginMenu("File")) return;
  if (ImGui::MenuItem("New", "Ctrl+N")) NewModelOp(*c);
  ImGui::Separator();

  // Open / Save As / Import first offer the host's native OS dialog; an inline
  // path field remains as a submenu fallback (headless builds, quick typing).
  if (ImGui::MenuItem("Open...", "Ctrl+O")) {
    c->file_dialog.Request(FileDialogState::Kind::Open, c->source_path);
  }
  if (ImGui::BeginMenu("Open path...")) {
    static std::string path;
    if (InlinePathField("##open", "path to MJCF (.xml)", &path)) {
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
  if (ImGui::MenuItem("Save As...", "Ctrl+Shift+S", false, c->model_ready)) {
    c->file_dialog.Request(FileDialogState::Kind::SaveAs, c->source_path);
  }
  if (ImGui::BeginMenu("Save As path...", c->model_ready)) {
    static std::string path;
    if (path.empty()) path = c->source_path;
    if (InlinePathField("##saveas", "path to write (.xml)", &path)) {
      SaveExternalize(c, path);
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndMenu();
  }
  ImGui::Separator();

  if (ImGui::MenuItem("Import Mesh...", nullptr, false, c->model_ready)) {
    c->file_dialog.Request(FileDialogState::Kind::ImportMesh, c->base_dir);
  }
  if (ImGui::MenuItem("Import Meshes...", nullptr, false, c->model_ready)) {
    c->file_dialog.Request(FileDialogState::Kind::ImportMeshes, c->base_dir);
  }
  if (ImGui::MenuItem("Import Folder...", nullptr, false, c->model_ready)) {
    c->file_dialog.Request(FileDialogState::Kind::ImportFolder, c->base_dir);
  }
  if (ImGui::BeginMenu("Import Mesh path...", c->model_ready)) {
    static std::string path;
    static std::string status;
    if (InlinePathField("##mesh", "path to .obj / .stl / .msh", &path)) {
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

// Dispatches the outcome of a native file dialog the host serviced. Runs once per
// delivered result, driven from the toolbar hook each frame.
void DrainFileDialog(EditorContext* c) {
  if (!c->file_dialog.HasResult()) return;
  const FileDialogState::Result r = c->file_dialog.TakeResult();
  if (!r.accepted || r.path.empty()) return;
  switch (r.kind) {
    case FileDialogState::Kind::Open:
      c->pending.Request(r.path);
      break;
    case FileDialogState::Kind::SaveAs:
      SaveExternalize(c, r.path);
      break;
    case FileDialogState::Kind::ImportMesh: {
      MeshImportResult m = ImportMesh(*c, r.path, nullptr);
      c->status_toast.Post(
          m.ok ? "imported mesh" : ("mesh import failed: " + m.error),
          m.ok ? StatusToast::Kind::Info : StatusToast::Kind::Error,
          ImGui::GetTime());
      break;
    }
    case FileDialogState::Kind::ImportMeshes: {
      // The host joined the multi-selection with '\n'.
      std::vector<std::string> paths;
      std::size_t start = 0;
      while (start <= r.path.size()) {
        const std::size_t nl = r.path.find('\n', start);
        const std::string p =
            r.path.substr(start, nl == std::string::npos ? std::string::npos
                                                         : nl - start);
        if (!p.empty()) paths.push_back(p);
        if (nl == std::string::npos) break;
        start = nl + 1;
      }
      MultiMeshImportResult m = ImportMeshes(*c, paths);
      c->status_toast.Post(
          m.ok ? ("imported " + std::to_string(m.imported) + " mesh(es)" +
                  (m.skipped ? ", " + std::to_string(m.skipped) + " skipped"
                             : ""))
               : ("mesh import failed: " + m.error),
          m.ok ? StatusToast::Kind::Info : StatusToast::Kind::Error,
          ImGui::GetTime());
      break;
    }
    case FileDialogState::Kind::ImportFolder: {
      MultiMeshImportResult m = ImportMeshFolder(*c, r.path);
      c->status_toast.Post(
          m.ok ? ("imported " + std::to_string(m.imported) + " mesh(es) from folder")
               : ("folder import failed: " + m.error),
          m.ok ? StatusToast::Kind::Info : StatusToast::Kind::Error,
          ImGui::GetTime());
      break;
    }
    case FileDialogState::Kind::None:
      break;
  }
}

void DrawEditMenu(EditorContext* c) {
  if (!ImGui::BeginMenu("Edit")) return;
  if (ImGui::MenuItem("Undo", "Ctrl+Z", false, c->history.can_undo())) Undo(*c);
  if (ImGui::MenuItem("Redo", "Ctrl+Y", false, c->history.can_redo())) Redo(*c);
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

// Transform-tool icons. These FontAwesome glyphs live in [U+F000, U+F3FF], the
// range the host merges into its UI font, so they rasterize (no fallback boxes).
constexpr const char* kIconSelect = "\xEF\x89\x85";  // mouse-pointer  U+F245
constexpr const char* kIconMove = "\xEF\x81\x87";    // arrows         U+F047
constexpr const char* kIconRotate = "\xEF\x80\x9E";  // repeat         U+F01E
constexpr const char* kIconScale = "\xEF\x82\xB2";   // arrows-alt     U+F0B2

// An icon toggle for a transform tool: highlighted when active, with a tooltip
// carrying the tool name and its keyboard shortcut. `id` keeps the ImGui id
// stable and unique independent of the glyph.
void ToolButton(const char* icon, const char* id, GizmoTool tool,
                GizmoSettings& g, const char* tip) {
  const bool active = g.tool == tool;
  if (active) {
    ImGui::PushStyleColor(ImGuiCol_Button,
                          ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
  }
  ImGui::PushID(id);
  if (ImGui::Button(icon)) g.tool = tool;
  ImGui::PopID();
  if (active) ImGui::PopStyleColor();
  ImGui::SetItemTooltip("%s", tip);
  ImGui::SameLine();
}

void DrawAddDropdown(EditorContext* c) {
  if (!ImGui::BeginMenu("+ Add")) return;
  const std::uint64_t sel = c->selected_serial;
  if (ImGui::BeginMenu("Body / geom (world)")) {
    // Batch count: N>1 drops a row of geoms in one undo entry (single-add default).
    static int count = 1;
    ImGui::SetNextItemWidth(90);
    ImGui::InputInt("count", &count);
    if (count < 1) count = 1;
    if (count > 64) count = 64;
    ImGui::Separator();
    if (ImGui::MenuItem("Body")) AddBodyOp(*c, 0);
    if (ImGui::MenuItem("Sphere")) AddGeomsOp(*c, 0, mj::GeomType::sphere, count);
    if (ImGui::MenuItem("Box")) AddGeomsOp(*c, 0, mj::GeomType::box, count);
    if (ImGui::MenuItem("Capsule"))
      AddGeomsOp(*c, 0, mj::GeomType::capsule, count);
    if (ImGui::MenuItem("Cylinder"))
      AddGeomsOp(*c, 0, mj::GeomType::cylinder, count);
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

  // The toolbar draws every frame, so it is the per-frame hook that dispatches a
  // native-dialog result the host delivered since the last frame.
  DrainFileDialog(c);

  ImGui::BeginDisabled(!c->model_ready);
  ToolButton(kIconSelect, "tool_select", GizmoTool::Select, g, "Select (Q)");
  ToolButton(kIconMove, "tool_move", GizmoTool::Translate, g, "Move (W)");
  ToolButton(kIconRotate, "tool_rotate", GizmoTool::Rotate, g,
             "Rotate (E) / joint: reorient axis");
  ToolButton(kIconScale, "tool_scale", GizmoTool::Scale, g, "Scale (R)");

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

  // Outside the disabled block: which mode we are in is exactly what a user
  // with no model loaded still needs to read.
  DrawModeChip(*c);
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

// Physics run/pause, pushed by the host every frame. Edit and paused-Play are
// both "paused", so the pause bit alone cannot tell them apart -- Play/Stop
// latch which of the two we are in. But a sim that is *advancing* is Play by
// definition, however it was started (Space bypasses set_mode entirely), so
// that case re-latches the mode rather than trusting the latch.
void OnSetPaused(EditorShellPlugin* self, bool paused) {
  EditorContext* c = Ctx(self->data);
  if (!paused) {
    c->mode = EditorMode::Play;
    c->play_paused = false;
  } else if (c->mode == EditorMode::Play) {
    c->play_paused = true;
  }
}

// --- Native file-dialog bridge (FileDialogPlugin) ------------------------- //
// The editor posts a request into c->file_dialog; the host polls it here, opens
// the matching OS dialog, and delivers the outcome back for DrainFileDialog.

int FileDialogPoll(FileDialogPlugin* self, char* hint, int hint_size) {
  EditorContext* c = Ctx(self->data);
  const FileDialogState::Kind k = c->file_dialog.Poll();
  if (k == FileDialogState::Kind::None) return 0;
  if (hint && hint_size > 0) {
    std::snprintf(hint, hint_size, "%s", c->file_dialog.start_hint().c_str());
  }
  return static_cast<int>(k);
}

void FileDialogDeliver(FileDialogPlugin* self, int, const char* path,
                       bool accepted) {
  Ctx(self->data)->file_dialog.Deliver(path ? path : "", accepted);
}

bool FileDialogIsSave(FileDialogPlugin*, int kind) {
  return kind == static_cast<int>(FileDialogState::Kind::SaveAs);
}

bool FileDialogIsMulti(FileDialogPlugin*, int kind) {
  return kind == static_cast<int>(FileDialogState::Kind::ImportMeshes);
}

bool FileDialogIsFolder(FileDialogPlugin*, int kind) {
  return kind == static_cast<int>(FileDialogState::Kind::ImportFolder);
}

}  // namespace

void RegisterEditorShell(EditorContext& ctx) {
  MainMenuPlugin menu;
  menu.name = "ProtoSpec Menus";
  menu.draw = OnMainMenu;
  menu.data = &ctx;
  RegisterPlugin<MainMenuPlugin>(menu);

  FileDialogPlugin fd;
  fd.name = "ProtoSpec File Dialog";
  fd.poll = FileDialogPoll;
  fd.deliver = FileDialogDeliver;
  fd.is_save = FileDialogIsSave;
  fd.is_multi = FileDialogIsMulti;
  fd.is_folder = FileDialogIsFolder;
  fd.data = &ctx;
  RegisterPlugin<FileDialogPlugin>(fd);

  ToolbarPlugin bar;
  bar.name = "ProtoSpec Tools";
  bar.draw = OnToolbar;
  bar.data = &ctx;
  RegisterPlugin<ToolbarPlugin>(bar);

  EditorShellPlugin shell;
  shell.name = "ProtoSpec Mode";
  shell.set_mode = OnSetMode;
  shell.set_paused = OnSetPaused;
  shell.is_dirty = [](EditorShellPlugin* self) { return Ctx(self->data)->dirty; };
  shell.error_count = [](EditorShellPlugin* self) {
    return DiagnosticErrorCount(Ctx(self->data)->diagnostics);
  };
  shell.focus_diagnostics = [](EditorShellPlugin* self) {
    Ctx(self->data)->focus_diagnostics_request = true;
  };
  shell.data = &ctx;
  RegisterPlugin<EditorShellPlugin>(shell);
}

}  // namespace ps::studio
