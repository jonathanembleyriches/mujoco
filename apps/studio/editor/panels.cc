// ProtoSpec Studio: the remaining editor Gui panels + commands (ps::studio,
// ours). Hierarchy lives in hierarchy_panel.cc and the generated Details panel in
// details_panel.cc (SE1b); this TU keeps Assets / Diagnostics, the File menu
// (load / save), and the Ctrl+Z / Ctrl+Y / Ctrl+S key handlers.

#include <array>
#include <string>
#include <variant>
#include <vector>

#include <imgui.h>
#include <imgui_stdlib.h>

#include "editor/asset_import.h"
#include "editor/authoring_ops.h"
#include "editor/editor_context.h"
#include "editor/editor_ops.h"
#include "editor/plugins.h"
#include "platform/ux/plugin.h"

namespace ps::studio {

namespace mj = ps::mjcf;

static EditorContext* Ctx(void* data) {
  return static_cast<EditorContext*>(data);
}

// Every material / texture in the model, in document order (across asset blocks).
static std::vector<mj::Material*> AllMaterials(mj::Model& tree) {
  std::vector<mj::Material*> out;
  for (auto& a : tree.assets) {
    if (!a) continue;
    for (auto& m : a->materials) {
      if (m) out.push_back(m.get());
    }
  }
  return out;
}
static std::vector<mj::Texture*> AllTextures(mj::Model& tree) {
  std::vector<mj::Texture*> out;
  for (auto& a : tree.assets) {
    if (!a) continue;
    for (auto& t : a->textures) {
      if (t) out.push_back(t.get());
    }
  }
  return out;
}

static const char* kBuiltinLabels[] = {"none", "gradient", "checker", "flat"};

// A small rgba colour square (the browsable-list swatch).
static void ColorSwatch(const char* id, const std::array<float, 4>& c) {
  ImGui::ColorButton(id, ImVec4(c[0], c[1], c[2], c[3]),
                     ImGuiColorEditFlags_AlphaPreviewHalf, ImVec2(16, 16));
}

// The "New Material" modal: name + rgba + PBR sliders + a texture-role dropdown
// that assigns an existing texture to the material's rgb role.
static void NewMaterialModal(EditorContext* c) {
  if (!ImGui::BeginPopupModal("New Material", nullptr,
                              ImGuiWindowFlags_AlwaysAutoResize)) {
    return;
  }
  static MaterialSpec spec;
  ImGui::SetNextItemWidth(200);
  ImGui::InputText("name", &spec.name);
  ImGui::ColorEdit4("rgba", spec.rgba.data(),
                    ImGuiColorEditFlags_AlphaBar |
                        ImGuiColorEditFlags_AlphaPreviewHalf);
  ImGui::SliderFloat("specular", &spec.specular, 0.0f, 1.0f);
  ImGui::SliderFloat("shininess", &spec.shininess, 0.0f, 1.0f);
  ImGui::SliderFloat("reflectance", &spec.reflectance, 0.0f, 1.0f);
  ImGui::SliderFloat("metallic", &spec.metallic, 0.0f, 1.0f);
  ImGui::SliderFloat("roughness", &spec.roughness, 0.0f, 1.0f);

  // Texture-role dropdown: assign an existing texture to the rgb role.
  const std::vector<mj::Texture*> texs = AllTextures(*c->tree);
  const std::string preview =
      spec.texture_rgb.empty() ? "(none)" : spec.texture_rgb;
  if (ImGui::BeginCombo("rgb texture", preview.c_str())) {
    if (ImGui::Selectable("(none)", spec.texture_rgb.empty())) {
      spec.texture_rgb.clear();
    }
    for (mj::Texture* t : texs) {
      const std::string tn = t->name ? *t->name : std::string();
      if (tn.empty()) continue;
      if (ImGui::Selectable(tn.c_str(), tn == spec.texture_rgb)) {
        spec.texture_rgb = tn;
      }
    }
    ImGui::EndCombo();
  }

  ImGui::Separator();
  const bool valid = MaterialSpecValid(spec);
  ImGui::BeginDisabled(!valid);
  if (ImGui::Button("Create")) {
    CreateMaterialOp(*c, spec);
    spec = MaterialSpec{};
    ImGui::CloseCurrentPopup();
  }
  ImGui::EndDisabled();
  ImGui::SameLine();
  if (ImGui::Button("Cancel")) ImGui::CloseCurrentPopup();
  if (!valid) ImGui::TextDisabled("Enter a name to create.");
  ImGui::EndPopup();
}

// The "New Texture" modal: builtin (checker/gradient/flat) with rgb1/rgb2/markrgb
// colours + width/height, OR a file path.
static void NewTextureModal(EditorContext* c) {
  if (!ImGui::BeginPopupModal("New Texture", nullptr,
                              ImGuiWindowFlags_AlwaysAutoResize)) {
    return;
  }
  static TextureSpec spec;
  static float rgb1[3] = {0.8f, 0.8f, 0.8f};
  static float rgb2[3] = {0.2f, 0.2f, 0.2f};
  static float markrgb[3] = {0, 0, 0};
  ImGui::SetNextItemWidth(200);
  ImGui::InputText("name", &spec.name);

  ImGui::RadioButton("builtin", spec.builtin);
  if (ImGui::IsItemClicked()) spec.builtin = true;
  ImGui::SameLine();
  ImGui::RadioButton("file", !spec.builtin);
  if (ImGui::IsItemClicked()) spec.builtin = false;

  if (spec.builtin) {
    int bi = static_cast<int>(spec.builtin_type);
    ImGui::SetNextItemWidth(140);
    if (ImGui::Combo("pattern", &bi, kBuiltinLabels,
                     IM_ARRAYSIZE(kBuiltinLabels))) {
      spec.builtin_type = static_cast<mj::TextureBuiltin>(bi);
    }
    ImGui::ColorEdit3("rgb1", rgb1);
    ImGui::ColorEdit3("rgb2", rgb2);
    ImGui::ColorEdit3("markrgb", markrgb);
    ImGui::SetNextItemWidth(90);
    ImGui::InputInt("width", &spec.width);
    ImGui::SetNextItemWidth(90);
    ImGui::InputInt("height", &spec.height);
  } else {
    ImGui::SetNextItemWidth(280);
    ImGui::InputTextWithHint("file", "path to image (.png)...", &spec.file);
  }

  ImGui::Separator();
  const bool valid = TextureSpecValid(spec);
  ImGui::BeginDisabled(!valid);
  if (ImGui::Button("Create")) {
    for (int i = 0; i < 3; ++i) {
      spec.rgb1[i] = rgb1[i];
      spec.rgb2[i] = rgb2[i];
      spec.markrgb[i] = markrgb[i];
    }
    CreateTextureOp(*c, spec);
    spec = TextureSpec{};
    ImGui::CloseCurrentPopup();
  }
  ImGui::EndDisabled();
  ImGui::SameLine();
  if (ImGui::Button("Cancel")) ImGui::CloseCurrentPopup();
  if (spec.name.empty()) {
    ImGui::TextDisabled("Enter a name to create.");
  } else if (spec.builtin && (spec.width < 1 || spec.height < 1)) {
    ImGui::TextDisabled("Width and height must be at least 1.");
  } else if (!spec.builtin && spec.file.empty()) {
    ImGui::TextDisabled("Enter an image path to create.");
  }
  ImGui::EndPopup();
}

// A shared "Import Mesh..." control: a path input (no OS dialog dependency in the
// vendored shell) that imports the file and auto-builds a mesh geom at the origin.
static void ImportMeshControl(EditorContext* c) {
  static std::string mesh_path;
  static std::string import_status;
  ImGui::SetNextItemWidth(-1);
  const bool enter = ImGui::InputTextWithHint(
      "##meshpath", "path to .obj / .stl / .msh...", &mesh_path,
      ImGuiInputTextFlags_EnterReturnsTrue);
  ImGui::BeginDisabled(!c->model_ready || mesh_path.empty());
  if ((ImGui::Button("Import Mesh...") || enter) && !mesh_path.empty()) {
    MeshImportResult r = ImportMesh(*c, mesh_path, nullptr);
    import_status = r.ok ? ("imported '" + mesh_path + "'" +
                            (r.vfs ? " (in-memory; externalized on save)" : ""))
                         : ("import failed: " + r.error);
  }
  ImGui::EndDisabled();
  if (!import_status.empty()) ImGui::TextWrapped("%s", import_status.c_str());
}

// Assets: meshes / textures / materials, with creation + a browsable swatch list.
static void AssetsUpdate(GuiPlugin* self) {
  EditorContext* c = Ctx(self->data);
  ImGui::TextDisabled("Assets (meshes / textures / materials / hfields)");
  ImGui::Separator();
  if (!c->tree) {
    ImGui::TextDisabled("No model loaded.");
    return;
  }

  // --- Mesh import (single inline + native multi / folder dialogs) --------- //
  ImportMeshControl(c);
  ImGui::BeginDisabled(!c->model_ready);
  if (ImGui::Button("Import Meshes...")) {
    c->file_dialog.Request(FileDialogState::Kind::ImportMeshes, c->base_dir);
  }
  ImGui::SameLine();
  if (ImGui::Button("Import Folder...")) {
    c->file_dialog.Request(FileDialogState::Kind::ImportFolder, c->base_dir);
  }
  ImGui::EndDisabled();
  ImGui::Separator();

  // --- Material / texture creation ---------------------------------------- //
  if (ImGui::Button("New Material")) ImGui::OpenPopup("New Material");
  ImGui::SameLine();
  if (ImGui::Button("New Texture")) ImGui::OpenPopup("New Texture");
  NewMaterialModal(c);
  NewTextureModal(c);
  ImGui::Separator();

  // --- Browsable materials list (rgba swatch + name) ---------------------- //
  const std::vector<mj::Material*> mats = AllMaterials(*c->tree);
  ImGui::Text("Materials (%zu)", mats.size());
  for (mj::Material* m : mats) {
    ImGui::PushID(static_cast<int>(m->serial));
    std::array<float, 4> col{0.8f, 0.8f, 0.8f, 1.0f};
    if (m->rgba) col = *m->rgba;
    ColorSwatch("##matsw", col);
    ImGui::SameLine();
    const std::string name = m->name ? *m->name : ("material#" +
                                std::to_string(m->serial));
    if (ImGui::Selectable(name.c_str(), c->selected_serial == m->serial)) {
      SelectBySerial(*c, m->serial);
    }
    ImGui::PopID();
  }
  ImGui::Spacing();

  // --- Browsable textures list (builtin preview / file badge) ------------- //
  const std::vector<mj::Texture*> texs = AllTextures(*c->tree);
  ImGui::Text("Textures (%zu)", texs.size());
  for (mj::Texture* t : texs) {
    ImGui::PushID(static_cast<int>(t->serial));
    // A builtin texture previews as its rgb1 swatch; a file texture as a marker.
    std::array<float, 4> col{0.5f, 0.5f, 0.5f, 1.0f};
    const char* kind = "file";
    if (t->source && std::holds_alternative<mj::TextureBuiltin>(*t->source)) {
      kind = kBuiltinLabels[static_cast<int>(
          std::get<mj::TextureBuiltin>(*t->source))];
      if (t->rgb1) {
        col = {static_cast<float>((*t->rgb1)[0]),
               static_cast<float>((*t->rgb1)[1]),
               static_cast<float>((*t->rgb1)[2]), 1.0f};
      }
    }
    ColorSwatch("##texsw", col);
    ImGui::SameLine();
    const std::string name = t->name ? *t->name : ("texture#" +
                                std::to_string(t->serial));
    const std::string label = name + "  [" + kind + "]";
    if (ImGui::Selectable(label.c_str(), c->selected_serial == t->serial)) {
      SelectBySerial(*c, t->serial);
    }
    ImGui::PopID();
  }
}

// The "+ Add" panel: model-level structural adds (deliverable 1). Target-aware --
// an actuator/sensor wires to the current selection when it is a valid target.
static void AddMenuUpdate(GuiPlugin* self) {
  EditorContext* c = Ctx(self->data);
  if (!c->tree) {
    ImGui::TextDisabled("No model loaded.");
    return;
  }
  const std::uint64_t sel = c->selected_serial;
  ImGui::TextDisabled("Model-level elements");
  ImGui::Separator();

  if (ImGui::BeginMenu("Actuator")) {
    struct A { const char* label; ActuatorSpelling sp; };
    const A acts[] = {{"Motor", ActuatorSpelling::Motor},
                      {"Position", ActuatorSpelling::Position},
                      {"Velocity", ActuatorSpelling::Velocity},
                      {"IntVelocity", ActuatorSpelling::IntVelocity},
                      {"Damper", ActuatorSpelling::Damper},
                      {"Cylinder", ActuatorSpelling::Cylinder},
                      {"Muscle", ActuatorSpelling::Muscle},
                      {"Adhesion", ActuatorSpelling::Adhesion},
                      {"DcMotor", ActuatorSpelling::DcMotor},
                      {"General", ActuatorSpelling::General}};
    for (const A& a : acts) {
      if (ImGui::MenuItem(a.label)) AddActuatorOp(*c, a.sp, sel);
    }
    ImGui::EndMenu();
  }
  if (ImGui::BeginMenu("Sensor")) {
    struct S { const char* label; SensorSpelling sp; };
    const S sens[] = {{"Jointpos", SensorSpelling::Jointpos},
                      {"Jointvel", SensorSpelling::Jointvel},
                      {"Framepos", SensorSpelling::Framepos},
                      {"Framequat", SensorSpelling::Framequat},
                      {"Gyro", SensorSpelling::Gyro},
                      {"Accelerometer", SensorSpelling::Accelerometer},
                      {"Velocimeter", SensorSpelling::Velocimeter},
                      {"Force", SensorSpelling::Force},
                      {"Torque", SensorSpelling::Torque},
                      {"Touch", SensorSpelling::Touch}};
    for (const S& s : sens) {
      if (ImGui::MenuItem(s.label)) AddSensorOp(*c, s.sp, sel);
    }
    ImGui::EndMenu();
  }
  if (ImGui::MenuItem("Tendon (fixed)")) AddTendonOp(*c);
  if (ImGui::MenuItem("Equality (weld)")) AddEqualityWeldOp(*c);
  if (ImGui::MenuItem("Contact Pair")) AddPairOp(*c);
  if (ImGui::MenuItem("Contact Exclude")) AddExcludeOp(*c);
  if (ImGui::MenuItem("Keyframe")) AddKeyframeOp(*c);
  static std::string class_name;
  ImGui::Separator();
  ImGui::SetNextItemWidth(120);
  ImGui::InputTextWithHint("##class", "class name", &class_name);
  ImGui::SameLine();
  if (ImGui::Button("Add Default Class")) {
    AddDefaultClassOp(*c, class_name);
    class_name.clear();
  }
}

// Diagnostics: live Validate/Compile log + status line (§3). Rows are severity
// coloured; a row that carries a serial is clickable and selects the element it
// blames (SelectBySerial -> Hierarchy/viewport highlight); a row that carries a
// SourceLoc shows its file:line origin.
static ImVec4 SeverityColor(DiagEntry::Severity s) {
  switch (s) {
    case DiagEntry::Severity::Error:
      return ImVec4(0.95f, 0.45f, 0.42f, 1.0f);  // red
    case DiagEntry::Severity::Warning:
      return ImVec4(0.95f, 0.78f, 0.35f, 1.0f);  // amber
    case DiagEntry::Severity::Info:
      break;
  }
  return ImGui::GetStyleColorVec4(ImGuiCol_Text);
}

static void DiagnosticsUpdate(GuiPlugin* self) {
  EditorContext* c = Ctx(self->data);
  // The status-bar error chip focuses this panel by posting a request the panel
  // honours from inside its own window (one-shot).
  if (c->focus_diagnostics_request) {
    ImGui::SetWindowFocus();
    c->focus_diagnostics_request = false;
  }
  if (!c->status_line.empty()) {
    ImGui::TextWrapped("%s", c->status_line.c_str());
    ImGui::SameLine();
  }
  ImGui::BeginDisabled(c->diagnostics.empty());
  if (ImGui::SmallButton("Clear")) {
    c->ClearDiagnostics();
  }
  ImGui::EndDisabled();
  ImGui::Separator();

  ImGui::BeginChild("##diag_log");
  int row = 0;
  for (const DiagEntry& d : c->diagnostics) {
    ImGui::PushID(row++);
    std::string label;
    if (d.loc) {
      label = d.loc->file;
      if (d.loc->line > 0) label += ":" + std::to_string(d.loc->line);
      label += ": ";
    }
    label += d.message;

    ImGui::PushStyleColor(ImGuiCol_Text, SeverityColor(d.severity));
    // Only serial-bearing rows are actionable; the rest render as inert text so
    // a stray click never changes the selection.
    if (d.serial) {
      if (ImGui::Selectable(label.c_str())) {
        SelectBySerial(*c, *d.serial);
      }
    } else {
      ImGui::TextUnformatted(label.c_str());
    }
    ImGui::PopStyleColor();
    ImGui::PopID();
  }
  if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY()) {
    ImGui::SetScrollHereY(1.0f);
  }
  ImGui::EndChild();
}

// File menu: a simple path input for load / save (no vendored file dialog in
// SE0). Lives in the main menu bar; drag-drop and the CLI arg still feed loads.
// Save `c` to `path` and externalize any in-memory (imported) assets beside it.
static void SaveAndExternalize(EditorContext* c, const std::string& path) {
  if (SaveModel(*c, path)) {
    ExternalizeVfsAssets(*c, path);
  }
}

static void FileMenuUpdate(GuiPlugin* self) {
  EditorContext* c = Ctx(self->data);
  static std::string load_path;
  static std::string save_path;

  if (ImGui::Button("New")) {
    NewModelOp(*c);
  }
  ImGui::SameLine();
  ImGui::TextDisabled("empty starter scene (ground + light)");
  ImGui::Separator();

  ImGui::TextUnformatted("Import mesh (auto-builds a mesh geom):");
  ImportMeshControl(c);
  ImGui::Separator();

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
    SaveAndExternalize(c, c->source_path);
  }
  ImGui::EndDisabled();
  ImGui::SameLine();
  ImGui::BeginDisabled(!can_save);
  if (ImGui::Button("Save As")) {
    SaveAndExternalize(c, save_path);
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
  // Two-panel default (deliverable 3): only Hierarchy + Details stand open (plus
  // the central Viewport). Assets folds into the Hierarchy's asset section /
  // "+ Asset" creation; Diagnostics folds into the status-bar error chip (click
  // brings this panel forward). Both stay registered so the Plugins/View menu can
  // still toggle them for power users -- they just no longer dock open by default.
  RegisterGuiPanel("Assets", AssetsUpdate, false, ctx);
  RegisterGuiPanel("Diagnostics", DiagnosticsUpdate, false, ctx);
  RegisterGuiPanel("File", FileMenuUpdate, false, ctx);
  RegisterGuiPanel("+ Add", AddMenuUpdate, false, ctx);

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
          SaveAndExternalize(c, c->source_path);
        }
      },
      ctx);
}

}  // namespace ps::studio
