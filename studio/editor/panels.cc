// ProtoSpec Studio: the remaining editor Gui panels + commands (ps::studio,
// ours). Hierarchy lives in hierarchy_panel.cc and the generated Details panel in
// details_panel.cc (SE1b); this TU keeps Assets / Diagnostics, the File menu
// (load / save), and the Ctrl+Z / Ctrl+Y / Ctrl+S key handlers.

#include <array>
#include <cstdio>
#include <string>
#include <variant>
#include <vector>

#include <imgui.h>
#include <imgui_stdlib.h>

#include "editor/asset_import.h"
#include "editor/authoring_ops.h"
#include "editor/editor_context.h"
#include "editor/editor_ops.h"
#include "editor/layers.h"
#include "editor/plugin_abi.h"
#include "editor/plugins.h"
#include "platform/ux/plugin.h"

namespace ps::studio {

namespace mj = ps::mjcf;

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

// Layers: provenance tags over the ONE authored tree. A row is a distinct
// loc.file key; enabled rows stay in the compile input, the active row is the
// edit scope. Rows connected by cross-layer references render grouped, and a
// depended-on layer's enable toggle locks while any enabled layer needs it.
struct LayersUiState {
  std::string load_path;
  std::string export_path;
  std::string status;
  int confirm_remove = -1;   // layer index awaiting the delete-elements confirm
  int confirm_count = 0;
  bool open_confirm = false;
};

static void DrawLayerRow(EditorContext* c, int i, int& activate, int& move_idx,
                         int& move_delta, int& remove_req) {
  Layer& l = c->layers[i];
  ImGui::PushID(i);

  // Enabled toggle, locked while an enabled layer depends on this one.
  const LayerLock lock = LayerLockInfo(*c, i);
  ImGui::BeginDisabled(lock.locked);
  if (ImGui::Checkbox("##on", &l.enabled)) c->RequestRecompile();
  ImGui::EndDisabled();
  if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
    if (lock.locked) {
      ImGui::SetTooltip("Locked: %s depend%s on this layer\ne.g. %s",
                        lock.dependents.c_str(),
                        lock.dependents.find(',') == std::string::npos ? "s" : "",
                        lock.example.c_str());
    } else {
      ImGui::SetTooltip("Include this layer in the compiled model");
    }
  }
  ImGui::SameLine();

  // The edit scope. Radio: exactly one layer is active.
  if (ImGui::RadioButton("##active", c->active_layer == i)) activate = i;
  ImGui::SetItemTooltip("%s", "Author into this layer (edit scope)");
  ImGui::SameLine();

  ImGui::TextUnformatted(l.name.c_str());
  ImGui::SetItemTooltip("%s", l.key.c_str());

  // Structural containment annotation: this layer's content sits inside
  // another layer's element (nested <include>). Distinct from the reference
  // grouping -- and it has teeth: disabling the CONTAINER prunes this layer's
  // nested content with it, whatever this row's own toggle says.
  const std::vector<int>& inside = c->layer_graph.inside;
  if (i < static_cast<int>(inside.size()) && inside[i] >= 0 &&
      inside[i] < static_cast<int>(c->layers.size())) {
    const Layer& container = c->layers[inside[i]];
    ImGui::SameLine();
    if (!container.enabled) {
      ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.2f, 1.0f),
                         "(pruned with '%s')", container.name.c_str());
    } else {
      ImGui::TextDisabled("(in %s)", container.name.c_str());
    }
    if (ImGui::IsItemHovered() &&
        i < static_cast<int>(c->layer_graph.inside_example.size())) {
      ImGui::SetTooltip(
          "This layer's content lives inside '%s':\n%s\nDisabling '%s' prunes "
          "it too.",
          container.name.c_str(),
          c->layer_graph.inside_example[i].c_str(), container.name.c_str());
    }
  }

  // Order / remove controls, right-aligned-ish.
  ImGui::SameLine();
  ImGui::BeginDisabled(i == 0);
  if (ImGui::SmallButton("^")) { move_idx = i; move_delta = -1; }
  ImGui::EndDisabled();
  ImGui::SameLine();
  ImGui::BeginDisabled(i + 1 >= static_cast<int>(c->layers.size()));
  if (ImGui::SmallButton("v")) { move_idx = i; move_delta = 1; }
  ImGui::EndDisabled();
  ImGui::SameLine();
  ImGui::BeginDisabled(c->layers.size() <= 1);
  if (ImGui::SmallButton("x")) remove_req = i;
  ImGui::EndDisabled();
  ImGui::SetItemTooltip("%s", "Remove this layer (its elements are deleted)");

  ImGui::PopID();
}

static void LayersUpdate(GuiPlugin* self) {
  EditorContext* c = ctx_cast<EditorContext>(self);
  static LayersUiState st;

  ImGui::TextDisabled("Layers -- provenance tags over one tree");
  ImGui::Separator();

  if (c->layers.empty()) {
    ImGui::TextDisabled("No model loaded.");
    return;
  }

  const int n = static_cast<int>(c->layers.size());
  int activate = -1, move_idx = -1, move_delta = 0, remove_req = -1;

  // Grouped rendering: layers connected by dependency edges draw together
  // inside a frame; independents draw standalone. Display order is preserved
  // inside and across groups (a group sits where its first member sits).
  // Contained layers (graph.inside) draw indented directly under their
  // container so "partial is part of nested" reads at a glance; a containment
  // cycle (possible with multi-region layers) falls back to flat rows via the
  // final sweep.
  const std::vector<int>& comp = c->layer_graph.group;
  const std::vector<int>& inside = c->layer_graph.inside;
  const bool comp_ok = static_cast<int>(comp.size()) == n;
  const bool inside_ok = static_cast<int>(inside.size()) == n;
  std::vector<bool> drawn(n, false);

  // Draws row i, then every layer contained in it, indented beneath.
  std::function<void(int, int)> draw_with_nested = [&](int i, int depth) {
    if (drawn[i]) return;
    drawn[i] = true;
    DrawLayerRow(c, i, activate, move_idx, move_delta, remove_req);
    if (!inside_ok || depth >= 8) return;
    for (int j = 0; j < n; ++j) {
      if (!drawn[j] && inside[j] == i) {
        ImGui::Indent();
        draw_with_nested(j, depth + 1);
        ImGui::Unindent();
      }
    }
  };

  for (int i = 0; i < n; ++i) {
    if (drawn[i]) continue;
    // A contained layer draws under its container, not at top level.
    if (inside_ok && inside[i] >= 0 && inside[i] < n && !drawn[inside[i]]) {
      continue;
    }
    std::vector<int> members{i};
    if (comp_ok) {
      for (int j = i + 1; j < n; ++j) {
        if (comp[j] == comp[i]) members.push_back(j);
      }
    }
    if (members.size() > 1) {
      ImGui::PushStyleColor(ImGuiCol_ChildBg,
                            ImGui::GetStyleColorVec4(ImGuiCol_FrameBg));
      ImGui::BeginChild(ImGui::GetID(("##group" + std::to_string(i)).c_str()),
                        ImVec2(0, 0),
                        ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_Borders);
      ImGui::TextDisabled("linked by references");
      for (int m : members) {
        draw_with_nested(m, 0);
      }
      ImGui::EndChild();
      ImGui::PopStyleColor();
    } else {
      draw_with_nested(i, 0);
    }
  }
  // Cycle fallback: anything not reachable through a container draws flat.
  for (int i = 0; i < n; ++i) {
    if (!drawn[i]) DrawLayerRow(c, i, activate, move_idx, move_delta, remove_req);
  }

  // Apply after the loop: each of these mutates the vector being drawn.
  if (activate >= 0) c->SetActiveLayer(activate);
  if (move_idx >= 0) MoveLayer(*c, move_idx, move_delta);
  if (remove_req >= 0) {
    const int count = CountLayerElements(*c, remove_req);
    if (count == 0) {
      RemoveLayer(*c, remove_req, /*delete_elements=*/false);
    } else {
      st.confirm_remove = remove_req;
      st.confirm_count = count;
      st.open_confirm = true;
    }
  }

  // Non-empty layer removal deletes its ELEMENTS from the tree: confirm it.
  if (st.open_confirm) {
    ImGui::OpenPopup("Remove layer?##layers");
    st.open_confirm = false;
  }
  if (ImGui::BeginPopupModal("Remove layer?##layers", nullptr,
                             ImGuiWindowFlags_AlwaysAutoResize)) {
    const bool valid = st.confirm_remove >= 0 &&
                       st.confirm_remove < static_cast<int>(c->layers.size());
    ImGui::Text("Remove layer '%s'?",
                valid ? c->layers[st.confirm_remove].name.c_str() : "?");
    ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.2f, 1.0f),
                       "Its %d element(s) will be DELETED from the model.",
                       st.confirm_count);
    ImGui::Separator();
    if (ImGui::Button("Delete elements and remove") && valid) {
      RemoveLayer(*c, st.confirm_remove, /*delete_elements=*/true);
      st.confirm_remove = -1;
      ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel")) {
      st.confirm_remove = -1;
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }

  ImGui::Separator();
  if (ImGui::Button("+ Layer")) AddEmptyLayer(*c, "");
  ImGui::SetItemTooltip("%s",
                        "Add an empty authored layer and make it the edit scope");

  ImGui::SetNextItemWidth(240);
  ImGui::InputTextWithHint("##layerpath", "path to MJCF to add as a layer",
                           &st.load_path);
  ImGui::SameLine();
  if (ImGui::Button("Add") && !st.load_path.empty()) {
    std::string err;
    st.status = AddLayerFromFile(*c, st.load_path, &err)
                    ? ("added layer '" + st.load_path + "'")
                    : ("add failed: " + err);
    if (err.empty()) st.load_path.clear();
  }

  ImGui::Separator();
  ImGui::TextDisabled("Export: root <include> document + one file per layer");
  ImGui::SetNextItemWidth(240);
  ImGui::InputTextWithHint("##exportpath", "path to write (.xml)",
                           &st.export_path);
  ImGui::SameLine();
  if (ImGui::Button("Export") && !st.export_path.empty()) {
    std::string err;
    st.status = ExportLayeredMjcf(*c, st.export_path, &err)
                    ? ("exported -> " + st.export_path)
                    : ("export failed: " + err);
  }
  if (!st.status.empty()) ImGui::TextWrapped("%s", st.status.c_str());
}

// Assets: meshes / textures / materials, with creation + a browsable swatch list.
static void AssetsUpdate(GuiPlugin* self) {
  EditorContext* c = ctx_cast<EditorContext>(self);
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
    ImGui::PushID(ImGuiSerialId(m->serial));
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
    ImGui::PushID(ImGuiSerialId(t->serial));
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

// Model-level creation items, shared between the "+ Add" panel and the
// Hierarchy's per-section context menus (an empty family's section row is the
// bootstrap for its first element).
void DrawAddActuatorItems(EditorContext& ctx, std::uint64_t target) {
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
    if (ImGui::MenuItem(a.label)) AddActuatorOp(ctx, a.sp, target);
  }
}
void DrawAddSensorItems(EditorContext& ctx, std::uint64_t target) {
  struct S { const char* label; SensorSpelling sp; };
  const S sens[] = {{"Jointpos", SensorSpelling::Jointpos},
                    {"Jointvel", SensorSpelling::Jointvel},
                    {"Actuatorpos", SensorSpelling::Actuatorpos},
                    {"Actuatorvel", SensorSpelling::Actuatorvel},
                    {"Framepos", SensorSpelling::Framepos},
                    {"Framequat", SensorSpelling::Framequat},
                    {"Gyro", SensorSpelling::Gyro},
                    {"Accelerometer", SensorSpelling::Accelerometer},
                    {"Velocimeter", SensorSpelling::Velocimeter},
                    {"Force", SensorSpelling::Force},
                    {"Torque", SensorSpelling::Torque},
                    {"Touch", SensorSpelling::Touch}};
  for (const S& s : sens) {
    if (ImGui::MenuItem(s.label)) AddSensorOp(ctx, s.sp, target);
  }
}

// The "+ Add" panel: model-level structural adds (deliverable 1). Target-aware --
// an actuator/sensor wires to the current selection when it is a valid target.
static void AddMenuUpdate(GuiPlugin* self) {
  EditorContext* c = ctx_cast<EditorContext>(self);
  if (!c->tree) {
    ImGui::TextDisabled("No model loaded.");
    return;
  }
  const std::uint64_t sel = c->selected_serial;
  ImGui::TextDisabled("Model-level elements");
  ImGui::Separator();

  if (ImGui::BeginMenu("Actuator")) {
    DrawAddActuatorItems(*c, sel);
    ImGui::EndMenu();
  }
  if (ImGui::BeginMenu("Sensor")) {
    DrawAddSensorItems(*c, sel);
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
  EditorContext* c = ctx_cast<EditorContext>(self);
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

// File panel: path inputs for load / save (the vendored shell carries no native
// file dialog). A dockable panel, not a menu-bar item; drag-drop and the CLI arg
// still feed loads.
// Save `c` to `path` and externalize any in-memory (imported) assets beside it.
static void SaveAndExternalize(EditorContext* c, const std::string& path) {
  if (SaveModel(*c, path)) {
    ExternalizeVfsAssets(*c, path);
  }
}

static void FileMenuUpdate(GuiPlugin* self) {
  EditorContext* c = ctx_cast<EditorContext>(self);
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

// ==========================================================================
// SE4 editor shell (R1: collapsed from the fork's host/shell.cc into the editor).
// The File/Edit menus, the transform toolbar + Edit/Play mode + status chip, and
// the native file dialog all live here now, driven from a ModelPlugin do_update
// that runs INSIDE the active ImGui frame with no window wrapper -- so it can call
// ImGui::BeginMainMenuBar (appends to the host's real menu bar; the host's own
// File menu remains alongside, an accepted degradation) without the nested
// Begin/End a GuiPlugin window would impose. The dialog is serviced in-plugin
// (Linux zenity subprocess) instead of a host FileDialogPlugin.
// ==========================================================================

static void ShellSaveExternalize(EditorContext* c, const std::string& path) {
  if (SaveModel(*c, path)) ExternalizeVfsAssets(*c, path);
}

// Blocking zenity subprocess (mirrors the host's own file_dialog_zenity popen;
// a whole-frame stall is acceptable and no worse than upstream). A missing zenity
// delivers a clean rejection + a diagnostic. Behind FileDialogState so a future
// host/remote service can replace it. Linux only (the fork build target).
static void ServiceFileDialog(EditorContext* c) {
  const FileDialogState::Kind k = c->file_dialog.Poll();
  if (k == FileDialogState::Kind::None) return;
#ifdef __linux__
  std::string cmd = "zenity --file-selection";
  switch (k) {
    case FileDialogState::Kind::SaveAs:
      cmd += " --save --confirm-overwrite";
      break;
    case FileDialogState::Kind::ImportMeshes:
      cmd += " --multiple --separator='\\n'";
      break;
    case FileDialogState::Kind::ImportFolder:
      cmd += " --directory";
      break;
    default:
      break;
  }
  const std::string& hint = c->file_dialog.start_hint();
  if (!hint.empty()) cmd += " --filename=\"" + hint + "\"";
  cmd += " 2>/dev/null";
  FILE* p = popen(cmd.c_str(), "r");
  if (!p) {
    c->file_dialog.Deliver("", false);
    c->status_toast.Post("file dialog unavailable (install zenity)",
                         StatusToast::Kind::Error, ImGui::GetTime());
    return;
  }
  std::string out;
  char buf[4096];
  std::size_t n;
  while ((n = std::fread(buf, 1, sizeof(buf), p)) > 0) out.append(buf, n);
  const int rc = pclose(p);
  while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) {
    out.pop_back();
  }
  c->file_dialog.Deliver(out, rc == 0 && !out.empty());
#else
  c->file_dialog.Deliver("", false);
  c->status_toast.Post("native file dialog is Linux-only in this build",
                       StatusToast::Kind::Error, ImGui::GetTime());
#endif
}

// Dispatch a delivered dialog outcome (once per delivered result).
static void DrainFileDialog(EditorContext* c) {
  if (!c->file_dialog.HasResult()) return;
  const FileDialogState::Result r = c->file_dialog.TakeResult();
  if (!r.accepted || r.path.empty()) return;
  switch (r.kind) {
    case FileDialogState::Kind::Open:
      c->pending.Request(r.path);  // serviced by the model pipeline
      break;
    case FileDialogState::Kind::SaveAs:
      ShellSaveExternalize(c, r.path);
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
      std::vector<std::string> paths;
      std::size_t start = 0;
      while (start <= r.path.size()) {
        const std::size_t nl = r.path.find('\n', start);
        const std::string p = r.path.substr(
            start, nl == std::string::npos ? std::string::npos : nl - start);
        if (!p.empty()) paths.push_back(p);
        if (nl == std::string::npos) break;
        start = nl + 1;
      }
      MultiMeshImportResult m = ImportMeshes(*c, paths);
      c->status_toast.Post(
          m.ok ? ("imported " + std::to_string(m.imported) + " mesh(es)")
               : ("mesh import failed: " + m.error),
          m.ok ? StatusToast::Kind::Info : StatusToast::Kind::Error,
          ImGui::GetTime());
      break;
    }
    case FileDialogState::Kind::ImportFolder: {
      MultiMeshImportResult m = ImportMeshFolder(*c, r.path);
      c->status_toast.Post(
          m.ok ? ("imported " + std::to_string(m.imported) + " mesh(es)")
               : ("folder import failed: " + m.error),
          m.ok ? StatusToast::Kind::Info : StatusToast::Kind::Error,
          ImGui::GetTime());
      break;
    }
    case FileDialogState::Kind::None:
      break;
  }
}

static void ShellFileMenu(EditorContext* c) {
  if (!ImGui::BeginMenu("File")) return;
  if (ImGui::MenuItem("New", "Ctrl+N")) NewModelOp(*c);
  ImGui::Separator();
  if (ImGui::MenuItem("Open...", "Ctrl+O")) {
    c->file_dialog.Request(FileDialogState::Kind::Open, c->source_path);
  }
  ImGui::Separator();
  const bool can_save = c->model_ready && !c->source_path.empty();
  if (ImGui::MenuItem("Save", "Ctrl+S", false, can_save)) {
    ShellSaveExternalize(c, c->source_path);
  }
  if (ImGui::MenuItem("Save As...", "Ctrl+Shift+S", false, c->model_ready)) {
    c->file_dialog.Request(FileDialogState::Kind::SaveAs, c->source_path);
  }
  ImGui::Separator();
  if (ImGui::MenuItem("Import Mesh...", nullptr, false, c->CanEdit())) {
    c->file_dialog.Request(FileDialogState::Kind::ImportMesh, c->base_dir);
  }
  if (ImGui::MenuItem("Import Meshes...", nullptr, false, c->CanEdit())) {
    c->file_dialog.Request(FileDialogState::Kind::ImportMeshes, c->base_dir);
  }
  if (ImGui::MenuItem("Import Folder...", nullptr, false, c->CanEdit())) {
    c->file_dialog.Request(FileDialogState::Kind::ImportFolder, c->base_dir);
  }
  ImGui::EndMenu();
}

static void ShellEditMenu(EditorContext* c) {
  if (!ImGui::BeginMenu("Edit")) return;
  const bool editable = c->CanEdit();
  if (ImGui::MenuItem("Undo", "Ctrl+Z", false,
                      editable && c->history.can_undo())) {
    Undo(*c);
  }
  if (ImGui::MenuItem("Redo", "Ctrl+Y", false,
                      editable && c->history.can_redo())) {
    Redo(*c);
  }
  ImGui::Separator();
  const bool has_sel = editable && c->selected_serial != 0 &&
                       SerialInActiveLayer(*c, c->selected_serial);
  if (ImGui::MenuItem("Duplicate", "Ctrl+D", false, has_sel)) {
    DuplicateOp(*c, c->selected_serial);
  }
  if (ImGui::MenuItem("Delete", "Del", false, has_sel)) {
    c->delete_request_serial = c->selected_serial;
  }
  ImGui::EndMenu();
}

// Transform-tool icon toggle (FontAwesome glyphs in [U+F000,U+F3FF]).
static void ShellToolButton(const char* icon, const char* id, GizmoTool tool,
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

static void ShellAddDropdown(EditorContext* c) {
  if (!ImGui::BeginMenu("+ Add")) return;
  const std::uint64_t sel = c->selected_serial;
  if (ImGui::BeginMenu("Body / geom (world)")) {
    static int count = 1;
    ImGui::SetNextItemWidth(90);
    ImGui::InputInt("count", &count);
    count = count < 1 ? 1 : (count > 64 ? 64 : count);
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
  if (ImGui::BeginMenu("Actuator")) {
    DrawAddActuatorItems(*c, sel);
    ImGui::EndMenu();
  }
  if (ImGui::BeginMenu("Sensor")) {
    DrawAddSensorItems(*c, sel);
    ImGui::EndMenu();
  }
  if (ImGui::MenuItem("Keyframe")) AddKeyframeOp(*c);
  ImGui::EndMenu();
}

// The transform toolbar + single Edit toggle + status chip, drawn as a top-level
// window (dockable). Mode is self-owned (EditorContext.mode); the do_update
// freeze in the model plugin holds physics at reset while in Edit.
static void ShellToolbar(EditorContext* c) {
  constexpr const char* kIconSelect = "\xEF\x89\x85";
  constexpr const char* kIconMove = "\xEF\x81\x87";
  constexpr const char* kIconRotate = "\xEF\x80\x9E";
  constexpr const char* kIconScale = "\xEF\x82\xB2";
  GizmoSettings& g = c->gizmo;

  if (!ImGui::Begin("Transform")) {
    ImGui::End();
    return;
  }
  // Transport: ONE Edit toggle (there is no editor Play/pause -- MuJoCo Studio
  // owns that). Edit ON freezes at qpos0; toggling OFF hands the sim to the
  // host's own play/pause (toolbar / Space). Highlighted == Edit is ON. Space
  // toggles the same thing; the host's built-in Space is shadowed while the
  // editor is loaded (the host toolbar play/pause still governs Play mode).
  const bool in_edit = c->mode == EditorMode::Edit;
  if (in_edit) {
    ImGui::PushStyleColor(ImGuiCol_Button,
                          ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
  }
  if (ImGui::Button(in_edit ? "Edit (on)" : "Edit (off)")) c->ToggleEditMode();
  if (in_edit) ImGui::PopStyleColor();
  ImGui::SetItemTooltip(
      "%s", in_edit ? "Edit ON: physics frozen at qpos0, authoring enabled.\n"
                      "Click or press Space to hand off to the host transport."
                    : "Edit OFF: the host play/pause governs the sim.\n"
                      "Click or press Space to re-enter Edit (resets to qpos0).");
  ImGui::SameLine();
  ImGui::TextUnformatted("|");
  ImGui::SameLine();

  ImGui::BeginDisabled(!c->CanEdit());
  ShellToolButton(kIconSelect, "tool_select", GizmoTool::Select, g, "Select (Q)");
  ShellToolButton(kIconMove, "tool_move", GizmoTool::Translate, g, "Move (W)");
  ShellToolButton(kIconRotate, "tool_rotate", GizmoTool::Rotate, g, "Rotate (E)");
  ShellToolButton(kIconScale, "tool_scale", GizmoTool::Scale, g, "Scale (R)");
  if (ImGui::Button(g.world_space ? "World" : "Local")) {
    g.world_space = !g.world_space;
  }
  ImGui::SameLine();
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
    ImGui::Checkbox("absolute grid", &g.grid_absolute);
    ImGui::Checkbox("surface snap", &g.surf_snap);
    ImGui::Checkbox("align to surface", &g.surf_align);
    ImGui::EndPopup();
  }
  ImGui::SameLine();
  if (g.surf_snap) {
    ImGui::PushStyleColor(ImGuiCol_Button,
                          ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
  }
  if (ImGui::Button("Surf")) g.surf_snap = !g.surf_snap;
  if (g.surf_snap) ImGui::PopStyleColor();
  ImGui::SameLine();
  if (c->show_all_joints) {
    ImGui::PushStyleColor(ImGuiCol_Button,
                          ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
  }
  if (ImGui::Button("Joints")) c->show_all_joints = !c->show_all_joints;
  if (c->show_all_joints) ImGui::PopStyleColor();
  ImGui::SameLine();
  ShellAddDropdown(c);
  ImGui::EndDisabled();

  // Status chip: dirty + error count (folds the old EditorShell bridge).
  const int errs = DiagnosticErrorCount(c->diagnostics);
  ImGui::SameLine();
  ImGui::TextUnformatted("|");
  ImGui::SameLine();
  if (c->dirty) {
    ImGui::TextColored(ImVec4(0.95f, 0.78f, 0.35f, 1.0f), "unsaved");
    ImGui::SameLine();
  }
  if (errs > 0) {
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.5f, 0.15f, 0.15f, 1.0f));
    if (ImGui::SmallButton((std::to_string(errs) + " error(s)").c_str())) {
      c->focus_diagnostics_request = true;
    }
    ImGui::PopStyleColor();
  } else {
    ImGui::TextDisabled("%s", c->status_line.c_str());
  }
  ImGui::End();
}

// The shell driver: runs each tick inside the ImGui frame, no window wrapper.
static bool ShellUpdate(ModelPlugin* self, mjModel*, mjData*) {
  EditorContext* c = ctx_cast<EditorContext>(self);
  ServiceFileDialog(c);  // open + deliver any pending native dialog (blocking)
  DrainFileDialog(c);    // dispatch a delivered result
  if (ImGui::BeginMainMenuBar()) {
    ShellFileMenu(c);
    ShellEditMenu(c);
    ImGui::EndMainMenuBar();
  }
  ShellToolbar(c);
  return false;  // never affects host stepping
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
  // The SE4 shell (menu bar + toolbar + mode + native dialog) runs as a
  // ModelPlugin do_update so it draws inside the ImGui frame with no window
  // wrapper (BeginMainMenuBar appends to the host's real bar). Returns false.
  ModelPlugin shell;
  shell.name = "ProtoSpec Shell";
  shell.do_update = ShellUpdate;
  shell.data = &ctx;
  RegisterPlugin<ModelPlugin>(shell);

  RegisterGuiPanel("Layers", LayersUpdate, true, ctx);
  RegisterGuiPanel("Assets", AssetsUpdate, false, ctx);
  RegisterGuiPanel("Diagnostics", DiagnosticsUpdate, false, ctx);
  RegisterGuiPanel("File", FileMenuUpdate, false, ctx);
  RegisterGuiPanel("+ Add", AddMenuUpdate, false, ctx);

  RegisterKey(
      "Undo", ImGuiMod_Ctrl | ImGuiKey_Z,
      [](KeyHandlerPlugin* self) {
        EditorContext* c = ctx_cast<EditorContext>(self);
        if (c->CanEdit()) Undo(*c);
      }, ctx);
  RegisterKey(
      "Redo", ImGuiMod_Ctrl | ImGuiKey_Y,
      [](KeyHandlerPlugin* self) {
        EditorContext* c = ctx_cast<EditorContext>(self);
        if (c->CanEdit()) Redo(*c);
      }, ctx);
  RegisterKey(
      "Save", ImGuiMod_Ctrl | ImGuiKey_S,
      [](KeyHandlerPlugin* self) {
        EditorContext* c = ctx_cast<EditorContext>(self);
        if (c->model_ready && !c->source_path.empty()) {
          SaveAndExternalize(c, c->source_path);
        }
      },
      ctx);

  // Space toggles the Edit transport (play/stop), matching the toolbar button.
  // WHY a KeyHandler that CONSUMES Space (the host's own play/pause hotkey):
  // KeyHandlerPlugins run before the host's built-in keys AND the host treats any
  // matched plugin chord as handled (it early-returns, skipping its own Space).
  // We must consume it: while the editor freezes in Edit the host sits UNPAUSED
  // (verified -- our do_update returns true so the host never reaches
  // StepControl::Advance), so *letting* the host process Space would toggle it to
  // PAUSED, and exiting Edit would then show a paused sim, not a playing one.
  // Consuming instead leaves the host unpaused, so Edit-OFF plays immediately --
  // "host play wins over edit" in one press. The host toolbar play/pause is
  // untouched and governs Play mode; only the Space KEY is repurposed.
  // RESIDUAL GAP (unavoidable plugin-side): clicking the host toolbar PLAY button
  // while in Edit is invisible to us -- it only calls StepControl::SetPauseState
  // (host-internal, no plugin-visible signal) and our freeze already blocks
  // stepping -- so it does nothing until Edit is toggled off. Closing that would
  // need a tiny studio-land hook (e.g. a plugin-visible transport-intent flag);
  // not landed.
  RegisterKey(
      "Toggle Edit (Space)", ImGuiKey_Space,
      [](KeyHandlerPlugin* self) {
        EditorContext* c = ctx_cast<EditorContext>(self);
        if (c->model_ready) c->ToggleEditMode();
      },
      ctx);
}

}  // namespace ps::studio
