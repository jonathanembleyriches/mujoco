// ProtoSpec Studio Hierarchy panel: ImGui draw + context menus (ps::studio,
// ours). The windowless tree-model builder lives in hierarchy_model.cc; this TU
// only renders it and wires clicks/rename/delete through the editor ops.

#include <cfloat>
#include <cstring>
#include <string>
#include <utility>

#include <cstdint>

#include <imgui.h>
#include <imgui_stdlib.h>

#include "editor/authoring_ops.h"
#include "editor/editor_ops.h"
#include "editor/hierarchy_icons.h"
#include "editor/hierarchy_panel.h"
#include "editor/layers.h"
#include "editor/plugins.h"
#include "platform/ux/plugin.h"
#include "types.h"

namespace ps::studio {

namespace {

namespace mj = ps::mjcf;

// Row glyphs (FontAwesome, merged into the host font). Section headers get a
// folder; the filter's clear affordance is a small times glyph.
constexpr const char* kSectionIcon = "\xEF\x81\xBB";  // U+F07B folder
constexpr const char* kClearIcon = "\xEF\x80\x8D";    // U+F00D times

struct HierUiState {
  std::string search;
  std::string new_class;        // "New Default class" input buffer
  bool keep_world_pose = true;  // reparent "keep world pose" toggle (deliverable 4)

  std::uint64_t rename_serial = 0;
  std::string rename_buf;
  bool rename_focus = false;

  bool open_delete = false;
  std::uint64_t delete_serial = 0;
  std::string delete_desc;
  std::vector<std::string> delete_dangling;
};

// The "Add child ▸" submenu on a body-context container (a Body, a Frame, or the
// world -- parent_serial == 0). Enables the SE1 placeholder.
void DrawAddChildMenu(EditorContext& ctx, std::uint64_t parent) {
  if (!ImGui::BeginMenu("Add child")) return;
  if (ImGui::MenuItem("Body")) AddBodyOp(ctx, parent);
  if (ImGui::BeginMenu("Geom")) {
    if (ImGui::MenuItem("sphere")) AddGeomOp(ctx, parent, mj::GeomType::sphere);
    if (ImGui::MenuItem("box")) AddGeomOp(ctx, parent, mj::GeomType::box);
    if (ImGui::MenuItem("capsule")) AddGeomOp(ctx, parent, mj::GeomType::capsule);
    if (ImGui::MenuItem("cylinder"))
      AddGeomOp(ctx, parent, mj::GeomType::cylinder);
    if (ImGui::MenuItem("ellipsoid"))
      AddGeomOp(ctx, parent, mj::GeomType::ellipsoid);
    if (ImGui::MenuItem("plane")) AddGeomOp(ctx, parent, mj::GeomType::plane);
    ImGui::EndMenu();
  }
  if (ImGui::BeginMenu("Joint")) {
    static const double kX[3] = {1, 0, 0}, kY[3] = {0, 1, 0}, kZ[3] = {0, 0, 1};
    if (ImGui::BeginMenu("hinge")) {
      if (ImGui::MenuItem("axis X")) AddJointAxisOp(ctx, parent, mj::JointType::hinge, kX);
      if (ImGui::MenuItem("axis Y")) AddJointAxisOp(ctx, parent, mj::JointType::hinge, kY);
      if (ImGui::MenuItem("axis Z")) AddJointAxisOp(ctx, parent, mj::JointType::hinge, kZ);
      ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("slide")) {
      if (ImGui::MenuItem("axis X")) AddJointAxisOp(ctx, parent, mj::JointType::slide, kX);
      if (ImGui::MenuItem("axis Y")) AddJointAxisOp(ctx, parent, mj::JointType::slide, kY);
      if (ImGui::MenuItem("axis Z")) AddJointAxisOp(ctx, parent, mj::JointType::slide, kZ);
      ImGui::EndMenu();
    }
    if (ImGui::MenuItem("ball")) AddJointOp(ctx, parent, mj::JointType::ball);
    if (ImGui::MenuItem("free (freejoint)"))
      AddJointOp(ctx, parent, mj::JointType::free);
    ImGui::EndMenu();
  }
  if (ImGui::MenuItem("Site")) AddSiteOp(ctx, parent);
  if (ImGui::MenuItem("Camera")) AddCameraOp(ctx, parent);
  if (ImGui::MenuItem("Light")) AddLightOp(ctx, parent);
  if (ImGui::MenuItem("Frame")) AddFrameOp(ctx, parent);
  ImGui::EndMenu();
}

// The Assets-section creation verbs (folded in from the retired Assets panel):
// instant-create a default material/texture -- named uniquely, edited in Details,
// no modal -- and request the host's mesh-import dialog. Shared by the Assets
// section's right-click and the panel-header "+ Asset" button (the latter works
// even before any asset exists, when the Assets section is not yet present).
void DrawNewAssetMenu(EditorContext& ctx) {
  if (ImGui::MenuItem("New Material")) CreateMaterialOp(ctx, MaterialSpec{});
  if (ImGui::MenuItem("New Texture")) CreateTextureOp(ctx, TextureSpec{});
  ImGui::Separator();
  if (ImGui::MenuItem("Import Mesh(es)...")) {
    ctx.file_dialog.Request(FileDialogState::Kind::ImportMeshes, ctx.base_dir);
  }
}

bool IsContainerNode(const HierNode& n) {
  return n.type == mj::ElementType::Body || n.type == mj::ElementType::Frame;
}

// The layer edit-scope gate (layers.h), O(1) off the node's recorded key.
// Mutating an EXISTING element requires its layer to be active; creating a NEW
// element is always fine (it is stamped with the active key at the compile
// seam), so "Add child" stays open on foreign containers -- that is exactly
// the nested-include case the tag design exists to express.
bool NodeInActiveLayer(EditorContext& ctx, const HierNode& n) {
  if (ctx.layers.size() <= 1 || n.layer_key.empty()) return true;
  const int li = LayerIndexForKey(ctx, n.layer_key);
  return li < 0 || li == ctx.active_layer;
}

void ClearSelectionIf(EditorContext& ctx, std::uint64_t serial) {
  if (ctx.selected_serial == serial) {
    ctx.selected_serial = 0;
    ctx.selected_desc.clear();
  }
}

void CommitDelete(EditorContext& ctx, std::uint64_t serial, bool cascade,
                  const std::string& desc) {
  ctx.BeginEdit();
  DeleteBySerial(ctx, serial, cascade);
  ctx.CommitEdit("Delete " + desc);
  ClearSelectionIf(ctx, serial);
}

void DrawRenameInput(EditorContext& ctx, const HierNode& node, HierUiState& st) {
  if (st.rename_focus) {
    ImGui::SetKeyboardFocusHere();
    st.rename_focus = false;
  }
  ImGui::SetNextItemWidth(-FLT_MIN);
  const bool enter = ImGui::InputText(
      "##rename", &st.rename_buf,
      ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll);
  if (enter) {
    if (!st.rename_buf.empty() && st.rename_buf != node.name) {
      ctx.BeginEdit();
      RenameBySerial(ctx, node.serial, st.rename_buf);
      ctx.CommitEdit("Rename " + node.name + " -> " + st.rename_buf);
      SelectBySerial(ctx, node.serial);
    }
    st.rename_serial = 0;
  } else if (ImGui::IsItemDeactivated()) {
    st.rename_serial = 0;  // clicked away / escaped
  }
}

void DrawContextMenu(EditorContext& ctx, const HierNode& node, HierUiState& st) {
  ImGui::TextDisabled("%s", node.label.c_str());
  ImGui::Separator();
  // Elements of an inactive layer stay browsable but not editable; the menu
  // says who owns them and offers the switch.
  const bool in_active = NodeInActiveLayer(ctx, node);
  if (!in_active) {
    const int owner = LayerIndexForKey(ctx, node.layer_key);
    const std::string owner_name =
        owner >= 0 ? ctx.layers[owner].name : node.layer_key;
    ImGui::TextDisabled("Owned by layer '%s'", owner_name.c_str());
    if (owner >= 0 &&
        ImGui::MenuItem(("Edit layer '" + owner_name + "'").c_str())) {
      ctx.SetActiveLayer(owner);
    }
    ImGui::Separator();
  }
  ImGui::BeginDisabled(!in_active);
  if (ImGui::MenuItem("Rename")) {
    st.rename_serial = node.serial;
    st.rename_buf = node.name;
    st.rename_focus = true;
  }
  if (ImGui::MenuItem("Delete")) {
    std::vector<std::string> dangling = PreviewDeleteReferrers(ctx, node.serial);
    if (dangling.empty()) {
      CommitDelete(ctx, node.serial, /*cascade=*/false, node.label);
    } else {
      st.delete_serial = node.serial;
      st.delete_desc = node.label;
      st.delete_dangling = std::move(dangling);
      st.open_delete = true;
    }
  }
  if (ImGui::MenuItem("Duplicate", "Ctrl+D")) {
    DuplicateOp(ctx, node.serial);
  }
  ImGui::EndDisabled();  // !in_active gate: tool select / add-child stay open
  // Gizmo tool selection as a right-click verb (mirrors the Q/W/E/R hotkeys), so
  // the transform tools live on the element, not only the toolbar.
  if (ImGui::BeginMenu("Transform")) {
    if (ImGui::MenuItem("Select", "Q", ctx.gizmo.tool == GizmoTool::Select)) {
      ctx.gizmo.tool = GizmoTool::Select;
    }
    if (ImGui::MenuItem("Move", "W", ctx.gizmo.tool == GizmoTool::Translate)) {
      ctx.gizmo.tool = GizmoTool::Translate;
    }
    if (ImGui::MenuItem("Rotate", "E", ctx.gizmo.tool == GizmoTool::Rotate)) {
      ctx.gizmo.tool = GizmoTool::Rotate;
    }
    if (ImGui::MenuItem("Scale", "R", ctx.gizmo.tool == GizmoTool::Scale)) {
      ctx.gizmo.tool = GizmoTool::Scale;
    }
    ImGui::EndMenu();
  }
  if (IsContainerNode(node)) {
    DrawAddChildMenu(ctx, node.serial);
  }
  // Geom material assignment: pick from the model's materials (or clear).
  // Mutates the geom, so it obeys the layer edit scope.
  if (node.type == mj::ElementType::Geom) {
    ImGui::BeginDisabled(!in_active);
    if (ImGui::BeginMenu("Assign material")) {
      if (ImGui::MenuItem("(none)")) {
        AssignGeomMaterialOp(ctx, node.serial, "");
      }
      bool any = false;
      for (auto& a : ctx.tree->assets) {
        if (!a) continue;
        for (auto& m : a->materials) {
          if (!m || !m->name) continue;
          any = true;
          if (ImGui::MenuItem(m->name->c_str())) {
            AssignGeomMaterialOp(ctx, node.serial, *m->name);
          }
        }
      }
      if (!any) ImGui::TextDisabled("no materials -- add one in Assets");
      ImGui::EndMenu();
    }
    ImGui::EndDisabled();
  }
  // Quick-rig: wire an actuator to a selected joint (surfaces AddActuatorOp).
  if (node.type == mj::ElementType::Joint ||
      node.type == mj::ElementType::FreeJoint) {
    if (ImGui::BeginMenu("Add actuator for joint")) {
      if (ImGui::MenuItem("motor"))
        AddActuatorOp(ctx, ActuatorSpelling::Motor, node.serial);
      if (ImGui::MenuItem("position"))
        AddActuatorOp(ctx, ActuatorSpelling::Position, node.serial);
      if (ImGui::MenuItem("velocity"))
        AddActuatorOp(ctx, ActuatorSpelling::Velocity, node.serial);
      ImGui::EndMenu();
    }
  }
}

// Drag-drop reparent: every element node is a drag source; body/frame nodes are
// drop targets. Dropping element `src` onto container `dst` runs ReparentOp with
// the panel's keep-world-pose toggle (cycles / invalid kinds are rejected there).
void DrawReparentDragDrop(EditorContext& ctx, const HierNode& node,
                          const HierUiState& st) {
  if (!ctx.CanEdit()) return;  // reparenting is an edit like any other
  if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)) {
    // Moving an element mutates it, so the drag source obeys the layer edit
    // scope. Dropping INTO a foreign container is fine (nested-include case).
    if (!NodeInActiveLayer(ctx, node)) {
      // A payload type no target accepts: the drag shows why it will not
      // move, and every drop site refuses it.
      const std::uint64_t s = node.serial;
      ImGui::SetDragDropPayload("PS_ELEMENT_LOCKED", &s, sizeof(s));
      ImGui::TextUnformatted(
          (node.label + "  --  owned by an inactive layer").c_str());
      ImGui::EndDragDropSource();
      return;
    }
    const std::uint64_t s = node.serial;
    ImGui::SetDragDropPayload("PS_ELEMENT", &s, sizeof(s));
    ImGui::TextUnformatted(node.label.c_str());
    ImGui::EndDragDropSource();
  }
  if (IsContainerNode(node) && ImGui::BeginDragDropTarget()) {
    if (const ImGuiPayload* pl = ImGui::AcceptDragDropPayload("PS_ELEMENT")) {
      std::uint64_t src = 0;
      std::memcpy(&src, pl->Data, sizeof(src));
      if (src != 0 && src != node.serial) {
        ReparentOp(ctx, src, node.serial, st.keep_world_pose);
      }
    }
    ImGui::EndDragDropTarget();
  }
}

void DrawNode(EditorContext& ctx, const HierNode& node, HierUiState& st) {
  if (node.is_section) {
    const std::string section_label =
        std::string(kSectionIcon) + "  " + node.label;
    const bool open = ImGui::TreeNodeEx(section_label.c_str(),
                                        ImGuiTreeNodeFlags_DefaultOpen |
                                            ImGuiTreeNodeFlags_SpanAvailWidth);
    // The Body Tree header adds world-parented (top-level) elements.
    if (node.tag == std::string("Body Tree") &&
        ImGui::BeginPopupContextItem("##bodytree_add")) {
      ImGui::BeginDisabled(!ctx.CanEdit());
      DrawAddChildMenu(ctx, 0);
      ImGui::EndDisabled();
      ImGui::EndPopup();
    }
    // The Assets header folds the old Assets panel's creation in: instant-create
    // (a default material/texture the user then edits in Details, no modal) and
    // the mesh-import dialog request.
    if (node.tag == std::string("Assets") &&
        ImGui::BeginPopupContextItem("##assets_add")) {
      ImGui::BeginDisabled(!ctx.CanEdit());
      DrawNewAssetMenu(ctx);
      ImGui::EndDisabled();
      ImGui::EndPopup();
    }
    // The Defaults header adds a new default class.
    if (node.tag == std::string("Defaults") &&
        ImGui::BeginPopupContextItem("##defaults_add")) {
      static std::string cls;
      ImGui::BeginDisabled(!ctx.CanEdit());
      ImGui::SetNextItemWidth(160);
      ImGui::InputTextWithHint("##newclass", "class name", &cls);
      ImGui::SameLine();
      if (ImGui::Button("Add class") && !cls.empty()) {
        AddDefaultClassOp(ctx, cls);
        cls.clear();
        ImGui::CloseCurrentPopup();
      }
      ImGui::EndDisabled();
      ImGui::EndPopup();
    }
    if (open) {
      for (const HierNode& c : node.children) {
        DrawNode(ctx, c, st);
      }
      ImGui::TreePop();
    }
    return;
  }

  ImGui::PushID(static_cast<int>(node.serial));

  if (st.rename_serial == node.serial) {
    DrawRenameInput(ctx, node, st);
    ImGui::PopID();
    return;
  }

  const bool leaf = node.children.empty();
  ImGuiTreeNodeFlags flags =
      ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_OpenOnArrow;
  if (leaf) {
    flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
  }
  const bool selected = node.serial == ctx.selected_serial;
  if (selected) {
    flags |= ImGuiTreeNodeFlags_Selected;
  }

  // A family glyph leads every element row so kinds read at a glance; the
  // selected row is lifted to the brightest text colour for prominence.
  std::string label =
      std::string(IconForElementType(node.type)) + "  " + node.label;
  if (node.is_macro) {
    label += "  [macro]";
  }
  if (selected) {
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
  }
  const bool open = ImGui::TreeNodeEx(label.c_str(), flags);
  if (selected) {
    ImGui::PopStyleColor();
  }

  DrawReparentDragDrop(ctx, node, st);

  if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
    SelectBySerial(ctx, node.serial);
  }
  if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
    ctx.focus_request_serial = node.serial;  // consumed by the viewport (SE2)
    ctx.Diagnose(DiagEntry{DiagEntry::Severity::Info,
                           "focus request -> serial " + std::to_string(node.serial),
                           node.serial, {}});
  }
  if (ImGui::BeginPopupContextItem()) {
    ImGui::BeginDisabled(!ctx.CanEdit());
    DrawContextMenu(ctx, node, st);
    ImGui::EndDisabled();
    ImGui::EndPopup();
  }

  if (open && !leaf) {
    for (const HierNode& c : node.children) {
      DrawNode(ctx, c, st);
    }
    ImGui::TreePop();
  }
  ImGui::PopID();
}

void DrawDeleteModal(EditorContext& ctx, HierUiState& st) {
  if (st.open_delete) {
    ImGui::OpenPopup("Delete?##hier");
    st.open_delete = false;
  }
  if (ImGui::BeginPopupModal("Delete?##hier", nullptr,
                             ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::Text("Delete %s and its subtree?", st.delete_desc.c_str());
    ImGui::Spacing();
    ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.2f, 1.0f),
                       "%zu reference(s) would be left dangling:",
                       st.delete_dangling.size());
    for (const std::string& d : st.delete_dangling) {
      ImGui::BulletText("%s", d.c_str());
    }
    ImGui::Separator();
    if (ImGui::Button("Delete and clear references")) {
      CommitDelete(ctx, st.delete_serial, /*cascade=*/true, st.delete_desc);
      ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel")) {
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }
}

void HierarchyUpdate(GuiPlugin* self) {
  static HierUiState st;
  EditorContext* c = static_cast<EditorContext*>(self->data);
  if (!c->tree) {
    ImGui::TextUnformatted("No model loaded.");
    return;
  }

  const HierNode model = BuildHierarchyModel(*c->tree);

  // Filter field with a clear-x affordance and a live match count.
  const bool filtering = !st.search.empty();
  const float btn_w = ImGui::GetFrameHeight();
  const float spacing = ImGui::GetStyle().ItemSpacing.x;
  ImGui::SetNextItemWidth(filtering ? -(btn_w + spacing) : -FLT_MIN);
  ImGui::InputTextWithHint("##search", "filter by name...", &st.search);
  if (filtering) {
    ImGui::SameLine(0, spacing);
    if (ImGui::Button((std::string(kClearIcon) + "##clearfilter").c_str(),
                      ImVec2(btn_w, btn_w))) {
      st.search.clear();
    }
    ImGui::SetItemTooltip("%s", "Clear filter");
    const int matches = CountHierarchyMatches(model, st.search);
    ImGui::TextDisabled("%d match%s", matches, matches == 1 ? "" : "es");
  }
  ImGui::Checkbox("Keep world pose on reparent", &st.keep_world_pose);

  // Authoring affordances close outside the reset pose; browsing, filtering and
  // selection stay live so the hierarchy remains readable during a sim.
  const bool editable = c->CanEdit();
  ImGui::BeginDisabled(!editable);

  // New Default class (also reachable via the Defaults section context menu, but
  // kept here so it works before any class exists / the section is present).
  ImGui::SetNextItemWidth(140);
  ImGui::InputTextWithHint("##newdefclass", "new default class...", &st.new_class);
  ImGui::SameLine();
  if (ImGui::Button("+ Class") && !st.new_class.empty()) {
    AddDefaultClassOp(*c, st.new_class);
    st.new_class.clear();
  }
  // "+ Asset" folds the retired Assets panel's creation into the Hierarchy; it is
  // always available (the Assets section only appears once an asset exists, so a
  // section-only right-click could not bootstrap the first one).
  ImGui::SameLine();
  if (ImGui::Button("+ Asset")) ImGui::OpenPopup("##new_asset_hdr");
  ImGui::EndDisabled();
  if (ImGui::BeginPopup("##new_asset_hdr")) {
    DrawNewAssetMenu(*c);
    ImGui::EndPopup();
  }
  if (!editable) {
    ImGui::TextDisabled("Simulation running or advanced -- press Stop to edit.");
  }
  ImGui::Separator();

  const HierNode view = FilterHierarchy(model, st.search);

  ImGui::BeginChild("##hier_tree");
  for (const HierNode& section : view.children) {
    DrawNode(*c, section, st);
  }
  ImGui::EndChild();

  // Service a viewport Del request through the same referrer-confirm flow.
  // The layer edit scope is enforced HERE, the one funnel both the Del key and
  // the Edit menu route through (their own gates are advisory greying).
  if (c->delete_request_serial != 0 &&
      !SerialInActiveLayer(*c, c->delete_request_serial)) {
    const int owner = LayerOfSerial(*c, c->delete_request_serial);
    c->delete_request_serial = 0;
    c->status_toast.Post(
        "Owned by layer '" +
            (owner >= 0 ? c->layers[owner].name : std::string("?")) +
            "' -- activate that layer to delete",
        StatusToast::Kind::Warning, ImGui::GetTime());
  }
  if (c->delete_request_serial != 0) {
    const std::uint64_t sn = c->delete_request_serial;
    c->delete_request_serial = 0;
    std::string desc = c->selected_desc.empty()
                           ? ("serial " + std::to_string(sn))
                           : c->selected_desc;
    std::vector<std::string> dangling = PreviewDeleteReferrers(*c, sn);
    if (dangling.empty()) {
      CommitDelete(*c, sn, /*cascade=*/false, desc);
    } else {
      st.delete_serial = sn;
      st.delete_desc = desc;
      st.delete_dangling = std::move(dangling);
      st.open_delete = true;
    }
  }

  DrawDeleteModal(*c, st);
}

}  // namespace

void RegisterHierarchyPanel(EditorContext& ctx) {
  GuiPlugin plugin;
  plugin.name = "Hierarchy";
  plugin.update = HierarchyUpdate;
  plugin.active = true;
  plugin.data = &ctx;
  RegisterPlugin<GuiPlugin>(plugin);
}

}  // namespace ps::studio
