// ProtoSpec Studio Hierarchy panel: ImGui draw + context menus (ps::studio,
// ours). The windowless tree-model builder lives in hierarchy_model.cc; this TU
// only renders it and wires clicks/rename/delete through the editor ops.

#include <cfloat>
#include <string>
#include <utility>

#include <imgui.h>
#include <imgui_stdlib.h>

#include "editor/editor_ops.h"
#include "editor/hierarchy_panel.h"
#include "editor/plugins.h"
#include "platform/ux/plugin.h"

namespace ps::studio {

namespace {

struct HierUiState {
  std::string search;

  std::uint64_t rename_serial = 0;
  std::string rename_buf;
  bool rename_focus = false;

  bool open_delete = false;
  std::uint64_t delete_serial = 0;
  std::string delete_desc;
  std::vector<std::string> delete_dangling;
};

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
  ImGui::BeginDisabled();
  ImGui::MenuItem("Duplicate");   // SE3
  ImGui::MenuItem("Add child");   // SE3
  ImGui::EndDisabled();
}

void DrawNode(EditorContext& ctx, const HierNode& node, HierUiState& st) {
  if (node.is_section) {
    if (ImGui::TreeNodeEx(node.label.c_str(),
                          ImGuiTreeNodeFlags_DefaultOpen |
                              ImGuiTreeNodeFlags_SpanAvailWidth)) {
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
  if (node.serial == ctx.selected_serial) {
    flags |= ImGuiTreeNodeFlags_Selected;
  }

  std::string label = node.label;
  if (node.is_macro) {
    label += "  [macro]";
  }
  const bool open = ImGui::TreeNodeEx(label.c_str(), flags);

  if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
    SelectBySerial(ctx, node.serial);
  }
  if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
    ctx.focus_request_serial = node.serial;  // consumed by the viewport (SE2)
    ctx.Log("focus request -> serial " + std::to_string(node.serial));
  }
  if (ImGui::BeginPopupContextItem()) {
    DrawContextMenu(ctx, node, st);
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

  ImGui::SetNextItemWidth(-FLT_MIN);
  ImGui::InputTextWithHint("##search", "filter by name...", &st.search);
  ImGui::Separator();

  const HierNode model = BuildHierarchyModel(*c->tree);
  const HierNode view = FilterHierarchy(model, st.search);

  ImGui::BeginChild("##hier_tree");
  for (const HierNode& section : view.children) {
    DrawNode(*c, section, st);
  }
  ImGui::EndChild();

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
