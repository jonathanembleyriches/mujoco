// ProtoSpec Studio Hierarchy panel (ps::studio, ours): the authored ProtoSpec
// tree (§3), not the compiled mjModel.
//
// The tree-MODEL builder (BuildHierarchyModel / FilterHierarchy) is windowless
// and unit-tested; the ImGui draw + context menus live in hierarchy_panel.cc and
// are registered as a GuiPlugin. Keeping the builder free of ImGui lets the ctest
// target assert section counts and macro-node singularity with no window.

#ifndef PS_STUDIO_EDITOR_HIERARCHY_PANEL_H_
#define PS_STUDIO_EDITOR_HIERARCHY_PANEL_H_

#include <cstdint>
#include <string>
#include <vector>

#include "editor/editor_context.h"
#include "types.h"

namespace ps::studio {

// One row of the authored tree. A node is either a real element (serial != 0) or
// a synthetic section header (is_section, serial == 0). Macro nodes
// (replicate / composite / attach / flexcomp) are shown as a single row and are
// never expanded (their generated children are compiled artifacts, not authored
// tree, DR §3).
struct HierNode {
  std::uint64_t serial = 0;                            // element serial, 0 = section
  ps::mjcf::ElementType type = ps::mjcf::ElementType::Model;
  std::string tag;    // element XML tag ("body", "geom", ...) or section title
  std::string name;   // authored name, empty when unnamed
  std::string label;  // display label ("body: torso", "geom", section title)
  std::string layer_key;  // the element's loc.file tag (its layer); see layers.h
  bool is_section = false;
  bool is_macro = false;
  std::vector<HierNode> children;
};

// The root of the authored tree: a synthetic node whose children are the §3
// sections (Body Tree / Actuators / Sensors / Tendons / Equality / Contact /
// Defaults / Keyframes / Assets / Custom). Empty sections are omitted.
HierNode BuildHierarchyModel(const ps::mjcf::Model& model);

// A pruned copy keeping only nodes whose name or tag contains `query`
// (case-insensitive substring), plus every ancestor of a match and, for a
// matched element, its whole subtree. Empty query returns the tree unchanged.
HierNode FilterHierarchy(const HierNode& root, const std::string& query);

// Number of element (non-section) nodes whose name or tag matches `query`
// (case-insensitive substring). Zero for an empty query. Counts genuine matches,
// not the ancestor/subtree rows FilterHierarchy keeps for context.
int CountHierarchyMatches(const HierNode& root, const std::string& query);

// Registers the Hierarchy GuiPlugin (replaces the SE0 flat-body placeholder).
void RegisterHierarchyPanel(EditorContext& ctx);

}  // namespace ps::studio

#endif  // PS_STUDIO_EDITOR_HIERARCHY_PANEL_H_
