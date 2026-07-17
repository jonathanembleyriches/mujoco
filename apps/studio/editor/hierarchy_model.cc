// ProtoSpec Studio Hierarchy tree-model builder (ps::studio, ours). Windowless;
// see hierarchy_panel.h. No ImGui here so the ctest target can assert the tree
// shape directly.

#include <algorithm>
#include <cctype>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "editor/hierarchy_panel.h"
#include "protospec/detail.h"  // ps::sdk::detail::NameOf
#include "reflect.h"
#include "types.h"
#include "visit.h"

namespace ps::studio {

namespace mj = ps::mjcf;
namespace reflect = ps::mjcf::reflect;
namespace sdk_detail = ps::sdk::detail;

namespace {

bool IsMacroType(mj::ElementType t) {
  return t == mj::ElementType::Composite || t == mj::ElementType::Flexcomp ||
         t == mj::ElementType::Replicate || t == mj::ElementType::Attach;
}

template <class E>
HierNode MakeNode(const E& e) {
  HierNode n;
  n.serial = e.serial;
  n.type = mj::element_type_of<E>::value;
  n.tag = std::string(reflect::Describe(n.type).xml);
  if (const std::string* nm = sdk_detail::NameOf(e)) {
    n.name = *nm;
  }
  n.is_macro = IsMacroType(n.type);
  n.layer_key = e.loc.file;
  n.label = n.name.empty() ? n.tag : (n.tag + ": " + n.name);
  return n;
}

// Visits an element's authored child / union-child lists, appending one HierNode
// per nested element and recursing (except into macro nodes, DR §3).
struct NodeBuilder {
  HierNode* parent;

  template <class U>
  void field(int, const char*, const U&) {}

  template <class C>
  void child(int, const char*, const C& list) {
    for (const auto& p : list) {
      if (p) Emit(*p);
    }
  }

  template <class C>
  void union_child(int, const char*, const C& list) {
    for (const auto& item : list) {
      std::visit([&](const auto& p) { if (p) Emit(*p); }, item.node);
    }
  }

  template <class E>
  void Emit(const E& e) {
    HierNode n = MakeNode(e);
    if (!n.is_macro) {
      NodeBuilder sub{&n};
      mj::Visit(e, sub);
    }
    parent->children.push_back(std::move(n));
  }
};

HierNode Section(const char* title) {
  HierNode s;
  s.is_section = true;
  s.tag = title;
  s.label = title;
  return s;
}

template <class T>
void EmitPtrVec(NodeBuilder& b, const std::vector<std::unique_ptr<T>>& v) {
  for (const auto& p : v) {
    if (p) b.Emit(*p);
  }
}

template <class VecAny>
void EmitUnionVec(NodeBuilder& b, const VecAny& v) {
  for (const auto& item : v) {
    std::visit([&](const auto& p) { if (p) b.Emit(*p); }, item.node);
  }
}

void PushIfNonEmpty(HierNode& root, HierNode&& section) {
  if (!section.children.empty()) {
    root.children.push_back(std::move(section));
  }
}

std::string ToLower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

bool ContainsCI(const std::string& hay, const std::string& needle_lc) {
  if (needle_lc.empty()) return true;
  return ToLower(hay).find(needle_lc) != std::string::npos;
}

bool FilterNode(const HierNode& in, const std::string& needle_lc, HierNode& out) {
  HierNode node = in;
  node.children.clear();

  // Element (non-section) nodes match on name or tag; a match keeps the node
  // and its whole authored subtree. Sections never self-match (they are kept
  // only to carry matching children).
  const bool self_match =
      !in.is_section && (ContainsCI(in.name, needle_lc) || ContainsCI(in.tag, needle_lc));
  if (self_match) {
    node.children = in.children;
    out = std::move(node);
    return true;
  }

  bool kept_child = false;
  for (const HierNode& c : in.children) {
    HierNode child_out;
    if (FilterNode(c, needle_lc, child_out)) {
      node.children.push_back(std::move(child_out));
      kept_child = true;
    }
  }
  if (kept_child) {
    out = std::move(node);
    return true;
  }
  return false;
}

}  // namespace

HierNode BuildHierarchyModel(const mj::Model& m) {
  HierNode root;
  root.is_section = true;
  root.tag = "model";
  root.label = m.model ? *m.model : std::string("model");

  {
    // worldbody[0] is the implicit world Body; its authored children are the
    // top-level scene elements. Hoist them so the section root IS the world
    // (Unity-style scene root) rather than a redundant "body" wrapper node.
    HierNode s = Section("Body Tree");
    for (const auto& world : m.worldbody) {
      if (world) {
        NodeBuilder hoist{&s};
        mj::Visit(*world, hoist);
      }
    }
    PushIfNonEmpty(root, std::move(s));
  }
  {
    HierNode s = Section("Actuators");
    NodeBuilder b{&s};
    for (const auto& act : m.actuators) {
      if (act) EmitUnionVec(b, act->actuators);
    }
    PushIfNonEmpty(root, std::move(s));
  }
  {
    HierNode s = Section("Sensors");
    NodeBuilder b{&s};
    for (const auto& sen : m.sensors) {
      if (sen) EmitUnionVec(b, sen->sensors);
    }
    PushIfNonEmpty(root, std::move(s));
  }
  {
    HierNode s = Section("Tendons");
    NodeBuilder b{&s};
    for (const auto& t : m.tendons) {
      if (t) EmitUnionVec(b, t->tendons);
    }
    PushIfNonEmpty(root, std::move(s));
  }
  {
    HierNode s = Section("Equality");
    NodeBuilder b{&s};
    for (const auto& eq : m.equalitys) {
      if (eq) EmitUnionVec(b, eq->equalities);
    }
    PushIfNonEmpty(root, std::move(s));
  }
  {
    HierNode s = Section("Contact");
    NodeBuilder b{&s};
    for (const auto& c : m.contacts) {
      if (c) {
        EmitPtrVec(b, c->pairs);
        EmitPtrVec(b, c->excludes);
      }
    }
    PushIfNonEmpty(root, std::move(s));
  }
  {
    HierNode s = Section("Defaults");
    NodeBuilder b{&s};
    EmitPtrVec(b, m.defaults);
    PushIfNonEmpty(root, std::move(s));
  }
  {
    HierNode s = Section("Keyframes");
    NodeBuilder b{&s};
    for (const auto& kf : m.keyframes) {
      if (kf) EmitPtrVec(b, kf->keys);
    }
    PushIfNonEmpty(root, std::move(s));
  }
  {
    HierNode s = Section("Assets");
    NodeBuilder b{&s};
    for (const auto& a : m.assets) {
      if (a) {
        EmitPtrVec(b, a->meshs);
        EmitPtrVec(b, a->hfields);
        EmitPtrVec(b, a->skins);
        EmitPtrVec(b, a->textures);
        EmitPtrVec(b, a->materials);
        EmitPtrVec(b, a->modelAssets);
      }
    }
    PushIfNonEmpty(root, std::move(s));
  }
  {
    HierNode s = Section("Custom");
    NodeBuilder b{&s};
    for (const auto& cu : m.customs) {
      if (cu) {
        EmitPtrVec(b, cu->numerics);
        EmitPtrVec(b, cu->texts);
        EmitPtrVec(b, cu->tuples);
      }
    }
    PushIfNonEmpty(root, std::move(s));
  }

  return root;
}

static int CountSelfMatches(const HierNode& n, const std::string& needle_lc) {
  int count = 0;
  if (!n.is_section &&
      (ContainsCI(n.name, needle_lc) || ContainsCI(n.tag, needle_lc))) {
    ++count;
  }
  for (const HierNode& c : n.children) {
    count += CountSelfMatches(c, needle_lc);
  }
  return count;
}

int CountHierarchyMatches(const HierNode& root, const std::string& query) {
  if (query.empty()) return 0;
  return CountSelfMatches(root, ToLower(query));
}

HierNode FilterHierarchy(const HierNode& root, const std::string& query) {
  if (query.empty()) {
    return root;
  }
  const std::string needle = ToLower(query);
  HierNode out;
  if (!FilterNode(root, needle, out)) {
    // Nothing matched: return an empty root (keeps identity/title, no children).
    HierNode empty = root;
    empty.children.clear();
    return empty;
  }
  return out;
}

}  // namespace ps::studio
