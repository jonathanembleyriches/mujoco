// ProtoSpec SDK: traversal and lookup.
//
// The tree stores no parent pointers (DR-2), so upward queries build a parent
// map on demand. Downward queries (Find, ForEach*) are direct walks over the
// generated Visit hook. All of it is read-only: nothing here mutates the model.
#ifndef PROTOSPEC_SDK_TRAVERSAL_H
#define PROTOSPEC_SDK_TRAVERSAL_H

#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "protospec/detail.h"
#include "reflect.h"
#include "types.h"

namespace ps::sdk {

namespace mj = ps::mjcf;

// --- Element identity ----------------------------------------------------- //
// Uniform name / type access over any element, so a consumer never reaches into
// ps::sdk::detail for the basics. `Name` returns the authored name (a Default's
// identity is its dclass), or nullptr for the nameless (Replicate, Config, ...).

template <class E>
const std::string* Name(const E& e) {
  return detail::NameOf(e);
}

// Set (create) an element's name field; a no-op for nameless element types.
template <class E>
void SetName(E& e, const std::string& name) {
  detail::SetName(e, name);
}

// The runtime element-type tag of a concrete element type.
template <class E>
mj::ElementType TypeOf(const E&) {
  return mj::element_type_of<std::decay_t<E>>::value;
}

// --- Element walks -------------------------------------------------------- //
// Invoke `fn(element&)` for `root` and every element beneath it, document order.
template <class E, class Fn>
void WalkSubtree(E& root, Fn&& fn) {
  detail::WalkTree(root, std::forward<Fn>(fn));
}

// Invoke `fn(element&)` for every element in the model, the <default> class tree
// included. Const overload walks a const Model.
template <class Fn>
void WalkModel(mj::Model& model, Fn&& fn) {
  detail::WalkModelAll(model, std::forward<Fn>(fn));
}
template <class Fn>
void WalkModel(const mj::Model& model, Fn&& fn) {
  detail::WalkModelAll(model, std::forward<Fn>(fn));
}

// A parent lookup + path index over a Model, built once and queried many times.
// Construction is a single whole-tree walk; every element (including class
// elements under <default>) gets an entry. Element identity is the pointer
// (stable under sibling mutation, DR-2). The map is a snapshot: rebuild it after
// structural edits.
class ParentMap {
 public:
  struct Node {
    const void* parent = nullptr;         // owning element, null for the root
    mj::ElementType type{};               // this element's type
    std::string name;                     // authored name (or dclass), else ""
    std::string childclass;               // body-context childclass, else ""
  };

  explicit ParentMap(const mj::Model& model) {
    const void* root = &model;
    nodes_[root] = Node{nullptr, mj::ElementType::Model, ModelName(model), ""};
    Record(model, root);
  }

  // Parent of an element, or null when it is the Model root or not indexed.
  template <class E>
  const void* ParentOf(const E& e) const {
    auto it = nodes_.find(&e);
    return it == nodes_.end() ? nullptr : it->second.parent;
  }

  const Node* Lookup(const void* ptr) const {
    auto it = nodes_.find(ptr);
    return it == nodes_.end() ? nullptr : &it->second;
  }

  // A "/"-joined path from the root to the element, each step the element's XML
  // tag plus its name in brackets when it has one (e.g.
  // "mujoco/worldbody/body[torso]/geom[shin]"). For diagnostics.
  template <class E>
  std::string PathTo(const E& e) const {
    return PathToPtr(&e);
  }

  std::string PathToPtr(const void* ptr) const {
    std::vector<const Node*> chain;
    for (const void* p = ptr; p;) {
      const Node* n = Lookup(p);
      if (!n) break;
      chain.push_back(n);
      p = n->parent;
    }
    std::string out;
    for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
      if (!out.empty()) out += '/';
      out += mj::reflect::Describe((*it)->type).xml;
      if (!(*it)->name.empty()) {
        out += '[';
        out += (*it)->name;
        out += ']';
      }
    }
    return out;
  }

 private:
  static std::string ModelName(const mj::Model& m) {
    return m.model ? *m.model : std::string();
  }

  // A recording visitor: registers each child of the element it is applied to
  // as pointing back at `self`. A member class (not function-local) so its
  // visit methods can be templates.
  struct Rec {
    ParentMap* pm;
    const void* self;
    template <class U>
    void field(int, const char*, const U&) {}
    template <class C>
    void child(int, const char*, const C& list) {
      for (const auto& p : list)
        if (p) pm->Add(*p, self);
    }
    template <class C>
    void union_child(int, const char*, const C& list) {
      for (const auto& item : list)
        std::visit(
            [&](const auto& p) {
              if (p) pm->Add(*p, self);
            },
            item.node);
    }
  };

  // Record children of `e` (whose handle is `self`) as pointing back at `self`,
  // then recurse.
  template <class E>
  void Record(const E& e, const void* self) {
    Rec rec{this, self};
    mj::Visit(const_cast<E&>(e), rec);
  }

  template <class E>
  void Add(const E& e, const void* parent) {
    Node n;
    n.parent = parent;
    n.type = mj::element_type_of<E>::value;
    if (const std::string* nm = detail::NameOf(e)) n.name = *nm;
    n.childclass = ChildClass(e);
    nodes_[&e] = std::move(n);
    Record(e, &e);
  }

  // childclass propagates defaults down the body tree; only body-context
  // elements carry it. Frame spells its class attribute `dclass`.
  template <class E>
  static std::string ChildClass(const E& e) {
    if constexpr (std::is_same_v<E, mj::Body>) {
      return e.childclass ? e.childclass->name : std::string();
    } else if constexpr (std::is_same_v<E, mj::Replicate>) {
      return e.childclass ? e.childclass->name : std::string();
    } else if constexpr (std::is_same_v<E, mj::Frame>) {
      return e.dclass ? e.dclass->name : std::string();
    } else {
      return std::string();
    }
  }

  std::unordered_map<const void*, Node> nodes_;
};

// --- Find ----------------------------------------------------------------- //

// The first element of type T with the given name, or nullptr. Searches the
// whole model, class elements included. Names are unique per element type in a
// valid model (Q-NAMES), so this is effectively a keyed lookup.
template <class T>
T* Find(mj::Model& model, std::string_view name) {
  T* found = nullptr;
  detail::WalkModelAll(model, [&](auto& e) {
    using E = std::decay_t<decltype(e)>;
    if constexpr (std::is_same_v<E, T>) {
      if (!found) {
        const std::string* nm = detail::NameOf(e);
        if (nm && *nm == name) found = &e;
      }
    }
  });
  return found;
}

template <class T>
const T* Find(const mj::Model& model, std::string_view name) {
  return Find<T>(const_cast<mj::Model&>(model), name);
}

// --- Typed visitors ------------------------------------------------------- //

// Call `fn(T&)` for every T anywhere in the model (class elements included).
template <class T, class Fn>
void ForEachOfType(mj::Model& model, Fn&& fn) {
  detail::WalkModelAll(model, [&](auto& e) {
    using E = std::decay_t<decltype(e)>;
    if constexpr (std::is_same_v<E, T>) fn(e);
  });
}

namespace detail {

// Walk the ordered BodyChildAny subtree, emitting each T. When `recursive`,
// descend the body-context children (Body/Frame) that carry their own subtree;
// a shallow walk stops at the immediate level, so a shallow ForEachGeom never
// crosses into a child body.
template <class T, class Fn>
void EmitFromSubtree(std::vector<mj::BodyChildAny>& subtree, bool recursive,
                     Fn& fn) {
  for (auto& item : subtree) {
    std::visit(
        [&](auto& p) {
          if (!p) return;
          using E = std::decay_t<decltype(*p)>;
          if constexpr (std::is_same_v<E, T>) fn(*p);
          if (recursive) {
            if constexpr (std::is_same_v<E, mj::Body> ||
                          std::is_same_v<E, mj::Frame>) {
              EmitFromSubtree<T>(p->subtree, recursive, fn);
            }
          }
        },
        item.node);
  }
}

}  // namespace detail

// Each Geom directly in `body` (recursive=false) or anywhere beneath it.
template <class Fn>
void ForEachGeom(mj::Body& body, bool recursive, Fn&& fn) {
  detail::EmitFromSubtree<mj::Geom>(body.subtree, recursive, fn);
}
template <class Fn>
void ForEachJoint(mj::Body& body, bool recursive, Fn&& fn) {
  detail::EmitFromSubtree<mj::Joint>(body.subtree, recursive, fn);
}
template <class Fn>
void ForEachSite(mj::Body& body, bool recursive, Fn&& fn) {
  detail::EmitFromSubtree<mj::Site>(body.subtree, recursive, fn);
}
// Each Body directly in `body` (its immediate child bodies) or, recursively,
// every descendant body.
template <class Fn>
void ForEachBody(mj::Body& body, bool recursive, Fn&& fn) {
  detail::EmitFromSubtree<mj::Body>(body.subtree, recursive, fn);
}

}  // namespace ps::sdk

#endif  // PROTOSPEC_SDK_TRAVERSAL_H
