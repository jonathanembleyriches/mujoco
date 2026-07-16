// ProtoSpec SDK: attach (namespaced subtree splice).
//
// Attach deep-clones a body subtree from one model (or another point in the
// same model) into a parent body, prefixing every name AND every internal typed
// reference of the clone so the graft is self-contained: joints/sites/geoms the
// clone refers to resolve within the clone, never colliding with the host. This
// mirrors MuJoCo's mjs_attach namespacing but is a pure tree operation -- no
// compile, no source mutation (the source is cloned, not moved, unlike
// mjs_attach which mutates its source child).
//
// A name collision against the host is reported and blocks the attach (the host
// is left untouched), rather than producing two elements of one type sharing a
// name (which a valid model forbids, Q-NAMES). Assets and default classes the
// source relied on are NOT copied; a clone's prefixed asset/class references are
// left for the caller to satisfy (bring the assets over, or clear the refs).
#ifndef PROTOSPEC_SDK_ATTACH_H
#define PROTOSPEC_SDK_ATTACH_H

#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <variant>
#include <vector>

#include "protospec/builders.h"
#include "protospec/detail.h"
#include "protospec/refs.h"
#include "protospec/traversal.h"
#include "types.h"
#include "visit.h"

namespace ps::sdk {

namespace mj = ps::mjcf;

struct AttachResult {
  mj::Body* attached = nullptr;         // the grafted clone, when ok
  bool ok = false;                      // false when a collision blocked it
  std::vector<std::string> collisions;  // "type:name" clashes with the host
};

namespace detail {

// Prefix every typed reference name in one element (names handled separately).
struct RefPrefixer {
  const std::string* prefix;
  template <class U>
  void field(int, const char*, U& v) {
    using D = std::decay_t<U>;
    if constexpr (opt_ref<D>::value) {
      if (v.has_value() && !v->name.empty()) v->name = *prefix + v->name;
    }
  }
  template <class C>
  void child(int, const char*, C&) {}
  template <class C>
  void union_child(int, const char*, C&) {}
};

// Prefix every name and every internal reference throughout a cloned subtree.
inline void PrefixSubtree(mj::Body& clone, const std::string& prefix) {
  WalkTree(clone, [&](auto& e) {
    if (const std::string* nm = NameOf(e)) SetName(e, prefix + *nm);
    RefPrefixer p{&prefix};
    mj::Visit(e, p);
  });
}

inline std::string Key(mj::ElementType t, const std::string& name) {
  return std::to_string(static_cast<int>(t)) + ':' + name;
}

// Every (type,name) an element of the host already uses.
inline std::unordered_set<std::string> HostNames(const mj::Model& model) {
  std::unordered_set<std::string> names;
  WalkModelAll(const_cast<mj::Model&>(model), [&](auto& e) {
    using E = std::decay_t<decltype(e)>;
    if (const std::string* nm = NameOf(e))
      names.insert(Key(mj::element_type_of<E>::value, *nm));
  });
  return names;
}

}  // namespace detail

// Clone `src` (a body subtree), prefix all its names and internal references,
// check for collisions against the host `model`, and on success splice it under
// `parent`. On collision nothing is attached and the clashes are reported.
inline AttachResult Attach(mj::Model& model, mj::Body& parent,
                           const mj::Body& src, const std::string& prefix) {
  AttachResult result;
  std::unique_ptr<mj::Body> clone = mj::Clone(src);
  detail::PrefixSubtree(*clone, prefix);

  const std::unordered_set<std::string> host = detail::HostNames(model);
  detail::WalkTree(*clone, [&](auto& e) {
    using E = std::decay_t<decltype(e)>;
    if (const std::string* nm = detail::NameOf(e)) {
      if (host.count(detail::Key(mj::element_type_of<E>::value, *nm)))
        result.collisions.push_back(
            std::string(mj::reflect::Describe(mj::element_type_of<E>::value)
                            .name) +
            ":" + *nm);
    }
  });

  if (!result.collisions.empty()) {
    result.ok = false;
    return result;  // host untouched
  }

  result.attached =
      &detail::PushUnion<mj::Body>(parent.subtree, std::move(clone));
  result.ok = true;
  return result;
}

// Attach every top-level body of another model's worldbody under `parent`,
// sharing one prefix. Collisions from any body abort the whole splice (nothing
// is attached) and are aggregated in the result.
inline AttachResult AttachModel(mj::Model& model, mj::Body& parent,
                                const mj::Model& src,
                                const std::string& prefix) {
  AttachResult result;
  if (src.worldbody.empty() || !src.worldbody.front()) {
    result.ok = true;
    return result;
  }

  // Dry-run collision scan across all source bodies first.
  const std::unordered_set<std::string> host = detail::HostNames(model);
  std::vector<std::unique_ptr<mj::Body>> clones;
  for (const auto& item : src.worldbody.front()->subtree) {
    std::visit(
        [&](const auto& p) {
          using E = std::decay_t<decltype(*p)>;
          if constexpr (std::is_same_v<E, mj::Body>) {
            if (p) {
              auto clone = mj::Clone(*p);
              detail::PrefixSubtree(*clone, prefix);
              detail::WalkTree(*clone, [&](auto& e) {
                using X = std::decay_t<decltype(e)>;
                if (const std::string* nm = detail::NameOf(e)) {
                  if (host.count(
                          detail::Key(mj::element_type_of<X>::value, *nm)))
                    result.collisions.push_back(
                        std::string(
                            mj::reflect::Describe(mj::element_type_of<X>::value)
                                .name) +
                        ":" + *nm);
                }
              });
              clones.push_back(std::move(clone));
            }
          }
        },
        item.node);
  }

  if (!result.collisions.empty()) {
    result.ok = false;
    return result;
  }
  for (auto& c : clones)
    result.attached =
        &detail::PushUnion<mj::Body>(parent.subtree, std::move(c));
  result.ok = true;
  return result;
}

// --- Duplicate (same-model deep clone) ------------------------------------ //

namespace detail {

// MuJoCo names are unique within an object category. Every element type is its
// own category except the two joint spellings, which share one (so a duplicated
// joint does not collide with a free joint of the same name and vice versa).
inline int NameCategory(mj::ElementType t) {
  if (t == mj::ElementType::Joint || t == mj::ElementType::FreeJoint) return -1;
  return static_cast<int>(t);
}

// Insert a fresh-serial deep clone of the element at `target` immediately after
// it in its owning child/union list, and report the clone's (stable heap)
// address. The concrete element type is recovered inside the generated Visit
// hook, so mj::Clone is the only per-type code -- no whole-model walk is
// instantiated per element type.
struct Duplicator {
  const void* target;
  void** clone_out;
  bool* done;

  template <class U>
  void field(int, const char*, U&) {}

  template <class C>
  void child(int, const char*, C& list) {
    if (*done) return;
    for (std::size_t i = 0; i < list.size(); ++i) {
      if (list[i] && static_cast<const void*>(list[i].get()) == target) {
        auto clone = mj::Clone(*list[i]);
        *clone_out = clone.get();
        list.insert(list.begin() + static_cast<std::ptrdiff_t>(i) + 1,
                    std::move(clone));
        *done = true;
        return;
      }
    }
    for (auto& p : list) {
      if (*done) return;
      if (p) {
        Duplicator d{target, clone_out, done};
        mj::Visit(*p, d);
      }
    }
  }

  template <class C>
  void union_child(int, const char*, C& list) {
    if (*done) return;
    for (std::size_t i = 0; i < list.size(); ++i) {
      bool match = false;
      std::visit(
          [&](auto& p) {
            if (p && static_cast<const void*>(p.get()) == target) match = true;
          },
          list[i].node);
      if (match) {
        typename C::value_type node;
        std::visit(
            [&](auto& p) {
              auto clone = mj::Clone(*p);
              *clone_out = clone.get();
              node.node = std::move(clone);
            },
            list[i].node);
        list.insert(list.begin() + static_cast<std::ptrdiff_t>(i) + 1,
                    std::move(node));
        *done = true;
        return;
      }
    }
    for (auto& item : list) {
      if (*done) return;
      std::visit(
          [&](auto& p) {
            if (p) {
              Duplicator d{target, clone_out, done};
              mj::Visit(*p, d);
            }
          },
          item.node);
    }
  }
};

// Uniquely rename every named element of the freshly-cloned subtree rooted at
// `clone`, and remap the clone's INTERNAL typed references to the new names;
// references pointing OUTSIDE the clone are left untouched. All bookkeeping runs
// through generic-lambda walks, never a per-type templated op with an embedded
// walk.
inline void RemapClone(mj::Model& model, const void* clone) {
  ParentMap pm(model);
  auto in_sub = [&](const void* p) -> bool {
    for (const void* q = p; q;) {
      if (q == clone) return true;
      const ParentMap::Node* n = pm.Lookup(q);
      if (!n) break;
      q = n->parent;
    }
    return false;
  };

  struct SubElem {
    mj::ElementType type;
    std::string name;
    std::function<void(const std::string&)> set;
  };
  std::vector<SubElem> sub;
  std::map<int, std::unordered_set<std::string>> reserved;  // outside names
  WalkModelAll(model, [&](auto& e) {
    using E = std::decay_t<decltype(e)>;
    if constexpr (!std::is_same_v<E, mj::Model>) {
      const std::string* nm = NameOf(e);
      if (!nm) return;
      const mj::ElementType t = mj::element_type_of<E>::value;
      if (in_sub(&e)) {
        auto* ep = &e;
        sub.push_back(
            {t, *nm, [ep](const std::string& s) { SetName(*ep, s); }});
      } else {
        reserved[NameCategory(t)].insert(*nm);
      }
    }
  });

  struct Ren {
    mj::ElementType type;
    std::string oldn;
    std::string newn;
  };
  std::vector<Ren> renamed;
  for (auto& se : sub) {
    const int c = NameCategory(se.type);
    std::string cand = se.name;
    for (int k = 1; reserved[c].count(cand); ++k)
      cand = se.name + "_" + std::to_string(k);
    reserved[c].insert(cand);
    if (cand != se.name) {
      renamed.push_back({se.type, se.name, cand});
      se.set(cand);
    }
  }
  if (renamed.empty()) return;

  WalkModelAll(model, [&](auto& e) {
    using E = std::decay_t<decltype(e)>;
    if constexpr (!std::is_same_v<E, mj::Model>) {
      if (!in_sub(&e)) return;
      ScanRefs(e, [&](int, const char*, std::string& rn,
                      const std::vector<mj::ElementType>& tgts) {
        for (const Ren& r : renamed) {
          if (r.oldn == rn && Contains(tgts, r.type)) {
            rn = r.newn;
            break;
          }
        }
      });
    }
  });
}

}  // namespace detail

// Deep-clone `elem` (a whole subtree) as its next sibling in the same model,
// with fresh serials. Names in the clone are re-uniqued against the rest of the
// model, and references INTERNAL to the clone are remapped to the new names;
// references pointing outside the clone are preserved. Returns the clone's root
// element (type-erased; its ElementType matches `elem`'s), or nullptr when
// `elem` is not found. Unlike Attach, this is a same-model, unprefixed
// duplicate -- the everyday "copy this element" verb.
inline void* Duplicate(mj::Model& model, const void* elem) {
  void* clone = nullptr;
  bool done = false;
  detail::Duplicator d{elem, &clone, &done};
  mj::Visit(model, d);
  if (!done || !clone) return nullptr;
  detail::RemapClone(model, clone);
  return clone;
}

// --- Reparent (pure-tree move) -------------------------------------------- //

struct ReparentResult {
  bool ok = false;
  std::string reason;  // why the move was rejected (empty on success)
};

namespace detail {

// Locate the BodyChildAny node whose held element is `target`, returning the
// owning list + index. Searches a subtree recursively (Body/Frame carry their
// own subtree).
inline bool FindOwningList(std::vector<mj::BodyChildAny>& sub, const void* target,
                           std::vector<mj::BodyChildAny>** out_list,
                           std::size_t* out_i) {
  for (std::size_t i = 0; i < sub.size(); ++i) {
    bool match = false;
    std::visit(
        [&](auto& up) {
          if (up && static_cast<const void*>(up.get()) == target) match = true;
        },
        sub[i].node);
    if (match) {
      *out_list = &sub;
      *out_i = i;
      return true;
    }
    bool rec = false;
    std::visit(
        [&](auto& up) {
          using T = std::decay_t<decltype(*up)>;
          if constexpr (std::is_same_v<T, mj::Body> ||
                        std::is_same_v<T, mj::Frame>) {
            if (up && FindOwningList(up->subtree, target, out_list, out_i))
              rec = true;
          }
        },
        sub[i].node);
    if (rec) return true;
  }
  return false;
}

inline bool FindOwningListInModel(mj::Model& model, const void* target,
                                  std::vector<mj::BodyChildAny>** out_list,
                                  std::size_t* out_i) {
  for (auto& w : model.worldbody) {
    if (w && FindOwningList(w->subtree, target, out_list, out_i)) return true;
  }
  return false;
}

// The subtree list of a container: a Body, a Frame, or the world (parent ==
// nullptr, or the world body pointer). Returns nullptr when `parent` is neither.
inline std::vector<mj::BodyChildAny>* SubtreeOf(mj::Model& model,
                                                const void* parent) {
  if (parent == nullptr) return &World(model).subtree;
  std::vector<mj::BodyChildAny>* out = nullptr;
  WalkModelAll(model, [&](auto& e) {
    using E = std::decay_t<decltype(e)>;
    if (out) return;
    if constexpr (std::is_same_v<E, mj::Body> || std::is_same_v<E, mj::Frame>) {
      if (static_cast<const void*>(&e) == parent) out = &e.subtree;
    }
  });
  return out;
}

inline void CollectSubtreePtrs(mj::BodyChildAny& node,
                               std::unordered_set<const void*>& out) {
  std::visit(
      [&](auto& up) {
        if (up)
          WalkTree(*up, [&](auto& e) {
            out.insert(static_cast<const void*>(&e));
          });
      },
      node.node);
}

}  // namespace detail

// Move the body-context child `elem` (Body / Geom / Joint / FreeJoint / Site /
// Camera / Light / Frame) out of its current container and into `new_parent`
// (a Body or Frame; nullptr == the world body). This is a PURE TREE operation:
// the element keeps its authored local pose, so its WORLD pose changes with the
// new parent -- pose-preserving reparent is a compile-aware concern that stays
// with the bridge/editor (it needs the compiled parent world pose, which the SDK
// deliberately does not compute). Rejects, leaving the model untouched:
//   * `elem` is not a movable body-context child;
//   * `new_parent` is not a Body or Frame (nor the world);
//   * `new_parent` is `elem` itself or lies inside `elem`'s subtree (a cycle).
inline ReparentResult Reparent(mj::Model& model, const void* elem,
                               void* new_parent) {
  ReparentResult r;

  std::vector<mj::BodyChildAny>* src_list = nullptr;
  std::size_t src_i = 0;
  if (!detail::FindOwningListInModel(model, elem, &src_list, &src_i)) {
    r.reason = "element is not a movable body-context child";
    return r;
  }
  std::vector<mj::BodyChildAny>* dst = detail::SubtreeOf(model, new_parent);
  if (!dst) {
    r.reason = "target is not a body or frame";
    return r;
  }

  // Cycle rejection: the destination container must be neither the moved element
  // nor anything inside its subtree.
  std::unordered_set<const void*> subptrs;
  detail::CollectSubtreePtrs((*src_list)[src_i], subptrs);
  if (new_parent != nullptr && subptrs.count(new_parent)) {
    r.reason = "cannot reparent into its own subtree";
    return r;
  }

  mj::BodyChildAny moved = std::move((*src_list)[src_i]);
  src_list->erase(src_list->begin() + static_cast<std::ptrdiff_t>(src_i));
  dst->push_back(std::move(moved));
  r.ok = true;
  return r;
}

}  // namespace ps::sdk

#endif  // PROTOSPEC_SDK_ATTACH_H
