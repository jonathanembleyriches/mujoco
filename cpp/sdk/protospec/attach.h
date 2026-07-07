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

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "protospec/builders.h"
#include "protospec/detail.h"
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

}  // namespace ps::sdk

#endif  // PROTOSPEC_SDK_ATTACH_H
