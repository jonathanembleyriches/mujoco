// ProtoSpec: shared model-core machinery (ps::sdk::internal).
//
// This is the internal seam SHARED between the ProtoSpec SDK's public verbs and
// the in-tree native compiler (cpp/compile/{native,build}.cc). It is NOT a
// stable public surface -- external consumers program against the public verbs
// aggregated in sdk.h (Name, WalkModel, WalkSubtree, Rename, ...). But unlike
// detail.h (genuinely private to the SDK), this header is a real, named contract
// with a second in-tree consumer: the compiler legitimately needs the generic
// tree walk, name access, reference-shape traits, the <default> class index and
// the reference prefixer to lower an authored Model to MuJoCo.
//
// CONTRACT: any change to a symbol here must update BOTH consumers in the same
// change -- the SDK headers (traversal.h / refs.h / classes.h / attach.h /
// builders.h) and cpp/compile. It carries no compatibility guarantee beyond
// that; it is not versioned and not exported.
//
// The object model is plain owned values (DR-2) with a generated Visit(elem, V)
// hook (visit.h) that hands a visitor every field by id + name + typed reference
// and every child / union-child list. That single hook is enough to build a
// whole-tree walk, per-field probes, name access and typed-reference detection
// with no element-specific code, so both the SDK and the compiler stay thin and
// never need regenerating when the schema grows.
#ifndef PROTOSPEC_SDK_MODEL_CORE_H
#define PROTOSPEC_SDK_MODEL_CORE_H

#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "protospec/core.h"
#include "reflect.h"
#include "types.h"
#include "visit.h"

namespace ps::sdk::internal {

namespace mj = ps::mjcf;

// --- Small type traits ---------------------------------------------------- //

template <class T>
struct is_opt : std::false_type {};
template <class T>
struct is_opt<ps::opt<T>> : std::true_type {
  using inner = T;
};

template <class T>
struct is_ref : std::false_type {};
template <class T>
struct is_ref<ps::Ref<T>> : std::true_type {
  using target = T;
};

// True when U is `opt<Ref<Target>>` (the storage shape of every typed cross
// reference in the model, DR-8); pulls the referenced element/union type out.
template <class U>
struct opt_ref {
  static constexpr bool value = false;
};
template <class T>
struct opt_ref<ps::opt<ps::Ref<T>>> {
  static constexpr bool value = true;
  using target = T;
};

// True when U is `opt<std::vector<Ref<Target>>>` -- the storage shape of a
// reference LIST (`ref<T>[]`, one space-separated attribute of names, e.g.
// <flex body="b1 b2">). Everything that scans refs generically handles both
// shapes: one name per list entry.
template <class U>
struct opt_ref_list {
  static constexpr bool value = false;
};
template <class T>
struct opt_ref_list<ps::opt<std::vector<ps::Ref<T>>>> {
  static constexpr bool value = true;
  using target = T;
};

// --- Per-field probe by id ------------------------------------------------ //
// Reads the field with a given id out of an element, matched by expected type
// U. Field ids are per-element and identical across two instances of the same
// type (same generated Visit order), so this recovers "the same field" on a
// sibling/clone. Only `field` callbacks participate (child ids are a separate
// namespace and are ignored). Used by the SDK's class-merge and by the editor's
// reflection-driven inspector (details_panel).
template <class U>
struct ConstFieldGrab {
  int target;
  const U* out = nullptr;
  template <class W>
  void field(int id, const char*, const W& v) {
    if (id == target) {
      if constexpr (std::is_same_v<W, U>) out = &v;
    }
  }
  template <class C>
  void child(int, const char*, const C&) {}
  template <class C>
  void union_child(int, const char*, const C&) {}
};

template <class U>
struct MutFieldGrab {
  int target;
  U* out = nullptr;
  template <class W>
  void field(int id, const char*, W& v) {
    if (id == target) {
      if constexpr (std::is_same_v<W, U>) out = &v;
    }
  }
  template <class C>
  void child(int, const char*, C&) {}
  template <class C>
  void union_child(int, const char*, C&) {}
};

template <class T, class U>
const U* FieldAt(const T& e, int id) {
  ConstFieldGrab<U> g{id};
  mj::Visit(e, g);
  return g.out;
}
template <class T, class U>
U* FieldAt(T& e, int id) {
  MutFieldGrab<U> g{id};
  mj::Visit(e, g);
  return g.out;
}

// --- Reference target types ----------------------------------------------- //
// The member element types of a union, straight from the generated union
// descriptor. The single source of truth for a union's spellings is the schema
// (reflect::DescribeUnion); nothing here re-lists them.
inline std::vector<mj::ElementType> UnionMemberTypes(std::string_view name) {
  const mj::reflect::UnionDescriptor& u = mj::reflect::DescribeUnion(name);
  return {u.members, u.members + u.member_count};
}

// The element types a `Ref<Target>` can name. Every ref names a single element
// type except the union targets: `Ref<TendonAny>` (the two tendon spellings)
// and `Ref<ActuatorAny>` (every actuator spelling -- one MuJoCo actuator
// namespace), whose member sets are read from the union descriptor.
template <class Target>
inline std::vector<mj::ElementType> RefTargetTypes() {
  if constexpr (std::is_same_v<Target, mj::TendonAny>) {
    return UnionMemberTypes("TendonAny");
  } else if constexpr (std::is_same_v<Target, mj::ActuatorAny>) {
    return UnionMemberTypes("ActuatorAny");
  } else {
    return {mj::element_type_of<Target>::value};
  }
}

inline bool Contains(const std::vector<mj::ElementType>& v, mj::ElementType t) {
  for (auto x : v)
    if (x == t) return true;
  return false;
}

// --- Whole-tree walk ------------------------------------------------------ //
// Visits `root`, then every element nested beneath it in document order, in a
// single generic traversal. `fn` is a callable accepting `auto& element` (its
// constness follows the walked object's). Scalar/variant fields are ignored;
// only child and union-child lists descend, which is exactly the set of nested
// elements (nested elements never live in a plain field).
template <class Fn>
struct Walker {
  Fn* fn;

  template <class U>
  void field(int, const char*, U&) {}

  template <class C>
  void child(int, const char*, C& list) {
    for (auto& p : list)
      if (p) Descend(*p);
  }

  template <class C>
  void union_child(int, const char*, C& list) {
    for (auto& item : list)
      std::visit(
          [&](auto& p) {
            if (p) Descend(*p);
          },
          item.node);
  }

  template <class E>
  void Descend(E& e) {
    (*fn)(e);
    Walker w{fn};
    mj::Visit(e, w);
  }
};

// Walk a single element subtree (root included).
template <class E, class Fn>
void WalkTree(E& root, Fn&& fn) {
  fn(root);
  Walker<std::remove_reference_t<Fn>> w{&fn};
  mj::Visit(root, w);
}

// Walk a list of owned elements and their subtrees.
template <class T, class Fn>
void WalkList(std::vector<std::unique_ptr<T>>& list, Fn&& fn) {
  for (auto& p : list)
    if (p) WalkTree(*p, fn);
}
template <class T, class Fn>
void WalkList(const std::vector<std::unique_ptr<T>>& list, Fn&& fn) {
  for (auto& p : list)
    if (p) WalkTree(*p, fn);
}

// Walk every "live" element of a Model -- every top-level section except the
// <default> tree. Class-defining elements live only under `defaults` and are
// authoring templates, not model content; operations that act on real elements
// (flatten, find-by-name outside classes, referrer scans against live refs)
// use this to skip them. `defaults` is walked explicitly where a query needs
// the class tree itself.
template <class M, class Fn>
void WalkModelLive(M& m, Fn&& fn) {
  WalkList(m.compilers, fn);
  WalkList(m.options, fn);
  WalkList(m.sizes, fn);
  WalkList(m.visuals, fn);
  WalkList(m.statistics, fn);
  WalkList(m.extensions, fn);
  WalkList(m.customs, fn);
  WalkList(m.assets, fn);
  WalkList(m.worldbody, fn);
  WalkList(m.deformables, fn);
  WalkList(m.contacts, fn);
  WalkList(m.equalitys, fn);
  WalkList(m.tendons, fn);
  WalkList(m.actuators, fn);
  WalkList(m.sensors, fn);
  WalkList(m.keyframes, fn);
}

// Walk every element of a Model, defaults included.
template <class M, class Fn>
void WalkModelAll(M& m, Fn&& fn) {
  fn(m);
  Walker<std::remove_reference_t<Fn>> w{&fn};
  mj::Visit(m, w);
}

// --- Name access ---------------------------------------------------------- //
// Most elements carry `opt<std::string> name`; a Default carries `opt<string>
// dclass` as its identity. These uniformly expose "the name" for whichever an
// element has, and nullptr for the nameless (Replicate, Config, ...).
template <class E>
const std::string* NameOf(const E& e) {
  if constexpr (std::is_same_v<E, mj::Default>) {
    return e.dclass ? &*e.dclass : nullptr;
  } else if constexpr (requires { e.name; }) {
    using NT = std::decay_t<decltype(e.name)>;
    if constexpr (is_opt<NT>::value &&
                  std::is_same_v<typename is_opt<NT>::inner, std::string>) {
      return e.name ? &*e.name : nullptr;
    } else {
      return nullptr;
    }
  } else {
    return nullptr;
  }
}

template <class E>
bool HasNameField() {
  if constexpr (std::is_same_v<E, mj::Default>) {
    return true;
  } else if constexpr (requires(E e) { e.name; }) {
    using NT = std::decay_t<decltype(std::declval<E>().name)>;
    return is_opt<NT>::value &&
           std::is_same_v<typename is_opt<NT>::inner, std::string>;
  } else {
    return false;
  }
}

// Set an element's name (creates the value if the field exists). No-op when the
// element has no name field.
template <class E>
void SetName(E& e, const std::string& name) {
  if constexpr (std::is_same_v<E, mj::Default>) {
    e.dclass = name;
  } else if constexpr (requires { e.name; }) {
    using NT = std::decay_t<decltype(e.name)>;
    if constexpr (is_opt<NT>::value &&
                  std::is_same_v<typename is_opt<NT>::inner, std::string>) {
      e.name = name;
    }
  }
}

// --- Reference prefixing -------------------------------------------------- //
// Prefix every typed reference name in one element (names handled separately).
// Cross-model namespacing (mjs_attach mirror) prefixes an entire cloned subtree;
// the compiler prefixes attach-flattened content the same way.
struct RefPrefixer {
  const std::string* prefix;
  template <class U>
  void field(int, const char*, U& v) {
    using D = std::decay_t<U>;
    if constexpr (opt_ref<D>::value) {
      if (v.has_value() && !v->name.empty()) v->name = *prefix + v->name;
    } else if constexpr (opt_ref_list<D>::value) {
      if (v.has_value())
        for (auto& r : *v)
          if (!r.name.empty()) r.name = *prefix + r.name;
    }
  }
  template <class C>
  void child(int, const char*, C&) {}
  template <class C>
  void union_child(int, const char*, C&) {}
};

}  // namespace ps::sdk::internal

#endif  // PROTOSPEC_SDK_MODEL_CORE_H
