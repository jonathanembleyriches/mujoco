// ProtoSpec SDK internals: the generic tree machinery every public SDK
// operation is built on.
//
// The object model is plain owned values (DR-2) with a generated Visit(elem, V)
// hook (visit.h) that hands a visitor every field by id + name + typed
// reference and every child / union-child list. That single hook is enough to
// build a whole-tree walk, per-field probes, name access, and typed-reference
// detection without any element-specific code -- so the SDK stays a thin,
// uniform layer that never needs regenerating when the schema grows.
//
// Nothing here is part of the public surface; see the sibling headers
// (builders.h / traversal.h / refs.h / classes.h / attach.h) for that.
#ifndef PROTOSPEC_SDK_DETAIL_H
#define PROTOSPEC_SDK_DETAIL_H

#include <cstdint>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include "protospec/core.h"
#include "reflect.h"
#include "types.h"
#include "visit.h"

namespace ps::sdk::detail {

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

// --- Per-field probe by id ------------------------------------------------ //
// Reads the field with a given id out of an element, matched by expected type
// U. Field ids are per-element and identical across two instances of the same
// type (same generated Visit order), so this recovers "the same field" on a
// sibling/clone. Only `field` callbacks participate (child ids are a separate
// namespace and are ignored).
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

// --- Type-erased element handle ------------------------------------------- //
// (pointer, runtime element type) pair, for parent maps and diagnostics where
// heterogeneous elements share one container.
struct Handle {
  const void* ptr = nullptr;
  mj::ElementType type{};
  bool operator==(const Handle& o) const {
    return ptr == o.ptr && type == o.type;
  }
  explicit operator bool() const { return ptr != nullptr; }
};

template <class E>
Handle MakeHandle(const E& e) {
  return Handle{&e, mj::element_type_of<std::decay_t<E>>::value};
}

// --- Reference target types ----------------------------------------------- //
// The element types a `Ref<Target>` can name. Every ref names a single element
// type except the union targets: `Ref<TendonAny>` (the two tendon spellings)
// and `Ref<ActuatorAny>` (every actuator spelling -- one MuJoCo actuator
// namespace).
template <class Target>
inline std::vector<mj::ElementType> RefTargetTypes() {
  if constexpr (std::is_same_v<Target, mj::TendonAny>) {
    return {mj::ElementType::Spatial, mj::ElementType::Fixed};
  } else if constexpr (std::is_same_v<Target, mj::ActuatorAny>) {
    return {mj::ElementType::ActuatorGeneral, mj::ElementType::Motor,
            mj::ElementType::Position,        mj::ElementType::Velocity,
            mj::ElementType::IntVelocity,     mj::ElementType::Damper,
            mj::ElementType::Cylinder,        mj::ElementType::Muscle,
            mj::ElementType::Adhesion,        mj::ElementType::DcMotor,
            mj::ElementType::ActuatorPlugin};
  } else {
    return {mj::element_type_of<Target>::value};
  }
}

inline bool Contains(const std::vector<mj::ElementType>& v, mj::ElementType t) {
  for (auto x : v)
    if (x == t) return true;
  return false;
}

// The element types a DYNAMIC reference can name, given the runtime keyword its
// sibling type field holds (objtype="body" -> Body, ...). Keywords follow
// MuJoCo's mju_str2Type object names. An unknown or empty keyword returns the
// empty set: the name is not scannable, and callers must treat it as opaque
// rather than guess a namespace.
inline std::vector<mj::ElementType> DynRefTargetTypes(std::string_view keyword) {
  if (keyword == "body" || keyword == "xbody") return {mj::ElementType::Body};
  if (keyword == "joint")
    return {mj::ElementType::Joint, mj::ElementType::FreeJoint};
  if (keyword == "geom") return {mj::ElementType::Geom};
  if (keyword == "site") return {mj::ElementType::Site};
  if (keyword == "camera") return {mj::ElementType::Camera};
  if (keyword == "light") return {mj::ElementType::Light};
  if (keyword == "flex") return {mj::ElementType::Flex};
  if (keyword == "mesh") return {mj::ElementType::Mesh};
  if (keyword == "skin") return {mj::ElementType::Skin};
  if (keyword == "hfield") return {mj::ElementType::Hfield};
  if (keyword == "texture") return {mj::ElementType::Texture};
  if (keyword == "material") return {mj::ElementType::Material};
  if (keyword == "tendon")
    return {mj::ElementType::Spatial, mj::ElementType::Fixed};
  if (keyword == "actuator") return RefTargetTypes<mj::ActuatorAny>();
  if (keyword == "numeric") return {mj::ElementType::Numeric};
  if (keyword == "text") return {mj::ElementType::Text};
  if (keyword == "tuple") return {mj::ElementType::Tuple};
  if (keyword == "key") return {mj::ElementType::Keyframe};
  if (keyword == "plugin") return {mj::ElementType::PluginInstance};
  return {};
}

}  // namespace ps::sdk::detail

#endif  // PROTOSPEC_SDK_DETAIL_H
