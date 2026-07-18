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

// --- Union membership (generated-derived) --------------------------------- //
// A union element type (TendonAny, ActuatorAny, ...) stores its alternatives in
// a `std::variant<std::unique_ptr<Members>...> node`; a concrete element type
// has no such member. These traits read that variant directly, so ref-target
// checking tracks the schema with no hand-listed member set.
template <class E, class V>
struct VariantHoldsPtr : std::false_type {};
template <class E, class... Ts>
struct VariantHoldsPtr<E, std::variant<Ts...>>
    : std::disjunction<std::is_same<std::unique_ptr<E>, Ts>...> {};

template <class T, class = void>
struct union_node {
  using type = void;
};
template <class T>
struct union_node<T, std::void_t<decltype(std::declval<T&>().node)>> {
  using type = std::decay_t<decltype(std::declval<T&>().node)>;
};

// True when element type E is a valid target of a `Ref<T>`: E == T for a
// concrete target, or E a member of the union T (Ref<TendonAny> accepts
// Spatial/Fixed; Ref<ActuatorAny> accepts every actuator spelling). Consumed by
// SetRef's compile-time target check, so a target of the wrong element type is
// a hard error rather than a silent mismatch.
template <class T, class E>
constexpr bool RefAcceptsTargetImpl() {
  using Node = typename union_node<T>::type;
  if constexpr (std::is_same_v<Node, void>)
    return std::is_same_v<E, T>;
  else
    return VariantHoldsPtr<E, Node>::value;
}
template <class T, class E>
inline constexpr bool ref_accepts_target = RefAcceptsTargetImpl<T, E>();

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

// --- Name categories (shared MuJoCo namespaces) --------------------------- //
// MuJoCo names are unique within an object *category*, not per element type.
// Most element types are their own category; the spelling families share one
// namespace each: the two joint spellings, the two tendon spellings, and every
// actuator / sensor / equality spelling. Collision checks, rename rejection and
// clone re-uniquing key on the category so e.g. a FreeJoint "j" does collide
// with a Joint "j".
inline bool InUnionNamespace(std::string_view union_name, mj::ElementType t) {
  const mj::reflect::UnionDescriptor& u = mj::reflect::DescribeUnion(union_name);
  for (std::size_t i = 0; i < u.member_count; ++i)
    if (u.members[i] == t) return true;
  return false;
}

inline int NameCategory(mj::ElementType t) {
  if (t == mj::ElementType::Joint || t == mj::ElementType::FreeJoint) return -1;
  if (InUnionNamespace("TendonAny", t)) return -2;
  if (InUnionNamespace("ActuatorAny", t)) return -3;
  if (InUnionNamespace("SensorAny", t)) return -4;
  if (InUnionNamespace("EqualityAny", t)) return -5;
  return static_cast<int>(t);
}

// The keyword -> target-type table backing dynamic references. Keywords follow
// MuJoCo's mju_str2Type object names; the union-valued arms (tendon / actuator /
// sensor) read their member sets from the union descriptors so they never drift
// from the schema. Built once. The table is also the single set the coverage
// guard (DynRefKeywordGaps) checks against, so keyword lookup and drift
// detection cannot disagree.
inline const std::vector<
    std::pair<std::string_view, std::vector<mj::ElementType>>>&
DynRefTable() {
  using ET = mj::ElementType;
  static const std::vector<
      std::pair<std::string_view, std::vector<mj::ElementType>>>
      table = [] {
        std::vector<std::pair<std::string_view, std::vector<mj::ElementType>>> t;
        t.push_back({"body", {ET::Body}});
        t.push_back({"xbody", {ET::Body}});
        t.push_back({"joint", {ET::Joint, ET::FreeJoint}});
        t.push_back({"geom", {ET::Geom}});
        t.push_back({"site", {ET::Site}});
        t.push_back({"camera", {ET::Camera}});
        t.push_back({"light", {ET::Light}});
        t.push_back({"flex", {ET::Flex}});
        t.push_back({"mesh", {ET::Mesh}});
        t.push_back({"skin", {ET::Skin}});
        t.push_back({"hfield", {ET::Hfield}});
        t.push_back({"texture", {ET::Texture}});
        t.push_back({"material", {ET::Material}});
        t.push_back({"tendon", UnionMemberTypes("TendonAny")});
        t.push_back({"actuator", UnionMemberTypes("ActuatorAny")});
        t.push_back({"sensor", UnionMemberTypes("SensorAny")});
        t.push_back({"numeric", {ET::Numeric}});
        t.push_back({"text", {ET::Text}});
        t.push_back({"tuple", {ET::Tuple}});
        t.push_back({"key", {ET::Keyframe}});
        t.push_back({"plugin", {ET::PluginInstance}});
        return t;
      }();
  return table;
}

// The element types a DYNAMIC reference can name, given the runtime keyword its
// sibling type field holds (objtype="body" -> Body, ...). An unknown or empty
// keyword returns the empty set: the name is not scannable, and callers must
// treat it as opaque rather than guess a namespace.
inline std::vector<mj::ElementType> DynRefTargetTypes(std::string_view keyword) {
  for (const auto& [k, v] : DynRefTable())
    if (k == keyword) return v;
  return {};
}

// Drift guard for the dynamic keyword table. Every element type that some typed
// `Ref<T>` field in the schema names AND that is a MuJoCo runtime object must be
// reachable through at least one dynamic keyword; otherwise a frame sensor could
// name that object by objtype but the SDK's rename / delete / referrer scan
// would silently skip it. Returns the reachable-but-orphaned target types, empty
// when the table covers every referenceable runtime object.
//
// Excluded (authoring-time reference kinds, never runtime mjOBJ objects, so no
// objtype keyword names them): Default (a class/childclass naming a <default>)
// and ModelAsset (a `model` naming an attached submodel).
//
// Limitation: the authoritative keyword universe is MuJoCo's mju_str2Type, which
// this MuJoCo-independent layer cannot enumerate. This guard therefore catches a
// new referenceable family added to the SCHEMA; a keyword added to mju_str2Type
// with no schema ref target is out of its reach and must be caught where MuJoCo
// is linked (cpp/compile/native.cc consumes str2Type directly).
inline std::vector<mj::ElementType> DynRefKeywordGaps() {
  std::vector<mj::ElementType> reachable;
  for (const auto& [k, v] : DynRefTable())
    for (mj::ElementType t : v)
      if (!Contains(reachable, t)) reachable.push_back(t);

  auto resolve = [](std::string_view type_name) -> std::vector<mj::ElementType> {
    for (std::size_t i = 0; i < mj::reflect::UnionCount(); ++i) {
      const mj::reflect::UnionDescriptor& u = mj::reflect::UnionAt(i);
      if (u.name == type_name) return {u.members, u.members + u.member_count};
    }
    if (const mj::reflect::ElementDescriptor* d =
            mj::reflect::DescribeByName(type_name)) {
      return {d->type};
    }
    return {};
  };

  std::vector<mj::ElementType> gaps;
  for (std::size_t i = 0; i < mj::reflect::ElementCount(); ++i) {
    const mj::reflect::ElementDescriptor& e = mj::reflect::ElementAt(i);
    for (std::size_t f = 0; f < e.field_count; ++f) {
      const mj::reflect::FieldDescriptor& fd = e.fields[f];
      if (fd.kind != mj::reflect::FieldKind::Ref) continue;
      for (mj::ElementType t : resolve(fd.type_name)) {
        if (t == mj::ElementType::Default || t == mj::ElementType::ModelAsset)
          continue;
        if (!Contains(reachable, t) && !Contains(gaps, t)) gaps.push_back(t);
      }
    }
  }
  return gaps;
}

}  // namespace ps::sdk::detail

#endif  // PROTOSPEC_SDK_DETAIL_H
