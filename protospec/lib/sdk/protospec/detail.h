// ProtoSpec SDK internals: genuinely-private helpers, used ONLY within the SDK
// headers themselves. Nothing here is part of the public surface (see the
// sibling headers builders.h / traversal.h / refs.h / classes.h / attach.h),
// and no in-tree consumer outside cpp/sdk refers to these private symbols.
//
// The generic tree machinery that the SDK's public verbs AND the in-tree native
// compiler both program against (the whole-tree walk, name access, reference
// traits, the <default> class index, the ref prefixer) is NOT here: it lives in
// model_core.h under ps::sdk::internal, a named shared seam with its own
// contract. This header carries the reflection-derived pieces the SDK keeps to
// itself: per-field probes, the union/ref-target descriptors, the name-category
// folding and the dynamic-keyword table + its drift guard.
//
// For continuity, the moved ps::sdk::internal names the SDK headers still spell
// as `detail::` are re-exported into this namespace below -- a spelling
// convenience for in-tree SDK/test code, not a widening of the surface.
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
#include "protospec/model_core.h"
#include "reflect.h"
#include "types.h"
#include "visit.h"

namespace ps::sdk::detail {

namespace mj = ps::mjcf;

// --- Shared-core re-exports ----------------------------------------------- //
// The generic machinery moved to ps::sdk::internal (model_core.h), re-exported
// under the `detail::` spelling the SDK headers and in-tree tests already use.
// The canonical home is model_core.h; these are aliases, not definitions.
using internal::Contains;
using internal::ConstFieldGrab;
using internal::FieldAt;
using internal::HasNameField;
using internal::is_opt;
using internal::is_ref;
using internal::MutFieldGrab;
using internal::NameOf;
using internal::opt_ref;
using internal::opt_ref_list;
using internal::RefPrefixer;
using internal::RefTargetTypes;
using internal::SetName;
using internal::UnionMemberTypes;
using internal::Walker;
using internal::WalkList;
using internal::WalkModelAll;
using internal::WalkModelLive;
using internal::WalkTree;

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
