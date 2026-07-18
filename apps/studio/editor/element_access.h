// ProtoSpec Studio: typed element access over a Model tree (ps::studio, ours).
//
// One home for the two element-lookup shapes the editor repeats everywhere:
//   * find the element carrying a creation serial (FindSerial / FindSerialAs), and
//   * dispatch a type-erased spatial pointer to its concrete type (DispatchSpatial).
//
// Windowless and header-only (templates over the generated tree); no ImGui, no
// EditorContext, so both the pure gizmo math (transform_math) and the ctx-aware
// ops (editor_ops / authoring_ops) can share it. FindSerial is the single walk a
// later swap to the SDK's own find verb collapses to.

#ifndef PS_STUDIO_EDITOR_ELEMENT_ACCESS_H_
#define PS_STUDIO_EDITOR_ELEMENT_ACCESS_H_

#include <cstdint>
#include <type_traits>
#include <utility>

#include "protospec/detail.h"  // ps::sdk::detail::WalkModelAll
#include "types.h"             // mj::Model, element_type_of, ElementType

namespace ps::studio {

namespace mj = ps::mjcf;

// The element carrying `serial` (any section, defaults included) paired with its
// ElementType, or {nullptr, Model} when none does. A const `model` yields a
// const pointer. serial 0 is the reserved "no element" sentinel and never
// matches. This is THE walk-by-serial for the editor; the field-reading finders
// (SelectBySerial's desc, LayerOfSerial's loc) keep their own typed walks.
template <class Model>
auto FindSerialWithType(Model& model, std::uint64_t serial) {
  using Ptr = std::conditional_t<std::is_const_v<Model>, const void*, void*>;
  std::pair<Ptr, mj::ElementType> out{nullptr, mj::ElementType::Model};
  if (serial == 0) return out;
  ps::sdk::detail::WalkModelAll(model, [&](auto& e) {
    using E = std::decay_t<decltype(e)>;
    if (out.first) return;
    if constexpr (!std::is_same_v<E, mj::Model>) {
      if constexpr (requires { e.serial; }) {
        if (e.serial == serial) {
          out.first = &e;
          out.second = mj::element_type_of<E>::value;
        }
      }
    }
  });
  return out;
}

// Type-erased pointer to the element carrying `serial`, or nullptr.
template <class Model>
auto FindSerial(Model& model, std::uint64_t serial) {
  return FindSerialWithType(model, serial).first;
}

// Typed pointer to the element carrying `serial`, iff it is a `T` (else
// nullptr). A const `model` yields a `const T*`.
template <class T, class Model>
auto FindSerialAs(Model& model, std::uint64_t serial)
    -> std::conditional_t<std::is_const_v<Model>, const T*, T*> {
  std::conditional_t<std::is_const_v<Model>, const T*, T*> out = nullptr;
  if (serial == 0) return out;
  ps::sdk::detail::WalkModelAll(model, [&](auto& e) {
    using E = std::decay_t<decltype(e)>;
    if constexpr (std::is_same_v<E, T>) {
      if (!out && e.serial == serial) out = &e;
    }
  });
  return out;
}

// The six spatial families the gizmo drives, in one place. Any element outside
// this set has no pos/orient a transform can act on.
inline bool IsSpatialType(mj::ElementType t) {
  return t == mj::ElementType::Body || t == mj::ElementType::Geom ||
         t == mj::ElementType::Site || t == mj::ElementType::Camera ||
         t == mj::ElementType::Light || t == mj::ElementType::Frame;
}

// Cast a type-erased spatial `ptr` to its concrete type per `type` and hand it
// to `visit` (a generic lambda callable with mj::Body/Geom/Site/Camera/Light/
// Frame&). `ptr` MUST point to an element of `type`. Returns false and does not
// call `visit` for a non-spatial `type`. Replaces the six-case
// switch->static_cast ladder the transform ops each spelled out.
template <class Visitor>
bool DispatchSpatial(mj::ElementType type, void* ptr, Visitor&& visit) {
  switch (type) {
    case mj::ElementType::Body:   visit(*static_cast<mj::Body*>(ptr));   return true;
    case mj::ElementType::Geom:   visit(*static_cast<mj::Geom*>(ptr));   return true;
    case mj::ElementType::Site:   visit(*static_cast<mj::Site*>(ptr));   return true;
    case mj::ElementType::Camera: visit(*static_cast<mj::Camera*>(ptr)); return true;
    case mj::ElementType::Light:  visit(*static_cast<mj::Light*>(ptr));  return true;
    case mj::ElementType::Frame:  visit(*static_cast<mj::Frame*>(ptr));  return true;
    default: return false;
  }
}

}  // namespace ps::studio

#endif  // PS_STUDIO_EDITOR_ELEMENT_ACCESS_H_
