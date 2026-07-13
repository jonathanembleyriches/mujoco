// ProtoSpec Studio: the generated Details panel (ps::studio, ours).
//
// One reflection-driven inspector renders any of the schema's element types with
// zero per-type UI code (plan Section 4). This header holds the windowless core:
// the FieldDescriptor -> widget classifier, the reference-combo population and
// dangling detection, enum keyword enumeration, InlineVec bound helpers, and the
// presence-layer classifier (authored / class-inherited / IDL-default / unset).
// The ImGui rendering that consumes all of this lives in details_panel.cc; the
// split keeps every mapping decision unit-testable without a window.
#ifndef PS_STUDIO_EDITOR_DETAILS_PANEL_H_
#define PS_STUDIO_EDITOR_DETAILS_PANEL_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "editor/editor_context.h"
#include "keywords.h"
#include "protospec/detail.h"
#include "protospec/traversal.h"
#include "reflect.h"
#include "types.h"
#include "visit.h"

namespace ps::studio::details {

namespace mj = ps::mjcf;
namespace reflect = ps::mjcf::reflect;
namespace sdkd = ps::sdk::detail;

// --- Value-shape traits ---------------------------------------------------- //
// The generic renderer dispatches on the stored C++ type of each field. The opt
// and Ref shapes come from the SDK (sdkd::is_opt / is_ref); the container and
// variant shapes are recognised here.

template <class T>
struct is_std_array : std::false_type {};
template <class X, std::size_t N>
struct is_std_array<std::array<X, N>> : std::true_type {
  using elem = X;
  static constexpr std::size_t size = N;
};

template <class T>
struct is_inline_vec : std::false_type {};
template <class X, std::size_t N>
struct is_inline_vec<ps::InlineVec<X, N>> : std::true_type {
  using elem = X;
  static constexpr std::size_t capacity = N;
};

template <class T>
struct is_std_vector : std::false_type {};
template <class X>
struct is_std_vector<std::vector<X>> : std::true_type {
  using elem = X;
};

template <class T>
struct is_variant : std::false_type {};
template <class... A>
struct is_variant<std::variant<A...>> : std::true_type {};

// --- Widget classification (the reflection-coverage contract) --------------- //
// Every FieldDescriptor maps to exactly one widget shape. `Unhandled` is only
// returned for a (kind, arity) pair the classifier does not know; the coverage
// test asserts the whole schema never produces it, so a schema change that adds
// a new shape surfaces as a test failure naming the field, not a silent gap.
enum class WidgetKind {
  Checkbox,          // Bool scalar
  IntScalar,         // Int32 / Uint64 scalar
  RealScalar,        // Float / Double scalar
  Text,              // String scalar
  EnumCombo,         // Enum scalar
  EnumSet,           // Enum, variable arity: keyword-set checkbox row
  RefCombo,          // Ref<T> scalar
  Variant,           // orientation / shape / intrinsics / inertia / source
  IntRow,            // Int32, fixed / range / unbounded
  RealRow,           // Float / Double, fixed / range / unbounded
  Unhandled,
};

inline WidgetKind ClassifyField(const reflect::FieldDescriptor& fd) {
  const bool scalar = fd.arity == reflect::ArityKind::Scalar;
  switch (fd.kind) {
    case reflect::FieldKind::Bool:
      return WidgetKind::Checkbox;
    case reflect::FieldKind::String:
      return WidgetKind::Text;
    case reflect::FieldKind::Ref:
      return WidgetKind::RefCombo;
    case reflect::FieldKind::Variant:
      return WidgetKind::Variant;
    case reflect::FieldKind::Int32:
    case reflect::FieldKind::Uint64:
      return scalar ? WidgetKind::IntScalar : WidgetKind::IntRow;
    case reflect::FieldKind::Float:
    case reflect::FieldKind::Double:
      return scalar ? WidgetKind::RealScalar : WidgetKind::RealRow;
    case reflect::FieldKind::Enum:
      return scalar ? WidgetKind::EnumCombo : WidgetKind::EnumSet;
  }
  return WidgetKind::Unhandled;
}

inline std::string_view WidgetKindName(WidgetKind w) {
  switch (w) {
    case WidgetKind::Checkbox: return "Checkbox";
    case WidgetKind::IntScalar: return "IntScalar";
    case WidgetKind::RealScalar: return "RealScalar";
    case WidgetKind::Text: return "Text";
    case WidgetKind::EnumCombo: return "EnumCombo";
    case WidgetKind::EnumSet: return "EnumSet";
    case WidgetKind::RefCombo: return "RefCombo";
    case WidgetKind::Variant: return "Variant";
    case WidgetKind::IntRow: return "IntRow";
    case WidgetKind::RealRow: return "RealRow";
    case WidgetKind::Unhandled: return "Unhandled";
  }
  return "?";
}

// --- Field grouping -------------------------------------------------------- //
// Transform-ish fields lead the panel under their own header (plan Section 4);
// everything else follows in schema order.
inline bool IsTransformField(const reflect::FieldDescriptor& fd) {
  return fd.xml == "pos" || fd.xml == "orient" || fd.xml == "size";
}

// --- Reference targets ----------------------------------------------------- //
// A Ref field's `type_name` is either a single element type or a union of them
// (only Ref<TendonAny> today). Expand it to the concrete element types a combo
// should offer.
inline std::vector<mj::ElementType> RefTargets(std::string_view type_name) {
  for (std::size_t i = 0; i < reflect::UnionCount(); ++i) {
    const reflect::UnionDescriptor& u = reflect::UnionAt(i);
    if (u.name == type_name) {
      return std::vector<mj::ElementType>(u.members, u.members + u.member_count);
    }
  }
  if (const reflect::ElementDescriptor* d = reflect::DescribeByName(type_name)) {
    return {d->type};
  }
  return {};
}

// Every authored name of an element whose type is one of `targets`, in document
// order -- the candidate list a Ref combo offers. Class elements are included
// (a Ref<Default> resolves against the class tree).
inline std::vector<std::string> RefCandidates(
    const mj::Model& model, const std::vector<mj::ElementType>& targets) {
  std::vector<std::string> out;
  sdkd::WalkModelAll(const_cast<mj::Model&>(model), [&](auto& e) {
    using E = std::decay_t<decltype(e)>;
    if constexpr (requires { mj::element_type_of<E>::value; }) {
      if (sdkd::Contains(targets, mj::element_type_of<E>::value)) {
        if (const std::string* nm = sdkd::NameOf(e); nm && !nm->empty()) {
          out.push_back(*nm);
        }
      }
    }
  });
  return out;
}

// A non-empty reference whose name matches no candidate is dangling (the
// referent was renamed or deleted); the panel keeps the text and flags it.
inline bool RefIsDangling(const std::vector<std::string>& candidates,
                          std::string_view name) {
  if (name.empty()) return false;
  for (const std::string& c : candidates) {
    if (c == name) return false;
  }
  return true;
}

// --- Enum keyword enumeration ---------------------------------------------- //
// The IDL enums are contiguous from 0 and ToMjcf() returns "" past the last
// value, so the keyword table itself enumerates the domain -- no separate values
// table is generated or needed.
template <class E>
std::vector<std::string_view> EnumLabels() {
  std::vector<std::string_view> out;
  for (int i = 0; i < 256; ++i) {
    std::string_view s = mj::ToMjcf(static_cast<E>(i));
    if (s.empty()) break;
    out.push_back(s);
  }
  return out;
}

template <class E>
int EnumIndexOf(E value) {
  return static_cast<int>(value);
}

// --- InlineVec bounds ------------------------------------------------------ //
// A range-arity attribute stores up to its capacity; the panel lets the filled
// count grow or shrink within [arity_min, capacity]. Range bounds are advisory
// (validation, not a type invariant, per InlineVec's contract), so the panel
// clamps display edits to the storage capacity and surfaces arity_min as the
// floor.
inline bool InlineVecCanGrow(std::size_t filled, std::size_t capacity) {
  return filled < capacity;
}
inline bool InlineVecCanShrink(std::size_t filled, int arity_min) {
  return static_cast<int>(filled) > arity_min;
}

// --- Presence layers (the differentiator) ---------------------------------- //
// A defaultable field draws from four layers, highest priority first: authored
// on the element, inherited from the class chain, the IDL default, or nothing.
// The classifier is pure over three booleans the caller reads off the element
// and its class-only / full sdk::Effective() copies, so both the renderer and
// the tests share one definition of the badge state.
enum class Presence {
  Required,     // non-optional field: always present, no badge
  Authored,     // set on the element itself
  Inherited,    // unset here, supplied by the class chain
  DefaultIdl,   // unset here and in the class chain, supplied by the IDL default
  Unset,        // unset everywhere; no effective value to show
};

inline Presence PresenceFromLayers(bool optional, bool authored, bool inherited,
                                   bool has_default) {
  if (!optional) return Presence::Required;
  if (authored) return Presence::Authored;
  if (inherited) return Presence::Inherited;
  if (has_default) return Presence::DefaultIdl;
  return Presence::Unset;
}

inline std::string_view PresenceBadge(Presence p) {
  switch (p) {
    case Presence::Required: return "";
    case Presence::Authored: return "";
    case Presence::Inherited: return "inherited";
    case Presence::DefaultIdl: return "default";
    case Presence::Unset: return "unset";
  }
  return "";
}

// --- Panel registration ---------------------------------------------------- //
// Self-registers the "Details" GuiPlugin against the shared context (mirrors the
// SE0 panel registration, but owned by this translation unit so plugins.h /
// panels.cc stay untouched).
void RegisterDetailsPanel(EditorContext& ctx);

}  // namespace ps::studio::details

#endif  // PS_STUDIO_EDITOR_DETAILS_PANEL_H_
