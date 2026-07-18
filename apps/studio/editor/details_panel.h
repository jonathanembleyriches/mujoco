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
#include <memory>
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

// --- Numeric widget width (display fidelity) ------------------------------- //
// A drag widget must read and write exactly the storage type's bytes. Choosing a
// wider host data type than the field -- an int64 drag over an int32 -- makes the
// widget read the adjacent word as the value's high bits and display an
// uninitialised garbage number. The width and signedness are derived from the
// C++ storage type here (window-free and unit-tested); details_panel.cc maps this
// enum 1:1 to the ImGui data-type constant, so the widget can never be wider than
// the field it edits.
enum class NumericWidget { S8, S16, S32, S64, U8, U16, U32, U64, F32, F64 };

template <class X>
constexpr NumericWidget NumericWidgetOf() {
  static_assert(std::is_arithmetic_v<X> && !std::is_same_v<X, bool>,
                "NumericWidgetOf is for non-bool arithmetic storage types");
  if constexpr (std::is_floating_point_v<X>) {
    return sizeof(X) == 4 ? NumericWidget::F32 : NumericWidget::F64;
  } else if constexpr (std::is_signed_v<X>) {
    return sizeof(X) == 1   ? NumericWidget::S8
           : sizeof(X) == 2 ? NumericWidget::S16
           : sizeof(X) == 4 ? NumericWidget::S32
                            : NumericWidget::S64;
  } else {
    return sizeof(X) == 1   ? NumericWidget::U8
           : sizeof(X) == 2 ? NumericWidget::U16
           : sizeof(X) == 4 ? NumericWidget::U32
                            : NumericWidget::U64;
  }
}

// Bytes the widget reads/writes for a given host data type; must equal
// sizeof(storage) for every numeric field.
constexpr std::size_t NumericWidgetBytes(NumericWidget w) {
  switch (w) {
    case NumericWidget::S8:
    case NumericWidget::U8:
      return 1;
    case NumericWidget::S16:
    case NumericWidget::U16:
      return 2;
    case NumericWidget::S32:
    case NumericWidget::U32:
    case NumericWidget::F32:
      return 4;
    case NumericWidget::S64:
    case NumericWidget::U64:
    case NumericWidget::F64:
      return 8;
  }
  return 0;
}

// --- Colour fields --------------------------------------------------------- //
// A handful of fields are RGB(A) colours and read far better as an ImGui colour
// swatch/picker than as a bare numeric row. They are recognised by field name
// (an authoring convenience -- a real color-space hint would be schema-driven),
// and only when the storage is actually a fixed 3- or 4-wide float/double array,
// so a same-named scalar (e.g. Material.specular) never trips the classifier.
enum class ColorKind {
  None,
  Rgb3,   // ImGui::ColorEdit3
  Rgba4,  // ImGui::ColorEdit4
};

inline ColorKind ColorKindByName(std::string_view xml) {
  if (xml == "rgba") return ColorKind::Rgba4;
  if (xml == "rgb" || xml == "rgb1" || xml == "rgb2" || xml == "markrgb" ||
      xml == "ambient" || xml == "diffuse" || xml == "specular" ||
      xml == "emission" || xml == "fog" || xml == "haze") {
    return ColorKind::Rgb3;
  }
  return ColorKind::None;
}

// The classifier proper: a colour field must also be a fixed-arity real row of
// the matching width, so the widget can bind directly to the array's floats.
inline ColorKind ColorKindOf(const reflect::FieldDescriptor& fd) {
  const ColorKind by_name = ColorKindByName(fd.xml);
  if (by_name == ColorKind::None) return ColorKind::None;
  const bool real = fd.kind == reflect::FieldKind::Float ||
                    fd.kind == reflect::FieldKind::Double;
  const bool fixed = fd.arity == reflect::ArityKind::Fixed;
  if (!real || !fixed) return ColorKind::None;
  if (by_name == ColorKind::Rgba4 && fd.arity_min == 4) return ColorKind::Rgba4;
  if (by_name == ColorKind::Rgb3 && fd.arity_min == 3) return ColorKind::Rgb3;
  return ColorKind::None;
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

// The value a presence-aware row seeds its edit temp with: the authored value if
// set, else the class-inherited value, else the IDL default, else a zero-init of
// the type. It never leaves the temp uninitialised, so the widget always shows a
// value drawn from a real layer rather than stack garbage. The renderer seeds
// every optional row through this, so what the panel displays for an unset field
// is exactly the layered value the tests compute independently.
template <class Inner>
Inner SeedValue(bool authored, const ps::opt<Inner>& slot,
                const ps::opt<Inner>* inherited, const ps::opt<Inner>* full) {
  if (authored) return *slot;
  if (inherited && inherited->has_value()) return **inherited;
  if (full && full->has_value()) return **full;
  return Inner{};
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

// --- Material texture layers ----------------------------------------------- //
// A Material's <layer> children are an owned vector<unique_ptr<MaterialLayer>>,
// so the generic field visitor -- which walks scalar/array fields only -- never
// reaches them: this is the one appearance surface the reflection panel misses.
// These pure mutators are the whole editing contract the Details panel drives for
// a material's texture layers. The panel wraps each in one BeginEdit/CommitEdit
// so it is a single undo step; keeping the mutation here (not inside the ImGui
// callback) lets the exact same code be exercised windowless.

// The display name of a layer's texture reference ("" when unset).
inline std::string LayerTextureName(const mj::MaterialLayer& layer) {
  return layer.texture ? layer.texture->name : std::string();
}

// A layer's role as an enum index; an unset role reads as rgb (index 0), the role
// a freshly added layer carries.
inline int LayerRoleIndex(const mj::MaterialLayer& layer) {
  return layer.role ? static_cast<int>(*layer.role) : 0;
}

// Point a layer at a texture by name; an empty name clears the reference.
inline void SetLayerTexture(mj::MaterialLayer& layer, std::string_view texture) {
  if (texture.empty()) {
    layer.texture.reset();
  } else {
    layer.texture = ps::Ref<mj::Texture>(std::string(texture));
  }
}

inline void SetLayerRole(mj::MaterialLayer& layer, mj::TexRole role) {
  layer.role = role;
}

// Append a texture layer (role rgb, referencing `texture` when non-empty) and
// return its index.
inline std::size_t AddMaterialLayer(mj::Material& mat,
                                    std::string_view texture = {}) {
  auto layer = std::make_unique<mj::MaterialLayer>();
  layer->role = mj::TexRole::rgb;
  if (!texture.empty()) {
    layer->texture = ps::Ref<mj::Texture>(std::string(texture));
  }
  mat.layers.push_back(std::move(layer));
  return mat.layers.size() - 1;
}

// Remove the layer at `idx`; a no-op returning false when out of range.
inline bool RemoveMaterialLayer(mj::Material& mat, std::size_t idx) {
  if (idx >= mat.layers.size()) return false;
  mat.layers.erase(mat.layers.begin() + idx);
  return true;
}

// --- Panel registration ---------------------------------------------------- //
// Self-registers the "Details" GuiPlugin against the shared context (mirrors the
// SE0 panel registration, but owned by this translation unit so plugins.h /
// panels.cc stay untouched).
void RegisterDetailsPanel(EditorContext& ctx);

}  // namespace ps::studio::details

#endif  // PS_STUDIO_EDITOR_DETAILS_PANEL_H_
