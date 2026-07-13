// MJCF reader: tinyxml2 document -> ps::Model, driven by the generated
// xml_binding tables (attrs/variants/children) with the generated Visit hook for
// typed field access. Hand code appears only at the quirks the tables cannot
// carry: numeric parsing (Q-NUM, in numeric.cc), the surviving geom-shape
// aliasing group (Q-FROMTO) and texture source, variable arity (Q-ARITY), and
// the strict support boundary (unsupported-element reporting).
//
// Q-ORIENT / Q-INERTIA canonicalization (docs/plan_canonicalization.md, Wave A):
// orientation is stored as a single canonical quaternion and inertia as
// diaginertia + iquat. The reader still ACCEPTS every authored spelling (quat/
// euler/axisangle/xyaxes/zaxis via the binding's input aliases; diaginertia/
// fullinertia), but folds them into the canonical field. Inertia has no angular
// context, so fullinertia is eigendecomposed inline at read (core::FullInertia
// ToDiag). Orientation resolution IS deferred to parse end: MJCF admits
// <compiler> blocks anywhere (even after <worldbody>) and euler/axisangle depend
// on compiler.angle/eulerseq, so resolving mid-parse would make the result
// depend on document order. The reader collects each authored orientation as a
// pending fold (with a stable pointer to its destination quat field -- every
// orientation-bearing element is heap-owned, DR-2) and resolves them all in one
// pass over the effective, document-order-folded compiler context (core::Resolve
// Orientation, the same math MuJoCo compiles). Element-wins-atomic: an element's
// authored orientation always wins over a class-inherited alt spelling (a
// deliberate, documented divergence from MuJoCo's ReadAlternative precedence
// wart; no corpus witness -- docs/plan_canonicalization.md Section 3).
//
// Q-ANGLE is handled by form preservation, not read-time conversion: angle
// values are stored exactly as authored and the compiler's angle unit round
// trips verbatim, so MuJoCo performs every degree->radian conversion at compile
// time. This is necessary, not merely tidy: MuJoCo converts joint range only
// for LIMITED hinge/ball joints (user_objects.cc:3207-3224) and ref/springref
// only for hinge (:3279-3282) -- conditions resolved per consuming joint at
// compile. A default class (data, DR-1) can be shared by joints that resolve
// those conditions differently (e.g. auto_limits.xml's `range_defined` feeds
// both a limited hinge, converted, and a free joint, not), so no single
// pre-converted stored value is correct. Preserving the authored unit and
// deferring to MuJoCo is the only round-trip-exact option.
//
// Presence (DR-1): only authored attributes set fields; a missing attribute
// leaves the opt<T> empty. Provenance (DR-9): every element records {file, line}.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "defaults.h"
#include "include.h"
#include "keywords.h"
#include "mjcf.h"
#include "numeric.h"
#include "protospec/core.h"
#include "resolve.h"
#include "tinyxml2.h"
#include "types.h"
#include "visit.h"
#include "xml_binding.h"

namespace ps::mjcf::io {
namespace {

using tinyxml2::XMLDocument;
using tinyxml2::XMLElement;
using xmlbind::AttrBinding;
using xmlbind::Bind;
using xmlbind::ElementBinding;
using xmlbind::VariantBinding;

// --- Storage-shape traits (mirror the emitter's type mapping) ------------- //
template <class T>
struct is_opt : std::false_type {};
template <class T>
struct is_opt<ps::opt<T>> : std::true_type {
  using inner = T;
};

template <class T>
struct is_ref : std::false_type {};
template <class T>
struct is_ref<ps::Ref<T>> : std::true_type {};

template <class T>
struct is_std_array : std::false_type {};
template <class T, std::size_t N>
struct is_std_array<std::array<T, N>> : std::true_type {};

template <class T>
struct is_inline_vec : std::false_type {};
template <class T, std::size_t N>
struct is_inline_vec<ps::InlineVec<T, N>> : std::true_type {};

template <class T>
struct is_vector : std::false_type {};
template <class T>
struct is_vector<std::vector<T>> : std::true_type {};

// The supported families: Model-level blocks, the body tree, defaults, assets,
// and (wave 2) the contact/equality/tendon/actuator sections. Everything else
// is a well-formed but unsupported element (skip signal), never a malformed
// input.
bool IsSupported(ElementType t) {
  switch (t) {
    case ElementType::Model:
    case ElementType::Compiler:
    case ElementType::LengthRange:
    case ElementType::Option:
    case ElementType::Flag:
    case ElementType::Size:
    case ElementType::Visual:
    case ElementType::VisualGlobal:
    case ElementType::VisualQuality:
    case ElementType::VisualHeadlight:
    case ElementType::VisualMap:
    case ElementType::VisualScale:
    case ElementType::VisualRgba:
    case ElementType::Statistic:
    case ElementType::Body:
    case ElementType::Inertial:
    case ElementType::Joint:
    case ElementType::FreeJoint:
    case ElementType::Geom:
    case ElementType::Site:
    case ElementType::Camera:
    case ElementType::Light:
    case ElementType::Frame:
    // Defaults (family c): the class tree and its per-family sub-elements.
    case ElementType::Default:
    case ElementType::Pair:
    case ElementType::EqualityDefault:
    case ElementType::TendonDefault:
    case ElementType::ActuatorGeneral:
    case ElementType::Motor:
    case ElementType::Position:
    case ElementType::Velocity:
    case ElementType::IntVelocity:
    case ElementType::Damper:
    case ElementType::Cylinder:
    case ElementType::Muscle:
    case ElementType::Adhesion:
    case ElementType::DcMotor:
    // Contact / equality / tendon sections (family e).
    case ElementType::Contact:
    case ElementType::Exclude:
    case ElementType::Equality:
    case ElementType::Connect:
    case ElementType::Weld:
    case ElementType::EqualityJoint:
    case ElementType::EqualityTendon:
    case ElementType::EqualityFlex:
    case ElementType::Flexvert:
    case ElementType::Flexstrain:
    case ElementType::Tendon:
    case ElementType::Spatial:
    case ElementType::SpatialSite:
    case ElementType::SpatialGeom:
    case ElementType::Pulley:
    case ElementType::Fixed:
    case ElementType::FixedJoint:
    // Actuator section (family f).
    case ElementType::Actuator:
    case ElementType::ActuatorPlugin:
    case ElementType::Config:
    // Assets (family d).
    case ElementType::Asset:
    case ElementType::Mesh:
    case ElementType::Hfield:
    case ElementType::Skin:
    case ElementType::SkinBone:
    case ElementType::Texture:
    case ElementType::Material:
    case ElementType::MaterialLayer:
    case ElementType::ModelAsset:
    // Sensors (family g): the full <sensor> section as an ordered union.
    case ElementType::Sensor:
    case ElementType::Touch:
    case ElementType::Accelerometer:
    case ElementType::Velocimeter:
    case ElementType::Gyro:
    case ElementType::Force:
    case ElementType::Torque:
    case ElementType::Magnetometer:
    case ElementType::Camprojection:
    case ElementType::Rangefinder:
    case ElementType::Jointpos:
    case ElementType::Jointvel:
    case ElementType::Tendonpos:
    case ElementType::Tendonvel:
    case ElementType::Actuatorpos:
    case ElementType::Actuatorvel:
    case ElementType::Actuatorfrc:
    case ElementType::Jointactuatorfrc:
    case ElementType::Tendonactuatorfrc:
    case ElementType::Ballquat:
    case ElementType::Ballangvel:
    case ElementType::Jointlimitpos:
    case ElementType::Jointlimitvel:
    case ElementType::Jointlimitfrc:
    case ElementType::Tendonlimitpos:
    case ElementType::Tendonlimitvel:
    case ElementType::Tendonlimitfrc:
    case ElementType::Framepos:
    case ElementType::Framequat:
    case ElementType::Framexaxis:
    case ElementType::Frameyaxis:
    case ElementType::Framezaxis:
    case ElementType::Framelinvel:
    case ElementType::Frameangvel:
    case ElementType::Framelinacc:
    case ElementType::Frameangacc:
    case ElementType::Subtreecom:
    case ElementType::Subtreelinvel:
    case ElementType::Subtreeangmom:
    case ElementType::Insidesite:
    case ElementType::Distance:
    case ElementType::Normal:
    case ElementType::Fromto:
    case ElementType::SensorContact:
    case ElementType::EPotential:
    case ElementType::EKinetic:
    case ElementType::Clock:
    case ElementType::Tactile:
    case ElementType::SensorUser:
    case ElementType::SensorPlugin:
    // Custom + keyframe + extension (family h).
    case ElementType::Custom:
    case ElementType::Numeric:
    case ElementType::Text:
    case ElementType::Tuple:
    case ElementType::TupleElement:
    case ElementType::Keyframe:
    case ElementType::Key:
    case ElementType::Extension:
    case ElementType::PluginDef:
    case ElementType::PluginInstance:
    case ElementType::PluginRef:
    // Macros + deformable (family i): first-class pass-through (DR-7).
    case ElementType::Composite:
    case ElementType::CompositeJoint:
    case ElementType::CompositeSkin:
    case ElementType::CompositeGeom:
    case ElementType::CompositeSite:
    case ElementType::Flexcomp:
    case ElementType::FlexcompEdge:
    case ElementType::FlexElasticity:
    case ElementType::FlexContact:
    case ElementType::FlexcompPin:
    case ElementType::Attach:
    case ElementType::Replicate:
    case ElementType::Deformable:
    case ElementType::Flex:
    case ElementType::FlexEdge:
      return true;
    default:
      return false;
  }
}

// The element type carried by the I-th arm of a union item's node variant
// (each arm is a std::unique_ptr<Member>).
template <class Var, std::size_t I>
using union_member_t =
    typename std::variant_alternative_t<I, Var>::element_type;

// Append (member tag, is-supported) for every arm of a union node variant.
// Drives both child classification (unknown vs unsupported) and the ordered
// union reader below off the same compile-time membership.
template <class Var, std::size_t... Is>
void AddUnionClaimsImpl(
    std::vector<std::pair<std::string_view, bool>>& claims,
    std::index_sequence<Is...>) {
  (claims.push_back(
       {Bind(element_type_of<union_member_t<Var, Is>>::value).tag,
        IsSupported(element_type_of<union_member_t<Var, Is>>::value)}),
   ...);
}

template <class U>
void AddUnionClaims(std::vector<std::pair<std::string_view, bool>>& claims) {
  using Var = std::decay_t<decltype(std::declval<U>().node)>;
  AddUnionClaimsImpl<Var>(claims,
                          std::make_index_sequence<std::variant_size_v<Var>>{});
}

// The eleven actuator spellings share the common transmission attributes and so
// the same target-exclusivity quirk (Q-ACT); this tags them for the hook.
template <class E>
struct is_actuator : std::false_type {};
template <>
struct is_actuator<ActuatorGeneral> : std::true_type {};
template <>
struct is_actuator<Motor> : std::true_type {};
template <>
struct is_actuator<Position> : std::true_type {};
template <>
struct is_actuator<Velocity> : std::true_type {};
template <>
struct is_actuator<IntVelocity> : std::true_type {};
template <>
struct is_actuator<Damper> : std::true_type {};
template <>
struct is_actuator<Cylinder> : std::true_type {};
template <>
struct is_actuator<Muscle> : std::true_type {};
template <>
struct is_actuator<Adhesion> : std::true_type {};
template <>
struct is_actuator<DcMotor> : std::true_type {};
template <>
struct is_actuator<ActuatorPlugin> : std::true_type {};

// Frame sensors carrying the generic objtype/objname slot plus the optional
// reftype/refname pair (Sensor(), xml_native_reader.cc:4348-4432). framelinacc
// and frameangacc omit the reference pair and are excluded here.
template <class E>
struct is_frame_ref_sensor : std::false_type {};
template <>
struct is_frame_ref_sensor<Framepos> : std::true_type {};
template <>
struct is_frame_ref_sensor<Framequat> : std::true_type {};
template <>
struct is_frame_ref_sensor<Framexaxis> : std::true_type {};
template <>
struct is_frame_ref_sensor<Frameyaxis> : std::true_type {};
template <>
struct is_frame_ref_sensor<Framezaxis> : std::true_type {};
template <>
struct is_frame_ref_sensor<Framelinvel> : std::true_type {};
template <>
struct is_frame_ref_sensor<Frameangvel> : std::true_type {};

// Elements that hold a <config> key/value list (a plugin reference or instance):
// MuJoCo rejects duplicate config keys (ReadPluginConfigs,
// xml_native_reader.cc:128-153).
template <class E>
struct has_config_children : std::false_type {};
template <>
struct has_config_children<PluginInstance> : std::true_type {};
template <>
struct has_config_children<PluginRef> : std::true_type {};
template <>
struct has_config_children<ActuatorPlugin> : std::true_type {};
template <>
struct has_config_children<SensorPlugin> : std::true_type {};

// The canonical quaternion field of an orientation-bearing element (Q-ORIENT),
// or nullptr for every other element. Posed elements (Body/Geom/Site/Camera/
// Frame) and Flexcomp store it as `quat`; Inertial stores the inertial frame as
// `iquat`. The template fallback keeps ReadElement instantiable for every type.
using QuatOpt = ps::opt<std::array<double, 4>>;
template <class E>
QuatOpt* QuatField(E&) { return nullptr; }
inline QuatOpt* QuatField(Body& e) { return &e.quat; }
inline QuatOpt* QuatField(Geom& e) { return &e.quat; }
inline QuatOpt* QuatField(Site& e) { return &e.quat; }
inline QuatOpt* QuatField(Camera& e) { return &e.quat; }
inline QuatOpt* QuatField(Frame& e) { return &e.quat; }
inline QuatOpt* QuatField(Flexcomp& e) { return &e.quat; }
inline QuatOpt* QuatField(Inertial& e) { return &e.iquat; }

// A collected-but-unresolved orientation: an authored spelling awaiting the
// document-order-folded compiler context (parse-end resolution, Q-ORIENT). The
// destination pointer is stable because every orientation-bearing element is
// heap-owned (DR-2), and `xml` stays valid for the whole parse.
struct PendingOrient {
  const XMLElement* xml;
  QuatOpt* dest;
  ps::core::OrientKind kind;
  double raw[6];
};

// The canonical springlength pair of a tendon-bearing element (Q-SPRINGLENGTH,
// docs/plan_canonicalization.md Wave B #5 / R3), or nullptr otherwise. Spatial,
// Fixed and TendonDefault store the resting length as a two-value pair; MJCF
// accepts one or two values and duplicates a lone value into the second slot
// (xml_native_reader.cc:2372-2374). Resolved inline at read (no compiler
// context), like inertia.
using PairOpt = ps::opt<std::array<double, 2>>;
template <class E>
PairOpt* SpringlengthField(E&) { return nullptr; }
inline PairOpt* SpringlengthField(Spatial& e) { return &e.springlength; }
inline PairOpt* SpringlengthField(Fixed& e) { return &e.springlength; }
inline PairOpt* SpringlengthField(TendonDefault& e) { return &e.springlength; }

// --------------------------------------------------------------------------- //
// Reader                                                                        //
// --------------------------------------------------------------------------- //
class Reader {
 public:
  Reader(std::string filename, std::vector<Diagnostic>& errors,
         std::vector<Diagnostic>& warnings,
         const ProvenanceMap* provenance = nullptr)
      : filename_(std::move(filename)),
        errors_(errors),
        warnings_(warnings),
        provenance_(provenance) {}

  // Provenance (DR-9): elements spliced from an included file carry that file's
  // path and line via the pre-pass map; top-level elements take theirs from
  // tinyxml2 against the model filename.
  ps::SourceLoc Loc(const XMLElement* e) const {
    if (provenance_) {
      auto it = provenance_->find(e);
      if (it != provenance_->end()) return it->second;
    }
    return ps::SourceLoc{filename_, e->GetLineNum()};
  }

  void Err(const XMLElement* e, std::string msg) {
    errors_.push_back({Diagnostic::Kind::MalformedInput, std::move(msg), Loc(e),
                       ""});
  }
  void Unsupported(const XMLElement* e, std::string tag) {
    std::string msg = "unsupported element '" + tag + "'";
    errors_.push_back(
        {Diagnostic::Kind::UnsupportedElement, std::move(msg), Loc(e), tag});
  }
  void Warn(const XMLElement* e, std::string msg) {
    warnings_.push_back({Diagnostic::Kind::MalformedInput, std::move(msg),
                         Loc(e), ""});
  }

  // Read a concrete element from an XML node: attributes + variants, the
  // per-element quirk hooks, then supported children.
  template <class E>
  void ReadElement(XMLElement* xml, E& out) {
    const ElementBinding& b = Bind(element_type_of<E>::value);
    out.loc = Loc(xml);

    CheckUnknownAttributes(xml, b);
    AttrVisitor<E> av{this, xml, &b};
    Visit(out, av);

    // Q-ORIENT: collect the authored orientation (resolved at parse end against
    // the folded compiler context). Orientation-bearing elements only.
    CollectOrientation(xml, out);
    // Q-INERTIA: fullinertia/diaginertia have no angular context, so resolve
    // inline (eigendecomposition for fullinertia -> diaginertia + iquat).
    if constexpr (std::is_same_v<E, Inertial>) {
      InertialOrientExclusion(xml);
      ResolveInertiaAttrs(xml, out);
    }
    // Wave B canonicalizations resolved inline at read (no compiler context).
    ResolveSpringlength(xml, out);  // #5: scalar or pair -> canonical pair
    if constexpr (std::is_same_v<E, Light>) ResolveLightType(xml, out);
    if constexpr (std::is_same_v<E, Cylinder>) ResolveCylinderArea(xml, out);
    if constexpr (std::is_same_v<E, Numeric>) ResolveNumericData(xml, out);
    if constexpr (std::is_same_v<E, Camera>) CameraIntrinsicExclusion(xml);
    if constexpr (std::is_same_v<E, Size>) MemoryFixup(xml, out);
    if constexpr (std::is_same_v<E, Connect>) ConnectExclusion(xml);
    if constexpr (std::is_same_v<E, Weld>) WeldExclusion(xml);
    if constexpr (is_actuator<E>::value) ActuatorTransmission(xml);
    if constexpr (std::is_same_v<E, Rangefinder>) RangefinderExclusion(xml);
    if constexpr (std::is_same_v<E, Distance> ||
                  std::is_same_v<E, Normal> || std::is_same_v<E, Fromto>)
      GeomDistanceExclusion(xml);
    if constexpr (std::is_same_v<E, SensorContact>) ContactSensorExclusion(xml);
    if constexpr (is_frame_ref_sensor<E>::value) FrameSensorRefPairing(xml);
    if constexpr (std::is_same_v<E, SensorUser>) UserSensorPairing(xml);
    if constexpr (std::is_same_v<E, SensorPlugin>) PluginSensorPairing(xml);
    if constexpr (has_config_children<E>::value) ConfigKeyUnique(xml);

    ClassifyChildren(xml, b, out);
    ChildVisitor<E> cv{this, xml, &b};
    Visit(out, cv);

    // #7: a material's `texture=` attribute is the RGB entry of the canonical
    // <layer> list. Folded after children are read so the mix-with-layers error
    // (xml_native_reader.cc:1849-1852) sees the authored layers, and the injected
    // RGB layer is prepended (first) exactly as MuJoCo slots mjTEXROLE_RGB.
    if constexpr (std::is_same_v<E, Material>) MaterialTextureFixup(xml, out);
  }

  // Parse-end orientation fold (Q-ORIENT). Computes the effective compiler
  // context by folding Model.compilers in document order (later authored
  // attributes win, matching MuJoCo's accumulate-into-one-spec), then resolves
  // every collected authored orientation against that single context -- so the
  // result is document-order independent even though <compiler> may appear
  // anywhere. Defaults when unauthored: angle="degree", eulerseq="xyz".
  void ResolvePendingOrientations(const Model& m) {
    ps::core::OrientContext ctx;  // degree=true, eulerseq="xyz"
    for (const auto& c : m.compilers) {
      if (c->angle) ctx.degree = (*c->angle == AngleUnit::degree);
      if (c->eulerseq) ctx.eulerseq = *c->eulerseq;
    }
    for (const PendingOrient& p : pending_orient_) {
      *p.dest = ps::core::ResolveOrientation(p.kind, p.raw, ctx);
    }
  }

  // Construct and read the union member whose tag matches `tag`, storing it in
  // item.node. Returns false (item untouched) for a tag that matches no member
  // or a member whose type is unsupported (both already diagnosed).
  template <class U>
  bool ReadUnionItem(XMLElement* c, std::string_view tag, U& item) {
    using Var = std::decay_t<decltype(item.node)>;
    bool done = false;
    ReadUnionArms<Var>(c, tag, item, done,
                       std::make_index_sequence<std::variant_size_v<Var>>{});
    return done;
  }

 private:
  template <class Var, class U, std::size_t... Is>
  void ReadUnionArms(XMLElement* c, std::string_view tag, U& item, bool& done,
                     std::index_sequence<Is...>) {
    (ReadUnionArm<Var, Is>(c, tag, item, done), ...);
  }

  template <class Var, std::size_t I, class U>
  void ReadUnionArm(XMLElement* c, std::string_view tag, U& item, bool& done) {
    if (done) return;
    using M = union_member_t<Var, I>;
    if (Bind(element_type_of<M>::value).tag != tag) return;
    if (!IsSupported(element_type_of<M>::value)) return;
    auto member = std::make_unique<M>();
    ReadElement(c, *member);
    item.node = std::move(member);
    done = true;
  }

  // ---- attribute / variant field reading ---------------------------------- //
  template <class E>
  struct AttrVisitor {
    Reader* r;
    XMLElement* xml;
    const ElementBinding* b;

    template <class T>
    void field(int id, const char*, T& value) {
      for (std::size_t i = 0; i < b->attr_count; ++i) {
        if (b->attrs[i].field_id == id) {
          // Resolved (canonicalized) fields -- quat/iquat/diaginertia -- are set
          // by the parse-end orientation fold / inline inertia resolution, not
          // the plain attr path (Q-ORIENT/Q-INERTIA).
          if (b->attrs[i].resolved) return;
          r->ReadAttr(xml, b->type, b->attrs[i], value);
          return;
        }
      }
      for (std::size_t i = 0; i < b->variant_count; ++i) {
        if (b->variants[i].field_id == id) {
          r->ReadVariant(xml, b->variants[i], value);
          return;
        }
      }
    }
    template <class T>
    void child(int, const char*, std::vector<std::unique_ptr<T>>&) {}
    template <class U>
    void union_child(int, const char*, std::vector<U>&) {}
  };

  template <class T>
  void ReadAttr(XMLElement* xml, ElementType type, const AttrBinding& ab,
                T& value) {
    std::string attr(ab.attr);
    const char* raw =
        ab.element_text ? xml->GetText() : xml->Attribute(attr.c_str());
    if (!raw) return;
    std::string_view text(raw);
    if constexpr (is_opt<T>::value) {
      typename is_opt<T>::inner tmp{};
      if (ReadInner(xml, type, ab, text, tmp)) value = std::move(tmp);
    } else {
      ReadInner(xml, type, ab, text, value);
    }
  }

  template <class Inner>
  bool ReadInner(XMLElement* xml, ElementType type, const AttrBinding& ab,
                 std::string_view text, Inner& out) {
    if constexpr (is_ref<Inner>::value) {
      out.name = std::string(text);
      return true;
    } else if constexpr (std::is_same_v<Inner, std::string>) {
      out = std::string(text);
      return true;
    } else if constexpr (std::is_same_v<Inner, bool>) {
      // The <flag> element spells its booleans enable/disable (MuJoCo
      // enable_map); every other boolean attribute is true/false (bool_map).
      const char* yes = type == ElementType::Flag ? "enable" : "true";
      const char* no = type == ElementType::Flag ? "disable" : "false";
      if (text == yes) {
        out = true;
      } else if (text == no) {
        out = false;
      } else {
        Err(xml, "invalid keyword '" + std::string(text) + "' for attribute '" +
                     std::string(ab.attr) + "' (expected " + yes + " or " + no +
                     ")");
        return false;
      }
      return true;
    } else if constexpr (std::is_enum_v<Inner>) {
      if (FromMjcf(text, out)) return true;
      Err(xml, "invalid keyword '" + std::string(text) + "' for attribute '" +
                   std::string(ab.attr) + "'");
      return false;
    } else if constexpr (is_std_array<Inner>::value) {
      return ReadFixed(xml, ab, text, out);
    } else if constexpr (is_inline_vec<Inner>::value) {
      return ReadRange(xml, ab, text, out);
    } else if constexpr (is_vector<Inner>::value) {
      return ReadVector(xml, ab, text, out);
    } else if constexpr (std::is_arithmetic_v<Inner>) {
      return ReadNumberN(xml, ab, text, &out, 1, 1) == 1;
    } else {
      // Variant field types never reach here (routed through ReadVariant); the
      // branch exists only so the field() dispatcher's odr-use type-checks.
      return false;
    }
  }

  // Fixed-arity array: exactly N values (Q-ARITY exact=true).
  template <class S, std::size_t N>
  bool ReadFixed(XMLElement* xml, const AttrBinding& ab, std::string_view text,
                 std::array<S, N>& out) {
    S buf[N]{};
    int got = ReadNumberN(xml, ab, text, buf, static_cast<int>(N),
                          static_cast<int>(N));
    if (got != static_cast<int>(N)) return false;
    for (std::size_t i = 0; i < N; ++i) out[i] = buf[i];
    return true;
  }

  // Range-arity inline vector: up to capacity values, fewer is fine (record the
  // filled count), more is an error (Q-ARITY exact=false).
  template <class S, std::size_t N>
  bool ReadRange(XMLElement* xml, const AttrBinding& ab, std::string_view text,
                 ps::InlineVec<S, N>& out) {
    S buf[N]{};
    int got = ReadNumberN(xml, ab, text, buf, 0, static_cast<int>(N));
    if (got < 0) return false;
    out.clear();
    for (int i = 0; i < got; ++i) out.push_back(buf[i]);
    return true;
  }

  // Unbounded vector: numeric list, or a space-separated keyword set (Q for
  // MapValues) when the element type is an enum.
  template <class S>
  bool ReadVector(XMLElement* xml, const AttrBinding& ab, std::string_view text,
                  std::vector<S>& out) {
    if constexpr (std::is_enum_v<S>) {
      std::unordered_set<std::string> seen;
      std::vector<S> vals;
      for (std::string_view tok : num::Tokens(text)) {
        std::string key(tok);
        if (!seen.insert(key).second) {
          Err(xml, "duplicate keyword '" + key + "' in attribute '" +
                       std::string(ab.attr) + "'");
          return false;
        }
        S v{};
        if (!FromMjcf(tok, v)) {
          Err(xml, "invalid keyword '" + key + "' for attribute '" +
                       std::string(ab.attr) + "'");
          return false;
        }
        vals.push_back(v);
      }
      // #9 keyword-set canonicalization: a keyword set is an order-insensitive
      // bitmask (MapValues); store it in enum-declaration order so the canonical
      // form is deterministic. The enum's underlying value is its declaration
      // index, which is also the MJCF map order (camout/raydata/condata) -- for
      // the contact sensor, whose reader demands that exact order
      // (xml_native_reader.cc:4517-4530), the strict-order check runs separately
      // on the raw input before this normalization.
      std::sort(vals.begin(), vals.end());
      out = std::move(vals);
      return true;
    } else {
      auto toks = num::Tokens(text);
      std::vector<S> vals(toks.size());
      for (std::size_t i = 0; i < toks.size(); ++i) {
        if (!ParseScalar(xml, ab, toks[i], vals[i])) return false;
      }
      out = std::move(vals);
      return true;
    }
  }

  // Parse min_count..max_count numbers into buf; returns the count, or -1 on
  // error (already reported). min_count>0 enforces "not enough data".
  template <class S>
  int ReadNumberN(XMLElement* xml, const AttrBinding& ab, std::string_view text,
                  S* buf, int min_count, int max_count) {
    auto toks = num::Tokens(text);
    const int n = static_cast<int>(toks.size());
    if (n < min_count) {
      Err(xml, "attribute '" + std::string(ab.attr) +
                   "' does not have enough data");
      return -1;
    }
    if (n > max_count) {
      Err(xml, "attribute '" + std::string(ab.attr) + "' has too much data");
      return -1;
    }
    for (int i = 0; i < n; ++i) {
      if (!ParseScalar(xml, ab, toks[i], buf[i])) return -1;
    }
    return n;
  }

  template <class S>
  bool ParseScalar(XMLElement* xml, const AttrBinding& ab, std::string_view tok,
                   S& out) {
    if constexpr (std::is_floating_point_v<S>) {
      bool is_nan = false;
      num::Status st = num::ParseFloat<S>(tok, out, is_nan);
      if (st == num::Status::Ok) {
        if (is_nan) {
          Warn(xml, "attribute '" + std::string(ab.attr) +
                        "' contains NaN; please check it carefully");
        }
        return true;
      }
      ReportNumError(xml, ab, st);
      return false;
    } else {
      num::Status st = num::ParseInt<S>(tok, out);
      if (st == num::Status::Ok) return true;
      ReportNumError(xml, ab, st);
      return false;
    }
  }

  void ReportNumError(XMLElement* xml, const AttrBinding& ab, num::Status st) {
    if (st == num::Status::Overflow) {
      Err(xml, "number is too large in attribute '" + std::string(ab.attr) +
                   "'");
    } else {
      Err(xml, "bad number in attribute '" + std::string(ab.attr) + "'");
    }
  }

  // ---- variant fields (DR-3 aliasing groups) ------------------------------ //
  template <class T>
  void ReadVariant(XMLElement* xml, const VariantBinding&, T& value) {
    if constexpr (is_opt<T>::value) {
      using V = typename is_opt<T>::inner;
      if constexpr (std::is_same_v<V, GeomShape>) {
        MarshalGeomShape(xml, value);
      } else if constexpr (std::is_same_v<V, TextureSource>) {
        MarshalTextureSource(xml, value);
      }
    }
  }

  bool Has(XMLElement* xml, const char* attr) {
    return xml->Attribute(attr) != nullptr;
  }

  // Read exactly n doubles from an attribute (verbatim, no unit conversion).
  bool ReadDoubleArr(XMLElement* xml, const char* attr, int n, double* out) {
    AttrBinding ab{};
    ab.attr = attr;
    return ReadNumberN(xml, ab, xml->Attribute(attr), out, n, n) == n;
  }

  // Q-ORIENT: collect the authored orientation spelling for later resolution
  // against the folded compiler context (parse end). Multiple-specifier and
  // zero-quaternion errors are reader-level (no compiler context needed) and are
  // reported here, verbatim as MuJoCo does (xml_base.cc:73-75). A no-op for
  // elements with no canonical quat field.
  template <class E>
  void CollectOrientation(XMLElement* xml, E& out) {
    QuatOpt* dest = QuatField(out);
    if (!dest) return;
    int count = Has(xml, "quat") + Has(xml, "axisangle") + Has(xml, "xyaxes") +
                Has(xml, "zaxis") + Has(xml, "euler");
    if (count == 0) return;
    if (count > 1) {
      Err(xml, "multiple orientation specifiers are not allowed");
      return;
    }
    PendingOrient p{};
    p.xml = xml;
    p.dest = dest;
    if (Has(xml, "quat")) {
      if (!ReadDoubleArr(xml, "quat", 4, p.raw)) return;
      if (p.raw[0] == 0 && p.raw[1] == 0 && p.raw[2] == 0 && p.raw[3] == 0) {
        Err(xml, "zero quaternion is not allowed");
        return;
      }
      p.kind = ps::core::OrientKind::Quat;
    } else if (Has(xml, "axisangle")) {
      if (!ReadDoubleArr(xml, "axisangle", 4, p.raw)) return;
      p.kind = ps::core::OrientKind::AxisAngle;
    } else if (Has(xml, "xyaxes")) {
      if (!ReadDoubleArr(xml, "xyaxes", 6, p.raw)) return;
      p.kind = ps::core::OrientKind::XYAxes;
    } else if (Has(xml, "zaxis")) {
      if (!ReadDoubleArr(xml, "zaxis", 3, p.raw)) return;
      p.kind = ps::core::OrientKind::ZAxis;
    } else {  // euler
      if (!ReadDoubleArr(xml, "euler", 3, p.raw)) return;
      p.kind = ps::core::OrientKind::Euler;
    }
    pending_orient_.push_back(p);
  }

  // Q-INERTIA: diaginertia stores as authored; fullinertia is eigendecomposed
  // into diaginertia + the inertial-frame quaternion (iquat) inline -- inertia
  // has no angular context, so no parse-end deferral is needed. Mutual exclusion
  // mirrors MuJoCo (user_objects.cc:2705-2716 / xml_native_reader.cc:3682-3684).
  void ResolveInertiaAttrs(XMLElement* xml, Inertial& out) {
    bool diag = Has(xml, "diaginertia");
    bool full = Has(xml, "fullinertia");
    if (diag && full) {
      Err(xml, "diaginertia and fullinertia cannot both be specified");
      return;
    }
    if (diag) {
      double d[3]{};
      if (!ReadDoubleArr(xml, "diaginertia", 3, d)) return;
      out.diaginertia = std::array<double, 3>{d[0], d[1], d[2]};
    } else if (full) {
      double f[6]{};
      if (!ReadDoubleArr(xml, "fullinertia", 6, f)) return;
      double diagv[3]{}, q[4]{};
      if (const char* e = ps::core::FullInertiaToDiag(f, diagv, q)) {
        Err(xml, e);
        return;
      }
      out.diaginertia = std::array<double, 3>{diagv[0], diagv[1], diagv[2]};
      // fullinertia carries the inertial frame in its eigenvectors (the inertial
      // orientation must be absent -- InertialOrientExclusion).
      out.iquat = std::array<double, 4>{q[0], q[1], q[2], q[3]};
    }
  }

  // #5 springlength: read one or two values; a lone value duplicates into the
  // second slot, exactly as MuJoCo (xml_native_reader.cc:2372-2374). Canonical
  // storage is the two-value pair. A no-op for elements with no springlength.
  template <class E>
  void ResolveSpringlength(XMLElement* xml, E& out) {
    PairOpt* dest = SpringlengthField(out);
    if (!dest) return;
    if (!Has(xml, "springlength")) return;
    double v[2]{};
    AttrBinding ab{};
    ab.attr = "springlength";
    int got = ReadNumberN(xml, ab, xml->Attribute("springlength"), v, 1, 2);
    if (got < 1) return;
    if (got == 1) v[1] = v[0];
    *dest = std::array<double, 2>{v[0], v[1]};
  }

  // #1 light type: `directional` (legacy bool) and `type` (enum) both fill the
  // canonical `type`, mutually exclusive (xml_native_reader.cc:2123-2131).
  // directional="true" -> directional, "false" -> spot.
  void ResolveLightType(XMLElement* xml, Light& out) {
    bool has_dir = Has(xml, "directional");
    bool has_type = Has(xml, "type");
    if (has_dir && has_type) {
      Err(xml, "type and directional cannot both be defined");
      return;
    }
    if (has_dir) {
      bool d{};
      AttrBinding ab{};
      ab.attr = "directional";
      if (ReadInner(xml, ElementType::Light, ab,
                    std::string_view(xml->Attribute("directional")), d)) {
        out.type = d ? LightType::directional : LightType::spot;
      }
    } else if (has_type) {
      LightType t{};
      if (FromMjcf(std::string_view(xml->Attribute("type")), t)) {
        out.type = t;
      } else {
        Err(xml, "invalid keyword '" + std::string(xml->Attribute("type")) +
                     "' for attribute 'type'");
      }
    }
  }

  // #6 cylinder area: `area` and `diameter` both fill gainprm[0]; a non-negative
  // diameter overrides area with pi/4 d^2 (mjs_setToCylinder, user_api.cc:
  // 1236-1248). Canonical storage is `area`. Element-wins-atomic (a class
  // diameter beating an element area is not emulated; docs/plan_canonicalization
  // .md Section 7, owner-approved).
  void ResolveCylinderArea(XMLElement* xml, Cylinder& out) {
    // mjPI verbatim (this module is MuJoCo-free) so the fold is bit-identical to
    // mjs_setToCylinder's `mjPI / 4 * diameter*diameter`.
    constexpr double kPi = 3.14159265358979323846;
    double diameter = -1;
    bool has_diam = Has(xml, "diameter");
    if (has_diam && !ReadDoubleArr(xml, "diameter", 1, &diameter)) return;
    if (has_diam && diameter >= 0) {
      out.area = kPi / 4 * diameter * diameter;
      return;
    }
    if (Has(xml, "area")) {
      double area{};
      if (ReadDoubleArr(xml, "area", 1, &area)) out.area = area;
    }
  }

  // #8 numeric data: MuJoCo materializes the data array to the authored `size`
  // (zero-padded / truncated), else to the data length (xml_native_reader.cc:
  // 3200-3217). Canonical storage is that materialized array; the `size` spelling
  // is erased. The 1..500 bound is kept.
  void ResolveNumericData(XMLElement* xml, Numeric& out) {
    std::vector<double> data;
    if (const char* raw = xml->Attribute("data")) {
      AttrBinding ab{};
      ab.attr = "data";
      if (!ReadVector(xml, ab, std::string_view(raw), data)) return;
    }
    int size = static_cast<int>(data.size());
    if (const char* sz = xml->Attribute("size")) {
      int parsed = 0;
      if (num::ParseInt<int>(sz, parsed) != num::Status::Ok) {
        Err(xml, "bad number in attribute 'size'");
        return;
      }
      size = parsed;
    }
    if (size < 1 || size > 500) {
      Err(xml, "custom field size must be between 1 and 500");
      return;
    }
    std::vector<double> materialized(static_cast<std::size_t>(size), 0.0);
    const int copy = std::min<int>(size, static_cast<int>(data.size()));
    for (int i = 0; i < copy; ++i) materialized[i] = data[i];
    out.data = std::move(materialized);
  }

  // #7 material texture attr -> canonical <layer role="rgb">. The attribute is a
  // read-only input alias (no field); it prepends an RGB layer, and mixing it
  // with authored <layer> children is an error (xml_native_reader.cc:1849-1852).
  void MaterialTextureFixup(XMLElement* xml, Material& out) {
    const char* tex = xml->Attribute("texture");
    if (!tex) return;
    if (!out.layers.empty()) {
      Err(xml,
          "a material with a texture attribute cannot have layer sub-elements");
      return;
    }
    auto layer = std::make_unique<MaterialLayer>();
    layer->loc = out.loc;
    layer->texture = ps::Ref<Texture>{std::string(tex)};
    layer->role = TexRole::rgb;
    out.layers.insert(out.layers.begin(), std::move(layer));
  }

  void MarshalGeomShape(XMLElement* xml, ps::opt<GeomShape>& out) {
    // The `size` attribute is a plain field; the shape variant captures the
    // fromto authoring alternative (the "explicit" arm has no distinct
    // attribute -- absence of fromto is the explicit form).
    if (!Has(xml, "fromto")) return;
    double ft[6]{};
    if (!ReadDoubleArr(xml, "fromto", 6, ft)) return;
    out = GeomShape{FromTo{{ft[0], ft[1], ft[2], ft[3], ft[4], ft[5]}}};
  }

  // Q-TEX: a texture's source is either a single image file or a procedural
  // builtin (gradient/checker/flat). MuJoCo reads `builtin` and `file` as
  // independent attributes (xml_native_reader.cc:3473-3486); ProtoSpec models
  // them as one variant, so `file` wins when both appear -- an explicit
  // builtin="none" alongside a file is a file texture, and no real file texture
  // sets a non-none builtin. Cube-face files (fileright/...) and buffer data
  // are separate plain attributes, not part of this variant.
  void MarshalTextureSource(XMLElement* xml, ps::opt<TextureSource>& out) {
    if (const char* file = xml->Attribute("file")) {
      out = TextureSource{TexFile{std::string(file)}};
    } else if (const char* builtin = xml->Attribute("builtin")) {
      TextureBuiltin b{};
      if (!FromMjcf(std::string_view(builtin), b)) {
        Err(xml, "invalid keyword '" + std::string(builtin) +
                     "' for attribute 'builtin'");
        return;
      }
      out = TextureSource{b};
    }
  }

  // ---- per-element quirk hooks -------------------------------------------- //
  // Q-NUM: the `memory` size is authored as an integer with an optional
  // K/M/G/T/P/E binary suffix; the reader stores the canonical byte count (as a
  // string, the field's storage). "-1" or absent means unset.
  void MemoryFixup(XMLElement* xml, Size& s) {
    const char* raw = xml->Attribute("memory");
    if (!raw) {
      s.memory.reset();
      return;
    }
    std::uint64_t bytes = 0;
    switch (num::ParseMemory(raw, bytes)) {
      case num::MemStatus::Ok:
        s.memory = std::to_string(bytes);
        break;
      case num::MemStatus::Unset:
        s.memory.reset();
        break;
      case num::MemStatus::Bad:
        Err(xml,
            "unsigned integer with an optional suffix {K,M,G,T,P,E} is expected "
            "in attribute 'memory'");
        break;
    }
  }

  // fullinertia carries the inertial frame in its eigenvectors, so an authored
  // inertial orientation (quat/euler/...) alongside it is a conflict, matching
  // MuJoCo (user_objects.cc:2705-2716). Checked on the raw XML before the
  // orientation collect, so the evidence is intact.
  void InertialOrientExclusion(XMLElement* xml) {
    if (!Has(xml, "fullinertia")) return;
    int orient = Has(xml, "quat") + Has(xml, "axisangle") + Has(xml, "xyaxes") +
                 Has(xml, "zaxis") + Has(xml, "euler");
    if (orient > 0) {
      Err(xml, "fullinertia and inertial orientation cannot both be specified");
    }
  }

  // R1: a camera's field of view (fovy) and its physical sensor size are the
  // exclusive intrinsic families on the same element (xml_native_reader.cc:
  // 2086-2090). fovy vs focal is NOT the real exclusion (a class fovy + element
  // sensorsize is legal and resolved at compile), so only fovy+sensorsize is a
  // reader error; focal/principal without sensorsize stays a compile-time lint.
  void CameraIntrinsicExclusion(XMLElement* xml) {
    if (Has(xml, "fovy") && Has(xml, "sensorsize")) {
      Err(xml, "fovy and sensor size cannot both be specified");
    }
  }

  // Q-EQ: connect/weld carry two mutually exclusive attribute sets -- a body
  // form (body1[+body2][+anchor]) and a site form (site1+site2). ProtoSpec
  // stores each attribute as a plain field, so the exclusivity is a read check
  // mirroring OneEquality (xml_native_reader.cc:2209-2285).
  void ConnectExclusion(XMLElement* xml) {
    bool s1 = Has(xml, "site1"), s2 = Has(xml, "site2");
    bool b1 = Has(xml, "body1"), b2 = Has(xml, "body2"),
         anchor = Has(xml, "anchor");
    bool maybe_site = s1 || s2;
    bool maybe_body = b1 || b2 || anchor;
    if (maybe_site && maybe_body) {
      Err(xml, "body and site semantics cannot be mixed");
      return;
    }
    bool site_semantic = s1 && s2;
    bool body_semantic = b1 && anchor;
    if (site_semantic == body_semantic) {
      Err(xml,
          "either both body1 and anchor must be defined, or both site1 and "
          "site2 must be defined");
    }
  }

  void WeldExclusion(XMLElement* xml) {
    bool s1 = Has(xml, "site1"), s2 = Has(xml, "site2");
    bool b1 = Has(xml, "body1"), b2 = Has(xml, "body2");
    bool anchor = Has(xml, "anchor"), relpose = Has(xml, "relpose");
    bool maybe_site = s1 || s2;
    bool maybe_body = b1 || b2 || anchor || relpose;
    if (maybe_site && maybe_body) {
      Err(xml, "body and site semantics cannot be mixed");
      return;
    }
    bool site_semantic = s1 && s2;
    bool body_semantic = b1;
    if (site_semantic == body_semantic) {
      Err(xml,
          "either body1 must be defined and optionally {body2, anchor, "
          "relpose}, or site1 and site2 must be defined");
    }
  }

  // Q-ACT: an actuator has at most one transmission target, and the
  // slidercrank-only (cranklength/slidersite) and site-only (refsite)
  // attributes require the matching transmission (xml_native_reader.cc:
  // 2415-2470). trntype is undefined with no target, slidercrank with
  // cranksite, site with site.
  void ActuatorTransmission(XMLElement* xml) {
    int cnt = Has(xml, "joint") + Has(xml, "jointinparent") +
              Has(xml, "tendon") + Has(xml, "cranksite") + Has(xml, "site") +
              Has(xml, "body");
    if (cnt > 1) {
      Err(xml, "actuator can have at most one of transmission target");
      return;
    }
    bool cranklength = Has(xml, "cranklength");
    bool slidersite = Has(xml, "slidersite");
    if ((cranklength || slidersite) && cnt == 1 && !Has(xml, "cranksite")) {
      Err(xml,
          "cranklength and slidersite can only be used in slidercrank "
          "transmission");
      return;
    }
    if (Has(xml, "refsite") && cnt == 1 && !Has(xml, "site")) {
      Err(xml, "refsite can only be used with site transmission");
    }
  }

  // Q-SENS: sensor object/reference slot exclusivity, mirroring the hand-coded
  // dispatch in Sensor() (xml_native_reader.cc:4181-4638). ProtoSpec stores each
  // slot as a plain field, so the "exactly one of" / "at most one of" rules are
  // read checks rather than an objtype/reftype tag assignment.

  // A rangefinder attaches to exactly one of site or camera (:4243-4249).
  void RangefinderExclusion(XMLElement* xml) {
    if (Has(xml, "site") == Has(xml, "camera")) {
      Err(xml, "rangefinder requires exactly one of 'site' or 'camera'");
    }
  }

  // distance/normal/fromto: exactly one of (geom1, body1) and exactly one of
  // (geom2, body2) (:4483-4501).
  void GeomDistanceExclusion(XMLElement* xml) {
    if (Has(xml, "geom1") == Has(xml, "body1")) {
      Err(xml, "exactly one of (geom1, body1) must be specified");
      return;
    }
    if (Has(xml, "geom2") == Has(xml, "body2")) {
      Err(xml, "exactly one of (geom2, body2) must be specified");
    }
  }

  // contact sensor: at most one first source (geom1/body1/subtree1/site) and at
  // most one second source (geom2/body2/subtree2), num positive (:4505-4560).
  void ContactSensorExclusion(XMLElement* xml) {
    if (Has(xml, "site") + Has(xml, "body1") + Has(xml, "subtree1") +
            Has(xml, "geom1") >
        1) {
      Err(xml, "at most one of (geom1, body1, subtree1, site) can be specified");
      return;
    }
    if (Has(xml, "body2") + Has(xml, "subtree2") + Has(xml, "geom2") > 1) {
      Err(xml, "at most one of (geom2, body2, subtree2) can be specified");
      return;
    }
    if (const char* num = xml->Attribute("num")) {
      int n = 0;
      if (num::ParseInt<int>(num, n) == num::Status::Ok && n <= 0) {
        Err(xml, "'num' must be positive in sensor");
      }
    }
    // #9: the contact sensor's `data` keywords must be authored in strict enum
    // order (xml_native_reader.cc:4517-4530) -- unlike camera/rangefinder keyword
    // sets (order-insensitive), the contact reader rejects any other order, so
    // enum order is the sole legal spelling. Checked on the raw input before the
    // reader's keyword-set canonicalization sorts it.
    if (const char* data = xml->Attribute("data")) {
      int prev = -1;
      for (std::string_view tok : num::Tokens(std::string_view(data))) {
        ContactData v{};
        if (!FromMjcf(tok, v)) break;  // invalid keyword: reported by the field path
        int cur = static_cast<int>(v);
        if (cur <= prev) {
          Err(xml,
              "data attributes must be in order: found, force, torque, dist, "
              "pos, normal, tangent");
          return;
        }
        prev = cur;
      }
    }
  }

  // Frame sensors: a refname requires a matching reftype (:4356-4361 and peers).
  void FrameSensorRefPairing(XMLElement* xml) {
    if (Has(xml, "refname") && !Has(xml, "reftype")) {
      Err(xml, "refname '" + std::string(xml->Attribute("refname")) +
                   "' given but reftype is missing");
    }
  }

  // user sensor: objtype and objname must both be present or both absent
  // (:4577-4586).
  void UserSensorPairing(XMLElement* xml) {
    bool objtype = Has(xml, "objtype"), objname = Has(xml, "objname");
    if (objtype && !objname) {
      Err(xml, "objtype '" + std::string(xml->Attribute("objtype")) +
                   "' given but objname is missing");
    } else if (objname && !objtype) {
      Err(xml, "objname '" + std::string(xml->Attribute("objname")) +
                   "' given but objtype is missing");
    }
  }

  // plugin sensor: objtype/objname and reftype/refname pair up (:4607-4626).
  void PluginSensorPairing(XMLElement* xml) {
    bool objtype = Has(xml, "objtype"), objname = Has(xml, "objname");
    if (objtype && !objname) {
      Err(xml, "objtype is specified but objname is not");
    } else if (objname && !objtype) {
      Err(xml, "objname is specified but objtype is not");
    }
    bool reftype = Has(xml, "reftype"), refname = Has(xml, "refname");
    if (reftype && !refname) {
      Err(xml, "reftype is specified but refname is not");
    } else if (refname && !reftype) {
      Err(xml, "refname is specified but reftype is not");
    }
  }

  // Q-PLUGIN: a plugin/instance element's <config> keys must be unique
  // (ReadPluginConfigs, xml_native_reader.cc:135-141).
  void ConfigKeyUnique(XMLElement* xml) {
    std::unordered_set<std::string> seen;
    for (XMLElement* c = xml->FirstChildElement("config"); c;
         c = c->NextSiblingElement("config")) {
      const char* key = c->Attribute("key");
      if (key && !seen.insert(key).second) {
        Err(c, "duplicate config key: " + std::string(key));
      }
    }
  }

  // ---- children ----------------------------------------------------------- //
  void CheckUnknownAttributes(XMLElement* xml, const ElementBinding& b) {
    std::unordered_set<std::string_view> known;
    for (std::size_t i = 0; i < b.attr_count; ++i) known.insert(b.attrs[i].attr);
    // Read-only input aliases are accepted spellings canonicalized on read:
    // euler/axisangle/xyaxes/zaxis, fullinertia (Q-ORIENT/Q-INERTIA) and the
    // Wave B scalar/slot aliases (directional->type, diameter->area, size->data,
    // material texture->layer).
    for (std::size_t i = 0; i < b.input_alias_count; ++i)
      known.insert(b.input_aliases[i].attr);
    for (std::size_t i = 0; i < b.variant_count; ++i) {
      const VariantBinding& v = b.variants[i];
      for (std::size_t a = 0; a < v.arm_count; ++a) known.insert(v.arms[a].attr);
    }
    for (const tinyxml2::XMLAttribute* a = xml->FirstAttribute(); a;
         a = a->Next()) {
      if (!known.count(a->Name())) {
        Err(xml, "unknown attribute '" + std::string(a->Name()) +
                     "' in element '" + std::string(b.tag) + "'");
      }
    }
  }

  // Collects every tag the element's child lists can hold, paired with whether
  // that child type is supported. Regular lists contribute their one tag; union
  // lists contribute every member tag (compile-time membership). This is the
  // only place with the type info to classify a union child, so it drives the
  // runtime scan below.
  struct ClaimVisitor {
    std::vector<std::pair<std::string_view, bool>>* claims;
    const ElementBinding* b;
    template <class T>
    void field(int, const char*, T&) {}
    template <class T>
    void child(int cid, const char*, std::vector<std::unique_ptr<T>>&) {
      claims->push_back(
          {b->children[cid].tag, IsSupported(element_type_of<T>::value)});
    }
    template <class U>
    void union_child(int, const char*, std::vector<U>&) {
      AddUnionClaims<U>(*claims);
    }
  };

  // Diagnose child tags that map to no child list (malformed, an error) or to
  // an unsupported child type (skip signal). Supported children are read by
  // ChildVisitor. Union members are classified by member tag, so an unsupported
  // member of a supported union still signals skip rather than a false unknown.
  template <class E>
  void ClassifyChildren(XMLElement* xml, const ElementBinding& b, E& out) {
    std::vector<std::pair<std::string_view, bool>> claims;
    ClaimVisitor v{&claims, &b};
    Visit(out, v);
    for (XMLElement* c = xml->FirstChildElement(); c;
         c = c->NextSiblingElement()) {
      std::string_view tag = c->Value();
      bool known = false, supported = false;
      for (const auto& [t, s] : claims) {
        if (t == tag) {
          known = true;
          supported = s;
          break;
        }
      }
      if (!known) {
        Err(c, "unknown element '" + std::string(tag) + "' in '" +
                   std::string(b.tag) + "'");
      } else if (!supported) {
        Unsupported(c, std::string(tag));
      }
    }
  }

  template <class E>
  struct ChildVisitor {
    Reader* r;
    XMLElement* xml;
    const ElementBinding* b;

    template <class T>
    void field(int, const char*, T&) {}

    template <class T>
    void child(int cid, const char*, std::vector<std::unique_ptr<T>>& list) {
      const xmlbind::ChildBinding& cb = b->children[cid];
      if (!IsSupported(element_type_of<T>::value)) return;
      for (XMLElement* c = xml->FirstChildElement(); c;
           c = c->NextSiblingElement()) {
        if (cb.tag != c->Value()) continue;
        auto child = std::make_unique<T>();
        r->ReadElement(c, *child);
        list.push_back(std::move(child));
      }
    }

    // Union child list (Section 6 interleave): a single ordered list holding
    // several element spellings, read in document order so MuJoCo's id
    // assignment is reproduced. Each XML child is routed to the member whose
    // tag it carries; unknown/unsupported tags were already diagnosed by
    // ClassifyChildren and are skipped here.
    template <class U>
    void union_child(int, const char*, std::vector<U>& list) {
      for (XMLElement* c = xml->FirstChildElement(); c;
           c = c->NextSiblingElement()) {
        U item;
        if (r->ReadUnionItem(c, c->Value(), item)) list.push_back(std::move(item));
      }
    }
  };

  std::string filename_;
  std::vector<Diagnostic>& errors_;
  std::vector<Diagnostic>& warnings_;
  const ProvenanceMap* provenance_ = nullptr;
  std::vector<PendingOrient> pending_orient_;
};

void ValidateEulerseq(Reader& r, XMLElement* root, const Model& m) {
  // eulerseq must be length 3 (Q-ANGLE); MuJoCo enforces this at read.
  for (const auto& c : m.compilers) {
    if (c->eulerseq && c->eulerseq->size() != 3) {
      XMLElement* ce = root->FirstChildElement("compiler");
      r.Err(ce ? ce : root, "euler format must have length 3");
    }
  }
}

// Drive a parsed document through the include pre-pass and the table-driven
// reader. `model_dir` is where relative includes resolve first (the top-level
// file's directory); empty for string input with no real path.
ParseResult ReadDocument(XMLDocument& doc, const std::string& filename,
                         const std::string& model_dir) {
  ParseResult result;

  XMLElement* root = doc.RootElement();
  if (!root || std::strcmp(root->Value(), "mujoco") != 0) {
    Diagnostic d;
    d.kind = Diagnostic::Kind::MalformedInput;
    d.loc = ps::SourceLoc{filename, root ? root->GetLineNum() : 0};
    d.message = "root element must be <mujoco>";
    result.errors.push_back(std::move(d));
    return result;
  }

  // Expand <include> before anything else, so the reader sees a flat document.
  // Include failures are structural errors and abort the parse (MuJoCo throws).
  IncludeResult inc = ExpandIncludes(doc, model_dir, filename);
  if (!inc.errors.empty()) {
    result.errors = std::move(inc.errors);
    return result;
  }

  Reader reader(filename, result.errors, result.warnings, &inc.provenance);

  auto model = std::make_unique<Model>();
  reader.ReadElement(root, *model);
  // Q-ORIENT: fold the effective compiler context and resolve every collected
  // orientation now that the whole (include-flattened) document is in the tree.
  reader.ResolvePendingOrientations(*model);
  ValidateEulerseq(reader, root, *model);
  ValidateDefaultClasses(*model, result.errors);

  result.model = std::move(model);
  return result;
}

}  // namespace

ParseResult ParseMjcfString(const std::string& xml, const std::string& filename) {
  ParseResult result;
  XMLDocument doc;
  if (doc.Parse(xml.c_str(), xml.size()) != tinyxml2::XML_SUCCESS) {
    Diagnostic d;
    d.kind = Diagnostic::Kind::MalformedInput;
    d.loc = ps::SourceLoc{filename, doc.ErrorLineNum()};
    const char* what = doc.ErrorStr();
    d.message =
        std::string("XML parse error: ") + (what ? what : "malformed document");
    result.errors.push_back(std::move(d));
    return result;
  }
  // A pseudo-name like "<string>" yields an empty parent path, disabling
  // relative include resolution; a real path resolves includes beside it.
  std::string dir = std::filesystem::path(filename).parent_path().string();
  return ReadDocument(doc, filename, dir);
}

ParseResult ParseMjcfFile(const std::string& path) {
  ParseResult result;
  XMLDocument doc;
  if (doc.LoadFile(path.c_str()) != tinyxml2::XML_SUCCESS) {
    Diagnostic d;
    d.kind = Diagnostic::Kind::MalformedInput;
    d.loc = ps::SourceLoc{path, doc.ErrorLineNum()};
    const char* what = doc.ErrorStr();
    d.message = std::string("cannot read file: ") + (what ? what : "load error");
    result.errors.push_back(std::move(d));
    return result;
  }
  std::string dir = std::filesystem::path(path).parent_path().string();
  return ReadDocument(doc, path, dir);
}

std::string Diagnostic::Render() const {
  std::string prefix = loc.file.empty() ? "<string>" : loc.file;
  if (loc.line > 0) prefix += ":" + std::to_string(loc.line);
  return prefix + ": " + message;
}

bool ParseResult::unsupported_only() const {
  bool any_unsupported = false;
  for (const auto& e : errors) {
    if (e.kind == Diagnostic::Kind::MalformedInput) return false;
    if (e.kind == Diagnostic::Kind::UnsupportedElement) any_unsupported = true;
  }
  return any_unsupported;
}

}  // namespace ps::mjcf::io
