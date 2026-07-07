// MJCF reader: tinyxml2 document -> ps::Model, driven by the generated
// xml_binding tables (attrs/variants/children) with the generated Visit hook for
// typed field access. Hand code appears only at the quirks the tables cannot
// carry: numeric parsing (Q-NUM, in numeric.cc), the orientation / geom-shape /
// inertia / camera-intrinsics aliasing groups (Q-ORIENT/Q-FROMTO/Q-INERTIA),
// variable arity (Q-ARITY), and the strict support boundary
// (unsupported-element reporting).
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

// The supported families: Model-level blocks, the body tree, defaults, and
// assets. Everything else is a well-formed but unsupported element (skip
// signal), never a malformed input. The default-only actuator/pair/equality/
// tendon sub-elements are supported here as data: their top-level sections
// (<actuator>/<contact>/<equality>/<tendon>) remain unsupported, so those
// elements are only ever reached inside a <default>, which short-circuits on
// its unsupported parent otherwise.
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
      return true;
    default:
      return false;
  }
}

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

    if constexpr (std::is_same_v<E, Inertial>) InertialExclusion(xml, out);
    if constexpr (std::is_same_v<E, Size>) MemoryFixup(xml, out);

    ClassifyChildren(xml, b);
    ChildVisitor<E> cv{this, xml, &b};
    Visit(out, cv);
  }

 private:
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
      if constexpr (std::is_same_v<V, Orientation>) {
        MarshalOrientation(xml, value);
      } else if constexpr (std::is_same_v<V, GeomShape>) {
        MarshalGeomShape(xml, value);
      } else if constexpr (std::is_same_v<V, InertiaSpec>) {
        MarshalInertiaSpec(xml, value);
      } else if constexpr (std::is_same_v<V, CameraIntrinsics>) {
        MarshalCameraIntrinsics(xml, value);
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

  // Orientation encodings are stored exactly as authored (Q-ANGLE form
  // preservation): euler/axisangle angles keep their unit, and MuJoCo applies
  // compiler.angle at compile.
  void MarshalOrientation(XMLElement* xml, ps::opt<Orientation>& out) {
    int count = Has(xml, "quat") + Has(xml, "axisangle") + Has(xml, "xyaxes") +
                Has(xml, "zaxis") + Has(xml, "euler");
    if (count > 1) {
      Err(xml, "multiple orientation specifiers are not allowed");
      return;
    }
    if (count == 0) return;
    if (Has(xml, "quat")) {
      double q[4]{};
      if (!ReadDoubleArr(xml, "quat", 4, q)) return;
      if (q[0] == 0 && q[1] == 0 && q[2] == 0 && q[3] == 0) {
        Err(xml, "zero quaternion is not allowed");
        return;
      }
      out = Orientation{Quat{q[0], q[1], q[2], q[3]}};
    } else if (Has(xml, "axisangle")) {
      double a[4]{};
      if (!ReadDoubleArr(xml, "axisangle", 4, a)) return;
      out = Orientation{AxisAngle{{a[0], a[1], a[2]}, a[3]}};
    } else if (Has(xml, "xyaxes")) {
      double v[6]{};
      if (!ReadDoubleArr(xml, "xyaxes", 6, v)) return;
      out = Orientation{XYAxes{{v[0], v[1], v[2], v[3], v[4], v[5]}}};
    } else if (Has(xml, "zaxis")) {
      double v[3]{};
      if (!ReadDoubleArr(xml, "zaxis", 3, v)) return;
      out = Orientation{ZAxis{{v[0], v[1], v[2]}}};
    } else {  // euler
      double e[3]{};
      if (!ReadDoubleArr(xml, "euler", 3, e)) return;
      out = Orientation{Euler{{e[0], e[1], e[2]}}};
    }
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

  void MarshalInertiaSpec(XMLElement* xml, ps::opt<InertiaSpec>& out) {
    bool diag = Has(xml, "diaginertia");
    bool full = Has(xml, "fullinertia");
    if (diag && full) {
      Err(xml, "diaginertia and fullinertia cannot both be specified");
      return;
    }
    if (diag) {
      double d[3]{};
      if (!ReadDoubleArr(xml, "diaginertia", 3, d)) return;
      out = InertiaSpec{DiagInertia{{d[0], d[1], d[2]}}};
    } else if (full) {
      double f[6]{};
      if (!ReadDoubleArr(xml, "fullinertia", 6, f)) return;
      out = InertiaSpec{FullInertia{{f[0], f[1], f[2], f[3], f[4], f[5]}}};
    }
  }

  void MarshalCameraIntrinsics(XMLElement* xml, ps::opt<CameraIntrinsics>& out) {
    bool has_fovy = Has(xml, "fovy");
    bool has_focal = Has(xml, "focal");
    if (has_fovy && has_focal) {
      Err(xml, "fovy and focal cannot both be specified");
      return;
    }
    if (has_fovy) {
      // Camera fovy is a field of view in degrees in MuJoCo (mjModel.cam_fovy),
      // not converted to radians by compiler.angle.
      double f[1]{};
      if (!ReadDoubleArr(xml, "fovy", 1, f)) return;
      out = CameraIntrinsics{Fovy{f[0]}};
    } else if (has_focal) {
      double f[2]{};
      if (!ReadDoubleArr(xml, "focal", 2, f)) return;
      out = CameraIntrinsics{Focal{{f[0], f[1]}}};
    }
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

  void InertialExclusion(XMLElement* xml, Inertial& in) {
    const bool full =
        in.inertia && std::holds_alternative<FullInertia>(*in.inertia);
    if (full && in.iorient) {
      Err(xml, "fullinertia and inertial orientation cannot both be specified");
    }
  }

  // ---- children ----------------------------------------------------------- //
  void CheckUnknownAttributes(XMLElement* xml, const ElementBinding& b) {
    std::unordered_set<std::string_view> known;
    for (std::size_t i = 0; i < b.attr_count; ++i) known.insert(b.attrs[i].attr);
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

  // Diagnose child tags that map to no child list (malformed) or to an
  // unsupported / union child list (skip signal). Supported element children
  // are read by ChildVisitor.
  void ClassifyChildren(XMLElement* xml, const ElementBinding& b) {
    for (XMLElement* c = xml->FirstChildElement(); c;
         c = c->NextSiblingElement()) {
      std::string_view tag = c->Value();
      const xmlbind::ChildBinding* cb = nullptr;
      for (std::size_t i = 0; i < b.child_count; ++i) {
        if (!b.children[i].is_union && b.children[i].tag == tag) {
          cb = &b.children[i];
          break;
        }
      }
      if (!cb) {
        Err(c, "unknown element '" + std::string(tag) + "' in '" +
                   std::string(b.tag) + "'");
      } else if (!IsSupported(cb->child_type)) {
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
    template <class U>
    void union_child(int, const char*, std::vector<U>&) {}
  };

  std::string filename_;
  std::vector<Diagnostic>& errors_;
  std::vector<Diagnostic>& warnings_;
  const ProvenanceMap* provenance_ = nullptr;
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
