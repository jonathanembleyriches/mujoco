// MJCF writer: ps::Model -> canonical MJCF string, driven by the same generated
// xml_binding tables and Visit hook as the reader. Deterministic output: 2-space
// indent, attributes in schema field order, children in schema child order,
// exactly the authored fields (DR-1, no default diffing). Angle values are
// emitted verbatim in their authored unit and the compiler's angle attribute
// round trips as read (Q-ANGLE form preservation); MuJoCo applies the
// degree->radian conversion at compile, per consuming element.
//
// Numeric formatting is shortest round-trip via std::to_chars (numeric.cc). This
// deviates from MuJoCo's fixed-precision writer; the differential harness
// accounts for the difference when comparing our output to MuJoCo's.

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

#include "keywords.h"
#include "mjcf.h"
#include "numeric.h"
#include "protospec/core.h"
#include "types.h"
#include "visit.h"
#include "xml_binding.h"

namespace ps::mjcf::io {
namespace {

using xmlbind::AttrBinding;
using xmlbind::Bind;
using xmlbind::ElementBinding;

// --- Storage-shape traits (shared shape with the reader) ------------------ //
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

std::string Escape(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    switch (c) {
      case '&': out += "&amp;"; break;
      case '<': out += "&lt;"; break;
      case '>': out += "&gt;"; break;
      case '"': out += "&quot;"; break;
      default: out += c;
    }
  }
  return out;
}

template <class S>
std::string FormatScalar(S v) {
  if constexpr (std::is_same_v<S, double>) {
    return num::FormatDouble(v);
  } else if constexpr (std::is_same_v<S, float>) {
    return num::FormatFloat(v);
  } else {
    return num::FormatInt(static_cast<std::int64_t>(v));
  }
}

template <class S>
std::string JoinScalars(const S* data, std::size_t n) {
  std::string out;
  for (std::size_t i = 0; i < n; ++i) {
    if (i) out += ' ';
    out += FormatScalar(data[i]);
  }
  return out;
}

// Format an authored field value into an attribute string, or nullopt to skip
// (empty variable-length values are omitted, matching a non-authored field).
template <class Inner>
std::optional<std::string> FormatValue(const Inner& v, bool keyword_set) {
  if constexpr (is_ref<Inner>::value) {
    if (v.name.empty()) return std::nullopt;
    return Escape(v.name);
  } else if constexpr (std::is_same_v<Inner, std::string>) {
    return Escape(v);
  } else if constexpr (std::is_same_v<Inner, bool>) {
    return std::string(v ? "true" : "false");
  } else if constexpr (std::is_enum_v<Inner>) {
    return std::string(ToMjcf(v));
  } else if constexpr (is_std_array<Inner>::value) {
    return JoinScalars(v.data(), v.size());
  } else if constexpr (is_inline_vec<Inner>::value) {
    if (v.size() == 0) return std::nullopt;
    std::string out;
    for (std::size_t i = 0; i < v.size(); ++i) {
      if (i) out += ' ';
      out += FormatScalar(v[i]);
    }
    return out;
  } else if constexpr (is_vector<Inner>::value) {
    if (v.empty()) return std::nullopt;
    using S = typename Inner::value_type;
    if constexpr (is_ref<S>::value) {
      // ref<T>[]: names joined by single spaces (the wire form they came from).
      std::string out;
      for (std::size_t i = 0; i < v.size(); ++i) {
        if (i) out += ' ';
        out += Escape(v[i].name);
      }
      return out;
    } else if constexpr (std::is_enum_v<S>) {
      std::string out;
      for (std::size_t i = 0; i < v.size(); ++i) {
        if (i) out += ' ';
        out += std::string(ToMjcf(v[i]));
      }
      return out;
    } else {
      (void)keyword_set;
      return JoinScalars(v.data(), v.size());
    }
  } else if constexpr (std::is_arithmetic_v<Inner>) {
    return FormatScalar(v);
  } else {
    // Variant field types are emitted via WriteVariantArm, not as a plain
    // attribute; this branch exists only for the dispatcher's odr-use.
    return std::nullopt;
  }
}

std::string Indent(int depth) { return std::string(2 * depth, ' '); }

template <class E>
void WriteElement(const E& e, std::string_view tag, int depth, std::string& out,
                  const AutoNames* names);

// Append ` name="value"` unless value is skipped.
void AppendAttr(std::string& attrs, std::string_view name,
                const std::optional<std::string>& value) {
  if (!value) return;
  attrs += ' ';
  attrs += name;
  attrs += "=\"";
  attrs += *value;
  attrs += '"';
}

// Emit the attributes contributed by a variant field (DR-3 aliasing group).
// Orientation, inertia and camera intrinsics are canonicalized to plain fields
// (quat/iquat, diaginertia, fovy/focal -- Q-ORIENT/Q-INERTIA/R1), written by the
// normal attribute path; only the surviving variants have an arm writer.
void WriteVariantArm(std::string& attrs, const GeomShape& s) {
  if (auto* ft = std::get_if<FromTo>(&s)) {
    AppendAttr(attrs, "fromto", JoinScalars(ft->fromto.data(), 6));
  }
  // The Explicit arm carries the plain `size` field's value; it is written by
  // the size attribute, not here (avoids a duplicate size attribute).
}

void WriteVariantArm(std::string& attrs, const TextureSource& s) {
  if (auto* b = std::get_if<TextureBuiltin>(&s)) {
    AppendAttr(attrs, "builtin", std::string(ToMjcf(*b)));
  } else if (auto* f = std::get_if<TexFile>(&s)) {
    AppendAttr(attrs, "file", Escape(f->file));
  }
}

template <class E>
struct WriterVisitor {
  const ElementBinding* b;
  ElementType type;
  int depth;
  std::string* attrs;
  std::string* children;
  const AutoNames* names;

  template <class T>
  void field(int id, const char*, T& value) {
    for (std::size_t i = 0; i < b->attr_count; ++i) {
      if (b->attrs[i].field_id == id) {
        WriteAttrField(b->attrs[i], value);
        return;
      }
    }
    for (std::size_t i = 0; i < b->variant_count; ++i) {
      if (b->variants[i].field_id == id) {
        WriteVariantField(value);
        return;
      }
    }
  }

  template <class T>
  void WriteAttrField(const AttrBinding& ab, const T& value) {
    auto emit = [&](const auto& inner) {
      using I = std::decay_t<decltype(inner)>;
      // The <flag> element spells booleans enable/disable, not true/false.
      if constexpr (std::is_same_v<I, bool>) {
        if (type == ElementType::Flag) {
          AppendAttr(*attrs, ab.attr,
                     std::string(inner ? "enable" : "disable"));
          return;
        }
      }
      AppendAttr(*attrs, ab.attr, FormatValue(inner, ab.keyword_set));
    };
    if constexpr (is_opt<T>::value) {
      if (value.has_value()) emit(*value);
    } else {
      emit(value);
    }
  }

  template <class T>
  void WriteVariantField(const T& value) {
    if constexpr (is_opt<T>::value) {
      using V = typename is_opt<T>::inner;
      if constexpr (std::is_same_v<V, GeomShape> ||
                    std::is_same_v<V, TextureSource>) {
        if (value.has_value()) WriteVariantArm(*attrs, *value);
      }
    }
  }

  template <class T>
  void child(int cid, const char*, std::vector<std::unique_ptr<T>>& list) {
    std::string_view tag = b->children[cid].tag;
    for (const auto& item : list) {
      if (item) WriteElement(*item, tag, depth + 1, *children, names);
    }
  }

  template <class U>
  void union_child(int, const char*, std::vector<U>& list) {
    for (auto& item : list) {
      std::visit(
          [&](auto& p) {
            using M = typename std::decay_t<decltype(p)>::element_type;
            if (p) {
              WriteElement(*p, Bind(element_type_of<M>::value).tag, depth + 1,
                           *children, names);
            }
          },
          item.node);
    }
  }
};

template <class E>
void WriteElement(const E& e, std::string_view tag, int depth, std::string& out,
                  const AutoNames* names) {
  const ElementBinding& b = Bind(element_type_of<E>::value);
  std::string attrs;
  std::string children;
  WriterVisitor<E> v{&b,       element_type_of<E>::value,
                     depth,    &attrs,
                     &children, names};
  Visit(const_cast<E&>(e), v);

  // Auto-name injection (DR-10): an unnamed bindable element whose serial the
  // bridge recorded is emitted with its reserved name. The tree is untouched;
  // the name exists only here in the emitted text. Prepended so it reads first.
  if (names != nullptr) {
    auto it = names->find(e.serial);
    if (it != names->end()) {
      std::string name_attr = " name=\"";
      name_attr += Escape(it->second);
      name_attr += '"';
      attrs.insert(0, name_attr);
    }
  }

  out += Indent(depth);
  out += '<';
  out += tag;
  out += attrs;
  if (children.empty()) {
    out += "/>\n";
  } else {
    out += ">\n";
    out += children;
    out += Indent(depth);
    out += "</";
    out += tag;
    out += ">\n";
  }
}

}  // namespace

std::string WriteMjcf(const Model& model) {
  std::string out;
  WriteElement(model, Bind(ElementType::Model).tag, 0, out, nullptr);
  return out;
}

std::string WriteMjcf(const Model& model, const AutoNames& auto_names) {
  std::string out;
  WriteElement(model, Bind(ElementType::Model).tag, 0, out, &auto_names);
  return out;
}

}  // namespace ps::mjcf::io
