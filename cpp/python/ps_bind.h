// ProtoSpec Python bindings: hand-written pybind11 support the generated glue
// (cpp/python/generated/) builds on. Three things live here:
//
//   1. pybind type_casters that make ProtoSpec's storage types Pythonic:
//        * every IDL enum  <-> its MJCF keyword string ("sphere", "hinge", ...)
//        * ps::Ref<T>      <-> the referenced name string
//        * ps::InlineVec   <-> a Python list (variable-arity attributes)
//      std::array / std::vector / std::variant reuse pybind11/stl.h.
//   2. field-registration helpers: OptField (opt<T> -> a None-able property),
//      PlainField / StructField (a plain read/write property), ChildList /
//      UnionList (a child list -> a sequence of typed element handles), and
//      ElementBase (serial + source-location read-only members).
//   3. the Augment(pyb::class_<E>&) customization hook -- a no-op by default,
//      overloaded in module.cc for the body-context elements (Body/Frame/
//      Replicate) and Model, which gain add_* builders and friendly child views.
//
// Every generated class registration is `OptField(...); ...; Augment(c);`, so
// the generator stays uniform and all the type knowledge lives in one place.
#ifndef PROTOSPEC_PYTHON_PS_BIND_H
#define PROTOSPEC_PYTHON_PS_BIND_H

#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "protospec/core.h"
#include "types.h"
#include "keywords.h"
#include "generated/py_enums.h"
#include "generated/py_nocopy.h"

// --------------------------------------------------------------------------- //
// Type casters (pybind11::detail). Declared before any binding TU uses the      //
// underlying types, so every field property picks them up.                      //
// --------------------------------------------------------------------------- //
namespace pybind11 {
namespace detail {

// pybind11 3.x ships a native caster for every C++ enum. Opt the IDL enums out
// of it so our keyword-string caster below is the only match (otherwise the two
// partial specializations are ambiguous).
template <class T>
struct type_caster_enum_type_enabled<T, std::enable_if_t<ps::py_gen::is_enum_v<T>>>
    : std::false_type {};

// IDL enum <-> MJCF keyword string. Selected for exactly the generated enums via
// ps::py_gen::is_enum. A Python user always sees/sets the string keyword.
template <class T>
struct type_caster<T, std::enable_if_t<ps::py_gen::is_enum_v<T>>> {
  PYBIND11_TYPE_CASTER(T, const_name("str"));

  bool load(handle src, bool) {
    if (!isinstance<str>(src)) return false;
    return ps::mjcf::FromMjcf(std::string(reinterpret_borrow<str>(src)), value);
  }
  static handle cast(T v, return_value_policy, handle) {
    return str(std::string(ps::mjcf::ToMjcf(v))).release();
  }
};

// ps::Ref<Target> <-> the name string it stores (DR-8). Assigning a Python str
// sets the referenced name; reading yields the name.
template <class Target>
struct type_caster<ps::Ref<Target>> {
  using R = ps::Ref<Target>;
  PYBIND11_TYPE_CASTER(R, const_name("str"));

  bool load(handle src, bool) {
    if (!isinstance<str>(src)) return false;
    value.name = std::string(reinterpret_borrow<str>(src));
    return true;
  }
  static handle cast(const R& r, return_value_policy, handle) {
    return str(r.name).release();
  }
};

// ps::InlineVec<T, N> <-> a Python list (a fixed-capacity, count-carrying vector
// for variable-arity MJCF attributes). Bounds are a validation concern, so the
// caster does not enforce the range here.
template <class T, std::size_t N>
struct type_caster<ps::InlineVec<T, N>> {
  using Vec = ps::InlineVec<T, N>;
  PYBIND11_TYPE_CASTER(Vec, const_name("list"));

  bool load(handle src, bool convert) {
    if (!isinstance<sequence>(src)) return false;
    auto seq = reinterpret_borrow<sequence>(src);
    if (seq.size() > N) return false;
    value.clear();
    for (handle item : seq) {
      make_caster<T> conv_item;
      if (!conv_item.load(item, convert)) return false;
      value.push_back(cast_op<T>(std::move(conv_item)));
    }
    return true;
  }
  static handle cast(const Vec& v, return_value_policy pol, handle parent) {
    list l;
    for (const T& x : v)
      l.append(reinterpret_steal<object>(make_caster<T>::cast(x, pol, parent)));
    return l.release();
  }
};

}  // namespace detail
}  // namespace pybind11

namespace ps::py {

namespace pyb = pybind11;
using namespace ps::mjcf;  // element/struct/enum types are all in ps::mjcf

// --- Field helpers -------------------------------------------------------- //

// opt<V> field -> a property that reads None when unset and clears on `= None`.
template <class E, class V>
void OptField(pyb::class_<E>& c, const char* name, ps::opt<V> E::*mp) {
  c.def_property(
      name,
      [mp](const E& e) -> pyb::object {
        const auto& o = e.*mp;
        return o.has_value() ? pyb::cast(*o) : pyb::none();
      },
      [mp](E& e, pyb::handle v) {
        if (v.is_none()) {
          (e.*mp).reset();
          return;
        }
        (e.*mp) = v.cast<V>();
      });
}

// Plain (required) element field -> a read/write property.
template <class E, class V>
void PlainField(pyb::class_<E>& c, const char* name, V E::*mp) {
  c.def_property(
      name, [mp](const E& e) { return pyb::cast(e.*mp); },
      [mp](E& e, const V& v) { e.*mp = v; });
}

// POD struct (variant-arm) field. Same as PlainField; a distinct name keeps the
// generated struct file readable and lets the two evolve independently.
template <class E, class V>
void StructField(pyb::class_<E>& c, const char* name, V E::*mp) {
  PlainField(c, name, mp);
}

// A child list (vector<unique_ptr<Child>>) -> a read-only Python list of live,
// typed element handles (reference_internal keeps the owning parent alive).
template <class E, class Child>
void ChildList(pyb::class_<E>& c, const char* name,
               std::vector<std::unique_ptr<Child>> E::*mp) {
  c.def_property_readonly(name, [mp](pyb::object self) {
    E& e = self.cast<E&>();
    pyb::list out;
    for (auto& p : e.*mp)
      if (p)
        out.append(pyb::cast(p.get(),
                             pyb::return_value_policy::reference_internal,
                             self));
    return out;
  });
}

// A union child list (ordered heterogeneous vector) -> a read-only Python list
// iterating the active member of each node, each as its typed handle.
template <class E, class Any>
void UnionList(pyb::class_<E>& c, const char* name,
               std::vector<Any> E::*mp) {
  c.def_property_readonly(name, [mp](pyb::object self) {
    E& e = self.cast<E&>();
    pyb::list out;
    for (auto& item : e.*mp)
      std::visit(
          [&](auto& p) {
            if (p)
              out.append(pyb::cast(
                  p.get(), pyb::return_value_policy::reference_internal, self));
          },
          item.node);
    return out;
  });
}

// serial (stable element identity, DR-10/11) + structured source location.
template <class E>
void ElementBase(pyb::class_<E>& c) {
  c.def_readonly("serial", &E::serial);
  c.def_property_readonly("source_file", [](const E& e) { return e.loc.file; });
  c.def_property_readonly("source_line", [](const E& e) { return e.loc.line; });
}

// --- Augment hook --------------------------------------------------------- //
// Default: nothing. Overloaded (non-template, so it wins overload resolution)
// in module.cc for the elements that gain ergonomic builders.
template <class E>
void Augment(pyb::class_<E>&) {}

void Augment(pyb::class_<Body>& c);
void Augment(pyb::class_<Frame>& c);
void Augment(pyb::class_<Replicate>& c);
void Augment(pyb::class_<Model>& c);

}  // namespace ps::py

#endif  // PROTOSPEC_PYTHON_PS_BIND_H
