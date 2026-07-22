// ProtoSpec Python bindings: hand-written pybind11 support the generated glue
// (protospec/lib/python/generated/) builds on. Three things live here:
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
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

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

// The `name` field, wrapped so a *rename* warns. Setting `.name` writes the
// field directly and does NOT rewrite referrers -- the property setter has no
// handle on the owning Model, so it cannot. Renaming an already-named element
// this way silently dangles every reference to the old name. This property
// emits a UserWarning in exactly that case, pointing at Model.rename(), which
// is the referrer-rewriting verb. Naming a previously-nameless element, or a
// no-op set, is silent: nothing could refer to it yet, so nothing can dangle.
// The generator routes the `name` field here (emit_py._field_line); every other
// opt<string> field stays a plain OptField.
template <class E>
void NameField(pyb::class_<E>& c, ps::opt<std::string> E::*mp) {
  c.def_property(
      "name",
      [mp](const E& e) -> pyb::object {
        const auto& o = e.*mp;
        return o.has_value() ? pyb::cast(*o) : pyb::none();
      },
      [mp](E& e, pyb::handle v) {
        auto& o = e.*mp;
        const bool was_named = o.has_value() && !o->empty();
        std::string newname = v.is_none() ? std::string() : v.cast<std::string>();
        if (was_named && newname != *o) {
          std::string msg = "setting .name renames element '" + *o +
                            "' but does NOT rewrite referrers; use "
                            "model.rename(elem, \"" + newname +
                            "\") to keep references in sync";
          // Propagate the warning-as-error case (warnings filter "error"): leave
          // the name untouched so a rejected rename does not half-apply.
          if (PyErr_WarnEx(PyExc_UserWarning, msg.c_str(), 1) < 0)
            throw pyb::error_already_set();
        }
        if (v.is_none())
          o.reset();
        else
          o = std::move(newname);
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
  // Internal identity the model edit verbs (rename/delete/duplicate/reparent)
  // key on: the element's stable heap address and its ElementType tag. Not part
  // of the public surface -- Model.rename(elem, ...) etc. read these; users pass
  // the element handle, never the raw pointer.
  c.def_property_readonly("_ptr", [](const E& e) {
    return reinterpret_cast<std::uintptr_t>(&e);
  });
  c.def_property_readonly("_etype", [](const E&) {
    return static_cast<int>(ps::mjcf::element_type_of<E>::value);
  });
}

// --- Builder helpers (shared with the generated py_builders.cc) ----------- //

// Wrap a freshly built child as a live handle internal to `self`, apply any
// keyword arguments as attribute assignments (reusing the generated field
// properties, so kwargs are validated by exactly the same casters as direct
// assignment), and return it. This is how every add_* builder turns
// `add_geom(type="box", size=[...])` into an authored element.
template <class Child>
pyb::object FinishChild(pyb::object self, Child& child, const pyb::kwargs& kw) {
  pyb::object obj =
      pyb::cast(&child, pyb::return_value_policy::reference_internal, self);
  for (auto item : kw) obj.attr(item.first) = item.second;
  return obj;
}

// Append a fresh Child to the owned list `member` of the model's `Section`
// wrapper (e.g. AddOwnedChild(m.assets, &Asset::materials)), creating the
// wrapper at the front if the model has none -- mirroring the SDK Ensure*
// helpers. The list of (section, member) pairs is emitted from the schema, so a
// new upstream asset/contact/custom family flows into Python at the next emit.
template <class Section, class Child>
Child& AddOwnedChild(std::vector<std::unique_ptr<Section>>& sections,
                     std::vector<std::unique_ptr<Child>> Section::*member) {
  if (sections.empty() || !sections.front())
    sections.insert(sections.begin(), std::make_unique<Section>());
  auto child = std::make_unique<Child>();
  Child& ref = *child;
  ((*sections.front()).*member).push_back(std::move(child));
  return ref;
}

// A keyword constructor: `ps.Material(name="steel", rgba=[...])` builds a fresh
// element and applies each keyword through the generated field properties (so
// ctor kwargs are validated by exactly the same casters as `.field = ...` and
// as the add_* builders). Registered alongside the no-arg init, so both
// `ps.Material()` and `ps.Material(name=...)` work. The transient reference
// wrapper is scoped so its registered-instance entry is gone before the owning
// instance pybind builds from the returned unique_ptr is registered.
template <class E>
void InitKwargs(pyb::class_<E>& c) {
  c.def(
      pyb::init([](const pyb::kwargs& kw) {
        auto e = std::make_unique<E>();
        if (kw && pyb::len(kw) > 0) {
          pyb::object o =
              pyb::cast(e.get(), pyb::return_value_policy::reference);
          for (auto item : kw) o.attr(item.first) = item.second;
        }
        return e;
      }),
      "Construct an element, applying field keywords (name=..., rgba=..., ...).");
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

// --- Generated builder surface (py_builders.cc) --------------------------- //
// Both emitted from the schema, so new element families / union members reach
// Python at the next `emit` without hand edits.

// Registers the schema-driven Model authoring verbs: the union-family builders
// (add_actuator / add_sensor / add_tendon / add_equality, dispatched by MJCF
// keyword) and the owned-list builders (add_material / add_mesh / add_pair /
// add_key / add_numeric / ...). Called from Augment(Model).
void RegisterModelBuilders(pyb::class_<Model>& c);

// Wrap a type-erased element pointer as its typed Python handle, keyed on its
// runtime ElementType. Used by Model.duplicate() to return the clone as the
// same element class as its source. Returns None for an unknown tag.
pyb::object WrapElement(int etype, void* p, pyb::object parent);

}  // namespace ps::py

#endif  // PROTOSPEC_PYTHON_PS_BIND_H
