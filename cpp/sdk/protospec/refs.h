// ProtoSpec SDK: typed references (DR-8).
//
// References are stored as name strings with a phantom target type; the tree
// holds no pointers. This module gives the ergonomic operations that make that
// safe: resolve a ref to the element it names, find every referrer of an
// element, rename an element and fix up all referrers, and delete a subtree
// while reporting (or cascading) the references it would leave dangling.
//
// A "referrer" is any authored `Ref<T>` field whose target-type set includes
// the referenced element's type and whose stored name matches. Untyped string
// cross-references (e.g. a sensor's `objname`, a `<subtreecom>` body string)
// are NOT tracked here -- only the schema's typed `ref<T>` fields, which are
// exactly what the generated Visit hook exposes.
#ifndef PROTOSPEC_SDK_REFS_H
#define PROTOSPEC_SDK_REFS_H

#include <string>
#include <string_view>
#include <vector>

#include "protospec/detail.h"
#include "protospec/traversal.h"
#include "types.h"
#include "visit.h"

namespace ps::sdk {

namespace mj = ps::mjcf;

// A single authored reference that names a particular element.
struct Referrer {
  detail::Handle element;  // the element carrying the ref
  std::string field;       // the ref field's IR name (e.g. "joint", "material")
  std::string refname;     // the stored name string
  std::string path;        // root-to-element path, for diagnostics
};

namespace detail {

// Invoke `on(field_id, field_name, ref_name_ref, target_types)` for every
// authored typed reference of `e`. `ref_name_ref` is the mutable/const
// `std::string&` inside the Ref, so callers can both inspect and edit it.
template <class OnRef>
struct RefScan {
  OnRef* on;
  template <class U>
  void field(int id, const char* name, U& v) {
    using D = std::decay_t<U>;
    if constexpr (opt_ref<D>::value) {
      using Tgt = typename opt_ref<D>::target;
      if (v.has_value() && !v->name.empty())
        (*on)(id, name, v->name, RefTargetTypes<Tgt>());
    }
  }
  template <class C>
  void child(int, const char*, C&) {}
  template <class C>
  void union_child(int, const char*, C&) {}
};

template <class E, class OnRef>
void ScanRefs(E& e, OnRef&& on) {
  RefScan<std::remove_reference_t<OnRef>> v{&on};
  mj::Visit(e, v);
}

}  // namespace detail

// --- Reference assignment (DR-8) ------------------------------------------ //
// Every typed cross-reference in the model is stored as `opt<Ref<T>>` -- a name
// string with a phantom target type, never a tree pointer. These set or clear
// that field without the consumer having to spell `ps::Ref<T>(...)` or know the
// opt/Ref storage shape. An empty name clears the field (unauthored); otherwise
// it names `name`. No lookup is done here (that is Resolve's job); assigning a
// name that resolves to nothing is a validation concern, not an error here.

template <class T>
void SetRef(ps::opt<ps::Ref<T>>& field, std::string_view name) {
  if (name.empty())
    field.reset();
  else
    field = ps::Ref<T>(std::string(name));
}

// Point a reference at a concrete element by its authored name. Returns false
// (leaving the field untouched) when the target has no usable name -- an unnamed
// element cannot be referred to in MJCF.
template <class T, class E>
bool SetRef(ps::opt<ps::Ref<T>>& field, const E& target) {
  const std::string* nm = detail::NameOf(target);
  if (!nm || nm->empty()) return false;
  field = ps::Ref<T>(*nm);
  return true;
}

template <class T>
void ClearRef(ps::opt<ps::Ref<T>>& field) {
  field.reset();
}

// --- Resolve -------------------------------------------------------------- //

// The element a reference names, as a type-erased handle, or a null handle when
// nothing matches. `Ref<TendonAny>` resolves against both tendon spellings.
template <class Target>
detail::Handle Resolve(mj::Model& model, const ps::Ref<Target>& ref) {
  if (ref.empty()) return {};
  const std::vector<mj::ElementType> types = detail::RefTargetTypes<Target>();
  detail::Handle found;
  detail::WalkModelAll(model, [&](auto& e) {
    if (found) return;
    using E = std::decay_t<decltype(e)>;
    if (detail::Contains(types, mj::element_type_of<E>::value)) {
      const std::string* nm = detail::NameOf(e);
      if (nm && *nm == ref.name) found = detail::MakeHandle(e);
    }
  });
  return found;
}

// Typed convenience: resolve directly to a T* (only valid when Target is a
// concrete element type, not a union).
template <class T>
T* ResolveTo(mj::Model& model, const ps::Ref<T>& ref) {
  detail::Handle h = Resolve(model, ref);
  return h ? static_cast<T*>(const_cast<void*>(h.ptr)) : nullptr;
}

// --- FindReferrers -------------------------------------------------------- //

// Every authored reference in the model that names an element of type
// `targetType` with name `targetName`.
inline std::vector<Referrer> FindReferrers(mj::Model& model,
                                           std::string_view targetName,
                                           mj::ElementType targetType) {
  std::vector<Referrer> out;
  ParentMap pm(model);
  detail::WalkModelAll(model, [&](auto& e) {
    detail::Handle h = detail::MakeHandle(e);
    detail::ScanRefs(e, [&](int, const char* fname, const std::string& rn,
                            const std::vector<mj::ElementType>& tgts) {
      if (rn == targetName && detail::Contains(tgts, targetType)) {
        out.push_back(Referrer{h, fname, rn, pm.PathToPtr(h.ptr)});
      }
    });
  });
  return out;
}

// Every referrer of a concrete element.
template <class E>
std::vector<Referrer> FindReferrers(mj::Model& model, const E& elem) {
  const std::string* nm = detail::NameOf(elem);
  if (!nm) return {};
  return FindReferrers(model, *nm, mj::element_type_of<E>::value);
}

// --- Rename --------------------------------------------------------------- //

// Rename an element and rewrite every referrer to match. Returns the number of
// referrer fields updated. A no-op (0) when the element is nameless or the new
// name equals the old.
template <class E>
int Rename(mj::Model& model, E& elem, const std::string& newname) {
  const std::string* cur = detail::NameOf(elem);
  const std::string oldname = cur ? *cur : std::string();
  if (oldname == newname) return 0;
  const mj::ElementType targetType = mj::element_type_of<E>::value;

  int updated = 0;
  detail::WalkModelAll(model, [&](auto& other) {
    detail::ScanRefs(other, [&](int, const char*, std::string& rn,
                                const std::vector<mj::ElementType>& tgts) {
      if (rn == oldname && detail::Contains(tgts, targetType)) {
        rn = newname;
        ++updated;
      }
    });
  });
  detail::SetName(elem, newname);
  return updated;
}

// --- DeleteRecursive ------------------------------------------------------ //

struct DeleteReport {
  bool removed = false;               // was the element found and unlinked
  std::vector<Referrer> dangling;     // references left pointing at nothing
  bool cascaded = false;              // dangling refs were cleared
};

namespace detail {

struct NameType {
  std::string name;
  mj::ElementType type;
};

// Remove the element identified by `target` (a pointer) from whichever child or
// union-child list owns it. Stops at the first match.
struct Remover {
  const void* target;
  bool* done;
  template <class U>
  void field(int, const char*, U&) {}
  template <class C>
  void child(int, const char*, C& list) {
    if (*done) return;
    for (auto it = list.begin(); it != list.end(); ++it) {
      if (it->get() == target) {
        list.erase(it);
        *done = true;
        return;
      }
    }
    for (auto& p : list) {
      if (*done) return;
      if (p) {
        Remover r{target, done};
        mj::Visit(*p, r);
      }
    }
  }
  template <class C>
  void union_child(int, const char*, C& list) {
    if (*done) return;
    for (auto it = list.begin(); it != list.end(); ++it) {
      bool match = false;
      std::visit(
          [&](auto& p) {
            if (p && static_cast<const void*>(p.get()) == target) match = true;
          },
          it->node);
      if (match) {
        list.erase(it);
        *done = true;
        return;
      }
    }
    for (auto& item : list) {
      if (*done) return;
      std::visit(
          [&](auto& p) {
            if (p) {
              Remover r{target, done};
              mj::Visit(*p, r);
            }
          },
          item.node);
    }
  }
};

inline bool RemoveByPtr(mj::Model& m, const void* target) {
  bool done = false;
  Remover r{target, &done};
  mj::Visit(m, r);
  return done;
}

inline bool IsDeleted(const std::vector<NameType>& deleted,
                      const std::string& name,
                      const std::vector<mj::ElementType>& tgts) {
  for (const auto& d : deleted)
    if (d.name == name && Contains(tgts, d.type)) return true;
  return false;
}

// Reset (make unauthored) every typed reference of an element that names a
// deleted element, so no empty-but-present ref lingers.
struct ClearRefs {
  const std::vector<NameType>* deleted;
  template <class U>
  void field(int, const char*, U& v) {
    using D = std::decay_t<U>;
    if constexpr (opt_ref<D>::value) {
      using Tgt = typename opt_ref<D>::target;
      if (v.has_value() && !v->name.empty() &&
          IsDeleted(*deleted, v->name, RefTargetTypes<Tgt>())) {
        v.reset();
      }
    }
  }
  template <class C>
  void child(int, const char*, C&) {}
  template <class C>
  void union_child(int, const char*, C&) {}
};

}  // namespace detail

// Remove `elem` and its whole subtree from the model. Any reference elsewhere
// that named a removed element is reported as dangling (with its path). When
// `cascade` is true those references are cleared (set unauthored) so the model
// is left with no silent danglers; otherwise the model is returned as-is with
// the danglers reported for the caller to resolve.
template <class E>
DeleteReport DeleteRecursive(mj::Model& model, E& elem, bool cascade = false) {
  DeleteReport report;

  std::vector<detail::NameType> deleted;
  detail::WalkTree(elem, [&](auto& e) {
    using X = std::decay_t<decltype(e)>;
    if (const std::string* nm = detail::NameOf(e))
      deleted.push_back({*nm, mj::element_type_of<X>::value});
  });

  const void* ptr = &elem;
  report.removed = detail::RemoveByPtr(model, ptr);
  if (!report.removed) return report;

  ParentMap pm(model);
  detail::WalkModelAll(model, [&](auto& other) {
    detail::Handle h = detail::MakeHandle(other);
    detail::ScanRefs(other, [&](int, const char* fname, std::string& rn,
                                const std::vector<mj::ElementType>& tgts) {
      if (detail::IsDeleted(deleted, rn, tgts)) {
        report.dangling.push_back(
            Referrer{h, fname, rn, pm.PathToPtr(h.ptr)});
      }
    });
  });

  if (cascade && !report.dangling.empty()) {
    detail::WalkModelAll(model, [&](auto& other) {
      detail::ClearRefs cr{&deleted};
      mj::Visit(other, cr);
    });
    report.cascaded = true;
  }
  return report;
}

}  // namespace ps::sdk

#endif  // PROTOSPEC_SDK_REFS_H
