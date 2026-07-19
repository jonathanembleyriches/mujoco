// ProtoSpec SDK: typed references (DR-8).
//
// References are stored as name strings with a phantom target type; the tree
// holds no pointers. This module gives the ergonomic operations that make that
// safe: resolve a ref to the element it names, find every referrer of an
// element, rename an element and fix up all referrers, and delete a subtree
// while reporting (or cascading) the references it would leave dangling.
//
// A "referrer" is any authored `Ref<T>` field (scalar or `ref<T>[]` list
// entry) whose target-type set includes the referenced element's type and
// whose stored name matches -- plus the schema's DYNAMIC refs: string fields
// annotated `(target_from=sibling)`, whose target type is the runtime keyword
// the sibling holds (a frame sensor's objtype/objname pair). All of them are
// declared in the schema, which is what lets this module stay generic.
#ifndef PROTOSPEC_SDK_REFS_H
#define PROTOSPEC_SDK_REFS_H

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "protospec/detail.h"
#include "protospec/traversal.h"
#include "reflect.h"
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
    } else if constexpr (opt_ref_list<D>::value) {
      // A ref list is one callback per entry; the mutable string& means the
      // rename fixup rewrites list entries exactly like scalar refs.
      using Tgt = typename opt_ref_list<D>::target;
      if (v.has_value())
        for (auto& r : *v)
          if (!r.name.empty()) (*on)(id, name, r.name, RefTargetTypes<Tgt>());
    }
  }
  template <class C>
  void child(int, const char*, C&) {}
  template <class C>
  void union_child(int, const char*, C&) {}
};

// The `opt<string>` field of `e` at reflect field id `id`, with `e`'s
// constness. (FieldAt with an explicitly const T would make its two overloads
// ambiguous; this picks the right one.)
template <class E>
auto StringFieldAt(E& e, int id) {
  using P = std::remove_const_t<E>;
  return FieldAt<P, ps::opt<std::string>>(e, id);
}

template <class E, class OnRef>
void ScanRefs(E& e, OnRef&& on) {
  RefScan<std::remove_reference_t<OnRef>> v{&on};
  mj::Visit(e, v);

  // Dynamic refs (target_from, docs/refs_design.md category D): the schema
  // marks string fields whose target type is the runtime value of a SIBLING
  // field (objname (target_from=objtype)). The reflect descriptor carries the
  // marking, so rename fixup / referrer scan / delete cleanup reach these
  // fields with no per-element code. An unset or unknown keyword yields no
  // targets and the name is left alone -- opaque, never guessed at.
  using P = std::remove_const_t<E>;
  if constexpr (!std::is_same_v<P, mj::Model>) {
    const mj::reflect::ElementDescriptor& desc =
        mj::reflect::Describe(mj::element_type_of<P>::value);
    for (int i = 0; i < static_cast<int>(desc.field_count); ++i) {
      const mj::reflect::FieldDescriptor& fd = desc.fields[i];
      if (fd.target_from.empty()) continue;
      int sib = -1;
      for (int j = 0; j < static_cast<int>(desc.field_count); ++j) {
        if (desc.fields[j].name == fd.target_from) {
          sib = j;
          break;
        }
      }
      if (sib < 0) continue;
      auto* kw = StringFieldAt(e, sib);
      auto* nm = StringFieldAt(e, i);
      if (!kw || !kw->has_value() || !nm || !nm->has_value() ||
          (*nm)->empty()) {
        continue;
      }
      std::vector<mj::ElementType> targets = DynRefTargetTypes(**kw);
      if (targets.empty()) continue;
      on(i, fd.name.data(), **nm, targets);
    }
  }
}

// True when `newname` is already held by an element other than `self` in the
// same shared name namespace as `type` (NameCategory folding: the joint /
// tendon / actuator / sensor / equality spelling unions each share one
// namespace). Renaming onto such a name would fuse two elements' referrer sets.
inline bool NameTakenByOther(mj::Model& model, const void* self,
                             mj::ElementType type, const std::string& newname) {
  const int cat = NameCategory(type);
  bool taken = false;
  WalkModelAll(model, [&](auto& e) {
    using E = std::decay_t<decltype(e)>;
    if constexpr (!std::is_same_v<E, mj::Model>) {
      if (taken || static_cast<const void*>(&e) == self) return;
      if (NameCategory(mj::element_type_of<E>::value) != cat) return;
      const std::string* nm = NameOf(e);
      if (nm && *nm == newname) taken = true;
    }
  });
  return taken;
}

// A name an element may be renamed to: non-empty and outside the reserved
// auto-name prefix (ps::kReservedNamePrefix, owned by compile-time binding
// names).
inline bool AssignableName(const std::string& name) {
  return !name.empty() &&
         !std::string_view(name).starts_with(ps::kReservedNamePrefix);
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
// element cannot be referred to in MJCF. `target`'s element type must be a valid
// target of `Ref<T>` (itself for a concrete ref, a union member for a union ref)
// -- otherwise this is a compile error, not a silent type-mismatched assignment.
template <class T, class E>
bool SetRef(ps::opt<ps::Ref<T>>& field, const E& target) {
  static_assert(detail::ref_accepts_target<T, std::decay_t<E>>,
                "SetRef target type is not a valid target for this Ref<T>");
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

// --- ScanRefs (reflection-driven reference visit) ------------------------- //

// Invoke `on` once for every authored typed reference OF a single element:
// scalar `Ref<T>` fields, each entry of a `ref<T>[]` list, and the schema's
// dynamic (target_from) references. `on` is called as
//
//   on(int field_id, const char* field_name, std::string& ref_name,
//      const std::vector<mj::ElementType>& target_types)
//
// where `ref_name` is the LIVE, mutable name string inside the reference --
// rewrite it to edit the reference in place (this is how Rename fixes referrers)
// or just read it for a scan -- and `target_types` is the set of element types
// the reference may name (a union ref names several). Reflection-driven: it
// needs no per-element code and tracks the schema automatically. Only `element`
// itself is visited (not its subtree); walk with WalkModel/WalkSubtree to reach
// every element. Never invoked for a Model (the root holds no references).
template <class E, class OnRef>
void ScanRefs(E& element, OnRef&& on) {
  detail::ScanRefs(element, std::forward<OnRef>(on));
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

// --- UniqueName ----------------------------------------------------------- //

// A name unique within `type`'s MuJoCo name namespace: returns `base` when free,
// else `base_1`, `base_2`, ... MuJoCo namespaces names by category, not by exact
// element type -- the two joint spellings share one namespace, and every
// actuator / sensor / tendon / equality spelling shares its union's namespace
// (detail::NameCategory) -- so a name unique here cannot collide at compile time.
// This is the generic form of the editor's per-add name uniquing.
inline std::string UniqueName(mj::Model& model, mj::ElementType type,
                             std::string_view base) {
  const int cat = detail::NameCategory(type);
  std::unordered_set<std::string> used;
  detail::WalkModelAll(model, [&](auto& e) {
    using E = std::decay_t<decltype(e)>;
    if constexpr (!std::is_same_v<E, mj::Model>) {
      if (detail::NameCategory(mj::element_type_of<E>::value) == cat)
        if (const std::string* nm = detail::NameOf(e)) used.insert(*nm);
    }
  });
  std::string b(base);
  if (!used.count(b)) return b;
  for (int k = 1;; ++k) {
    std::string c = b + "_" + std::to_string(k);
    if (!used.count(c)) return c;
  }
}

// --- Rename --------------------------------------------------------------- //
//
// RESULT-OBJECT CONVENTION. The structural verbs (Rename, Duplicate, Reparent,
// DeleteSubtree, Attach) all report through a small result object carrying `bool
// ok` (contextually convertible: `if (auto r = sdk::Rename(...))`) and, where a
// failure has a cause, a `std::string reason`. RenameResult additionally carries
// `updated` (referrer fields rewritten). The former scalar returns (`int` for
// Rename, `void*` for Duplicate) survive one release as deprecated conversions
// on the result object so existing call sites keep compiling; prefer `.ok` /
// `.updated` / `.clone`.
struct RenameResult {
  bool ok = false;         // the rename applied (or was an accepted no-op)
  int updated = 0;         // referrer fields rewritten (0 for a no-op / nameless
                           // element gaining its first name)
  std::string reason;      // why it was rejected (empty on success)
  explicit operator bool() const { return ok; }
  // DEPRECATED one-release compat with the former `int` return: the referrer
  // count on success, or -1 on rejection. Prefer `.ok` / `.updated`.
  operator int() const { return ok ? updated : -1; }  // NOLINT
};

// Rename an element and rewrite every referrer to match. On success `ok` is
// true and `updated` is the number of referrer fields rewritten -- 0 when the
// new name equals the old (an accepted no-op) or when the element was nameless
// (it gains the name; nothing can have referred to it). On rejection `ok` is
// false, `reason` says why, and the model is left untouched: `newname` is
// invalid (empty, inside the reserved auto-name prefix ps::kReservedNamePrefix)
// or already held by a DIFFERENT element in the same shared name namespace
// (renaming onto it would silently fuse the two elements' referrer sets). The
// pointer is excluded from this template so the runtime
// `Rename(model, const void*, newname)` below wins for a type-erased element
// pointer.
template <class E, std::enable_if_t<!std::is_pointer_v<E>, int> = 0>
RenameResult Rename(mj::Model& model, E& elem, const std::string& newname) {
  const std::string* cur = detail::NameOf(elem);
  const std::string oldname = cur ? *cur : std::string();
  if (oldname == newname) return {true, 0, ""};
  const mj::ElementType targetType = mj::element_type_of<E>::value;
  if (!detail::AssignableName(newname))
    return {false, 0, "name is empty or inside the reserved auto-name prefix"};
  if (detail::NameTakenByOther(model, &elem, targetType, newname))
    return {false, 0, "name already held by another element of this category"};

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
  return {true, updated, ""};
}

// --- DeleteRecursive ------------------------------------------------------ //

struct DeleteReport {
  bool removed = false;               // was the element found and unlinked
  std::vector<Referrer> dangling;     // references left pointing at nothing
  bool cascaded = false;              // dangling refs were cleared
  // Uniform result-object contract: truthy when the delete found and removed the
  // target (`if (sdk::DeleteSubtree(...)) ...`); `dangling` details the fallout.
  explicit operator bool() const { return removed; }
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
    } else if constexpr (opt_ref_list<D>::value) {
      // Drop only the deleted names from a ref list; an emptied list resets to
      // unauthored so the attribute disappears from the written MJCF.
      using Tgt = typename opt_ref_list<D>::target;
      if (v.has_value()) {
        auto& list = *v;
        std::erase_if(list, [&](const auto& r) {
          return !r.name.empty() &&
                 IsDeleted(*deleted, r.name, RefTargetTypes<Tgt>());
        });
        if (list.empty()) v.reset();
      }
    }
  }
  template <class C>
  void child(int, const char*, C&) {}
  template <class C>
  void union_child(int, const char*, C&) {}
};

// ClearRefs plus the dynamic fields the plain Visit cannot see: a target_from
// name whose sibling keyword resolves into the deleted set is reset alongside
// its typed siblings.
template <class E>
void ClearRefsOn(E& e, const std::vector<NameType>& deleted) {
  ClearRefs cr{&deleted};
  mj::Visit(e, cr);
  using P = std::remove_const_t<E>;
  if constexpr (!std::is_same_v<P, mj::Model>) {
    const mj::reflect::ElementDescriptor& desc =
        mj::reflect::Describe(mj::element_type_of<P>::value);
    for (int i = 0; i < static_cast<int>(desc.field_count); ++i) {
      const mj::reflect::FieldDescriptor& fd = desc.fields[i];
      if (fd.target_from.empty()) continue;
      int sib = -1;
      for (int j = 0; j < static_cast<int>(desc.field_count); ++j) {
        if (desc.fields[j].name == fd.target_from) {
          sib = j;
          break;
        }
      }
      if (sib < 0) continue;
      auto* kw = StringFieldAt(e, sib);
      auto* nm = StringFieldAt(e, i);
      if (!kw || !kw->has_value() || !nm || !nm->has_value() ||
          (*nm)->empty()) {
        continue;
      }
      if (IsDeleted(deleted, **nm, DynRefTargetTypes(**kw))) nm->reset();
    }
  }
}

}  // namespace detail

// Remove `elem` and its whole subtree from the model. Any reference elsewhere
// that named a removed element is reported as dangling (with its path). When
// `cascade` is true those references are cleared (set unauthored) so the model
// is left with no silent danglers; otherwise the model is returned as-is with
// the danglers reported for the caller to resolve.
template <class E, std::enable_if_t<!std::is_pointer_v<E>, int> = 0>
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
      detail::ClearRefsOn(other, deleted);
    });
    report.cascaded = true;
  }
  return report;
}

// --- Runtime-typed Rename / DeleteSubtree --------------------------------- //
//
// `Rename<E>` / `DeleteRecursive<E>` above are keyed on an element's STATIC
// type. Driving them from a runtime element pointer would instantiate each
// across all ~140 families, and each re-instantiates a whole-model walk -- a
// ~140x140 fan-out that makes a translation unit take minutes to compile. These
// variants are keyed on the element POINTER instead: one generic model walk
// locates it, recovers its runtime ElementType, and drives the same referrer
// bookkeeping through the runtime helpers. Same behaviour, a handful of walk
// instantiations rather than hundreds. This is the supported API for a consumer
// that resolves elements dynamically (by pick, serial, or path); reach for the
// `<E>` templates only when the static type is already in hand.

// Rename the element at `elem` (any element owned by `model`) to `newname` and
// rewrite every typed referrer. Same RenameResult contract as the `<E>` form:
// `ok` with `updated` referrer count on success (0 for a no-op / nameless gain),
// `ok == false` with a `reason` and the model untouched when `elem` is not in
// the model or `newname` is invalid (empty, reserved prefix, or already held by
// a DIFFERENT element in the same shared name namespace).
inline RenameResult Rename(mj::Model& model, const void* elem,
                          const std::string& newname) {
  bool found = false;
  std::string oldname;
  mj::ElementType type = mj::ElementType::Model;
  std::function<void()> set_name;
  detail::WalkModelAll(model, [&](auto& e) {
    using E = std::decay_t<decltype(e)>;
    if (found) return;
    if (static_cast<const void*>(&e) == elem) {
      found = true;
      type = mj::element_type_of<E>::value;
      if (const std::string* nm = detail::NameOf(e)) oldname = *nm;
      auto* ep = &e;
      set_name = [ep, &newname] { detail::SetName(*ep, newname); };
    }
  });
  if (!found) return {false, 0, "element is not in the model"};
  if (oldname == newname) return {true, 0, ""};
  if (!detail::AssignableName(newname))
    return {false, 0, "name is empty or inside the reserved auto-name prefix"};
  if (detail::NameTakenByOther(model, elem, type, newname))
    return {false, 0, "name already held by another element of this category"};

  int updated = 0;
  detail::WalkModelAll(model, [&](auto& other) {
    detail::ScanRefs(other, [&](int, const char*, std::string& rn,
                                const std::vector<mj::ElementType>& tgts) {
      if (rn == oldname && detail::Contains(tgts, type)) {
        rn = newname;
        ++updated;
      }
    });
  });
  set_name();
  return {true, updated, ""};
}

// Remove the subtree rooted at `elem` from `model`. Any reference elsewhere
// that named a removed element is reported as dangling (with its path). When
// `cascade`, those references are cleared (set unauthored) so the model has no
// silent danglers; otherwise the model is left as-is for the caller to resolve.
// `report.removed` is false when `elem` is not found. Same contract as
// `DeleteRecursive`, keyed on the runtime element pointer.
inline DeleteReport DeleteSubtree(mj::Model& model, const void* elem,
                                  bool cascade = false) {
  DeleteReport report;

  ParentMap pm(model);
  if (!pm.Lookup(elem)) return report;  // not in the model: removed = false
  auto in_subtree = [&](const void* p) -> bool {
    for (const void* q = p; q;) {
      if (q == elem) return true;
      const ParentMap::Node* n = pm.Lookup(q);
      if (!n) break;
      q = n->parent;
    }
    return false;
  };

  std::vector<detail::NameType> deleted;
  detail::WalkModelAll(model, [&](auto& e) {
    using E = std::decay_t<decltype(e)>;
    if (in_subtree(&e)) {
      if (const std::string* nm = detail::NameOf(e))
        deleted.push_back({*nm, mj::element_type_of<E>::value});
    }
  });

  report.removed = detail::RemoveByPtr(model, elem);
  if (!report.removed) return report;

  ParentMap after(model);
  detail::WalkModelAll(model, [&](auto& other) {
    detail::Handle h = detail::MakeHandle(other);
    detail::ScanRefs(other, [&](int, const char* fname, std::string& rn,
                                const std::vector<mj::ElementType>& tgts) {
      if (detail::IsDeleted(deleted, rn, tgts))
        report.dangling.push_back(Referrer{h, fname, rn, after.PathToPtr(h.ptr)});
    });
  });

  if (cascade && !report.dangling.empty()) {
    detail::WalkModelAll(model, [&](auto& other) {
      detail::ClearRefsOn(other, deleted);
    });
    report.cascaded = true;
  }
  return report;
}

// --- PruneSubtrees -------------------------------------------------------- //

namespace detail {

// Erase from every child / union-child list the elements a predicate selects,
// their subtrees going with them, then descend into the survivors. The generic
// form of an authored "drop these elements" pass (e.g. the editor's layer
// prune).
template <class Pred>
struct PruneVisitor {
  Pred* pred;
  template <class U>
  void field(int, const char*, U&) {}
  template <class T>
  void child(int, const char*, std::vector<std::unique_ptr<T>>& list) {
    std::erase_if(list,
                  [&](const std::unique_ptr<T>& p) { return p && (*pred)(*p); });
    for (auto& p : list)
      if (p) {
        PruneVisitor sub{pred};
        mj::Visit(*p, sub);
      }
  }
  template <class U>
  void union_child(int, const char*, std::vector<U>& list) {
    std::erase_if(list, [&](const U& item) {
      bool rm = false;
      std::visit([&](const auto& p) { if (p && (*pred)(*p)) rm = true; },
                 item.node);
      return rm;
    });
    for (auto& item : list)
      std::visit(
          [&](auto& p) {
            if (p) {
              PruneVisitor sub{pred};
              mj::Visit(*p, sub);
            }
          },
          item.node);
  }
};

}  // namespace detail

// Remove every element for which `pred` returns true, subtrees included: pruning
// a body prunes everything beneath it, and a kept parent is still descended so
// its own selected children go too. `pred` is any callable `bool(const auto&
// element)` invoked on each element in document order; the Model root is never
// offered to it (there is nothing to prune it from).
//
// A raw structural prune: unlike DeleteRecursive / DeleteSubtree it does NOT
// scan or clear references to removed elements, so a ref elsewhere may be left
// dangling. Use it for whole-partition drops (compile-input filtering, layer
// pruning) where dangling names are tolerated or reconciled separately; reach
// for the Delete verbs when referrer bookkeeping must stay consistent.
template <class Pred>
void PruneSubtrees(mj::Model& model, Pred&& pred) {
  detail::PruneVisitor<std::remove_reference_t<Pred>> v{&pred};
  mj::Visit(model, v);
}

}  // namespace ps::sdk

#endif  // PROTOSPEC_SDK_REFS_H
