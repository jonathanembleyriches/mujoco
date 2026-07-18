// ProtoSpec SDK: default-class operations (plan Section 7).
//
// Defaults are first-class data: a <default> class holds an `opt<>`-everything
// partial per defaultable family, and elements inherit unauthored fields from
// their class, its ancestor classes, `main`, and finally the IDL defaults, in
// that order (first authored value wins). These operations expose that layering
// and two tree transforms built on it.
//
// IMPORTANT: unlike Effective (a pure query returning a computed copy),
// FlattenDefaults and ExtractClass MUTATE the model. They are authoring
// operations, not part of Compile: FlattenDefaults bakes effective values into
// elements and drops the class tree; ExtractClass factors shared authored
// values out of a set of elements into a new class. Neither is reversible in
// place; clone the model first if you need the original.
//
// Scope: the layered merge is defined for families whose class partial has the
// same element type as the live element (geom, joint, site, camera, light,
// mesh, material, pair, and every actuator spelling that has a <default> block
// -- i.e. all but the plugin actuator, which has none). MuJoCo's equality/tendon
// defaults use a distinct partial type (EqualityDefault/TendonDefault) and are
// out of scope for these merges; such elements are passed through unchanged.
#ifndef PROTOSPEC_SDK_CLASSES_H
#define PROTOSPEC_SDK_CLASSES_H

#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "defaults.h"
#include "protospec/detail.h"
#include "protospec/traversal.h"
#include "types.h"
#include "visit.h"

namespace ps::sdk {

namespace mj = ps::mjcf;

// --- Family mapping: element type <-> the Default vector that holds its class
// partial ----------------------------------------------------------------- //

template <class T>
struct HasDefaultFamily : std::false_type {};

template <class T>
const std::vector<std::unique_ptr<T>>* DefaultVec(const mj::Default&) {
  return nullptr;
}
template <class T>
std::vector<std::unique_ptr<T>>* DefaultVecMut(mj::Default&) {
  return nullptr;
}

namespace detail {
// Runtime record of the element types PS_SDK_FAMILY maps, populated once at
// static init by the macro. Lets DefaultFamilyCoverageGaps compare what the
// class-merge actually covers against the generated <default> struct's families.
inline std::vector<mj::ElementType>& DefaultFamilyRegistry() {
  static std::vector<mj::ElementType> r;
  return r;
}
}  // namespace detail

#define PS_SDK_FAMILY(T, member)                                        \
  template <>                                                           \
  struct HasDefaultFamily<mj::T> : std::true_type {};                   \
  template <>                                                           \
  inline const std::vector<std::unique_ptr<mj::T>>* DefaultVec<mj::T>(  \
      const mj::Default& d) {                                           \
    return &d.member;                                                   \
  }                                                                     \
  template <>                                                           \
  inline std::vector<std::unique_ptr<mj::T>>* DefaultVecMut<mj::T>(     \
      mj::Default& d) {                                                 \
    return &d.member;                                                   \
  }                                                                     \
  inline const bool ps_sdk_family_reg_##T =                             \
      (detail::DefaultFamilyRegistry().push_back(                       \
           mj::element_type_of<mj::T>::value),                          \
       true);

PS_SDK_FAMILY(Geom, geom)
PS_SDK_FAMILY(Joint, joint)
PS_SDK_FAMILY(Site, site)
PS_SDK_FAMILY(Camera, camera)
PS_SDK_FAMILY(Light, light)
PS_SDK_FAMILY(Mesh, mesh)
PS_SDK_FAMILY(Material, material)
PS_SDK_FAMILY(Pair, pair)
PS_SDK_FAMILY(ActuatorGeneral, general)
PS_SDK_FAMILY(Motor, motor)
PS_SDK_FAMILY(Position, position)
PS_SDK_FAMILY(Velocity, velocity)
PS_SDK_FAMILY(IntVelocity, intvelocity)
PS_SDK_FAMILY(Damper, damper)
PS_SDK_FAMILY(Cylinder, cylinder)
PS_SDK_FAMILY(Muscle, muscle)
PS_SDK_FAMILY(Adhesion, adhesion)
PS_SDK_FAMILY(DcMotor, dcmotor)

#undef PS_SDK_FAMILY

// Every same-type-partial family the generated <default> struct exposes must
// have a PS_SDK_FAMILY mapping; without it the class-merge (Effective /
// FlattenDefaults / ExtractClass) silently passes that family through unmerged.
// This returns the <default> children that are same-type partials with no
// mapping -- empty unless the schema grew a defaultable family and the family
// table above was not extended to match.
//
// A <default> child is a same-type partial when its class element has the SAME
// element type as the live element it defaults (geom, joint, ...). The two
// distinct-partial families use a dedicated type named `<Family>Default`
// (EqualityDefault, TendonDefault) and the recursive `subclasses` child is a
// nested Default; all three carry the `Default` name suffix and are out of scope
// for the same-type merge, so they are skipped here.
inline std::vector<mj::ElementType> DefaultFamilyCoverageGaps() {
  const std::vector<mj::ElementType>& covered =
      detail::DefaultFamilyRegistry();
  std::vector<mj::ElementType> gaps;
  const mj::reflect::ElementDescriptor& d =
      mj::reflect::Describe(mj::ElementType::Default);
  for (std::size_t i = 0; i < d.child_count; ++i) {
    std::string_view target = d.children[i].target;
    if (target.size() >= 7 && target.substr(target.size() - 7) == "Default") {
      continue;  // Default / EqualityDefault / TendonDefault: not same-type
    }
    const mj::reflect::ElementDescriptor* cd =
        mj::reflect::DescribeByName(target);
    if (cd && !detail::Contains(covered, cd->type)) gaps.push_back(cd->type);
  }
  return gaps;
}

// --- Field-wise "fill unauthored" merge ----------------------------------- //

namespace detail {

template <class U>
void MergeField(U& dst, const U& src) {
  if constexpr (is_opt<U>::value) {
    if (!dst.has_value() && src.has_value()) dst = src;
  }
}

// Fill each unauthored field of `dst` from `src` (same element type). Required
// (non-opt) fields are left untouched -- a class never overrides structure.
template <class T>
struct MergeVisitor {
  const T* src;
  template <class U>
  void field(int id, const char*, U& dst) {
    const U* s = FieldAt<T, U>(*src, id);
    if (s) MergeField(dst, *s);
  }
  template <class C>
  void child(int, const char*, C&) {}
  template <class C>
  void union_child(int, const char*, C&) {}
};

template <class T>
void MergeUnset(T& dst, const T& src) {
  MergeVisitor<T> v{&src};
  mj::Visit(dst, v);
}

// Index of the <default> class tree: name -> node, node -> parent, plus the
// top-level blocks that make up the root `main` class.
//
// A model may carry SEVERAL top-level <default> blocks -- the norm once
// <include> merges two files' top-level blocks into one Model.defaults list.
// MuJoCo's reader has exactly one root default (`main`, created once in the
// mjCModel constructor) and feeds every top-level <default> section to it in
// document order: each block's direct element defaults merge field-wise into
// `main` (later blocks overwrite earlier per field) and each block's nested
// classes are added to the one shared, flat class namespace
// (xml/xml_native_reader.cc mjXReader::Default: at top level `def==nullptr`,
// so it takes `def = mjs_getSpecDefault(spec)` -- the single main -- rather than
// allocating a new one, and a top-level class name other than "" / "main" is
// rejected). Same-named classes across blocks are invalid: mjs_addDefault ->
// mjCModel::AddDefault checks the flat defaults_ list and returns null on a
// repeat, which the reader turns into "repeated default class name". The SDK
// does not validate, so it keeps the first occurrence of a duplicate key.
//
// A nested class snapshots `main` at the point it is parsed
// (mjCModel::AddDefault -> CopyWithoutChildren of the parent), so a class only
// inherits the root-level fields of the top-level blocks up to and including its
// own (later blocks' root fields never reach an earlier block's class). Roots()
// preserves that document order and RootRank() locates a block within it so
// MergeClassChain can reproduce the snapshot.
class DefaultIndex {
 public:
  explicit DefaultIndex(const mj::Model& m) {
    for (const auto& d : m.defaults) {
      if (!d) continue;
      roots_.push_back(d.get());
      Add(*d, nullptr);
    }
  }
  const mj::Default* ByNameOrRoot(const std::string& n) const {
    // "" and "main" both name the root; with multiple top-level blocks it is the
    // LAST one, so a class-free element sees `main` with every block merged in.
    if (n.empty() || n == "main") return root_;
    auto it = by_name_.find(n);
    return it != by_name_.end() ? it->second : root_;
  }
  const mj::Default* ParentOf(const mj::Default* d) const {
    auto it = parent_.find(d);
    return it != parent_.end() ? it->second : nullptr;
  }
  // Top-level <default> blocks in document order; together they form `main`.
  const std::vector<const mj::Default*>& Roots() const { return roots_; }
  // Document position of a top-level block in Roots(), or -1 if `d` is nested.
  int RootRank(const mj::Default* d) const {
    for (int i = 0; i < static_cast<int>(roots_.size()); ++i)
      if (roots_[i] == d) return i;
    return -1;
  }

 private:
  void Add(const mj::Default& d, const mj::Default* par) {
    std::string name = d.dclass ? *d.dclass : std::string();
    by_name_.emplace(name, &d);  // first occurrence wins (duplicate is invalid)
    parent_[&d] = par;
    if (name.empty() || name == "main") root_ = &d;
    for (const auto& s : d.subclasses)
      if (s) Add(*s, &d);
  }
  std::unordered_map<std::string, const mj::Default*> by_name_;
  std::unordered_map<const mj::Default*, const mj::Default*> parent_;
  std::vector<const mj::Default*> roots_;
  const mj::Default* root_ = nullptr;
};

// The class an element authored directly, "" when none.
template <class T>
std::string OwnClass(const T& e) {
  if constexpr (requires { e.dclass; }) {
    using DT = std::decay_t<decltype(e.dclass)>;
    if constexpr (is_opt<DT>::value) {
      using Inner = typename is_opt<DT>::inner;
      if constexpr (is_ref<Inner>::value) return e.dclass ? e.dclass->name : "";
    }
  }
  return "";
}

// The class governing an element: its own class, else the nearest enclosing
// body-context childclass, else "" (the root/main class).
inline std::string ResolveClassName(const ParentMap& pm,
                                     const std::string& own,
                                     const void* elemPtr) {
  if (!own.empty()) return own;
  const ParentMap::Node* n = pm.Lookup(elemPtr);
  for (const void* p = n ? n->parent : nullptr; p;) {
    const ParentMap::Node* pn = pm.Lookup(p);
    if (!pn) break;
    if (!pn->childclass.empty()) return pn->childclass;
    p = pn->parent;
  }
  return "";
}

// Fill `target`'s unauthored fields from one <default> node's class partial for
// family T (its single class element for that family, if any).
template <class T>
void MergeNode(const mj::Default& d, T& target) {
  if (const auto* vec = DefaultVec<T>(d))
    if (!vec->empty() && vec->front()) MergeUnset(target, *vec->front());
}

// Merge the class chain (className -> ... -> root) into `target`, highest
// priority first (MergeUnset fills only-still-unset fields, so an earlier merge
// wins). The chain climbs nested classes by parent pointer; its terminal is a
// top-level block, at which point `main` is the merge of every top-level block
// up to and including that one. Later blocks overwrite earlier per field
// (document order), so within the root layer we apply the highest-ranked block
// first. A class-free element resolves to the last block, hence merges them all;
// a class defined in block k sees only blocks 0..k (the parse-time snapshot).
template <class T>
void MergeClassChain(const DefaultIndex& idx, const std::string& className,
                     T& target) {
  for (const mj::Default* d = idx.ByNameOrRoot(className); d;
       d = idx.ParentOf(d)) {
    if (idx.ParentOf(d) == nullptr) {  // terminal: a top-level block -> `main`
      int rank = idx.RootRank(d);
      if (rank < 0) {
        MergeNode(*d, target);
      } else {
        const auto& roots = idx.Roots();
        for (int i = rank; i >= 0; --i)
          if (roots[i]) MergeNode(*roots[i], target);
      }
      break;
    }
    MergeNode(*d, target);
  }
}

inline mj::Default* EnsureRoot(mj::Model& model) {
  for (auto& d : model.defaults) {
    if (!d) continue;
    std::string n = d->dclass ? *d->dclass : std::string();
    if (n.empty() || n == "main") return d.get();
  }
  auto d = std::make_unique<mj::Default>();
  d->dclass = "main";
  mj::Default* raw = d.get();
  model.defaults.push_back(std::move(d));
  return raw;
}

}  // namespace detail

// --- Effective (query) ---------------------------------------------------- //

// The lookup state Effective needs, built once from a Model: the parent map
// (childclass resolution) and the <default> class index. A caller issuing many
// Effective queries against an unchanged model (a drag frame, a panel render)
// builds one context for the batch instead of paying both walks per call.
// Holds pointers into the model: any tree mutation invalidates it, so build it
// per frame/batch and discard -- never cache across edits.
class EffectiveContext {
 public:
  explicit EffectiveContext(const mj::Model& m) : pm_(m), idx_(m) {}
  const ParentMap& parents() const { return pm_; }
  const detail::DefaultIndex& classes() const { return idx_; }

 private:
  ParentMap pm_;
  detail::DefaultIndex idx_;
};

// The effective value of an element with all class layers resolved: a fresh
// copy of `e` with every unauthored field filled from its class chain and
// (when `apply_idl_defaults`) the IDL defaults. Does not mutate the model. For
// families without a same-type class partial, returns `e` plus IDL defaults.
// `ctx` must have been built from the model that owns `e`, after its last
// mutation.
template <class T>
std::unique_ptr<T> Effective(const EffectiveContext& ctx, const T& e,
                             bool apply_idl_defaults = true) {
  std::unique_ptr<T> out = mj::Clone(e);
  if constexpr (HasDefaultFamily<T>::value) {
    std::string cls =
        detail::ResolveClassName(ctx.parents(), detail::OwnClass(e), &e);
    detail::MergeClassChain(ctx.classes(), cls, *out);
  }
  // IDL defaults are the lowest priority layer: fill only fields still unset
  // after element + class. ApplyDefault assigns unconditionally, so route it
  // through a fill-only merge rather than calling it on `out` directly.
  if (apply_idl_defaults) {
    T defs;
    mj::ApplyDefault(defs);
    detail::MergeUnset(*out, defs);
  }
  return out;
}

// Single-query convenience: builds the lookup context for this one call.
template <class T>
std::unique_ptr<T> Effective(const mj::Model& model, const T& e,
                             bool apply_idl_defaults = true) {
  return Effective(EffectiveContext(model), e, apply_idl_defaults);
}

// --- FlattenDefaults (mutating) ------------------------------------------- //

// Bake each element's class-layered values into the element and drop the whole
// <default> tree. Only the authored class layers are baked (not the IDL
// defaults), so unauthored-everywhere fields stay unset and still resolve to
// MuJoCo's compiler defaults at compile. After this call the model has no
// classes and no `class`/`childclass` references, and compiles identically.
inline void FlattenDefaults(mj::Model& model) {
  ParentMap pm(model);
  detail::DefaultIndex idx(model);

  detail::WalkModelLive(model, [&](auto& e) {
    using E = std::decay_t<decltype(e)>;
    if constexpr (HasDefaultFamily<E>::value) {
      std::string cls = detail::ResolveClassName(pm, detail::OwnClass(e), &e);
      detail::MergeClassChain(idx, cls, e);
      if constexpr (requires { e.dclass; }) {
        using DT = std::decay_t<decltype(e.dclass)>;
        if constexpr (detail::is_opt<DT>::value) e.dclass.reset();
      }
    }
  });

  detail::WalkModelLive(model, [&](auto& e) {
    using E = std::decay_t<decltype(e)>;
    if constexpr (std::is_same_v<E, mj::Body> ||
                  std::is_same_v<E, mj::Replicate>) {
      e.childclass.reset();
    } else if constexpr (std::is_same_v<E, mj::Frame>) {
      e.dclass.reset();
    }
  });

  model.defaults.clear();
}

// --- ExtractClass (mutating) ---------------------------------------------- //

namespace detail {

// For each field authored identically across every element, move that value
// into the class element and clear it on each source element. Identity fields
// are exempt: `name` names the element (referrers depend on it) and `dclass`
// is the class link itself (rewritten by ExtractClass to point at the new
// class) -- neither may migrate into the class partial, where MJCF forbids
// them.
template <class T>
struct ExtractVisitor {
  const std::vector<T*>* elems;
  template <class U>
  void field(int id, const char* fname, U& clsField) {
    if constexpr (is_opt<U>::value) {
      const std::string_view f(fname);
      if (f == "name" || f == "dclass") return;
      const U* first = nullptr;
      bool all_equal = !elems->empty();
      for (T* e : *elems) {
        const U* f = FieldAt<T, U>(*e, id);
        if (!f || !f->has_value()) {
          all_equal = false;
          break;
        }
        if (!first)
          first = f;
        else if (!(*f == *first)) {
          all_equal = false;
          break;
        }
      }
      if (all_equal && first) {
        clsField = *first;
        for (T* e : *elems) {
          if (U* f = FieldAt<T, U>(*e, id)) f->reset();
        }
      }
    }
  }
  template <class C>
  void child(int, const char*, C&) {}
  template <class C>
  void union_child(int, const char*, C&) {}
};

}  // namespace detail

// Factor the fields shared (authored and equal) across `elems` into a new class
// `name`, clear those fields on each element, and point each element at the new
// class. The class is added under the root `main` default (created if absent).
// All elements must be the same defaultable family type. Returns the new class,
// or nullptr when `elems` is empty or the family has no class partial.
template <class T>
mj::Default* ExtractClass(mj::Model& model, const std::vector<T*>& elems,
                          const std::string& name) {
  if (elems.empty()) return nullptr;
  if constexpr (!HasDefaultFamily<T>::value) {
    return nullptr;
  } else {
    auto cls = std::make_unique<mj::Default>();
    cls->dclass = name;
    std::vector<std::unique_ptr<T>>* fam = DefaultVecMut<T>(*cls);
    fam->push_back(std::make_unique<T>());
    T& clsElem = *fam->back();

    detail::ExtractVisitor<T> v{&elems};
    mj::Visit(clsElem, v);

    mj::Default* root = detail::EnsureRoot(model);
    mj::Default* raw = cls.get();
    root->subclasses.push_back(std::move(cls));

    for (T* e : elems) {
      if constexpr (requires { e->dclass; }) {
        using DT = std::decay_t<decltype(e->dclass)>;
        if constexpr (detail::is_opt<DT>::value) {
          using Inner = typename detail::is_opt<DT>::inner;
          if constexpr (detail::is_ref<Inner>::value)
            e->dclass = ps::Ref<mj::Default>(name);
        }
      }
    }
    return raw;
  }
}

}  // namespace ps::sdk

#endif  // PROTOSPEC_SDK_CLASSES_H
