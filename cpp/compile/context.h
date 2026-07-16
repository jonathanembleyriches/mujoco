// CompileContext: the compile-scoped side tables the native compiler builds up
// stage by stage (impl-plan Section 2). This is the physical form of CDR-14
// purity: NativeCompile never writes into the ProtoSpec tree; every resolved,
// intermediate, and id-assignment quantity lives here, keyed by element
// identity, and dies with the compile.
//
// Identity key decision (impl-plan Section 2): during a compile the tree is
// const and single-instance, so element POINTERS are unique, stable, and free
// to obtain -- side tables are pointer-keyed. Binding entries additionally
// record the creation serial as the loggable/ABA-safe identity.
//
// Side-table shape: per-family ORDERED pointer lists (document order, which is
// id order until the NC4 transforms filter them) plus a pointer->slot map built
// during the Collect walk (stage S1). Every later per-element quantity becomes a
// vector parallel to the slot list -- linear, cache-friendly for large models,
// and id assignment falls out as "slot index within the filtered list" (CDR-4).
//
// This header is MuJoCo-free: it is pure ProtoSpec-side bookkeeping. The mjModel
// under construction is threaded separately from stage S11 on (not here yet).
#ifndef PROTOSPEC_COMPILE_CONTEXT_H
#define PROTOSPEC_COMPILE_CONTEXT_H

#include <cstddef>
#include <unordered_map>
#include <vector>

#include "report.h"   // ps::mjcf::Diagnostic sink (MuJoCo-free)
#include "types.h"     // generated ProtoSpec element structs

namespace ps::mjcf::compile {

// An ordered, pointer-keyed collection of one element family. `items` is
// document order (== id order pre-NC4); `slot` maps element pointer -> its index
// in `items` for O(1) ref resolution and Binding fill. Add() is idempotent-safe
// only in the sense that a family element is collected exactly once per walk.
template <class T>
class SlotTable {
 public:
  // Append an element, returning its slot index. The pointer becomes the key.
  int Add(const T* p) {
    const int idx = static_cast<int>(items_.size());
    items_.push_back(p);
    slot_.emplace(static_cast<const void*>(p), idx);
    return idx;
  }

  // Slot index for a previously-added element, or -1 if never collected.
  int SlotOf(const void* p) const {
    auto it = slot_.find(p);
    return it == slot_.end() ? -1 : it->second;
  }

  const std::vector<const T*>& items() const { return items_; }
  std::size_t size() const { return items_.size(); }
  const T* at(int slot) const { return items_[static_cast<std::size_t>(slot)]; }

 private:
  std::vector<const T*> items_;
  std::unordered_map<const void*, int> slot_;
};

// The stack-owned per-compile state. No compiler globals (thread-safety
// corollary, impl-plan Section 1): everything a stage needs beyond the const
// Model lives on an instance of this. Grown milestone by milestone; today it
// carries the family slot tables the Collect walk (S1) fills and the diagnostics
// sink. NC1+ adds the effective/resolved/tree/sizes/names groups (Section 2).
struct CompileContext {
  const Model* model = nullptr;

  // Per-family ordered pointer lists + slot maps (stage S1 output). The set of
  // families here is the NC1 rigid-body surface; NC2+ append their families.
  SlotTable<Body>    bodies;
  SlotTable<Joint>   joints;
  SlotTable<Geom>    geoms;
  SlotTable<Site>    sites;
  SlotTable<Camera>  cameras;
  SlotTable<Light>   lights;
  SlotTable<Mesh>    meshes;

  // Diagnostics drained into the CompileReport at the end of NativeCompile.
  std::vector<ps::mjcf::Diagnostic> diagnostics;
};

}  // namespace ps::mjcf::compile

#endif  // PROTOSPEC_COMPILE_CONTEXT_H
