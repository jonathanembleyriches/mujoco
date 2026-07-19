// ProtoSpec Studio undo/redo (ps::studio, ours). See undo.h.

#include "editor/undo.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "protospec/traversal.h"  // ps::sdk::WalkModel
#include "types.h"

namespace ps::studio {

namespace mj = ps::mjcf;

namespace {

// Collect every element's serial in document order. Only concrete elements (and
// the Model root) carry a serial; the union wrapper types the walk descends
// through do not and are transparently skipped by the `requires` guard.
void CollectSerials(const mj::Model& m, std::vector<std::uint64_t>& out) {
  ps::sdk::WalkModel(m, [&](const auto& e) {
    if constexpr (requires { e.serial; }) {
      out.push_back(e.serial);
    }
  });
}

void CollectSerialSlots(mj::Model& m, std::vector<std::uint64_t*>& out) {
  ps::sdk::WalkModel(m, [&](auto& e) {
    if constexpr (requires { e.serial; }) {
      out.push_back(&e.serial);
    }
  });
}

}  // namespace

std::unique_ptr<mj::Model> CloneWithSerials(const mj::Model& src) {
  std::unique_ptr<mj::Model> dst = mj::Clone(src);

  std::vector<std::uint64_t> src_serials;
  std::vector<std::uint64_t*> dst_slots;
  CollectSerials(src, src_serials);
  CollectSerialSlots(*dst, dst_slots);

  // Structural clone => identical walk order and count; pair one-to-one.
  const std::size_t n =
      src_serials.size() < dst_slots.size() ? src_serials.size() : dst_slots.size();
  for (std::size_t i = 0; i < n; ++i) {
    *dst_slots[i] = src_serials[i];
  }
  return dst;
}

void UndoStack::BeginEdit(const mj::Model& live) {
  pending_ = Entry{CloneWithSerials(live), std::string()};
}

void UndoStack::CommitEdit(std::string label) {
  if (!pending_) {
    return;
  }
  pending_->label = std::move(label);
  undo_.push_back(std::move(*pending_));
  pending_.reset();
  redo_.clear();
  while (undo_.size() > kMaxDepth) {
    undo_.pop_front();
  }
}

std::unique_ptr<mj::Model> UndoStack::TakePending() {
  if (!pending_) {
    return nullptr;
  }
  std::unique_ptr<mj::Model> out = std::move(pending_->model);
  pending_.reset();
  return out;
}

std::unique_ptr<mj::Model> UndoStack::Undo(std::unique_ptr<mj::Model> current,
                                           std::string* out_label) {
  if (undo_.empty()) {
    return current;
  }
  Entry e = std::move(undo_.back());
  undo_.pop_back();
  if (out_label) {
    *out_label = e.label;
  }
  redo_.push_back({std::move(current), e.label});
  return std::move(e.model);
}

std::unique_ptr<mj::Model> UndoStack::Redo(std::unique_ptr<mj::Model> current,
                                           std::string* out_label) {
  if (redo_.empty()) {
    return current;
  }
  Entry e = std::move(redo_.back());
  redo_.pop_back();
  if (out_label) {
    *out_label = e.label;
  }
  undo_.push_back({std::move(current), e.label});
  return std::move(e.model);
}

void UndoStack::Clear() {
  pending_.reset();
  undo_.clear();
  redo_.clear();
}

}  // namespace ps::studio
