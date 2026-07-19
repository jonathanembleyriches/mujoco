// ProtoSpec Studio undo/redo (ps::studio, ours). See undo.h.

#include "editor/undo.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "protospec/traversal.h"  // ps::sdk::CloneModelWithSerials
#include "types.h"

namespace ps::studio {

namespace mj = ps::mjcf;

std::unique_ptr<mj::Model> CloneWithSerials(const mj::Model& src) {
  // The serial-preserving clone is a promoted SDK verb (ps::sdk::traversal): it
  // deep-clones then copies each source serial onto its clone counterpart, and
  // ASSERTS the source/clone walks are a bijection instead of silently
  // min()-truncating a mismatch (the old fail-open). See the mechanism note in
  // undo.h and the contract in protospec/traversal.h.
  return ps::sdk::CloneModelWithSerials(src);
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
