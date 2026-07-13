// ProtoSpec Studio editor operations (ps::studio, ours): the pure, windowless
// core of the editor cluster. No SDL/ImGui/plugin-registry dependencies, so the
// ctest target links this directly and drives the ProtoSpec loop end to end.

#ifndef PS_STUDIO_EDITOR_EDITOR_OPS_H_
#define PS_STUDIO_EDITOR_EDITOR_OPS_H_

#include <cstdint>
#include <string>

#include "editor/editor_context.h"

namespace ps::studio {

// Loads a model through ProtoSpec (DR-S1 load path):
//   ParseMjcf(file) -> Validate(tiers 1-2, logged) -> Compile(Auto).
// On success, populates ctx.tree + ctx.compiled, sets ctx.model_ready, and logs
// the validation/compile summary. On failure, logs the errors, leaves any prior
// good compiled artifact untouched, and returns false.
bool LoadModel(EditorContext& ctx, const std::string& path);

// The element a pick resolved to, via Binding reverse lookup.
struct PickResolution {
  bool hit = false;
  std::string type;         // element type name ("Geom", "Body", ...)
  std::string name;         // authored name, or "<unnamed>"
  std::uint64_t serial = 0;
};

// Resolves a compiled geom/body id to a ProtoSpec element through ctx.compiled's
// Binding (geom preferred, else body), updates ctx.selected_*, and logs the hit
// to the Diagnostics panel. Pass -1 for a missing id. Proves the ProtoSpec loop
// (compiled id -> Binding -> tree element) end to end.
PickResolution ResolvePick(EditorContext& ctx, int geom_id, int body_id);

}  // namespace ps::studio

#endif  // PS_STUDIO_EDITOR_EDITOR_OPS_H_
