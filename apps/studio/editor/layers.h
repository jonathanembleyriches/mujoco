// ProtoSpec Studio: layer composition (ps::studio, ours).
//
// Every enabled layer is cloned and concatenated, in list order, into the one
// Model that compiles. Composition is ADDITIVE: layers contribute elements, they
// do not override each other. Two layers naming the same thing is a conflict,
// reported here rather than left to surface as a bare "duplicate name" from the
// compiler with no clue which layers collided.
//
// Serials survive the clone (CloneWithSerials), so a composed element carries the
// serial of the layer element it came from. That is what lets a pick on the
// composed model resolve back to the authored element it belongs to.

#ifndef PS_STUDIO_EDITOR_LAYERS_H_
#define PS_STUDIO_EDITOR_LAYERS_H_

#include <memory>
#include <string>
#include <vector>

#include "editor/editor_context.h"
#include "types.h"

namespace ps::studio {

// A (type, name) claimed by two enabled layers at once.
struct LayerConflict {
  std::string name;      // "material 'mug'"
  std::string first;     // layer that claimed it first
  std::string second;    // layer that collided
};

struct ComposeResult {
  std::unique_ptr<ps::mjcf::Model> model;
  std::vector<LayerConflict> conflicts;
  int layers_used = 0;
};

// Merge every enabled layer into one model. Never fails: conflicts are reported
// and the merge proceeds, so the compiler's own error stays the final word on
// whether the result is a valid model.
ComposeResult ComposeLayers(EditorContext& ctx);

// Compose and adopt the result into ctx.composed, logging conflicts. Returns the
// model to compile. Falls back to ctx.tree when no layers exist.
ps::mjcf::Model* BuildComposed(EditorContext& ctx);

// The layer index owning `serial`, or -1. Lets a pick on the composed model
// follow the selection back to the layer that authored the element.
int LayerOwningSerial(EditorContext& ctx, std::uint64_t serial);

// --- Layer list ops -------------------------------------------------------- //

// Append an empty authored layer and make it the edit target.
void AddEmptyLayer(EditorContext& ctx, const std::string& name);

// Append a layer parsed from an MJCF file. Returns false with `error` set.
bool AddLayerFromFile(EditorContext& ctx, const std::string& path,
                      std::string* error);

void RemoveLayer(EditorContext& ctx, int index);
void MoveLayer(EditorContext& ctx, int index, int delta);

// Collapse the context onto a single layer holding `tree` (a fresh load / New).
void ResetLayers(EditorContext& ctx, const std::string& name,
                 const std::string& path);

// --- Export ---------------------------------------------------------------- //

// Write the stack as MJCF: one file per enabled layer beside `root_path`, and a
// root document that pulls them in with <include> in composition order. This is
// the form the reader already understands -- ExpandIncludes splices the files
// back at load -- so an exported stack loads as the composed model in Studio,
// `simulate`, or anything else that reads MJCF.
//
//   scene.xml            <mujoco model="scene"><include file="scene.background.xml"/>...
//   scene.background.xml <mujoco> ... the layer's own elements ...
//
// Disabled layers are not written: the export is what the stack composes to.
// Returns false with `error` set; nothing is written on a failure to open.
bool ExportLayeredMjcf(EditorContext& ctx, const std::string& root_path,
                       std::string* error);

}  // namespace ps::studio

#endif  // PS_STUDIO_EDITOR_LAYERS_H_
