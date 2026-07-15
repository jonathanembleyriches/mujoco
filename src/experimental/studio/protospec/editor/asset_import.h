// ProtoSpec Studio SE3 mesh import (ps::studio, ours): windowless.
//
// "Import Mesh..." brings an OBJ/STL/MSH file into the authored model and auto-
// builds a body+geom(type=mesh) that renders it ("auto build mesh geoms"). Two
// backing modes, chosen by whether the model already lives on disk:
//
//   on-disk model  -> copy the file into the model's meshdir (created if the
//                     compiler authored one), add Mesh{file=<basename>}. The
//                     geom compiles straight off the file like any saved model.
//   in-memory model (File > New, never saved) -> register the raw bytes as a
//                     CompileOptions.vfs_asset on the EditorContext so every
//                     recompile resolves the mesh with no file on disk, and add
//                     Mesh{file=<basename>}. On the next Save the bytes are
//                     externalized next to the .xml (see ExternalizeVfsAssets)
//                     and the vfs store is cleared -- the refs already name the
//                     basename, so nothing is rewritten.
//
// The editor never parses the mesh; OBJ/STL/MSH are whatever the compile paths
// accept. The mesh element's name is the file stem (MuJoCo's convention).

#ifndef PS_STUDIO_EDITOR_ASSET_IMPORT_H_
#define PS_STUDIO_EDITOR_ASSET_IMPORT_H_

#include <cstdint>
#include <string>
#include <vector>

#include "editor/editor_context.h"

namespace ps::studio {

struct MeshImportResult {
  bool ok = false;
  std::string error;
  std::uint64_t mesh_serial = 0;  // the new <mesh> asset
  std::uint64_t body_serial = 0;  // the auto-built body carrying the mesh geom
  bool vfs = false;               // true when registered in-memory (unsaved model)
};

// Import `file_path` and auto-build a body+geom(mesh) at `world_point` (nullptr
// == origin). One labelled undo entry; the new body is selected. Returns ok=false
// (nothing changed) when the file cannot be read.
MeshImportResult ImportMesh(EditorContext& ctx, const std::string& file_path,
                            const double world_point[3]);

struct MultiMeshImportResult {
  bool ok = false;                          // at least one mesh imported
  std::string error;                        // per-file skip reasons (if any)
  int imported = 0;                         // meshes successfully added
  int skipped = 0;                          // files that could not be read
  std::vector<std::uint64_t> body_serials;  // one body per imported mesh
  std::uint64_t last_body_serial = 0;       // the selected (last) body
  bool vfs = false;                         // any asset registered in-memory
};

// Import every file in `file_paths` as ONE undo entry ("Import N meshes"): each
// becomes a Mesh asset + a Body + a mesh Geom, the bodies laid out on a grid in
// the XY plane (fixed step) so they do not overlap. The last imported body is
// selected. Unreadable files are skipped and counted; ok is true when at least
// one file imported. Nothing is committed (no undo entry) when none imported.
MultiMeshImportResult ImportMeshes(EditorContext& ctx,
                                   const std::vector<std::string>& file_paths);

// Glob `folder` (non-recursive) for *.obj / *.stl / *.msh and ImportMeshes them.
// Returns ok=false with an error when the folder holds no importable mesh files.
MultiMeshImportResult ImportMeshFolder(EditorContext& ctx,
                                       const std::string& folder);

// Write every in-memory vfs asset to disk relative to `xml_path` (into the
// model's meshdir when the compiler authored one, else beside the .xml), then
// clear the vfs store and repoint base_dir at the save directory. Called by
// SaveModel so a New-model-with-imported-mesh becomes a fully on-disk model on
// first save. No-op when there are no pending in-memory assets.
void ExternalizeVfsAssets(EditorContext& ctx, const std::string& xml_path);

}  // namespace ps::studio

#endif  // PS_STUDIO_EDITOR_ASSET_IMPORT_H_
