// ProtoSpec Studio SE3 mesh import (ps::studio, ours). See asset_import.h.

#include "editor/asset_import.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "compile.h"
#include "editor/authoring_ops.h"  // UniqueName
#include "editor/editor_ops.h"     // SelectBySerial
#include "protospec/builders.h"
#include "protospec/refs.h"
#include "types.h"

namespace ps::studio {

namespace mj = ps::mjcf;
namespace sdk = ps::sdk;
namespace fs = std::filesystem;

namespace {

// The model's authored <compiler meshdir>, or empty (files resolve beside the
// .xml). The last authored value wins, mirroring the compiler.
std::string MeshDir(const mj::Model& tree) {
  std::string dir;
  for (const auto& c : tree.compilers) {
    if (c && c->meshdir) dir = *c->meshdir;
  }
  return dir;
}

// The outcome of adding a single mesh entry to the (already snapshotted) tree.
struct OneMeshResult {
  bool ok = false;
  std::string error;
  std::uint64_t mesh_serial = 0;
  std::uint64_t body_serial = 0;
  bool vfs = false;
};

// Add one Mesh asset + a Body(at `world_point`) + a mesh Geom to `tree`, backing
// the mesh bytes on disk (copy into meshdir) or in the compile VFS for an unsaved
// model. Does NOT wrap an undo edit -- the caller owns BeginEdit/CommitEdit so a
// batch import is a single undo entry. Returns ok=false (nothing added) when the
// file cannot be read or copied.
OneMeshResult AddOneMesh(EditorContext& ctx, mj::Model& tree,
                         const std::string& file_path,
                         const double world_point[3]) {
  OneMeshResult r;
  const fs::path src(file_path);
  std::error_code ec;
  if (!fs::exists(src, ec)) {
    r.error = "file not found: " + file_path;
    return r;
  }
  std::ifstream in(file_path, std::ios::binary);
  if (!in) {
    r.error = "cannot read: " + file_path;
    return r;
  }
  std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                  std::istreambuf_iterator<char>());
  in.close();

  const std::string basename = src.filename().string();
  const std::string stem = src.stem().string();
  const bool on_disk = !ctx.source_path.empty();

  const std::string meshname =
      UniqueName(tree, {mj::ElementType::Mesh}, stem.empty() ? "mesh" : stem);

  if (on_disk) {
    // Copy the source file into the model's meshdir (created if authored).
    fs::path destdir(ctx.base_dir.empty() ? std::string(".") : ctx.base_dir);
    const std::string md = MeshDir(tree);
    if (!md.empty()) destdir /= md;
    fs::create_directories(destdir, ec);
    const fs::path dest = destdir / basename;
    const fs::path srcabs = fs::absolute(src, ec);
    const fs::path destabs = fs::absolute(dest, ec);
    if (srcabs != destabs) {
      fs::copy_file(src, dest, fs::copy_options::overwrite_existing, ec);
      if (ec) {
        r.error = "copy failed: " + ec.message();
        return r;
      }
    }
  } else {
    // Unsaved model: register the bytes in the compile VFS (basename-matched).
    ctx.vfs_assets.push_back(mj::VfsAsset{basename, std::move(bytes)});
    r.vfs = true;
  }

  r.mesh_serial = sdk::AddMesh(tree, meshname, basename).serial;

  // Auto-build the body+geom(type=mesh) that renders it.
  mj::Body& w = sdk::World(tree);
  const std::string bn = UniqueName(tree, {mj::ElementType::Body}, "body");
  mj::Body& b = sdk::AddBody(w, bn);
  if (world_point) {
    b.pos = std::array<double, 3>{world_point[0], world_point[1], world_point[2]};
  }
  const std::string gn = UniqueName(tree, {mj::ElementType::Geom}, "geom");
  mj::Geom& g = sdk::AddGeom(b, mj::GeomType::mesh, gn);
  g.mesh = ps::Ref<mj::Mesh>(meshname);
  r.body_serial = b.serial;
  r.ok = true;
  return r;
}

// A body-context, non-recursive glob of a folder for mesh files.
bool IsMeshExtension(const fs::path& p) {
  std::string ext = p.extension().string();
  for (char& ch : ext) ch = static_cast<char>(std::tolower((unsigned char)ch));
  return ext == ".obj" || ext == ".stl" || ext == ".msh";
}

}  // namespace

MeshImportResult ImportMesh(EditorContext& ctx, const std::string& file_path,
                            const double world_point[3]) {
  MeshImportResult r;
  if (!ctx.tree) {
    r.error = "no model";
    return r;
  }
  mj::Model& tree = *ctx.tree;
  ctx.BeginEdit();
  const OneMeshResult one = AddOneMesh(ctx, tree, file_path, world_point);
  if (!one.ok) {
    ctx.CancelEdit();
    r.error = one.error;
    return r;
  }
  ctx.CommitEdit("Import mesh");
  SelectBySerial(ctx, one.body_serial);
  r.mesh_serial = one.mesh_serial;
  r.body_serial = one.body_serial;
  r.vfs = one.vfs;
  r.ok = true;
  return r;
}

MultiMeshImportResult ImportMeshes(EditorContext& ctx,
                                   const std::vector<std::string>& file_paths) {
  MultiMeshImportResult r;
  if (!ctx.tree) {
    r.error = "no model";
    return r;
  }
  if (file_paths.empty()) {
    r.error = "no files";
    return r;
  }
  mj::Model& tree = *ctx.tree;

  // Row-major grid in the XY plane so the bodies do not stack at the origin.
  constexpr double kStep = 0.5;  // metres between adjacent bodies
  const int cols = std::max(
      1, static_cast<int>(std::ceil(std::sqrt(static_cast<double>(
             file_paths.size())))));

  ctx.BeginEdit();
  int idx = 0;
  std::string skipped_msg;
  for (const std::string& path : file_paths) {
    const int col = idx % cols;
    const int row = idx / cols;
    const double pos[3] = {col * kStep, row * kStep, 0.0};
    const OneMeshResult one = AddOneMesh(ctx, tree, path, pos);
    if (!one.ok) {
      ++r.skipped;
      if (!skipped_msg.empty()) skipped_msg += "; ";
      skipped_msg += one.error;
      continue;  // grid slot is consumed only by placed bodies
    }
    ++idx;
    ++r.imported;
    r.vfs = r.vfs || one.vfs;
    r.body_serials.push_back(one.body_serial);
    r.last_body_serial = one.body_serial;
  }

  if (r.imported == 0) {
    ctx.CancelEdit();
    r.error = skipped_msg.empty() ? "nothing imported" : skipped_msg;
    return r;
  }
  ctx.CommitEdit("Import " + std::to_string(r.imported) + " mesh" +
                 (r.imported == 1 ? "" : "es"));
  SelectBySerial(ctx, r.last_body_serial);
  r.error = skipped_msg;
  r.ok = true;
  return r;
}

MultiMeshImportResult ImportMeshFolder(EditorContext& ctx,
                                       const std::string& folder) {
  MultiMeshImportResult r;
  if (!ctx.tree) {
    r.error = "no model";
    return r;
  }
  std::error_code ec;
  const fs::path dir(folder);
  if (!fs::is_directory(dir, ec)) {
    r.error = "not a folder: " + folder;
    return r;
  }
  std::vector<std::string> files;
  for (const fs::directory_entry& de : fs::directory_iterator(dir, ec)) {
    if (de.is_regular_file(ec) && IsMeshExtension(de.path())) {
      files.push_back(de.path().string());
    }
  }
  std::sort(files.begin(), files.end());
  if (files.empty()) {
    r.error = "no .obj / .stl / .msh files in " + folder;
    return r;
  }
  return ImportMeshes(ctx, files);
}

void ExternalizeVfsAssets(EditorContext& ctx, const std::string& xml_path) {
  if (ctx.vfs_assets.empty()) return;
  const fs::path xml(xml_path);
  const fs::path dir = xml.parent_path();
  fs::path destdir = dir.empty() ? fs::path(".") : dir;
  if (ctx.tree) {
    const std::string md = MeshDir(*ctx.tree);
    if (!md.empty()) destdir /= md;
  }
  std::error_code ec;
  fs::create_directories(destdir, ec);
  for (const mj::VfsAsset& a : ctx.vfs_assets) {
    const fs::path dest = destdir / a.name;
    std::ofstream out(dest, std::ios::binary);
    if (out) {
      out.write(reinterpret_cast<const char*>(a.bytes.data()),
                static_cast<std::streamsize>(a.bytes.size()));
    }
  }
  ctx.vfs_assets.clear();
  // The model now lives on disk; recompiles resolve the meshes from files.
  ctx.base_dir = dir.empty() ? std::string() : dir.string();
}

}  // namespace ps::studio
