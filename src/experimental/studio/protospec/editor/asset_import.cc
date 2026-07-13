// ProtoSpec Studio SE3 mesh import (ps::studio, ours). See asset_import.h.

#include "editor/asset_import.h"

#include <array>
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
namespace bridge = ps::mjcf::bridge;
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

}  // namespace

MeshImportResult ImportMesh(EditorContext& ctx, const std::string& file_path,
                            const double world_point[3]) {
  MeshImportResult r;
  if (!ctx.tree) {
    r.error = "no model";
    return r;
  }
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

  mj::Model& tree = *ctx.tree;
  ctx.BeginEdit();

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
        ctx.CancelEdit();
        r.error = "copy failed: " + ec.message();
        return r;
      }
    }
  } else {
    // Unsaved model: register the bytes in the compile VFS (basename-matched).
    ctx.vfs_assets.push_back(bridge::VfsAsset{basename, std::move(bytes)});
    r.vfs = true;
  }

  mj::Mesh& mesh = sdk::AddMesh(tree, meshname, basename);
  r.mesh_serial = mesh.serial;

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

  ctx.CommitEdit("Import mesh " + meshname);
  SelectBySerial(ctx, b.serial);
  r.ok = true;
  return r;
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
  for (const bridge::VfsAsset& a : ctx.vfs_assets) {
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
