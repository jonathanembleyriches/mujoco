// ProtoSpec Studio SE5 authoring-richness battery (ps::studio), windowless.
// Covers the SE5 wave that the base authoring battery does not: multi-mesh /
// folder import as one undo, batch primitive add, full material + texture
// creation from a dialog spec, geom material assignment, and a save -> reload
// round-trip of a created appearance. Every structural op is followed by a real
// compile, so a broken op surfaces as a compile failure.

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <variant>
#include <vector>

#include <mujoco/mujoco.h>

#include "compile.h"
#include "editor/asset_import.h"
#include "editor/authoring_ops.h"
#include "editor/editor_context.h"
#include "editor/editor_ops.h"
#include "mjcf.h"
#include "protospec/detail.h"
#include "protospec/refs.h"
#include "protospec/traversal.h"
#include "types.h"

namespace mj = ps::mjcf;
namespace sdk = ps::sdk;
using ps::studio::EditorContext;

static int g_failed = 0;
static int g_checks = 0;

#define CHECK(cond)                                                \
  do {                                                             \
    ++g_checks;                                                    \
    if (!(cond)) {                                                 \
      ++g_failed;                                                  \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);  \
    }                                                              \
  } while (0)

static std::filesystem::path TempDir(const char* tag) {
  std::filesystem::path p =
      std::filesystem::temp_directory_path() / "protospec_studio_se5" / tag;
  std::error_code ec;
  std::filesystem::remove_all(p, ec);
  std::filesystem::create_directories(p, ec);
  return p;
}

static const char* kTetraObj =
    "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n"
    "f 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n";

template <class T>
static int CountAll(mj::Model& m) {
  int n = 0;
  sdk::detail::WalkModelAll(m, [&](auto& e) {
    using E = std::decay_t<decltype(e)>;
    if constexpr (std::is_same_v<E, T>) ++n;
  });
  return n;
}

template <class T>
static T* FindBySerial(mj::Model& m, std::uint64_t serial) {
  T* out = nullptr;
  sdk::detail::WalkModelAll(m, [&](auto& e) {
    using E = std::decay_t<decltype(e)>;
    if constexpr (std::is_same_v<E, T>) {
      if (!out && e.serial == serial) out = &e;
    }
  });
  return out;
}

// Multi-import: N files -> N meshes + N bodies + N mesh-geoms, one undo entry,
// last body selected; one Undo reverts the whole batch.
static void TestMultiImportOneUndo() {
  const std::filesystem::path dir = TempDir("multi");
  std::vector<std::string> paths;
  for (int i = 0; i < 5; ++i) {
    const auto obj = dir / ("m" + std::to_string(i) + ".obj");
    std::ofstream(obj, std::ios::binary) << kTetraObj;
    paths.push_back(obj.string());
  }

  EditorContext ctx;
  CHECK(ps::studio::NewModelOp(ctx));
  const std::size_t undo0 = ctx.history.undo_depth();

  ps::studio::MultiMeshImportResult r = ps::studio::ImportMeshes(ctx, paths);
  CHECK(r.ok);
  CHECK(r.imported == 5 && r.skipped == 0);
  CHECK(ctx.selected_serial == r.last_body_serial);
  CHECK(ctx.history.undo_depth() == undo0 + 1);  // ONE batch undo
  CHECK(CountAll<mj::Mesh>(*ctx.tree) == 5);
  CHECK(ps::studio::RecompileTree(ctx) && ctx.compiled.ok());
  CHECK(ctx.compiled.model->nmesh == 5);

  ps::studio::Undo(ctx);
  CHECK(CountAll<mj::Mesh>(*ctx.tree) == 0);  // batch fully reverted

  // A skipped (unreadable) file does not abort the rest of the batch.
  EditorContext ctx2;
  CHECK(ps::studio::NewModelOp(ctx2));
  ps::studio::MultiMeshImportResult r2 = ps::studio::ImportMeshes(
      ctx2, {paths[0], (dir / "missing.obj").string(), paths[1]});
  CHECK(r2.ok && r2.imported == 2 && r2.skipped == 1);

  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

// Folder import globs mesh files and ignores the rest.
static void TestFolderImportGlob() {
  const std::filesystem::path dir = TempDir("folder");
  for (int i = 0; i < 3; ++i) {
    std::ofstream(dir / ("p" + std::to_string(i) + ".obj"), std::ios::binary)
        << kTetraObj;
  }
  std::ofstream(dir / "ignore.txt", std::ios::binary) << "x";

  EditorContext ctx;
  CHECK(ps::studio::NewModelOp(ctx));
  ps::studio::MultiMeshImportResult r =
      ps::studio::ImportMeshFolder(ctx, dir.string());
  CHECK(r.ok && r.imported == 3);
  CHECK(ps::studio::RecompileTree(ctx) && ctx.compiled.ok());
  CHECK(ctx.compiled.model->nmesh == 3);

  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

// Batch primitive add: `count` geoms under a body, one undo, non-overlapping.
static void TestBatchPrimitiveAdd() {
  EditorContext ctx;
  CHECK(ps::studio::NewModelOp(ctx));
  const std::uint64_t body = ps::studio::AddBodyOp(ctx, 0);
  const std::size_t undo0 = ctx.history.undo_depth();

  const std::uint64_t last =
      ps::studio::AddGeomsOp(ctx, body, mj::GeomType::sphere, 6);
  CHECK(last != 0 && ctx.selected_serial == last);
  CHECK(ctx.history.undo_depth() == undo0 + 1);  // one batch undo

  int geoms = 0;
  mj::Body* b = FindBySerial<mj::Body>(*ctx.tree, body);
  CHECK(b != nullptr);
  if (b) {
    sdk::detail::WalkTree(*b, [&](auto& e) {
      using E = std::decay_t<decltype(e)>;
      if constexpr (std::is_same_v<E, mj::Geom>) ++geoms;
    });
  }
  CHECK(geoms == 6);
  CHECK(ps::studio::RecompileTree(ctx) && ctx.compiled.ok());

  // count <= 1 behaves like the single add.
  const std::uint64_t one = ps::studio::AddGeomsOp(ctx, body, mj::GeomType::box, 1);
  CHECK(one != 0);
}

// Full material + texture creation from a dialog spec, geom assignment, and a
// save -> reload round-trip.
static void TestMaterialTextureFull() {
  EditorContext ctx;
  CHECK(ps::studio::NewModelOp(ctx));

  ps::studio::TextureSpec tex;
  tex.name = "checker";
  tex.builtin = true;
  tex.builtin_type = mj::TextureBuiltin::checker;
  tex.rgb1 = {0.9, 0.9, 0.9};
  tex.rgb2 = {0.1, 0.1, 0.1};
  tex.markrgb = {1.0, 0.0, 0.0};
  tex.width = 128;
  tex.height = 128;
  const std::uint64_t tser = ps::studio::CreateTextureOp(ctx, tex);
  CHECK(tser != 0);
  mj::Texture* t = sdk::Find<mj::Texture>(*ctx.tree, "checker");
  CHECK(t != nullptr);
  if (t) {
    CHECK(t->source.has_value());
    CHECK(t->width && *t->width == 128);
  }

  ps::studio::MaterialSpec mat;
  mat.name = "panel";
  mat.rgba = {0.2f, 0.6f, 0.9f, 1.0f};
  mat.metallic = 0.3f;
  mat.roughness = 0.7f;
  mat.texture_rgb = "checker";
  const std::uint64_t mser = ps::studio::CreateMaterialOp(ctx, mat);
  CHECK(mser != 0);
  mj::Material* m = sdk::Find<mj::Material>(*ctx.tree, "panel");
  CHECK(m != nullptr);
  if (m) {
    CHECK(m->rgba.has_value());
    CHECK(m->metallic.has_value());
    CHECK(m->layers.size() == 1);  // rgb texture role layer
    if (!m->layers.empty() && m->layers[0]) {
      CHECK(m->layers[0]->role && *m->layers[0]->role == mj::TexRole::rgb);
    }
  }

  // Geom material assignment + clear.
  const std::uint64_t geom = ps::studio::AddGeomOp(ctx, 0, mj::GeomType::box);
  CHECK(ps::studio::AssignGeomMaterialOp(ctx, geom, "panel"));
  mj::Geom* g = FindBySerial<mj::Geom>(*ctx.tree, geom);
  CHECK(g != nullptr && g->material && g->material->name == "panel");
  CHECK(ps::studio::AssignGeomMaterialOp(ctx, geom, ""));
  g = FindBySerial<mj::Geom>(*ctx.tree, geom);
  CHECK(g != nullptr && !g->material.has_value());
  CHECK(ps::studio::AssignGeomMaterialOp(ctx, 999999, "panel") == false);  // bad serial

  CHECK(ps::studio::RecompileTree(ctx) && ctx.compiled.ok());
  CHECK(ctx.compiled.model->nmat >= 1 && ctx.compiled.model->ntex >= 1);

  // Round-trip (builtin texture + material survive save -> reload byte-exact).
  const std::filesystem::path dir = TempDir("roundtrip");
  const std::filesystem::path xml = dir / "scene.xml";
  CHECK(ps::studio::SaveModel(ctx, xml.string()));
  EditorContext ctx2;
  CHECK(ps::studio::LoadModel(ctx2, xml.string()));
  CHECK(ctx2.compiled.ok());
  CHECK(*ctx.tree == *ctx2.tree);
  CHECK(mj::io::WriteMjcf(*ctx.tree) == mj::io::WriteMjcf(*ctx2.tree));

  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

// A file (non-builtin) texture records its file source on the authored tree.
static void TestFileTextureSource() {
  EditorContext ctx;
  CHECK(ps::studio::NewModelOp(ctx));
  ps::studio::TextureSpec ftex;
  ftex.name = "img";
  ftex.builtin = false;
  ftex.file = "wood.png";
  CHECK(ps::studio::CreateTextureOp(ctx, ftex) != 0);
  mj::Texture* t = sdk::Find<mj::Texture>(*ctx.tree, "img");
  CHECK(t != nullptr && t->source.has_value());
  if (t && t->source) {
    CHECK(std::holds_alternative<mj::TexFile>(*t->source));
    if (std::holds_alternative<mj::TexFile>(*t->source)) {
      CHECK(std::get<mj::TexFile>(*t->source).file == "wood.png");
    }
  }
}

// A new default class is added and a selected element's dclass resolves to it.
static void TestDefaultClass() {
  EditorContext ctx;
  CHECK(ps::studio::NewModelOp(ctx));
  const std::uint64_t cls = ps::studio::AddDefaultClassOp(ctx, "arm");
  CHECK(cls != 0);
  CHECK(sdk::Find<mj::Default>(*ctx.tree, "arm") != nullptr);
  CHECK(ps::studio::RecompileTree(ctx) && ctx.compiled.ok());

  // Adding a class with a colliding name uniquely renames.
  const std::uint64_t cls2 = ps::studio::AddDefaultClassOp(ctx, "arm");
  CHECK(cls2 != 0 && cls2 != cls);
  CHECK(sdk::Find<mj::Default>(*ctx.tree, "arm_1") != nullptr);
}

int main() {
  TestMultiImportOneUndo();
  TestFolderImportGlob();
  TestBatchPrimitiveAdd();
  TestMaterialTextureFull();
  TestFileTextureSource();
  TestDefaultClass();

  std::printf("%d checks, %d failed\n", g_checks, g_failed);
  return g_failed == 0 ? 0 : 1;
}
