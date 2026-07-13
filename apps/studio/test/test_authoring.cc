// ProtoSpec Studio SE3 authoring tests (ps::studio), windowless. Drives the
// add / duplicate / reparent / mesh-import / new-model ops end to end: every
// structural op is followed by a real compile, so a broken op surfaces as a
// compile failure, and the full SE3 exit story (build an empty model into a
// small scene, simulate, save, reload, deep-compare) runs as one scripted test.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>

#include "binding.h"
#include "compile.h"
#include "editor/asset_import.h"
#include "editor/authoring_ops.h"
#include "editor/editor_context.h"
#include "editor/editor_ops.h"
#include "mjcf.h"
#include "protospec/detail.h"
#include "protospec/traversal.h"
#include "types.h"

namespace mj = ps::mjcf;
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

// --- Helpers -------------------------------------------------------------- //

static std::unique_ptr<mj::Model> Parse(const char* xml) {
  mj::io::ParseResult r = mj::io::ParseMjcfString(xml, "authoring_test");
  if (!r.ok()) {
    std::printf("FATAL parse:\n");
    for (const auto& e : r.errors) std::printf("  %s\n", e.Render().c_str());
  }
  return std::move(r.model);
}

// Install a tree and compile it so ctx.compiled is populated (mirrors a load).
static bool Adopt(EditorContext& ctx, std::unique_ptr<mj::Model> m) {
  ctx.tree = std::move(m);
  return ps::studio::RecompileTree(ctx);
}

template <class T>
static int CountInSubtree(mj::Body& body) {
  int n = 0;
  ps::sdk::detail::WalkTree(body, [&](auto& e) {
    using E = std::decay_t<decltype(e)>;
    if constexpr (std::is_same_v<E, T>) ++n;
  });
  return n;
}

static int GeomBindingId(EditorContext& ctx, std::uint64_t serial) {
  for (const auto& e : ctx.compiled.binding.entries()) {
    if (e.serial == serial && e.etype == mj::ElementType::Geom && e.id >= 0)
      return e.id;
  }
  return -1;
}

static bool GeomWorldPos(EditorContext& ctx, std::uint64_t serial,
                         double out[3]) {
  const int id = GeomBindingId(ctx, serial);
  if (id < 0 || !ctx.compiled.model) return false;
  const mjModel* m = ctx.compiled.model.get();
  mjData* d = mj_makeData(m);
  if (!d) return false;
  mj_resetData(m, d);
  mj_forward(m, d);
  out[0] = d->geom_xpos[3 * id];
  out[1] = d->geom_xpos[3 * id + 1];
  out[2] = d->geom_xpos[3 * id + 2];
  mj_deleteData(d);
  return true;
}

static std::filesystem::path TempDir() {
  std::filesystem::path p = std::filesystem::temp_directory_path() /
                            "protospec_studio_se3";
  std::error_code ec;
  std::filesystem::create_directories(p, ec);
  return p;
}

// A minimal tetrahedron OBJ (valid convex mesh for MuJoCo).
static const char* kTetraObj =
    "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n"
    "f 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n";

// --- Tests ---------------------------------------------------------------- //

// Each body-tree primitive add yields a compilable model with the expected
// element counts and a live binding id for the added geom.
static void TestAddPrimitivesCompile() {
  EditorContext ctx;
  CHECK(ps::studio::NewModelOp(ctx));  // ground + light, compiled
  CHECK(ctx.compiled.ok());

  const std::uint64_t body = ps::studio::AddBodyOp(ctx, 0);
  CHECK(body != 0);
  CHECK(ps::studio::RecompileTree(ctx));
  CHECK(ctx.compiled.ok());
  CHECK(ctx.selected_serial == body);

  const std::uint64_t geom =
      ps::studio::AddGeomOp(ctx, body, mj::GeomType::box);
  CHECK(geom != 0);
  CHECK(ps::studio::RecompileTree(ctx));
  CHECK(ctx.compiled.ok());
  CHECK(GeomBindingId(ctx, geom) >= 0);  // the added geom is bound

  // A joint makes the body dynamic; still compiles (mass comes from the geom).
  const std::uint64_t joint =
      ps::studio::AddJointOp(ctx, body, mj::JointType::hinge);
  CHECK(joint != 0);
  CHECK(ps::studio::RecompileTree(ctx));
  CHECK(ctx.compiled.ok());

  // Site / camera / light / frame under the body: each compiles.
  CHECK(ps::studio::AddSiteOp(ctx, body) != 0);
  CHECK(ps::studio::RecompileTree(ctx) && ctx.compiled.ok());
  CHECK(ps::studio::AddCameraOp(ctx, body) != 0);
  CHECK(ps::studio::RecompileTree(ctx) && ctx.compiled.ok());
  CHECK(ps::studio::AddLightOp(ctx, body) != 0);
  CHECK(ps::studio::RecompileTree(ctx) && ctx.compiled.ok());
  const std::uint64_t frame = ps::studio::AddFrameOp(ctx, body);
  CHECK(frame != 0);
  CHECK(ps::studio::RecompileTree(ctx) && ctx.compiled.ok());

  // A geom nested under the frame compiles (frame parent path works).
  CHECK(ps::studio::AddGeomOp(ctx, frame, mj::GeomType::sphere) != 0);
  CHECK(ps::studio::RecompileTree(ctx) && ctx.compiled.ok());

  // Actuator wired to the joint + a jointpos sensor: both compile.
  CHECK(ps::studio::AddActuatorOp(ctx, ps::studio::ActuatorSpelling::Motor,
                                  joint) != 0);
  CHECK(ps::studio::RecompileTree(ctx) && ctx.compiled.ok());
  CHECK(ctx.compiled.model->nu >= 1);
  CHECK(ps::studio::AddSensorOp(ctx, ps::studio::SensorSpelling::Jointpos,
                                joint) != 0);
  CHECK(ps::studio::RecompileTree(ctx) && ctx.compiled.ok());
}

// The viewport drop-add makes a world-parented body+geom at the click point.
static void TestDropAdd() {
  EditorContext ctx;
  CHECK(ps::studio::NewModelOp(ctx));
  const double pt[3] = {0.5, -0.25, 1.5};
  const std::uint64_t body =
      ps::studio::AddDropBodyGeomOp(ctx, mj::GeomType::sphere, pt);
  CHECK(body != 0);
  CHECK(ps::studio::RecompileTree(ctx) && ctx.compiled.ok());
  // The dropped body sits at the click point.
  mj::Body* b = nullptr;
  ps::sdk::detail::WalkModelAll(*ctx.tree, [&](auto& e) {
    using E = std::decay_t<decltype(e)>;
    if constexpr (std::is_same_v<E, mj::Body>) {
      if (e.serial == body) b = &e;
    }
  });
  CHECK(b != nullptr && b->pos.has_value());
  if (b && b->pos) {
    CHECK(std::fabs((*b->pos)[0] - 0.5) < 1e-9);
    CHECK(std::fabs((*b->pos)[2] - 1.5) < 1e-9);
  }
}

// Repeated adds of the same kind never collide on names.
static void TestNameUniquing() {
  EditorContext ctx;
  CHECK(ps::studio::NewModelOp(ctx));
  const std::uint64_t body = ps::studio::AddBodyOp(ctx, 0);
  ps::studio::AddGeomOp(ctx, body, mj::GeomType::box);
  ps::studio::AddGeomOp(ctx, body, mj::GeomType::box);
  ps::studio::AddGeomOp(ctx, body, mj::GeomType::box);
  CHECK(ps::studio::RecompileTree(ctx) && ctx.compiled.ok());
  CHECK(ps::sdk::Find<mj::Geom>(*ctx.tree, "geom") != nullptr);
  CHECK(ps::sdk::Find<mj::Geom>(*ctx.tree, "geom_1") != nullptr);
  CHECK(ps::sdk::Find<mj::Geom>(*ctx.tree, "geom_2") != nullptr);

  // UniqueName is stable when the base is free and suffixes otherwise.
  CHECK(ps::studio::UniqueName(*ctx.tree, {mj::ElementType::Geom}, "fresh") ==
        "fresh");
  CHECK(ps::studio::UniqueName(*ctx.tree, {mj::ElementType::Geom}, "geom") ==
        "geom_3");
}

static const char* kDupModel = R"MJCF(
<mujoco model="dup">
  <worldbody>
    <body name="torso" pos="0 0 1">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom name="torso_geom" type="box" size="0.2 0.2 0.2"/>
      <camera name="cam" mode="targetbody" target="arm"/>
      <body name="arm" pos="0.5 0 0">
        <geom name="arm_geom" type="capsule" size="0.05 0.2"/>
        <site name="tip"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="act1" joint="hinge"/>
  </actuator>
</mujoco>
)MJCF";

// Duplicate: fresh serials, unique names, internal ref remap, external ref
// preservation, subtree shape preserved, and the result compiles.
static void TestDuplicate() {
  EditorContext ctx;
  CHECK(Adopt(ctx, Parse(kDupModel)));
  const mj::Body* torso = ps::sdk::Find<mj::Body>(*ctx.tree, "torso");
  CHECK(torso != nullptr);
  const std::uint64_t torso_serial = torso ? torso->serial : 0;
  const int geoms_before =
      torso ? CountInSubtree<mj::Geom>(const_cast<mj::Body&>(*torso)) : -1;

  const std::uint64_t copy = ps::studio::DuplicateOp(ctx, torso_serial);
  CHECK(copy != 0);
  CHECK(copy != torso_serial);        // fresh serial
  CHECK(ctx.selected_serial == copy);

  // The clone and all its named elements exist under uniquified names.
  mj::Body* torso1 = ps::sdk::Find<mj::Body>(*ctx.tree, "torso_1");
  CHECK(torso1 != nullptr);
  CHECK(ps::sdk::Find<mj::Joint>(*ctx.tree, "hinge_1") != nullptr);
  CHECK(ps::sdk::Find<mj::Geom>(*ctx.tree, "torso_geom_1") != nullptr);
  CHECK(ps::sdk::Find<mj::Body>(*ctx.tree, "arm_1") != nullptr);
  CHECK(ps::sdk::Find<mj::Geom>(*ctx.tree, "arm_geom_1") != nullptr);
  CHECK(ps::sdk::Find<mj::Site>(*ctx.tree, "tip_1") != nullptr);

  // Fresh serials throughout: the clone's hinge differs from the original's.
  const mj::Joint* h0 = ps::sdk::Find<mj::Joint>(*ctx.tree, "hinge");
  const mj::Joint* h1 = ps::sdk::Find<mj::Joint>(*ctx.tree, "hinge_1");
  CHECK(h0 && h1 && h0->serial != h1->serial);

  // Internal ref (camera->arm) is remapped to the clone's body.
  const mj::Camera* cam1 = ps::sdk::Find<mj::Camera>(*ctx.tree, "cam_1");
  CHECK(cam1 != nullptr);
  CHECK(cam1 && cam1->target && cam1->target->name == "arm_1");
  // Original camera still points at the original body.
  const mj::Camera* cam0 = ps::sdk::Find<mj::Camera>(*ctx.tree, "cam");
  CHECK(cam0 && cam0->target && cam0->target->name == "arm");

  // External ref (actuator->hinge) is preserved: it still names the ORIGINAL.
  const mj::Motor* act = ps::sdk::Find<mj::Motor>(*ctx.tree, "act1");
  CHECK(act && act->joint && act->joint->name == "hinge");

  // Subtree shape preserved modulo names/serials.
  const int geoms_after = torso1 ? CountInSubtree<mj::Geom>(*torso1) : -2;
  CHECK(geoms_after == geoms_before);

  CHECK(ps::studio::RecompileTree(ctx) && ctx.compiled.ok());
}

static const char* kReparentModel = R"MJCF(
<mujoco model="reparent">
  <worldbody>
    <body name="A" pos="1 0 0" euler="0 0 30">
      <geom name="ga" type="sphere" size="0.1"/>
      <geom name="target" type="box" size="0.1 0.1 0.1" pos="0.3 0.1 0"/>
    </body>
    <body name="B" pos="-1 0.5 0.2" euler="0 45 0">
      <geom name="gb" type="sphere" size="0.1"/>
    </body>
  </worldbody>
</mujoco>
)MJCF";

// Reparent with keep-world-pose leaves the element's world pose invariant, both
// for a plain body-to-body move and a frame-wrapped one.
static void TestReparentKeepWorldPose() {
  EditorContext ctx;
  CHECK(Adopt(ctx, Parse(kReparentModel)));
  const mj::Geom* target = ps::sdk::Find<mj::Geom>(*ctx.tree, "target");
  const mj::Body* b = ps::sdk::Find<mj::Body>(*ctx.tree, "B");
  CHECK(target && b);
  const std::uint64_t gserial = target ? target->serial : 0;
  const std::uint64_t bserial = b ? b->serial : 0;

  double before[3];
  CHECK(GeomWorldPos(ctx, gserial, before));

  // Move the geom from A into B, keeping its world pose.
  ps::studio::ReparentResult rr =
      ps::studio::ReparentOp(ctx, gserial, bserial, /*keep_world_pose=*/true);
  CHECK(rr.ok);
  CHECK(ps::studio::RecompileTree(ctx) && ctx.compiled.ok());

  double after[3];
  CHECK(GeomWorldPos(ctx, gserial, after));
  for (int k = 0; k < 3; ++k) CHECK(std::fabs(before[k] - after[k]) < 1e-6);

  // The geom now lives under B (its authored pos was rewritten).
  bool under_b = false;
  if (mj::Body* bb = ps::sdk::Find<mj::Body>(*ctx.tree, "B")) {
    ps::sdk::ForEachGeom(*bb, /*recursive=*/true, [&](mj::Geom& g) {
      if (g.serial == gserial) under_b = true;
    });
  }
  CHECK(under_b);

  // Cycle rejection: dropping B into its own subtree is refused, model unchanged.
  ps::studio::ReparentResult cyc =
      ps::studio::ReparentOp(ctx, bserial, gserial, true);
  CHECK(!cyc.ok);
}

// Frame-wrapped keep-world-pose: element under a <frame> stays put across a move.
static const char* kFrameModel = R"MJCF(
<mujoco model="frames">
  <worldbody>
    <body name="A" pos="0.5 0 0.5">
      <frame pos="0.2 0.1 0" euler="0 0 90">
        <geom name="fg" type="box" size="0.05 0.05 0.05" pos="0.1 0 0"/>
      </frame>
    </body>
    <body name="dest" pos="-0.5 0.3 0.1" euler="10 20 30"/>
  </worldbody>
</mujoco>
)MJCF";

static void TestReparentFrameWrapped() {
  EditorContext ctx;
  CHECK(Adopt(ctx, Parse(kFrameModel)));
  const mj::Geom* fg = ps::sdk::Find<mj::Geom>(*ctx.tree, "fg");
  const mj::Body* dest = ps::sdk::Find<mj::Body>(*ctx.tree, "dest");
  CHECK(fg && dest);
  const std::uint64_t gs = fg ? fg->serial : 0;
  double before[3];
  CHECK(GeomWorldPos(ctx, gs, before));

  ps::studio::ReparentResult rr = ps::studio::ReparentOp(
      ctx, gs, dest ? dest->serial : 0, /*keep_world_pose=*/true);
  CHECK(rr.ok);
  CHECK(ps::studio::RecompileTree(ctx) && ctx.compiled.ok());
  double after[3];
  CHECK(GeomWorldPos(ctx, gs, after));
  for (int k = 0; k < 3; ++k) CHECK(std::fabs(before[k] - after[k]) < 1e-6);
}

// Mesh import into an unsaved model registers bytes via the compile VFS and the
// auto-built mesh geom compiles; Save externalizes the bytes to disk.
static void TestMeshImportVfsAndExternalize() {
  const std::filesystem::path dir = TempDir();
  const std::filesystem::path obj = dir / "tetra.obj";
  {
    std::ofstream out(obj, std::ios::binary);
    out << kTetraObj;
  }

  EditorContext ctx;
  CHECK(ps::studio::NewModelOp(ctx));  // unsaved -> vfs path
  CHECK(ctx.source_path.empty());

  ps::studio::MeshImportResult mr =
      ps::studio::ImportMesh(ctx, obj.string(), nullptr);
  CHECK(mr.ok);
  CHECK(mr.vfs);                         // registered in-memory
  CHECK(!ctx.vfs_assets.empty());
  CHECK(ps::sdk::Find<mj::Mesh>(*ctx.tree, "tetra") != nullptr);

  CHECK(ps::studio::RecompileTree(ctx));
  CHECK(ctx.compiled.ok());
  CHECK(ctx.compiled.model->nmesh >= 1);

  // Save + externalize: the mesh bytes land next to the .xml, the vfs store is
  // cleared, and a fresh load compiles straight from disk.
  const std::filesystem::path xml = dir / "scene.xml";
  CHECK(ps::studio::SaveModel(ctx, xml.string()));
  ps::studio::ExternalizeVfsAssets(ctx, xml.string());
  CHECK(ctx.vfs_assets.empty());
  CHECK(std::filesystem::exists(dir / "tetra.obj"));

  EditorContext ctx2;
  CHECK(ps::studio::LoadModel(ctx2, xml.string()));
  CHECK(ctx2.compiled.ok());
  CHECK(ctx2.compiled.model->nmesh >= 1);

  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

// The SE3 exit story as one scripted run: empty model -> ground/body/geom/joint/
// actuator -> compile -> step -> save -> reload -> deep compare.
static void TestExitStory() {
  EditorContext ctx;
  CHECK(ps::studio::NewModelOp(ctx));

  const std::uint64_t body = ps::studio::AddBodyOp(ctx, 0);
  ps::studio::AddGeomOp(ctx, body, mj::GeomType::box);
  const std::uint64_t joint =
      ps::studio::AddJointOp(ctx, body, mj::JointType::hinge);
  ps::studio::AddActuatorOp(ctx, ps::studio::ActuatorSpelling::Motor, joint);

  CHECK(ps::studio::RecompileTree(ctx));
  CHECK(ctx.compiled.ok());
  CHECK(ctx.compiled.model->nbody >= 2);
  CHECK(ctx.compiled.model->njnt >= 1);
  CHECK(ctx.compiled.model->nu >= 1);

  // Simulate a few steps (the scene actually runs).
  const mjModel* m = ctx.compiled.model.get();
  mjData* d = mj_makeData(m);
  CHECK(d != nullptr);
  if (d) {
    mj_resetData(m, d);
    const double t0 = d->time;
    for (int i = 0; i < 20; ++i) mj_step(m, d);
    CHECK(d->time > t0);
    mj_deleteData(d);
  }

  // Save, reload, deep-compare authored form (form-preserving round trip).
  const std::filesystem::path dir = TempDir();
  const std::filesystem::path xml = dir / "exit_story.xml";
  CHECK(ps::studio::SaveModel(ctx, xml.string()));

  EditorContext ctx2;
  CHECK(ps::studio::LoadModel(ctx2, xml.string()));
  CHECK(ctx2.compiled.ok());
  CHECK(*ctx.tree == *ctx2.tree);  // structural equality (serials aside)
  const std::string a = mj::io::WriteMjcf(*ctx.tree);
  const std::string b = mj::io::WriteMjcf(*ctx2.tree);
  CHECK(a == b);

  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

// New model is a compilable starter scene (ground plane + light), dirty/unsaved.
static void TestNewModel() {
  EditorContext ctx;
  CHECK(ps::studio::NewModelOp(ctx));
  CHECK(ctx.compiled.ok());
  CHECK(ctx.dirty);                 // unsaved
  CHECK(ctx.source_path.empty());
  CHECK(!ctx.history.can_undo());   // fresh history
  CHECK(ps::sdk::Find<mj::Geom>(*ctx.tree, "ground") != nullptr);
  CHECK(ps::sdk::Find<mj::Light>(*ctx.tree, "light") != nullptr);
}

int main() {
  TestNewModel();
  TestAddPrimitivesCompile();
  TestDropAdd();
  TestNameUniquing();
  TestDuplicate();
  TestReparentKeepWorldPose();
  TestReparentFrameWrapped();
  TestMeshImportVfsAndExternalize();
  TestExitStory();

  std::printf("%d checks, %d failed\n", g_checks, g_failed);
  return g_failed == 0 ? 0 : 1;
}
