// ProtoSpec Studio SE2 windowless tests: the DR-S6 delta rule and the gizmo
// projection / hit-testing math. The delta battery is the point of the milestone
// -- it compiles a model, applies a gizmo drag through the same transform_math
// the live gizmo drives, recompiles, and asserts the COMPILED world pose moved by
// exactly the world-space delta (with the mesh-frame bake proving it cancels).

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>

#include "binding.h"
#include "compile.h"
#include "editor/editor_context.h"
#include "editor/editor_ops.h"
#include "editor/gizmo_math.h"
#include "editor/joint_overlay.h"
#include "editor/transform_math.h"
#include "mjcf.h"
#include "protospec/traversal.h"
#include "types.h"

namespace mj = ps::mjcf;
using namespace ps::studio;

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

static bool Near(double a, double b, double eps = 1e-6) {
  return std::fabs(a - b) <= eps;
}

// MSVC rejects C compound literals; a tiny 3-vector builder keeps call sites terse.
struct V3 {
  double v[3];
  V3(double a, double b, double c) : v{a, b, c} {}
  operator const double*() const { return v; }
};
#define CHECK_NEAR(a, b) CHECK(Near((a), (b), 1e-5))

// A compiled scene the tests mutate + recompile.
struct Scene {
  std::unique_ptr<mj::Model> tree;
  mj::Compiled compiled;
  mjData* data = nullptr;

  explicit Scene(const char* xml) {
    mj::io::ParseResult r = mj::io::ParseMjcfString(xml, "gizmo_test");
    if (!r.ok()) {
      std::printf("FATAL parse:\n");
      for (const auto& e : r.errors) std::printf("  %s\n", e.Render().c_str());
    }
    tree = std::move(r.model);
    Recompile();
  }
  ~Scene() {
    if (data) mj_deleteData(data);
  }
  bool Recompile() {
    if (data) {
      mj_deleteData(data);
      data = nullptr;
    }
    mj::CompileOptions opts;
    opts.path = mj::CompilePath::Auto;
    compiled = mj::Compile(*tree, opts);
    if (!compiled.ok()) {
      std::printf("compile failed:\n");
      for (const auto& e : compiled.report.errors)
        std::printf("  %s\n", e.Render().c_str());
      return false;
    }
    data = mj_makeData(compiled.model.get());
    mj_resetData(compiled.model.get(), data);
    mj_forward(compiled.model.get(), data);
    return true;
  }
  const mjModel* m() const { return compiled.model.get(); }
};

// The compiled world position of the element with `serial`, via the Binding.
static bool WorldPosOf(const Scene& s, std::uint64_t serial, double out[3]) {
  for (const mj::Binding::Entry& e : s.compiled.binding.entries()) {
    if (e.serial != serial || e.id < 0) continue;
    const mjData* d = s.data;
    switch (e.etype) {
      case mj::ElementType::Geom:
        for (int i = 0; i < 3; ++i) out[i] = d->geom_xpos[3 * e.id + i];
        return true;
      case mj::ElementType::Body:
        for (int i = 0; i < 3; ++i) out[i] = d->xpos[3 * e.id + i];
        return true;
      case mj::ElementType::Site:
        for (int i = 0; i < 3; ++i) out[i] = d->site_xpos[3 * e.id + i];
        return true;
      default:
        break;
    }
  }
  return false;
}

// The compiled world orientation (quat) of a geom, from geom_xmat.
static bool WorldQuatOfGeom(const Scene& s, std::uint64_t serial, double q[4]) {
  for (const mj::Binding::Entry& e : s.compiled.binding.entries()) {
    if (e.serial != serial || e.id < 0 || e.etype != mj::ElementType::Geom)
      continue;
    mjtNum quat[4];
    mju_mat2Quat(quat, s.data->geom_xmat + 9 * e.id);
    for (int i = 0; i < 4; ++i) q[i] = quat[i];
    return true;
  }
  return false;
}

template <class T>
static std::uint64_t SerialOf(mj::Model& m, const char* name) {
  const T* e = ps::sdk::Find<T>(m, name);
  return e ? e->serial : 0;
}

// The compiled joint id for a serial, via the Binding.
static int JointIdOf(const Scene& s, std::uint64_t serial) {
  for (const mj::Binding::Entry& e : s.compiled.binding.entries()) {
    if (e.serial == serial && e.id >= 0 &&
        (e.etype == mj::ElementType::Joint ||
         e.etype == mj::ElementType::FreeJoint)) {
      return e.id;
    }
  }
  return -1;
}
// Compiled global joint anchor (mjData xanchor) and axis (xaxis).
static bool JointAnchorWorld(const Scene& s, std::uint64_t serial, double o[3]) {
  const int id = JointIdOf(s, serial);
  if (id < 0) return false;
  for (int i = 0; i < 3; ++i) o[i] = s.data->xanchor[3 * id + i];
  return true;
}
static bool JointAxisWorld(const Scene& s, std::uint64_t serial, double o[3]) {
  const int id = JointIdOf(s, serial);
  if (id < 0) return false;
  for (int i = 0; i < 3; ++i) o[i] = s.data->xaxis[3 * id + i];
  return true;
}

// ------------------------------------------------------------------------- //
// Rigid algebra + orientation resolution.
// ------------------------------------------------------------------------- //
static void TestRigidAlgebra() {
  Rigid a;
  QuatFromAxisAngle(V3(0, 0, 1), mjPI / 3, a.quat);
  a.pos[0] = 1;
  a.pos[1] = 2;
  a.pos[2] = 3;
  Rigid inv = Invert(a);
  Rigid id = Compose(a, inv);
  CHECK_NEAR(id.quat[0], 1.0);
  CHECK_NEAR(id.pos[0], 0.0);
  CHECK_NEAR(id.pos[1], 0.0);
  CHECK_NEAR(id.pos[2], 0.0);

  // Compose applies b first: (a.b)(x) == a(b(x)).
  Rigid b;
  b.pos[0] = 5;
  Rigid ab = Compose(a, b);
  double x[3] = {0, 0, 0};
  double bx[3], abx[3], a_bx[3];
  // b(x)
  QuatRotate(b.quat, x, bx);
  for (int i = 0; i < 3; ++i) bx[i] += b.pos[i];
  // a(b(x))
  QuatRotate(a.quat, bx, a_bx);
  for (int i = 0; i < 3; ++i) a_bx[i] += a.pos[i];
  // (a.b)(x)
  QuatRotate(ab.quat, x, abx);
  for (int i = 0; i < 3; ++i) abx[i] += ab.pos[i];
  for (int i = 0; i < 3; ++i) CHECK_NEAR(abx[i], a_bx[i]);
}

static void TestOrientationResolution() {
  // Q-ORIENT: orientation is canonicalized to a quat at read, so the studio only
  // copies + renormalizes the stored quat (the five-form resolver is tested in
  // the reader's core::ResolveOrientation, cpp/test/test_io.cc). QuatOf copies a
  // present quat (renormalizing) and yields identity for an absent one.
  double q[4];
  QuatOf(ps::opt<std::array<double, 4>>(
             std::array<double, 4>{0, 0, 0, 2}), q);  // renormalized
  CHECK_NEAR(q[3], 1.0);
  CHECK_NEAR(q[0], 0.0);

  QuatOf(ps::opt<std::array<double, 4>>{}, q);  // absent -> identity
  CHECK_NEAR(q[0], 1.0);
  CHECK_NEAR(q[1], 0.0);
}

// ------------------------------------------------------------------------- //
// Translate: a geom on a rotated body. The compiled world pos must move by
// exactly the world delta, and the authored pos by the parent-frame delta.
// ------------------------------------------------------------------------- //
static void TestTranslateGeomOnRotatedBody() {
  const char* xml = R"(
  <mujoco>
    <worldbody>
      <body name="b" pos="1 2 0.5" euler="0 0 90">
        <joint type="free"/>
        <geom name="g" type="box" size="0.1 0.1 0.1" pos="0.3 0 0"/>
      </body>
    </worldbody>
  </mujoco>)";
  Scene s(xml);
  CHECK(s.compiled.ok());
  const std::uint64_t g = SerialOf<mj::Geom>(*s.tree, "g");
  CHECK(g != 0);

  double before[3];
  CHECK(WorldPosOf(s, g, before));

  DragFrame f = BuildDragFrame(s.m(), s.data, s.compiled.binding, *s.tree, g);
  CHECK(f.valid);
  // Anchor == compiled world pos of the geom (frame origin, no mesh bake here).
  for (int i = 0; i < 3; ++i) CHECK_NEAR(f.anchor[i], before[i]);

  const double D[3] = {0.5, -0.25, 0.1};
  ApplyTranslate(*s.tree, g, f, D);

  // The authored pos changed by inv(P).quat * D. Body yaw is +90deg, so the
  // parent-frame delta rotates D by -90 about z: (Dx,Dy) -> (Dy,-Dx).
  const mj::Geom* gp = ps::sdk::Find<mj::Geom>(*s.tree, "g");
  CHECK(gp->pos.has_value());
  CHECK_NEAR((*gp->pos)[0], 0.3 + D[1]);   // +Dy
  CHECK_NEAR((*gp->pos)[1], 0.0 - D[0]);   // -Dx
  CHECK_NEAR((*gp->pos)[2], 0.0 + D[2]);

  CHECK(s.Recompile());
  double after[3];
  CHECK(WorldPosOf(s, g, after));
  for (int i = 0; i < 3; ++i) CHECK_NEAR(after[i], before[i] + D[i]);
}

// ------------------------------------------------------------------------- //
// Translate a nested body (parent at a pose).
// ------------------------------------------------------------------------- //
static void TestTranslateNestedBody() {
  const char* xml = R"(
  <mujoco>
    <worldbody>
      <body name="parent" pos="0 0 1" euler="0 90 0">
        <geom type="sphere" size="0.1"/>
        <body name="child" pos="0.5 0 0">
          <geom type="sphere" size="0.1"/>
        </body>
      </body>
    </worldbody>
  </mujoco>)";
  Scene s(xml);
  const std::uint64_t c = SerialOf<mj::Body>(*s.tree, "child");
  double before[3];
  CHECK(WorldPosOf(s, c, before));

  DragFrame f = BuildDragFrame(s.m(), s.data, s.compiled.binding, *s.tree, c);
  CHECK(f.valid);
  const double D[3] = {0.2, 0.3, -0.4};
  ApplyTranslate(*s.tree, c, f, D);
  CHECK(s.Recompile());
  double after[3];
  CHECK(WorldPosOf(s, c, after));
  for (int i = 0; i < 3; ++i) CHECK_NEAR(after[i], before[i] + D[i]);
}

// ------------------------------------------------------------------------- //
// Element inside a <frame>: P must compose the frame transform (frames are not
// in mjModel), so the authored delta lives in the frame's frame.
// ------------------------------------------------------------------------- //
static void TestTranslateGeomInFrame() {
  const char* xml = R"(
  <mujoco>
    <worldbody>
      <body name="b" pos="0 0 0">
        <frame pos="1 0 0" euler="0 0 90">
          <geom name="fg" type="box" size="0.1 0.1 0.1" pos="0.2 0 0"/>
        </frame>
      </body>
    </worldbody>
  </mujoco>)";
  Scene s(xml);
  const std::uint64_t g = SerialOf<mj::Geom>(*s.tree, "fg");
  double before[3];
  CHECK(WorldPosOf(s, g, before));

  DragFrame f = BuildDragFrame(s.m(), s.data, s.compiled.binding, *s.tree, g);
  CHECK(f.valid);
  // Frame is at (1,0,0) yaw 90; geom authored (0.2,0,0) inside it -> world:
  // body(0) . frame(1,0,0,yaw90) . (0.2,0,0) = (1, 0.2, 0).
  CHECK_NEAR(before[0], 1.0);
  CHECK_NEAR(before[1], 0.2);

  const double D[3] = {0.5, 0, 0};
  ApplyTranslate(*s.tree, g, f, D);
  // Authored delta = inv(frame).quat * D = rotate (0.5,0,0) by -90 about z =
  // (0, -0.5, 0). So authored pos -> (0.2, -0.5, 0).
  const mj::Geom* gp = ps::sdk::Find<mj::Geom>(*s.tree, "fg");
  CHECK_NEAR((*gp->pos)[0], 0.2);
  CHECK_NEAR((*gp->pos)[1], -0.5);

  CHECK(s.Recompile());
  double after[3];
  CHECK(WorldPosOf(s, g, after));
  for (int i = 0; i < 3; ++i) CHECK_NEAR(after[i], before[i] + D[i]);
}

// ------------------------------------------------------------------------- //
// THE mesh regression: the compiler bakes the mesh frame into geom_pos/geom_quat.
// A gizmo translate must still move the compiled world pose by exactly D (the
// baked suffix cancels), and never double-apply the mesh transform.
// ------------------------------------------------------------------------- //
static void TestMeshFrameCancellation() {
  // A mesh whose vertices are centred well away from the origin, so the compiler
  // re-centres it and stores a non-identity mesh_pos baked into the geom frame.
  const char* xml = R"(
  <mujoco>
    <asset>
      <mesh name="m" vertex="10 10 10  10.2 10 10  10 10.2 10  10 10 10.2"/>
    </asset>
    <worldbody>
      <body name="b" pos="0 0 1" euler="0 0 30">
        <geom name="mg" type="mesh" mesh="m" pos="0.4 0.1 0"/>
      </body>
    </worldbody>
  </mujoco>)";
  Scene s(xml);
  CHECK(s.compiled.ok());
  const std::uint64_t g = SerialOf<mj::Geom>(*s.tree, "mg");
  CHECK(g != 0);
  double before[3];
  CHECK(WorldPosOf(s, g, before));

  const mj::Geom* g0 = ps::sdk::Find<mj::Geom>(*s.tree, "mg");
  const std::array<double, 3> authored_before = *g0->pos;

  DragFrame f = BuildDragFrame(s.m(), s.data, s.compiled.binding, *s.tree, g);
  CHECK(f.valid);
  const double D[3] = {0.7, -0.3, 0.2};
  ApplyTranslate(*s.tree, g, f, D);

  // The authored field moved by the PARENT-FRAME delta (body yaw 30), NOT by D
  // plus a mesh jump. |authored_delta| == |D| (rigid rotation preserves length).
  const mj::Geom* g1 = ps::sdk::Find<mj::Geom>(*s.tree, "mg");
  double adelta[3];
  for (int i = 0; i < 3; ++i) adelta[i] = (*g1->pos)[i] - authored_before[i];
  const double alen = std::sqrt(adelta[0] * adelta[0] + adelta[1] * adelta[1] +
                                adelta[2] * adelta[2]);
  const double dlen =
      std::sqrt(D[0] * D[0] + D[1] * D[1] + D[2] * D[2]);
  CHECK_NEAR(alen, dlen);

  CHECK(s.Recompile());
  double after[3];
  CHECK(WorldPosOf(s, g, after));
  // The regression assertion: compiled world pose moved by EXACTLY D.
  for (int i = 0; i < 3; ++i) CHECK_NEAR(after[i], before[i] + D[i]);

  // And a second identical drag from the new state moves by D again (no drift /
  // no accumulating mesh jump across round trips).
  double before2[3];
  CHECK(WorldPosOf(s, g, before2));
  DragFrame f2 = BuildDragFrame(s.m(), s.data, s.compiled.binding, *s.tree, g);
  ApplyTranslate(*s.tree, g, f2, D);
  CHECK(s.Recompile());
  double after2[3];
  CHECK(WorldPosOf(s, g, after2));
  for (int i = 0; i < 3; ++i) CHECK_NEAR(after2[i], before2[i] + D[i]);
}

// ------------------------------------------------------------------------- //
// Rotate: the compiled orientation changes by the world delta; the world origin
// (position) is unchanged (pivot is the element's own frame origin).
// ------------------------------------------------------------------------- //
static void TestRotateGeom() {
  const char* xml = R"(
  <mujoco>
    <worldbody>
      <body name="b" pos="0 0 1" euler="0 0 45">
        <geom name="g" type="box" size="0.2 0.1 0.1" pos="0.3 0 0"/>
      </body>
    </worldbody>
  </mujoco>)";
  Scene s(xml);
  const std::uint64_t g = SerialOf<mj::Geom>(*s.tree, "g");
  double before[3], qbefore[4];
  CHECK(WorldPosOf(s, g, before));
  CHECK(WorldQuatOfGeom(s, g, qbefore));

  DragFrame f = BuildDragFrame(s.m(), s.data, s.compiled.binding, *s.tree, g);
  const double axis[3] = {0, 0, 1};
  const double angle = mjPI / 2;  // +90 about world z
  ApplyRotate(*s.tree, g, f, axis, angle);
  // Rotation materialises the canonical quat; pos left as authored.
  const mj::Geom* gp = ps::sdk::Find<mj::Geom>(*s.tree, "g");
  CHECK(gp->quat.has_value());
  CHECK_NEAR((*gp->pos)[0], 0.3);  // pos unchanged

  CHECK(s.Recompile());
  double after[3], qafter[4];
  CHECK(WorldPosOf(s, g, after));
  CHECK(WorldQuatOfGeom(s, g, qafter));
  // World position unchanged.
  for (int i = 0; i < 3; ++i) CHECK_NEAR(after[i], before[i]);
  // World orientation changed by qD applied on the left: qafter ~ qD * qbefore.
  double qd[4], expected[4];
  QuatFromAxisAngle(axis, angle, qd);
  QuatMul(qd, qbefore, expected);
  // Compare up to sign (q and -q are the same rotation).
  double dot = 0;
  for (int i = 0; i < 4; ++i) dot += qafter[i] * expected[i];
  if (dot < 0)
    for (int i = 0; i < 4; ++i) expected[i] = -expected[i];
  for (int i = 0; i < 4; ++i) CHECK_NEAR(qafter[i], expected[i]);
}

// ------------------------------------------------------------------------- //
// Joint rigging (SE4): translate the anchor + reorient the axis, hand-computed.
// ------------------------------------------------------------------------- //

// Move a hinge joint's anchor on a body rotated +90 about z. The compiled global
// anchor must move by exactly the world delta; the authored pos by the parent-
// frame delta conj(P).q * world_delta.
static void TestJointTranslateAnchor() {
  const char* xml = R"(
  <mujoco>
    <worldbody>
      <body name="b" pos="1 2 0.5" euler="0 0 90">
        <joint name="j" type="hinge" pos="0.1 0 0" axis="0 0 1"/>
        <geom name="g" type="box" size="0.1 0.1 0.1"/>
      </body>
    </worldbody>
  </mujoco>)";
  Scene s(xml);
  const std::uint64_t j = SerialOf<mj::Joint>(*s.tree, "j");
  CHECK(j != 0);
  double before[3];
  CHECK(JointAnchorWorld(s, j, before));
  CHECK_NEAR(before[0], 1.0);   // (1,2,0.5) + R_z90*(0.1,0,0) = (1, 2.1, 0.5)
  CHECK_NEAR(before[1], 2.1);
  CHECK_NEAR(before[2], 0.5);

  JointDragFrame f =
      BuildJointDragFrame(s.m(), s.data, s.compiled.binding, *s.tree, j);
  CHECK(f.valid && f.has_axis);
  const double wd[3] = {0.5, 0, 0};  // +x in world
  ApplyJointTranslate(*s.tree, j, f, wd);

  // Authored pos = (0.1,0,0) + R_z(-90)*(0.5,0,0) = (0.1, -0.5, 0).
  const mj::Joint* jp = ps::sdk::Find<mj::Joint>(*s.tree, "j");
  CHECK_NEAR((*jp->pos)[0], 0.1);
  CHECK_NEAR((*jp->pos)[1], -0.5);
  CHECK_NEAR((*jp->pos)[2], 0.0);

  CHECK(s.Recompile());
  double after[3];
  CHECK(JointAnchorWorld(s, j, after));
  CHECK_NEAR(after[0], before[0] + 0.5);
  CHECK_NEAR(after[1], before[1]);
  CHECK_NEAR(after[2], before[2]);
}

// Reorient a hinge axis by +90 about world x on a body rotated +90 about z. The
// authored axis becomes (-1,0,0) in the body frame; the compiled global axis
// becomes (0,-1,0).
static void TestJointReorientAxis() {
  const char* xml = R"(
  <mujoco>
    <worldbody>
      <body name="b" pos="0 0 0" euler="0 0 90">
        <joint name="j" type="hinge" pos="0 0 0" axis="0 0 1"/>
        <geom name="g" type="box" size="0.1 0.1 0.1"/>
      </body>
    </worldbody>
  </mujoco>)";
  Scene s(xml);
  const std::uint64_t j = SerialOf<mj::Joint>(*s.tree, "j");
  double axis_before[3];
  CHECK(JointAxisWorld(s, j, axis_before));
  CHECK_NEAR(axis_before[2], 1.0);  // world axis starts +z

  JointDragFrame f =
      BuildJointDragFrame(s.m(), s.data, s.compiled.binding, *s.tree, j);
  const double world_x[3] = {1, 0, 0};
  ApplyJointAxisRotate(*s.tree, j, f, world_x, mjPI / 2, /*snap=*/false);

  const mj::Joint* jp = ps::sdk::Find<mj::Joint>(*s.tree, "j");
  CHECK_NEAR((*jp->axis)[0], -1.0);
  CHECK_NEAR((*jp->axis)[1], 0.0);
  CHECK_NEAR((*jp->axis)[2], 0.0);

  CHECK(s.Recompile());
  double axis_after[3];
  CHECK(JointAxisWorld(s, j, axis_after));
  CHECK_NEAR(axis_after[0], 0.0);
  CHECK_NEAR(axis_after[1], -1.0);
  CHECK_NEAR(axis_after[2], 0.0);
}

// Axis snapping: an 80-degree tilt about world x lands the body-frame axis near
// -y; with snapping on it resolves exactly to the cardinal (0,-1,0).
static void TestJointAxisSnap() {
  const char* xml = R"(
  <mujoco>
    <worldbody>
      <body name="b">
        <joint name="j" type="hinge" pos="0 0 0" axis="0 0 1"/>
        <geom name="g" type="box" size="0.1 0.1 0.1"/>
      </body>
    </worldbody>
  </mujoco>)";
  Scene s(xml);
  const std::uint64_t j = SerialOf<mj::Joint>(*s.tree, "j");
  JointDragFrame f =
      BuildJointDragFrame(s.m(), s.data, s.compiled.binding, *s.tree, j);
  const double world_x[3] = {1, 0, 0};
  ApplyJointAxisRotate(*s.tree, j, f, world_x, 80.0 * mjPI / 180.0,
                       /*snap=*/true);
  const mj::Joint* jp = ps::sdk::Find<mj::Joint>(*s.tree, "j");
  CHECK_NEAR((*jp->axis)[0], 0.0);
  CHECK_NEAR((*jp->axis)[1], -1.0);
  CHECK_NEAR((*jp->axis)[2], 0.0);
}

// A ball joint has an anchor but no reorientable axis; a free joint has neither.
static void TestJointBallFreeNoAxis() {
  const char* xml = R"(
  <mujoco>
    <worldbody>
      <body name="bb" pos="0 0 1">
        <joint name="ball" type="ball" pos="0.2 0 0"/>
        <geom name="gb" type="sphere" size="0.1"/>
      </body>
      <body name="fb" pos="1 0 1">
        <freejoint name="fj"/>
        <geom name="gf" type="sphere" size="0.1"/>
      </body>
    </worldbody>
  </mujoco>)";
  Scene s(xml);
  const std::uint64_t ball = SerialOf<mj::Joint>(*s.tree, "ball");
  const std::uint64_t fj = SerialOf<mj::FreeJoint>(*s.tree, "fj");
  CHECK(IsJointSerial(*s.tree, ball));
  CHECK(IsJointSerial(*s.tree, fj));

  JointDragFrame bf =
      BuildJointDragFrame(s.m(), s.data, s.compiled.binding, *s.tree, ball);
  CHECK(bf.valid && !bf.has_axis);
  // Ball anchor still translates.
  const double wd[3] = {0.3, 0, 0};
  ApplyJointTranslate(*s.tree, ball, bf, wd);
  const mj::Joint* bp = ps::sdk::Find<mj::Joint>(*s.tree, "ball");
  CHECK_NEAR((*bp->pos)[0], 0.5);
  // Axis reorient is a no-op for a ball.
  const double wx[3] = {1, 0, 0};
  ApplyJointAxisRotate(*s.tree, ball, bf, wx, mjPI / 2, false);
  CHECK(!bp->axis.has_value());

  JointDragFrame ff =
      BuildJointDragFrame(s.m(), s.data, s.compiled.binding, *s.tree, fj);
  CHECK(ff.valid && !ff.has_axis);  // free joint: frame triad only
}

// Joint visualization collect: selecting a body yields its joints only; the
// selected joint is flagged; show_all yields joints across every body.
static void TestCollectJointVis() {
  const char* xml = R"(
  <mujoco>
    <worldbody>
      <body name="b1" pos="0 0 1">
        <joint name="h1" type="hinge" axis="0 1 0" range="-30 30" limited="true"/>
        <geom name="g1" type="box" size="0.1 0.1 0.1"/>
        <body name="b2" pos="0.3 0 0">
          <joint name="s2" type="slide" axis="1 0 0"/>
          <geom name="g2" type="sphere" size="0.1"/>
        </body>
      </body>
    </worldbody>
  </mujoco>)";
  Scene s(xml);
  const std::uint64_t b1 = SerialOf<mj::Body>(*s.tree, "b1");
  const std::uint64_t h1 = SerialOf<mj::Joint>(*s.tree, "h1");
  const std::uint64_t s2 = SerialOf<mj::Joint>(*s.tree, "s2");

  // Selecting body b1 shows only b1's joint (h1), not b2's.
  auto v1 = CollectJointVis(s.m(), s.data, s.compiled.binding, b1, false);
  CHECK(v1.size() == 1);
  CHECK(v1[0].serial == h1);
  CHECK(v1[0].type == mjJNT_HINGE);
  CHECK(v1[0].has_range);
  CHECK(!v1[0].selected);  // body is selected, not the joint -> not highlighted

  // Selecting the joint h1 flags it as selected.
  auto v2 = CollectJointVis(s.m(), s.data, s.compiled.binding, h1, false);
  CHECK(v2.size() == 1 && v2[0].serial == h1 && v2[0].selected);

  // show_all lists both joints regardless of selection.
  auto v3 = CollectJointVis(s.m(), s.data, s.compiled.binding, 0, true);
  CHECK(v3.size() == 2);
  bool saw_slide = false;
  for (const auto& jv : v3) if (jv.serial == s2) saw_slide = true;
  CHECK(saw_slide);

  // Nothing selected + show_all off -> empty.
  auto v4 = CollectJointVis(s.m(), s.data, s.compiled.binding, 0, false);
  CHECK(v4.empty());
}

// Local-vs-world: a local-frame translate along the element's own axis moves the
// authored pos along that local axis regardless of the parent rotation.
static void TestLocalVsWorldTranslate() {
  const char* xml = R"(
  <mujoco>
    <worldbody>
      <body name="b" pos="0 0 0" euler="0 0 90">
        <geom name="g" type="box" size="0.1 0.1 0.1" pos="0 0 0" euler="0 0 0"/>
      </body>
    </worldbody>
  </mujoco>)";
  Scene s(xml);
  const std::uint64_t g = SerialOf<mj::Geom>(*s.tree, "g");
  DragFrame f = BuildDragFrame(s.m(), s.data, s.compiled.binding, *s.tree, g);

  // Local +x axis of the element in world = world_quat * (1,0,0). The gizmo
  // forms the world delta along that axis; the authored pos then moves along the
  // element's own local x by the drag length.
  double lx[3];
  QuatRotate(f.world_quat, V3(1, 0, 0), lx);
  const double dist = 0.5;
  const double D[3] = {lx[0] * dist, lx[1] * dist, lx[2] * dist};
  ApplyTranslate(*s.tree, g, f, D);
  const mj::Geom* gp = ps::sdk::Find<mj::Geom>(*s.tree, "g");
  CHECK_NEAR((*gp->pos)[0], dist);  // moved along local x, not world x
  CHECK_NEAR((*gp->pos)[1], 0.0);
}

// Class-inherited pose materialisation: a geom with pos supplied by its default
// class materialises the field on first drag and moves correctly.
static void TestClassInheritedPoseMaterialize() {
  const char* xml = R"(
  <mujoco>
    <default>
      <default class="off">
        <geom pos="0.5 0 0" size="0.1"/>
      </default>
    </default>
    <worldbody>
      <body name="b" pos="0 0 1">
        <geom name="g" class="off" type="sphere"/>
      </body>
    </worldbody>
  </mujoco>)";
  Scene s(xml);
  const std::uint64_t g = SerialOf<mj::Geom>(*s.tree, "g");
  const mj::Geom* g0 = ps::sdk::Find<mj::Geom>(*s.tree, "g");
  CHECK(!g0->pos.has_value());  // unset -> inherited from class

  double before[3];
  CHECK(WorldPosOf(s, g, before));
  CHECK_NEAR(before[0], 0.5);  // class pos applied

  DragFrame f = BuildDragFrame(s.m(), s.data, s.compiled.binding, *s.tree, g);
  CHECK_NEAR(f.local.pos[0], 0.5);  // materialised from Effective
  const double D[3] = {0.25, 0, 0};
  ApplyTranslate(*s.tree, g, f, D);
  const mj::Geom* g1 = ps::sdk::Find<mj::Geom>(*s.tree, "g");
  CHECK(g1->pos.has_value());  // now materialised
  CHECK_NEAR((*g1->pos)[0], 0.75);

  CHECK(s.Recompile());
  double after[3];
  CHECK(WorldPosOf(s, g, after));
  CHECK_NEAR(after[0], before[0] + D[0]);
}

// Scale mapping: per-axis on a box, uniform on a sphere, mesh -> mesh asset.
static void TestScaleMapping() {
  const char* xml = R"(
  <mujoco>
    <asset>
      <mesh name="m" vertex="0 0 0  1 0 0  0 1 0  0 0 1" scale="1 1 1"/>
    </asset>
    <worldbody>
      <body name="b">
        <geom name="boxg" type="box" size="0.2 0.3 0.4"/>
        <geom name="sphg" type="sphere" size="0.5"/>
        <geom name="meshg" type="mesh" mesh="m"/>
      </body>
    </worldbody>
  </mujoco>)";
  Scene s(xml);
  const std::uint64_t box = SerialOf<mj::Geom>(*s.tree, "boxg");
  ScaleBase bb = BuildScaleBase(*s.tree, box);
  CHECK(bb.valid && bb.size_dofs == 3);
  const double f2[3] = {2, 1, 0.5};
  ApplyScale(*s.tree, box, bb, f2);
  const mj::Geom* bp = ps::sdk::Find<mj::Geom>(*s.tree, "boxg");
  CHECK_NEAR((*bp->size)[0], 0.4);
  CHECK_NEAR((*bp->size)[1], 0.3);
  CHECK_NEAR((*bp->size)[2], 0.2);

  const std::uint64_t sph = SerialOf<mj::Geom>(*s.tree, "sphg");
  ScaleBase sb = BuildScaleBase(*s.tree, sph);
  CHECK(sb.valid && sb.size_dofs == 1);  // uniform
  const double fu[3] = {3, 3, 3};
  ApplyScale(*s.tree, sph, sb, fu);
  const mj::Geom* sp = ps::sdk::Find<mj::Geom>(*s.tree, "sphg");
  CHECK(sp->size->size() == 1);
  CHECK_NEAR((*sp->size)[0], 1.5);

  const std::uint64_t meshg = SerialOf<mj::Geom>(*s.tree, "meshg");
  ScaleBase mb = BuildScaleBase(*s.tree, meshg);
  CHECK(mb.valid && mb.is_mesh);
  const double fm[3] = {2, 2, 2};
  ApplyScale(*s.tree, meshg, mb, fm);
  const mj::Mesh* mp = ps::sdk::Find<mj::Mesh>(*s.tree, "m");
  CHECK(mp->scale.has_value());
  CHECK_NEAR((*mp->scale)[0], 2.0);  // mesh asset scaled, not geom size
  CHECK(!ps::sdk::Find<mj::Geom>(*s.tree, "meshg")->size.has_value());
}

// ------------------------------------------------------------------------- //
// Gizmo projection + hit-testing math against a fixed camera fixture.
// ------------------------------------------------------------------------- //
static ViewProj FixtureCamera() {
  // Eye at (0,-5,0) looking down +y, up = +z, right = +x. Symmetric 60deg-ish
  // frustum, near=1.
  ViewProj vp;
  vp.eye[0] = 0; vp.eye[1] = -5; vp.eye[2] = 0;
  vp.forward[0] = 0; vp.forward[1] = 1; vp.forward[2] = 0;
  vp.up[0] = 0; vp.up[1] = 0; vp.up[2] = 1;
  vp.right[0] = 1; vp.right[1] = 0; vp.right[2] = 0;
  vp.zver[0] = 0.5; vp.zver[1] = 0.5;
  vp.zhor[0] = 0.5; vp.zhor[1] = 0.5;
  vp.zclip[0] = 1; vp.zclip[1] = 100;
  vp.aspect = 1.0;
  vp.orthographic = false;
  return vp;
}

static void TestGizmoProjection() {
  ViewProj vp = FixtureCamera();
  // The origin is straight ahead -> screen centre (0.5, 0.5).
  ScreenPt c = WorldToScreen(vp, V3(0, 0, 0));
  CHECK(c.visible);
  CHECK_NEAR(c.x, 0.5);
  CHECK_NEAR(c.y, 0.5);

  // A point up (+z) projects above centre (smaller y, y-from-top).
  ScreenPt up = WorldToScreen(vp, V3(0, 0, 1));
  CHECK(up.y < 0.5);
  // A point to the right (+x) projects right of centre.
  ScreenPt rt = WorldToScreen(vp, V3(1, 0, 0));
  CHECK(rt.x > 0.5);
  // Behind the camera -> not visible.
  ScreenPt bk = WorldToScreen(vp, V3(0, -10, 0));
  CHECK(!bk.visible);

  // Round trip: screen centre ray hits the forward axis; project a point on that
  // ray back to the centre.
  double ro[3], rd[3];
  ScreenToRay(vp, 0.5, 0.5, ro, rd);
  CHECK_NEAR(rd[0], 0.0);
  CHECK_NEAR(rd[1], 1.0);
  CHECK_NEAR(rd[2], 0.0);
  double onray[3] = {ro[0] + 3 * rd[0], ro[1] + 3 * rd[1], ro[2] + 3 * rd[2]};
  ScreenPt back = WorldToScreen(vp, onray);
  CHECK_NEAR(back.x, 0.5);
  CHECK_NEAR(back.y, 0.5);
}

static void TestGizmoHitTests() {
  ViewProj vp = FixtureCamera();
  // Axis handle from origin along +x to (1,0,0). A mouse near its screen segment
  // hits; one far away misses.
  ScreenPt a = WorldToScreen(vp, V3(0, 0, 0));
  ScreenPt b = WorldToScreen(vp, V3(1, 0, 0));
  const double midx = 0.5 * (a.x + b.x), midy = 0.5 * (a.y + b.y);
  CHECK(PointSegmentDist(midx, midy, a.x, a.y, b.x, b.y) < 1e-6);
  CHECK(PointSegmentDist(midx, midy + 0.3, a.x, a.y, b.x, b.y) > 0.1);

  // ClosestPointOnLine: the x-axis line and a vertical ray crossing at x=0.4.
  double p0[3] = {0, 0, 0}, pd[3] = {1, 0, 0};
  double q0[3] = {0.4, -1, 0}, qd[3] = {0, 1, 0};
  double t = 0;
  CHECK(ClosestPointOnLine(p0, pd, q0, qd, &t));
  CHECK_NEAR(t, 0.4);

  // RayPlaneIntersect: a ray down +y hits the z=0 plane's... use plane through
  // origin normal +x; ray from (-1, -5, 0) along +x hits at x=0.
  double ro[3] = {-1, 0, 0}, rd[3] = {1, 0, 0};
  double po[3] = {0, 0, 0}, pn[3] = {1, 0, 0};
  double hit[3];
  CHECK(RayPlaneIntersect(ro, rd, po, pn, &t, hit));
  CHECK_NEAR(hit[0], 0.0);
}

// ------------------------------------------------------------------------- //
// Mode machine (DR-S2): the recompile gate defers Play-mode edits.
// ------------------------------------------------------------------------- //
static void TestModeMachineRecompileGate() {
  EditorContext ctx;
  // Edit mode services a queued recompile.
  ctx.mode = EditorMode::Edit;
  ctx.recompile_requested = true;
  CHECK(ShouldServiceRecompile(ctx));
  // Play mode defers it (edits take effect on the next compile).
  ctx.mode = EditorMode::Play;
  CHECK(!ShouldServiceRecompile(ctx));
  // ...unless the one-shot apply_edits is set (the play transition).
  ctx.apply_edits = true;
  CHECK(ShouldServiceRecompile(ctx));
  ctx.apply_edits = false;
  // Nothing queued: no service.
  ctx.recompile_requested = false;
  ctx.mode = EditorMode::Edit;
  CHECK(!ShouldServiceRecompile(ctx));
}

// Gesture undo granularity: a whole drag (BeginEdit, many mutations, CommitEdit)
// is ONE undo entry; an Esc-cancelled drag records NO entry and reverts.
static void TestGestureUndoGranularity() {
  const char* xml = R"(
  <mujoco>
    <worldbody>
      <body name="b" pos="0 0 1">
        <geom name="g" type="box" size="0.1 0.1 0.1" pos="0 0 0"/>
      </body>
    </worldbody>
  </mujoco>)";
  Scene s(xml);
  EditorContext ctx;
  ctx.tree = std::move(s.tree);
  mj::CompileOptions opts;
  opts.path = mj::CompilePath::Auto;
  ctx.compiled = mj::Compile(*ctx.tree, opts);
  CHECK(ctx.compiled.ok());
  mjData* d = mj_makeData(ctx.compiled.model.get());
  mj_resetData(ctx.compiled.model.get(), d);
  mj_forward(ctx.compiled.model.get(), d);
  const std::uint64_t g = SerialOf<mj::Geom>(*ctx.tree, "g");
  const mj::Geom* g0 = ps::sdk::Find<mj::Geom>(*ctx.tree, "g");
  const std::array<double, 3> base = *g0->pos;

  // One gesture: BeginEdit once, five incremental applies (as a drag would),
  // CommitEdit once.
  CHECK(ctx.history.undo_depth() == 0);
  ctx.BeginEdit();
  DragFrame f = BuildDragFrame(ctx.compiled.model.get(), d, ctx.compiled.binding,
                               *ctx.tree, g);
  for (int i = 1; i <= 5; ++i) {
    const double D[3] = {0.02 * i, 0, 0};
    ApplyTranslate(*ctx.tree, g, f, D);  // recomputed from base each frame
  }
  ctx.CommitEdit("drag");
  CHECK(ctx.history.undo_depth() == 1);  // exactly one entry for the gesture
  const mj::Geom* g1 = ps::sdk::Find<mj::Geom>(*ctx.tree, "g");
  CHECK_NEAR((*g1->pos)[0], base[0] + 0.10);  // final cumulative delta, no drift

  // Undo restores the pre-gesture pose.
  CHECK(Undo(ctx));
  const mj::Geom* g2 = ps::sdk::Find<mj::Geom>(*ctx.tree, "g");
  CHECK_NEAR((*g2->pos)[0], base[0]);

  // Esc-cancelled gesture: BeginEdit, mutate, CancelEdit -> no new entry, revert.
  const std::size_t depth = ctx.history.undo_depth();
  ctx.BeginEdit();
  DragFrame f2 = BuildDragFrame(ctx.compiled.model.get(), d, ctx.compiled.binding,
                                *ctx.tree, g);
  const double big[3] = {0.5, 0, 0};
  ApplyTranslate(*ctx.tree, g, f2, big);
  ctx.CancelEdit();
  CHECK(ctx.history.undo_depth() == depth);  // no entry recorded
  const mj::Geom* g3 = ps::sdk::Find<mj::Geom>(*ctx.tree, "g");
  CHECK_NEAR((*g3->pos)[0], base[0]);  // reverted

  mj_deleteData(d);
}

// The corpus root holding the perf models; overridable so the gate runs against
// any MuJoCo checkout (or is skipped where the corpus is absent).
static std::string CorpusRoot() {
  if (const char* env = std::getenv("PROTOSPEC_CORPUS")) return env;
  return "C:/Users/jonat/Documents/Unreal Projects/url_proj/Plugins/"
         "UnrealRoboticsLab/third_party/MuJoCo/src";
}

struct PerfResult {
  bool ran = false;      // model found, parsed and compiled at least once
  double median_ms = 0;  // median wall time of a full recompile
  int nbody = 0;         // compiled body count (corpus-size signal)
};

// Median wall time of a full ProtoSpec recompile of `path` over `n` iterations.
// The debounced recompile IS the drag preview (DR-S3), so this is the interactive
// budget a drag pays each frame. A warmup compile primes caches; the reported
// number is the steady-state median, which is what a sustained drag sees.
static PerfResult MeasureRecompile(const std::string& path, int n) {
  PerfResult out;
  if (!std::filesystem::exists(path)) return out;
  mj::io::ParseResult r = mj::io::ParseMjcfFile(path);
  if (!r.ok()) return out;
  mj::CompileOptions opts;
  opts.path = mj::CompilePath::Auto;
  opts.base_dir = std::filesystem::path(path).parent_path().string();
  mj::Compiled warm = mj::Compile(*r.model, opts);
  if (!warm.ok()) return out;
  out.nbody = warm.model->nbody;
  std::vector<double> samples;
  samples.reserve(n);
  for (int i = 0; i < n; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    mj::Compiled c = mj::Compile(*r.model, opts);
    auto t1 = std::chrono::steady_clock::now();
    if (!c.ok()) return out;
    samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  std::sort(samples.begin(), samples.end());
  out.median_ms = samples[samples.size() / 2];
  out.ran = true;
  return out;
}

// Perf gate (G6): the drag-preview recompile must stay inside an interactive
// budget. Two enforced bounds, each skipped only when its corpus model is absent
// so the gate stays portable:
//   * humanoid       (nbody ~ 17): hard bound 10 ms median -- the everyday drag
//                    target. Measured ~3.3 ms on the reference build; well inside
//                    a 60 Hz (16.7 ms) frame, so the preview keeps up with a drag.
//   * humanoid200    (nbody ~ 217, 627 DOF): the largest loadable corpus model
//                    (humanoid + 200 free objects). Documented bound 120 ms median.
//                    A full recompile of a scene this large is NOT a per-frame
//                    interactive op -- it lands on the debounce tick, not every
//                    drag frame. Measured ~74 ms on the reference build; 120 ms
//                    leaves ~1.6x headroom. Raise deliberately (with a fresh
//                    measurement) if the compiler grows, never silently.
static void TestRecompilePerfGate() {
  const std::string root = CorpusRoot();

  const std::string humanoid =
      (std::filesystem::path(root) / "model" / "humanoid" / "humanoid.xml")
          .string();
  PerfResult h = MeasureRecompile(humanoid, 25);
  if (!h.ran) {
    std::printf("perf: humanoid corpus absent, gate skipped\n");
  } else {
    std::printf("perf: humanoid (nbody=%d) median %.2f ms/compile (%.0f Hz)\n",
                h.nbody, h.median_ms,
                h.median_ms > 0 ? 1000.0 / h.median_ms : 0.0);
    CHECK(h.median_ms <= 10.0);
  }

  const std::string large =
      (std::filesystem::path(root) / "test" / "benchmark" / "testdata" /
       "humanoid200.xml")
          .string();
  PerfResult big = MeasureRecompile(large, 15);
  if (!big.ran) {
    std::printf("perf: humanoid200 corpus absent, large-model gate skipped\n");
  } else {
    std::printf("perf: humanoid200 (nbody=%d) median %.2f ms/compile\n",
                big.nbody, big.median_ms);
    CHECK(big.median_ms <= 120.0);
  }
}

int main() {
  TestRecompilePerfGate();
  TestModeMachineRecompileGate();
  TestGestureUndoGranularity();
  TestRigidAlgebra();
  TestOrientationResolution();
  TestTranslateGeomOnRotatedBody();
  TestTranslateNestedBody();
  TestTranslateGeomInFrame();
  TestMeshFrameCancellation();
  TestRotateGeom();
  TestJointTranslateAnchor();
  TestJointReorientAxis();
  TestJointAxisSnap();
  TestJointBallFreeNoAxis();
  TestCollectJointVis();
  TestLocalVsWorldTranslate();
  TestClassInheritedPoseMaterialize();
  TestScaleMapping();
  TestGizmoProjection();
  TestGizmoHitTests();

  std::printf("%d checks, %d failed\n", g_checks, g_failed);
  return g_failed == 0 ? 0 : 1;
}
