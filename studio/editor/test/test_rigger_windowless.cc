// Windowless tests of the joint rigger core (studio/editor/joint_rig.cc). The
// rigger's correctness doctrine (docs/rigger_plan.md §3) is "every visual is a
// compiled or forwarded quantity"; these tests pin each visual against an
// INDEPENDENT mj_loadXML + mj_forward reference:
//
//   * scrub pose      == reference mj_forward at the same qpos (bitwise)
//   * snap-back       == qpos0 forward; dirty / undo untouched (no-commit)
//   * mirror          == host-data copy of the scrub kinematics
//   * ghost poses     == reference geom_x* at each jnt_range endpoint (bitwise);
//                        count == subtree geom count; alpha<1; mjCAT_DECOR
//   * arc/travel      == JointLimitChildPoint equals mj_forward at qpos=range[k]
//                        to 1e-12, on a DEGREE hinge, a RADIAN twin, and a SLIDE
//   * limited=auto    == m->jnt_limited pinned (autolimits on, limited="false" off)
//   * unit display    == JointDofToDisplay: degree fixture shows -30/45; radian
//                        verbatim; slide never converts
//
// The editor model comes through the ProtoSpec compile bridge (XmlPath, so it is
// literally mj_loadXML of the emitted bytes); the reference comes from a raw
// mj_loadXML of those same bytes -- identical code path, hence bitwise. Splices
// joint_rig.cc in so its functions are reachable; links libprotospec_core.a +
// libmujoco.so, same recipe as test_plugin_windowless. Driven by
// protospec/tests/test_rigger_windowless.py.

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>

#include "bridge.h"    // ps::mjcf::Compile / CompileToXml / CompileOptions
#include "editor/editor_context.h"
#include "editor/element_access.h"  // FindSerialAs
#include "mjcf.h"      // ps::mjcf::io::ParseMjcfString
#include "protospec/builders.h"   // ps::sdk::AddKey (keyframe capture)
#include "protospec/traversal.h"  // ps::sdk::FindBySerial (badge target)

// Splice the rigger core so its functions are reachable directly.
#include "editor/joint_rig.cc"
// Splice the PURE part of the P2 handle core (the ImGui controller is compiled
// out via PS_RIG_HANDLES_NO_CONTROLLER, leaving only the windowless cursor->dof
// mapping + the single unit-conversion write helper).
#define PS_RIG_HANDLES_NO_CONTROLLER
#include "editor/rig_handles.cc"
// The commit/undo/cancel tests exercise EditorContext's gesture flow, whose undo
// storage lives in undo.cc (not the linked core lib) -- splice it in too. It only
// needs the SDK serial-preserving clone (already in libprotospec_core.a).
#include "editor/undo.cc"

namespace ps::studio {
namespace {

namespace mj = ps::mjcf;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      return 1;                                                            \
    }                                                                      \
  } while (0)

// A real mjModel from an XML string (no on-disk assets in these fixtures).
mjModel* LoadStr(const char* name, const std::string& xml) {
  mjVFS vfs;
  mj_defaultVFS(&vfs);
  if (mj_addBufferVFS(&vfs, name, xml.data(), static_cast<int>(xml.size())) != 0) {
    mj_deleteVFS(&vfs);
    return nullptr;
  }
  char err[512] = {0};
  mjModel* m = mj_loadXML(name, &vfs, err, sizeof(err));
  mj_deleteVFS(&vfs);
  if (!m) std::fprintf(stderr, "mj_loadXML: %s\n", err);
  return m;
}

// The editor side of a fixture: the ProtoSpec tree + its compiled artifact +
// preview/scratch data wired into an EditorContext, exactly as the editor holds
// them. Plus a raw reference mjModel from the identical emitted bytes.
struct Fixture {
  EditorContext ctx;
  mjModel* ref = nullptr;  // independent mj_loadXML of the emitted bytes
  mjData* rd = nullptr;    // reference data
  std::uint64_t jserial = 0;
  int jid = -1;
  int qadr = -1;
  int jtype = 0;

  ~Fixture() {
    if (rd) mj_deleteData(rd);
    if (ref) mj_deleteModel(ref);
    if (ctx.sim_data) mj_deleteData(ctx.sim_data);
    if (ctx.ghost_data) mj_deleteData(ctx.ghost_data);
  }
};

// Build a Fixture from an MJCF string; picks the first (non-free) Joint entry.
bool BuildFixture(Fixture& f, const std::string& xml) {
  auto parsed = mj::io::ParseMjcfString(xml);
  if (!parsed.ok()) return false;
  f.ctx.tree = std::move(parsed.model);

  mj::CompileOptions opts;
  opts.path = mj::CompilePath::XmlPath;  // == mj_loadXML of the emitted bytes
  mj::Compiled c = mj::Compile(*f.ctx.tree, opts);
  if (!c.ok()) return false;
  mjModel* me = c.model.get();
  f.ctx.compiled = std::move(c);
  f.ctx.model_ready = true;
  f.ctx.sim_data = mj_makeData(me);
  mj_resetData(me, f.ctx.sim_data);
  mj_forward(me, f.ctx.sim_data);
  f.ctx.ghost_data = mj_makeData(me);

  // Reference from the SAME emitted bytes -> bitwise-identical mjModel.
  const std::string emitted = mj::CompileToXml(*f.ctx.tree, opts);
  f.ref = LoadStr("ref.xml", emitted);
  if (!f.ref) return false;
  f.rd = mj_makeData(f.ref);

  for (const mj::Binding::Entry& e : f.ctx.compiled.binding.entries()) {
    if (e.etype == mj::ElementType::Joint && e.id >= 0) {
      f.jserial = e.serial;
      f.jid = e.id;
      break;
    }
  }
  if (f.jid < 0) return false;
  f.qadr = me->jnt_qposadr[f.jid];
  f.jtype = me->jnt_type[f.jid];
  return true;
}

// Reference geom_xpos/xmat after forwarding the reference model at qpos[qadr]=q.
void RefForwardAt(Fixture& f, double q) {
  mj_resetData(f.ref, f.rd);
  f.rd->qpos[f.qadr] = q;
  mj_forward(f.ref, f.rd);
}

double MaxAbsDiff(const double* a, const double* b, int n) {
  double d = 0;
  for (int i = 0; i < n; ++i) d = std::max(d, std::fabs(a[i] - b[i]));
  return d;
}

// --- Fixtures --------------------------------------------------------------- //

// autolimits => `limited` is auto and range presence turns it on (m->jnt_limited
// == 1). Two subtree geoms on the arm; base geom is NOT in the hinge subtree.
constexpr char kDegHinge[] = R"(<mujoco model="armdeg">
  <compiler angle="degree" autolimits="true"/>
  <worldbody>
    <body name="base" pos="0 0 0.5">
      <geom name="basegeom" type="box" size="0.05 0.05 0.05"/>
      <body name="arm" pos="0 0 0">
        <joint name="hinge" type="hinge" axis="0 1 0" range="-30 45"/>
        <geom name="armgeom" type="capsule" fromto="0 0 0 0.4 0 0" size="0.03"/>
        <geom name="tipgeom" type="sphere" pos="0.4 0 0" size="0.04"/>
      </body>
    </body>
  </worldbody>
</mujoco>
)";

// Radian twin: same geometry, angle="radian", range = -30/45 deg in radians.
constexpr char kRadHinge[] = R"(<mujoco model="armrad">
  <compiler angle="radian" autolimits="true"/>
  <worldbody>
    <body name="base" pos="0 0 0.5">
      <geom name="basegeom" type="box" size="0.05 0.05 0.05"/>
      <body name="arm" pos="0 0 0">
        <joint name="hinge" type="hinge" axis="0 1 0" range="-0.5235987755982988 0.7853981633974483"/>
        <geom name="armgeom" type="capsule" fromto="0 0 0 0.4 0 0" size="0.03"/>
        <geom name="tipgeom" type="sphere" pos="0.4 0 0" size="0.04"/>
      </body>
    </body>
  </worldbody>
</mujoco>
)";

// Slide: range is metres and must never be converted.
constexpr char kSlide[] = R"(<mujoco model="slidefix">
  <compiler angle="degree" autolimits="true"/>
  <worldbody>
    <body name="base" pos="0 0 0.5">
      <geom name="basegeom" type="box" size="0.05 0.05 0.05"/>
      <body name="car" pos="0 0 0">
        <joint name="slider" type="slide" axis="1 0 0" range="-0.2 0.3"/>
        <geom name="cargeom" type="box" pos="0.1 0 0" size="0.05 0.05 0.05"/>
      </body>
    </body>
  </worldbody>
</mujoco>
)";

// limited="false" with a range set: m->jnt_limited must be 0 (no ghosts).
constexpr char kUnlimited[] = R"(<mujoco model="freehinge">
  <compiler angle="degree"/>
  <worldbody>
    <body name="base" pos="0 0 0.5">
      <geom name="basegeom" type="box" size="0.05 0.05 0.05"/>
      <body name="arm" pos="0 0 0">
        <joint name="hinge" type="hinge" axis="0 1 0" limited="false" range="-30 45"/>
        <geom name="armgeom" type="capsule" fromto="0 0 0 0.4 0 0" size="0.03"/>
      </body>
    </body>
  </worldbody>
</mujoco>
)";

// --- P3 fixtures ------------------------------------------------------------ //

// Ball joint, un-rotated body (local axes == world), tip off the local Z axis so
// PickArcReferenceGeom has a radius. range "0 60" -> 60deg swing-cone half-angle.
constexpr char kBall[] = R"(<mujoco model="ballfix">
  <compiler angle="degree" autolimits="true"/>
  <worldbody>
    <body name="base" pos="0 0 0.5">
      <geom name="basegeom" type="box" size="0.05 0.05 0.05"/>
      <body name="arm" pos="0 0 0">
        <joint name="ball" type="ball" range="0 60"/>
        <geom name="armgeom" type="capsule" fromto="0 0 0 0.4 0 0" size="0.03"/>
        <geom name="tipgeom" type="sphere" pos="0.4 0 0" size="0.04"/>
      </body>
    </body>
  </worldbody>
</mujoco>
)";

// Two hinges in a chain (nq == 2): the multi-joint held-composition + keyframe
// round-trip fixture.
constexpr char kTwoHinge[] = R"(<mujoco model="twohinge">
  <compiler angle="degree" autolimits="true"/>
  <worldbody>
    <body name="base" pos="0 0 0.5">
      <geom name="basegeom" type="box" size="0.05 0.05 0.05"/>
      <body name="upper" pos="0 0 0">
        <joint name="shoulder" type="hinge" axis="0 1 0" range="-90 90"/>
        <geom name="uppergeom" type="capsule" fromto="0 0 0 0.3 0 0" size="0.03"/>
        <body name="lower" pos="0.3 0 0">
          <joint name="elbow" type="hinge" axis="0 1 0" range="-120 0"/>
          <geom name="lowergeom" type="capsule" fromto="0 0 0 0.3 0 0" size="0.03"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
)";

// Hinge with stiffness + springref: qpos_spring differs from qpos0 -> a spring
// equilibrium ghost.
constexpr char kSpring[] = R"(<mujoco model="springfix">
  <compiler angle="degree" autolimits="true"/>
  <worldbody>
    <body name="base" pos="0 0 0.5">
      <geom name="basegeom" type="box" size="0.05 0.05 0.05"/>
      <body name="arm" pos="0 0 0">
        <joint name="hinge" type="hinge" axis="0 1 0" range="-90 90" stiffness="5" springref="30"/>
        <geom name="armgeom" type="capsule" fromto="0 0 0 0.4 0 0" size="0.03"/>
      </body>
    </body>
  </worldbody>
</mujoco>
)";

// Hinge driven by a position actuator AND wrapped by a fixed tendon: the badge
// resolution fixture.
constexpr char kDriven[] = R"(<mujoco model="drivenfix">
  <compiler angle="degree" autolimits="true"/>
  <worldbody>
    <body name="base" pos="0 0 0.5">
      <geom name="basegeom" type="box" size="0.05 0.05 0.05"/>
      <body name="arm" pos="0 0 0">
        <joint name="hinge" type="hinge" axis="0 1 0" range="-90 90"/>
        <geom name="armgeom" type="capsule" fromto="0 0 0 0.4 0 0" size="0.03"/>
      </body>
    </body>
  </worldbody>
  <tendon>
    <fixed name="ten0"><joint joint="hinge" coef="1"/></fixed>
  </tendon>
  <actuator>
    <position name="act0" joint="hinge"/>
  </actuator>
</mujoco>
)";

// The (serial, jid, qadr) of the joint named `jname` in the compiled fixture.
bool JointByName(Fixture& f, const char* jname, std::uint64_t* serial, int* jid,
                 int* qadr) {
  const mjModel* m = f.ctx.compiled.model.get();
  const int id = mj_name2id(m, mjOBJ_JOINT, jname);
  if (id < 0) return false;
  *jid = id;
  *qadr = m->jnt_qposadr[id];
  for (const mj::Binding::Entry& e : f.ctx.compiled.binding.entries()) {
    if (e.id == id && (e.etype == mj::ElementType::Joint ||
                       e.etype == mj::ElementType::FreeJoint)) {
      *serial = e.serial;
      return true;
    }
  }
  return false;
}

// The great-circle rotation angle (radians) of ball quat `q` relative to `q0`.
double BallSwingAngle(const double q0[4], const double q[4]) {
  double q0c[4] = {q0[0], -q0[1], -q0[2], -q0[3]};
  double delta[4];
  mju_mulQuat(delta, q0c, q);
  double w = std::fabs(delta[0]);
  if (w > 1.0) w = 1.0;
  return 2.0 * std::acos(w);
}

// --- Tests ------------------------------------------------------------------ //

int TestUnitsAndLimited() {
  // Degree hinge: m->jnt_range == {-30,45} deg in radians; display shows -30/45.
  Fixture fd;
  CHECK(BuildFixture(fd, kDegHinge));
  const mjModel* m = fd.ctx.compiled.model.get();
  CHECK(AngleIsDegree(*fd.ctx.tree));
  const double kPi = 3.14159265358979323846;
  CHECK(std::fabs(m->jnt_range[2 * fd.jid + 0] - (-30.0 * kPi / 180.0)) < 1e-12);
  CHECK(std::fabs(m->jnt_range[2 * fd.jid + 1] - (45.0 * kPi / 180.0)) < 1e-12);
  CHECK(std::fabs(JointDofToDisplay(fd.jtype, m->jnt_range[2 * fd.jid + 0], true) -
                  (-30.0)) < 1e-9);
  CHECK(std::fabs(JointDofToDisplay(fd.jtype, m->jnt_range[2 * fd.jid + 1], true) -
                  (45.0)) < 1e-9);
  // limited=auto is ON (autolimits + range present -> the compiled pin is 1).
  CHECK(m->jnt_limited[fd.jid] == 1);
  // Round trip through display->dof.
  CHECK(std::fabs(JointDisplayToDof(fd.jtype, -30.0, true) -
                  m->jnt_range[2 * fd.jid + 0]) < 1e-12);

  // Radian twin: display is verbatim (no conversion).
  Fixture fr;
  CHECK(BuildFixture(fr, kRadHinge));
  const mjModel* mr = fr.ctx.compiled.model.get();
  CHECK(!AngleIsDegree(*fr.ctx.tree));
  CHECK(std::fabs(JointDofToDisplay(fr.jtype, mr->jnt_range[2 * fr.jid + 1],
                                    false) -
                  mr->jnt_range[2 * fr.jid + 1]) < 1e-15);
  // The radian range equals the degree range in radians (same physical joint).
  CHECK(std::fabs(mr->jnt_range[2 * fr.jid + 0] - m->jnt_range[2 * fd.jid + 0]) <
        1e-12);

  // Slide: metres, no conversion even in a degree-angle model.
  Fixture fs;
  CHECK(BuildFixture(fs, kSlide));
  const mjModel* ms = fs.ctx.compiled.model.get();
  CHECK(fs.jtype == mjJNT_SLIDE);
  CHECK(std::fabs(ms->jnt_range[2 * fs.jid + 0] - (-0.2)) < 1e-12);
  CHECK(JointDofToDisplay(fs.jtype, ms->jnt_range[2 * fs.jid + 1], true) == 0.3);
  CHECK(JointDisplayToDof(fs.jtype, 0.3, true) == 0.3);

  // limited="false": the compiled pin is 0 -> the caller emits no ghosts.
  Fixture fu;
  CHECK(BuildFixture(fu, kUnlimited));
  CHECK(fu.ctx.compiled.model->jnt_limited[fu.jid] == 0);

  std::printf("  units + limited pins: ok\n");
  return 0;
}

int TestScrubAndSnapback() {
  Fixture f;
  CHECK(BuildFixture(f, kDegHinge));
  mjModel* m = f.ctx.compiled.model.get();
  mjData* d = f.ctx.sim_data;

  // Scrub to 20 deg (in radians) -- preview writes qpos + forwards sim_data.
  const double q = 20.0 * 3.14159265358979323846 / 180.0;
  CHECK(SetJointPreview(f.ctx, f.jserial, q));
  CHECK(f.ctx.rig_preview.active);
  CHECK(f.ctx.rig_preview.serial == f.jserial);
  // No-commit invariants: scrubbing never dirties, recompiles, or records undo.
  CHECK(f.ctx.dirty == false);
  CHECK(f.ctx.recompile_requested == false);
  CHECK(f.ctx.history.can_undo() == false);
  CHECK(std::fabs(d->qpos[f.qadr] - q) < 1e-15);

  // Scrub pose == independent reference mj_forward at the same qpos (bitwise).
  RefForwardAt(f, q);
  CHECK(MaxAbsDiff(d->geom_xpos, f.rd->geom_xpos, 3 * m->ngeom) == 0.0);
  CHECK(MaxAbsDiff(d->xpos, f.rd->xpos, 3 * m->nbody) == 0.0);
  CHECK(MaxAbsDiff(d->xquat, f.rd->xquat, 4 * m->nbody) == 0.0);

  // Mirror equality: a host-data copy of the scrub kinematics matches sim_data
  // (the exact array set MirrorDragKinematics copies for the render scene).
  mjData* hd = mj_makeData(m);
  mju_copy(hd->xpos, d->xpos, 3 * m->nbody);
  mju_copy(hd->xmat, d->xmat, 9 * m->nbody);
  mju_copy(hd->geom_xpos, d->geom_xpos, 3 * m->ngeom);
  mju_copy(hd->geom_xmat, d->geom_xmat, 9 * m->ngeom);
  CHECK(MaxAbsDiff(hd->geom_xpos, d->geom_xpos, 3 * m->ngeom) == 0.0);
  CHECK(MaxAbsDiff(hd->geom_xmat, d->geom_xmat, 9 * m->ngeom) == 0.0);
  mj_deleteData(hd);

  // Snap-back: ClearJointPreview restores qpos0 + forward; dirty/undo untouched.
  ClearJointPreview(f.ctx);
  CHECK(f.ctx.rig_preview.active == false);
  CHECK(MaxAbsDiff(d->qpos, m->qpos0, m->nq) == 0.0);
  RefForwardAt(f, m->qpos0[f.qadr]);
  CHECK(MaxAbsDiff(d->geom_xpos, f.rd->geom_xpos, 3 * m->ngeom) == 0.0);
  CHECK(f.ctx.dirty == false);
  CHECK(f.ctx.recompile_requested == false);
  CHECK(f.ctx.history.can_undo() == false);

  // Ball / free are not scrubbable (SetJointPreview declines).
  std::printf("  scrub + snap-back + mirror: ok\n");
  return 0;
}

int TestGhosts() {
  Fixture f;
  CHECK(BuildFixture(f, kDegHinge));
  const mjModel* m = f.ctx.compiled.model.get();

  // Subtree = the arm's two geoms only (the base body's geom is excluded).
  const std::vector<int> subtree = SubtreeGeoms(m, f.jid);
  CHECK(subtree.size() == 2);
  const int arm_body = m->jnt_bodyid[f.jid];
  const int base_body = m->body_parentid[arm_body];
  for (int g : subtree) CHECK(m->geom_bodyid[g] != base_body);

  const float rgba[4] = {0.3f, 0.6f, 1.0f, 0.25f};
  for (int end = 0; end < 2; ++end) {
    const double rk = m->jnt_range[2 * f.jid + end];
    // Forward the editor's ghost scratch at the endpoint qpos.
    std::vector<double> qpos;
    GhostQpos(m, f.jid, rk, qpos);
    CHECK(static_cast<int>(qpos.size()) == m->nq);
    CHECK(qpos[f.qadr] == rk);
    mj_resetData(m, f.ctx.ghost_data);
    for (int i = 0; i < m->nq; ++i) f.ctx.ghost_data->qpos[i] = qpos[i];
    mj_forward(m, f.ctx.ghost_data);

    const std::vector<mjvGeom> ghosts =
        CollectGhostGeoms(m, f.ctx.ghost_data, f.jid, rgba, 2000);
    // Count == subtree geom count (both arm geoms are primitives, none skipped).
    CHECK(ghosts.size() == subtree.size());

    // Reference forward for pose equality.
    RefForwardAt(f, rk);
    for (std::size_t i = 0; i < ghosts.size(); ++i) {
      const mjvGeom& g = ghosts[i];
      const int gid = subtree[i];
      // Pose equals reference geom_xpos/geom_xmat. mjvGeom stores pos/mat as
      // float (mjv_initGeom narrows), so the pin is exact against the float cast
      // of the reference double -- the editor ghost scratch and the reference are
      // the same forwarded model, hence bitwise-equal doubles before narrowing.
      for (int k = 0; k < 3; ++k) {
        CHECK(g.pos[k] == static_cast<float>(f.rd->geom_xpos[3 * gid + k]));
      }
      for (int k = 0; k < 9; ++k) {
        CHECK(g.mat[k] == static_cast<float>(f.rd->geom_xmat[9 * gid + k]));
      }
      // Transparent decor, excluded from pick/segmentation.
      CHECK(g.rgba[3] < 1.0f);
      CHECK(g.rgba[3] == 0.25f);
      CHECK(g.category == mjCAT_DECOR);
      CHECK(g.objtype == mjOBJ_UNKNOWN);
    }
  }
  std::printf("  ghosts (pose/count/alpha/decor): ok\n");
  return 0;
}

// The load-bearing arc/travel contract, on a hinge (degree), a radian twin, and a
// slide: JointLimitChildPoint(range[k]) == mj_forward(geom_xpos) at qpos=range[k].
int TestArcContractOne(const std::string& xml, const char* tag) {
  Fixture f;
  CHECK(BuildFixture(f, xml));
  const mjModel* m = f.ctx.compiled.model.get();
  mjData* d = f.ctx.sim_data;

  const int refg = PickArcReferenceGeom(m, d, f.jid);
  CHECK(refg >= 0);
  const double q_now = d->qpos[f.qadr];
  double p_ref[3];
  for (int k = 0; k < 3; ++k) p_ref[k] = d->geom_xpos[3 * refg + k];

  for (int end = 0; end < 2; ++end) {
    const double rk = m->jnt_range[2 * f.jid + end];
    double out[3];
    JointLimitChildPoint(f.jtype, d->xanchor + 3 * f.jid, d->xaxis + 3 * f.jid,
                         q_now, rk, p_ref, out);
    RefForwardAt(f, rk);
    const double diff = MaxAbsDiff(out, f.rd->geom_xpos + 3 * refg, 3);
    if (!(diff < 1e-12)) {
      std::fprintf(stderr, "arc[%s] endpoint %d diff=%.3e\n", tag, end, diff);
      return 1;
    }
  }
  std::printf("  arc/travel contract [%s]: ok\n", tag);
  return 0;
}

// --- P2: interactive handles (rig_handles.cc pure core) --------------------- //

constexpr double kPiT = 3.14159265358979323846;

// Recompile the (possibly mutated) tree through the same XmlPath bridge and read
// back the first Joint's compiled jnt_range -- the editor's post-commit compile.
bool RecompileFirstJointRange(mj::Model& tree, double out[2]) {
  mj::CompileOptions opts;
  opts.path = mj::CompilePath::XmlPath;
  mj::Compiled c = mj::Compile(tree, opts);
  if (!c.ok()) return false;
  const mjModel* m = c.model.get();
  for (const mj::Binding::Entry& e : c.binding.entries()) {
    if (e.etype == mj::ElementType::Joint && e.id >= 0) {
      out[0] = m->jnt_range[2 * e.id + 0];
      out[1] = m->jnt_range[2 * e.id + 1];
      return true;
    }
  }
  return false;
}

// Cursor->dof mapping: a synthetic camera ray whose plane/axis intersection is a
// CONSTRUCTED point at a known angle / axial position must map back to it.
int TestMappingFunctions() {
  // Hinge: axis +Y through origin, reference direction +X (u). A point at angle
  // theta in the (u, v=xaxis x u) plane, hit by a ray straight down the axis.
  const double anchor[3] = {0, 0, 0};
  const double yaxis[3] = {0, 1, 0};
  const double p_ref[3] = {1, 0, 0};
  for (double theta : {-0.9, -0.3, 0.0, 0.4, 1.1}) {
    // Construct the hit point via the pinned contract, then a ray onto it.
    double hit[3];
    JointLimitChildPoint(mjJNT_HINGE, anchor, yaxis, /*q_now=*/0.0, theta, p_ref,
                         hit);
    double ray_d[3] = {0, -1, 0};  // down the -axis, so it crosses the y=0 plane
    double ray_o[3] = {hit[0] - ray_d[0] * 5, hit[1] - ray_d[1] * 5,
                       hit[2] - ray_d[2] * 5};
    const double q = HingeDofFromRay(anchor, yaxis, p_ref, 0.0, ray_o, ray_d);
    CHECK(std::fabs(q - theta) < 1e-12);
  }
  // Slide: axis +X through the origin, p_ref at the anchor (s0=0); a ray crossing
  // the axis at x=s maps to q = s.
  const double xaxis[3] = {1, 0, 0};
  const double s_ref[3] = {0, 0, 0};
  const double origin[3] = {0, 0, 0};
  for (double s : {-0.4, 0.0, 0.25, 0.7}) {
    double ray_o[3] = {s, 2, 0};
    double ray_d[3] = {0, -1, 0};
    const double q = SlideDofFromRay(origin, xaxis, s_ref, 0.0, ray_o, ray_d);
    CHECK(std::fabs(q - s) < 1e-12);
  }
  // Snap: rounds to the step when on, identity when off.
  CHECK(SnapDof(mjJNT_HINGE, 0.20, false, 0.1, 0.1) == 0.20);
  CHECK(std::fabs(SnapDof(mjJNT_HINGE, 0.23, true, 0.1, 0.05) - 0.20) < 1e-12);
  CHECK(std::fabs(SnapDof(mjJNT_SLIDE, 0.23, true, 0.1, 0.05) - 0.25) < 1e-12);
  std::printf("  cursor->dof mapping + snap: ok\n");
  return 0;
}

// The commit round-trip (plan §3 P2): drag an endpoint to a radian/metre target,
// write the authored field through the ONE conversion (DofToAuthored), recompile
// -> m->jnt_range equals the target (no double-convert); undo restores authored.
int TestCommitRoundTripOne(const std::string& xml, const char* tag, double target,
                           int endpoint) {
  Fixture f;
  CHECK(BuildFixture(f, xml));
  mjModel* m = f.ctx.compiled.model.get();
  const int type = f.jtype;
  const bool deg = AngleIsDegree(*f.ctx.tree);
  mj::Joint* j = FindSerialAs<mj::Joint>(*f.ctx.tree, f.jserial);
  CHECK(j != nullptr && j->range.has_value());
  const std::array<double, 2> orig_auth = *j->range;

  // Gesture: BeginEdit, preview-patch the compiled range (authored untouched),
  // then on release write the authored field via the single conversion + commit.
  f.ctx.BeginEdit();
  m->jnt_range[2 * f.jid + endpoint] = target;  // live preview (compiled-only)
  std::array<double, 2> auth = *j->range;
  auth[endpoint] = DofToAuthored(type, m->jnt_range[2 * f.jid + endpoint], deg);
  j->range = auth;
  f.ctx.CommitEdit("joint range");
  CHECK(f.ctx.dirty);
  CHECK(f.ctx.recompile_requested);
  CHECK(f.ctx.history.can_undo());
  // The non-dragged endpoint keeps its exact authored number (no round-trip).
  CHECK((*j->range)[1 - endpoint] == orig_auth[1 - endpoint]);

  // Recompile: the dragged endpoint's compiled value equals the drag target.
  double rr[2];
  CHECK(RecompileFirstJointRange(*f.ctx.tree, rr));
  if (!(std::fabs(rr[endpoint] - target) < 1e-9)) {
    std::fprintf(stderr, "commit[%s] endpoint %d: got %.12g want %.12g\n", tag,
                 endpoint, rr[endpoint], target);
    return 1;
  }

  // Undo restores the prior authored range exactly.
  f.ctx.tree = f.ctx.history.Undo(std::move(f.ctx.tree));
  mj::Joint* j2 = FindSerialAs<mj::Joint>(*f.ctx.tree, f.jserial);
  CHECK(j2 != nullptr && j2->range.has_value());
  CHECK((*j2->range)[0] == orig_auth[0] && (*j2->range)[1] == orig_auth[1]);
  std::printf("  commit round-trip + undo [%s]: ok\n", tag);
  return 0;
}

// Esc mid-drag: the compiled preview patch is restored and the authored tree is
// untouched (the drag never mutated it), with no undo step recorded.
int TestEndpointCancel() {
  Fixture f;
  CHECK(BuildFixture(f, kDegHinge));
  mjModel* m = f.ctx.compiled.model.get();
  mj::Joint* j = FindSerialAs<mj::Joint>(*f.ctx.tree, f.jserial);
  CHECK(j != nullptr && j->range.has_value());
  const std::array<double, 2> orig_auth = *j->range;
  const double g0 = m->jnt_range[2 * f.jid + 0], g1 = m->jnt_range[2 * f.jid + 1];

  f.ctx.BeginEdit();
  m->jnt_range[2 * f.jid + 1] = 1.2;  // live preview
  // Esc cancel: restore the compiled patch + deferred CancelEdit (no undo step).
  m->jnt_range[2 * f.jid + 0] = g0;
  m->jnt_range[2 * f.jid + 1] = g1;
  f.ctx.CancelEdit();
  CHECK(m->jnt_range[2 * f.jid + 1] == g1);
  CHECK((*j->range)[0] == orig_auth[0] && (*j->range)[1] == orig_auth[1]);
  CHECK(f.ctx.history.can_undo() == false);
  std::printf("  endpoint cancel restores state: ok\n");
  return 0;
}

// Slider scrub and limb-drag scrub are the SAME qpos write: a limb-drag maps the
// cursor to q via JointDofFromRay, a slider hands q straight to SetJointPreview
// -- so both produce a bitwise-identical mirrored pose for the same value.
int TestSliderLimbEquivalence() {
  Fixture f;
  CHECK(BuildFixture(f, kDegHinge));
  mjModel* m = f.ctx.compiled.model.get();
  mjData* d = f.ctx.sim_data;
  const int refg = PickArcReferenceGeom(m, d, f.jid);
  CHECK(refg >= 0);
  double p_ref[3];
  for (int k = 0; k < 3; ++k) p_ref[k] = d->geom_xpos[3 * refg + k];
  const double q_now = d->qpos[f.qadr];
  const double target = 15.0 * kPiT / 180.0;

  // Limb-drag: construct a ray onto where p_ref lands at q=target, map it back.
  double hit[3];
  JointLimitChildPoint(mjJNT_HINGE, d->xanchor + 3 * f.jid, d->xaxis + 3 * f.jid,
                       q_now, target, p_ref, hit);
  double ray_d[3], ray_o[3];
  for (int k = 0; k < 3; ++k) {
    ray_d[k] = -d->xaxis[3 * f.jid + k];
    ray_o[k] = hit[k] - ray_d[k] * 5.0;
  }
  const double q_mapped =
      JointDofFromRay(mjJNT_HINGE, d->xanchor + 3 * f.jid, d->xaxis + 3 * f.jid,
                      p_ref, q_now, ray_o, ray_d);
  CHECK(std::fabs(q_mapped - target) < 1e-9);

  // Limb-scrub pose at the mapped q.
  CHECK(SetJointPreview(f.ctx, f.jserial, q_mapped));
  std::vector<double> limb_x(d->geom_xpos, d->geom_xpos + 3 * m->ngeom);
  std::vector<double> limb_q(d->qpos, d->qpos + m->nq);
  // Slider pose at the SAME value (independent reset inside SetJointPreview).
  CHECK(SetJointPreview(f.ctx, f.jserial, q_mapped));
  CHECK(MaxAbsDiff(limb_x.data(), d->geom_xpos, 3 * m->ngeom) == 0.0);
  CHECK(MaxAbsDiff(limb_q.data(), d->qpos, m->nq) == 0.0);
  std::printf("  slider == limb-scrub pose: ok\n");
  return 0;
}

// --- P3: ball parametrization + swing cone ---------------------------------- //

int TestBallParametrization() {
  Fixture f;
  CHECK(BuildFixture(f, kBall));
  const mjModel* m = f.ctx.compiled.model.get();
  mjData* d = f.ctx.sim_data;
  CHECK(f.jtype == mjJNT_BALL);
  CHECK(m->jnt_limited[f.jid] == 1);

  // Single-axis Euler == the pure axis-angle quaternion (documents axis order).
  const double kPi = 3.14159265358979323846;
  const double th = 0.5;
  const double axes[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  for (int a = 0; a < 3; ++a) {
    double e[3] = {0, 0, 0};
    e[a] = th;
    double got[4], want[4];
    BallQuatFromEuler(e, got);
    mju_axisAngle2Quat(want, axes[a], th);
    CHECK(MaxAbsDiff(got, want, 4) < 1e-12);
  }
  // Intrinsic XYZ == the qx (x) qy (x) qz product (regression pin of the order).
  {
    const double e[3] = {0.3, -0.4, 0.7};
    double qx[4], qy[4], qz[4], t[4], want[4], got[4];
    mju_axisAngle2Quat(qx, axes[0], e[0]);
    mju_axisAngle2Quat(qy, axes[1], e[1]);
    mju_axisAngle2Quat(qz, axes[2], e[2]);
    mju_mulQuat(t, qx, qy);
    mju_mulQuat(want, t, qz);
    BallQuatFromEuler(e, got);
    CHECK(MaxAbsDiff(got, want, 4) < 1e-12);
  }

  // Scrub pose: SetBallPreview writes the composed quat; the tip lands where an
  // independent mj_forward at the same qpos puts it (bitwise), AND -- for a pure
  // X swing, the un-rotated body's world axis == local X -- where the SAME pinned
  // JointLimitChildPoint contract predicts (ties ball scrub to the P1 pin).
  const int refg = PickArcReferenceGeom(m, d, f.jid);
  CHECK(refg >= 0);
  double p_ref0[3];
  for (int k = 0; k < 3; ++k) p_ref0[k] = d->geom_xpos[3 * refg + k];  // qpos0 tip
  const double xaxis[3] = {1, 0, 0};
  const double eul[3] = {0.4, 0, 0};
  CHECK(SetBallPreview(f.ctx, f.jserial, eul));
  CHECK(f.ctx.rig_preview.active);
  CHECK(f.ctx.rig_preview.overlay.count(f.jserial) == 1);
  // Independent reference forward at the exact composed qpos.
  mj_resetData(f.ref, f.rd);
  const int qadr = m->jnt_qposadr[f.jid];
  for (int k = 0; k < 4; ++k) f.rd->qpos[qadr + k] = d->qpos[qadr + k];
  mj_forward(f.ref, f.rd);
  CHECK(MaxAbsDiff(d->geom_xpos, f.rd->geom_xpos, 3 * m->ngeom) == 0.0);
  // Pinned contract: pure X swing rotates the tip about world X through anchor.
  double want[3];
  JointLimitChildPoint(mjJNT_HINGE, d->xanchor + 3 * f.jid, xaxis, 0.0, 0.4,
                       p_ref0, want);
  CHECK(MaxAbsDiff(f.rd->geom_xpos + 3 * refg, want, 3) < 1e-12);
  ClearJointPreview(f.ctx);
  CHECK(!f.ctx.rig_preview.active);
  CHECK(MaxAbsDiff(d->qpos, m->qpos0, m->nq) == 0.0);

  std::printf("  ball parametrization (Euler XYZ -> quat -> pose): ok\n");
  return 0;
}

int TestBallSwingCone() {
  Fixture f;
  CHECK(BuildFixture(f, kBall));
  const mjModel* m = f.ctx.compiled.model.get();
  mjData* d = f.ctx.sim_data;
  const int refg = PickArcReferenceGeom(m, d, f.jid);
  CHECK(refg >= 0);
  const int qadr = m->jnt_qposadr[f.jid];
  const double theta = m->jnt_range[2 * f.jid + 1];  // 60deg in rad
  double q0[4];
  for (int k = 0; k < 4; ++k) q0[k] = m->qpos0[qadr + k];

  // For sampled local axes in the XY plane, the swing qpos is a REAL limit pose:
  // the drawn rim point (scratch geom_xpos) equals an independent reference
  // forward (surface pinned), and the ball's rotation magnitude == range[1].
  const double kPi = 3.14159265358979323846;
  constexpr int K = 16;
  for (int s = 0; s < K; ++s) {
    const double phi = 2.0 * kPi * s / K;
    const double axis_local[3] = {std::cos(phi), std::sin(phi), 0.0};
    std::vector<double> qpos;
    BallSwingQpos(m, f.jid, axis_local, theta, qpos);
    CHECK(static_cast<int>(qpos.size()) == m->nq);
    // Editor scratch (the drawn surface source) at this qpos.
    mj_resetData(m, f.ctx.ghost_data);
    for (int i = 0; i < m->nq; ++i) f.ctx.ghost_data->qpos[i] = qpos[i];
    mj_forward(m, f.ctx.ghost_data);
    // Independent reference at the identical qpos.
    mj_resetData(f.ref, f.rd);
    for (int i = 0; i < m->nq; ++i) f.rd->qpos[i] = qpos[i];
    mj_forward(f.ref, f.rd);
    const double diff = MaxAbsDiff(f.ctx.ghost_data->geom_xpos + 3 * refg,
                                   f.rd->geom_xpos + 3 * refg, 3);
    CHECK(diff < 1e-12);
    // Every rim pose is exactly at the swing limit.
    double qb[4];
    for (int k = 0; k < 4; ++k) qb[k] = qpos[qadr + k];
    CHECK(std::fabs(BallSwingAngle(q0, qb) - theta) < 1e-9);
  }
  std::printf("  ball swing cone (rim == mj_forward at limit): ok\n");
  return 0;
}

// --- P3: multi-joint held composition + keyframe capture -------------------- //

int TestHeldComposition() {
  Fixture f;
  CHECK(BuildFixture(f, kTwoHinge));
  const mjModel* m = f.ctx.compiled.model.get();
  mjData* d = f.ctx.sim_data;
  CHECK(m->nq == 2);
  std::uint64_t s_sh, s_el;
  int j_sh, j_el, q_sh, q_el;
  CHECK(JointByName(f, "shoulder", &s_sh, &j_sh, &q_sh));
  CHECK(JointByName(f, "elbow", &s_el, &j_el, &q_el));

  const double kPi = 3.14159265358979323846;
  const double qs = 35.0 * kPi / 180.0, qe = -50.0 * kPi / 180.0;
  // Scrub the shoulder, then the elbow: the two COMPOSE (elbow no longer rebases
  // the shoulder, the P1 single-slot behavior superseded).
  CHECK(SetJointPreview(f.ctx, s_sh, qs));
  CHECK(SetJointPreview(f.ctx, s_el, qe));
  CHECK(f.ctx.rig_preview.overlay.size() == 2);
  CHECK(f.ctx.rig_preview.active);
  CHECK(std::fabs(d->qpos[q_sh] - qs) < 1e-15);
  CHECK(std::fabs(d->qpos[q_el] - qe) < 1e-15);

  // The composed pose == an independent forward at qpos0-with-both-dofs-set.
  mj_resetData(f.ref, f.rd);
  f.rd->qpos[q_sh] = qs;
  f.rd->qpos[q_el] = qe;
  mj_forward(f.ref, f.rd);
  CHECK(MaxAbsDiff(d->geom_xpos, f.rd->geom_xpos, 3 * m->ngeom) == 0.0);
  CHECK(MaxAbsDiff(d->xquat, f.rd->xquat, 4 * m->nbody) == 0.0);

  // RemoveJointPreview(elbow) snaps ONLY the elbow back; the shoulder stays held.
  RemoveJointPreview(f.ctx, s_el);
  CHECK(f.ctx.rig_preview.overlay.size() == 1);
  CHECK(std::fabs(d->qpos[q_sh] - qs) < 1e-15);
  CHECK(std::fabs(d->qpos[q_el] - m->qpos0[q_el]) < 1e-15);

  std::printf("  multi-joint held composition: ok\n");
  return 0;
}

int TestKeyframeRoundTrip() {
  Fixture f;
  CHECK(BuildFixture(f, kTwoHinge));
  const mjModel* m = f.ctx.compiled.model.get();
  mjData* d = f.ctx.sim_data;
  std::uint64_t s_sh, s_el;
  int j_sh, j_el, q_sh, q_el;
  CHECK(JointByName(f, "shoulder", &s_sh, &j_sh, &q_sh));
  CHECK(JointByName(f, "elbow", &s_el, &j_el, &q_el));

  const double kPi = 3.14159265358979323846;
  CHECK(SetJointPreview(f.ctx, s_sh, 20.0 * kPi / 180.0));
  CHECK(SetJointPreview(f.ctx, s_el, -75.0 * kPi / 180.0));
  // The held qpos captured verbatim (the core of CaptureKeyframeOp, minus the
  // ImGui-side DoAdd wiring): write sim_data->qpos into a new <key>.
  const std::vector<double> held(d->qpos, d->qpos + m->nq);
  mj::Key& key = ps::sdk::AddKey(*f.ctx.tree, "pose0");
  key.qpos = held;

  // Recompile through the same bridge: m->key_qpos == the held preview qpos.
  mj::CompileOptions opts;
  opts.path = mj::CompilePath::XmlPath;
  mj::Compiled c = mj::Compile(*f.ctx.tree, opts);
  CHECK(c.ok());
  const mjModel* mk = c.model.get();
  CHECK(mk->nkey >= 1);
  CHECK(mk->nq == static_cast<int>(held.size()));
  double maxd = 0.0;
  for (int i = 0; i < mk->nq; ++i) {
    maxd = std::max(maxd, std::fabs(mk->key_qpos[i] - held[i]));
  }
  // Tolerance covers mjs/xml float formatting on the qpos text round-trip.
  if (!(maxd < 1e-6)) {
    std::fprintf(stderr, "keyframe qpos round-trip diff=%.3e\n", maxd);
    return 1;
  }
  std::printf("  keyframe capture round-trip (key_qpos == held): ok\n");
  return 0;
}

// --- P3: springref ghost + driver badges ------------------------------------ //

int TestSpringrefGhost() {
  Fixture f;
  CHECK(BuildFixture(f, kSpring));
  const mjModel* m = f.ctx.compiled.model.get();
  const int qadr = m->jnt_qposadr[f.jid];
  CHECK(m->jnt_stiffness[f.jid] > 0.0);
  // qpos_spring differs from qpos0 (springref=30deg).
  CHECK(std::fabs(m->qpos_spring[qadr] - m->qpos0[qadr]) > 1e-6);
  const double kPi = 3.14159265358979323846;
  CHECK(std::fabs(m->qpos_spring[qadr] - 30.0 * kPi / 180.0) < 1e-9);

  // The spring ghost is the subtree pose at qpos_spring -- pinned like a range
  // ghost against an independent reference forward.
  std::vector<double> qpos;
  GhostQpos(m, f.jid, m->qpos_spring[qadr], qpos);
  mj_resetData(m, f.ctx.ghost_data);
  for (int i = 0; i < m->nq; ++i) f.ctx.ghost_data->qpos[i] = qpos[i];
  mj_forward(m, f.ctx.ghost_data);
  const float tint[4] = {0.85f, 0.35f, 0.95f, 0.28f};
  const std::vector<mjvGeom> ghosts =
      CollectGhostGeoms(m, f.ctx.ghost_data, f.jid, tint, 2000);
  const std::vector<int> subtree = SubtreeGeoms(m, f.jid);
  CHECK(ghosts.size() == subtree.size());
  RefForwardAt(f, m->qpos_spring[qadr]);
  for (std::size_t i = 0; i < ghosts.size(); ++i) {
    const int gid = subtree[i];
    for (int k = 0; k < 3; ++k) {
      CHECK(ghosts[i].pos[k] == static_cast<float>(f.rd->geom_xpos[3 * gid + k]));
    }
    CHECK(ghosts[i].rgba[3] < 1.0f);
    CHECK(ghosts[i].category == mjCAT_DECOR);
  }
  std::printf("  springref ghost pose: ok\n");
  return 0;
}

int TestDriverBadges() {
  Fixture f;
  CHECK(BuildFixture(f, kDriven));
  const mjModel* m = f.ctx.compiled.model.get();
  CHECK(m->nu == 1 && m->ntendon == 1);
  const std::vector<JointDriver> drivers =
      CollectJointDrivers(m, f.ctx.compiled.binding, f.jid);
  CHECK(drivers.size() == 2);
  bool saw_act = false, saw_ten = false;
  for (const JointDriver& drv : drivers) {
    if (!drv.is_tendon) {
      saw_act = true;
      CHECK(drv.objtype == mjOBJ_ACTUATOR);
      CHECK(drv.name == "act0");
      CHECK(drv.serial != 0);  // resolved through the Binding reverse mapping
      // The resolved serial really is the actuator (a click would select it).
      CHECK(ps::sdk::FindBySerial(*f.ctx.tree, drv.serial) != nullptr);
    } else {
      saw_ten = true;
      CHECK(drv.objtype == mjOBJ_TENDON);
      CHECK(drv.name == "ten0");
      CHECK(drv.serial != 0);
    }
  }
  CHECK(saw_act && saw_ten);

  // A joint driven by nothing reports no drivers.
  Fixture f2;
  CHECK(BuildFixture(f2, kDegHinge));
  CHECK(CollectJointDrivers(f2.ctx.compiled.model.get(),
                            f2.ctx.compiled.binding, f2.jid)
            .empty());
  std::printf("  actuator/tendon badge resolution: ok\n");
  return 0;
}

int RunTests() {
  if (TestUnitsAndLimited()) return 1;
  if (TestScrubAndSnapback()) return 1;
  if (TestGhosts()) return 1;
  if (TestArcContractOne(kDegHinge, "deg-hinge")) return 1;
  if (TestArcContractOne(kRadHinge, "rad-hinge")) return 1;
  if (TestArcContractOne(kSlide, "slide")) return 1;
  if (TestMappingFunctions()) return 1;
  if (TestCommitRoundTripOne(kDegHinge, "deg-hinge", 60.0 * kPiT / 180.0, 1))
    return 1;
  if (TestCommitRoundTripOne(kRadHinge, "rad-hinge", 60.0 * kPiT / 180.0, 1))
    return 1;
  if (TestCommitRoundTripOne(kSlide, "slide", 0.4, 1)) return 1;
  if (TestEndpointCancel()) return 1;
  if (TestSliderLimbEquivalence()) return 1;
  // P3.
  if (TestBallParametrization()) return 1;
  if (TestBallSwingCone()) return 1;
  if (TestHeldComposition()) return 1;
  if (TestKeyframeRoundTrip()) return 1;
  if (TestSpringrefGhost()) return 1;
  if (TestDriverBadges()) return 1;
  std::printf("rigger_windowless: all checks passed\n");
  return 0;
}

}  // namespace
}  // namespace ps::studio

int main() { return ps::studio::RunTests(); }
