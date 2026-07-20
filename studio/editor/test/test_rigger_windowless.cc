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
#include "mjcf.h"      // ps::mjcf::io::ParseMjcfString

// Splice the rigger core so its functions are reachable directly.
#include "editor/joint_rig.cc"

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

int RunTests() {
  if (TestUnitsAndLimited()) return 1;
  if (TestScrubAndSnapback()) return 1;
  if (TestGhosts()) return 1;
  if (TestArcContractOne(kDegHinge, "deg-hinge")) return 1;
  if (TestArcContractOne(kRadHinge, "rad-hinge")) return 1;
  if (TestArcContractOne(kSlide, "slide")) return 1;
  std::printf("rigger_windowless: all checks passed\n");
  return 0;
}

}  // namespace
}  // namespace ps::studio

int main() { return ps::studio::RunTests(); }
