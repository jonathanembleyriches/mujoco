// ProtoSpec Studio: interactive joint rigger core. See joint_rig.h.

#include "editor/joint_rig.h"

#include <cmath>

namespace ps::studio {

namespace {

constexpr double kPi = 3.14159265358979323846;

// A right-handed rotation of a vector about a unit axis (Rodrigues). Local so the
// rigger core needs no editor-lib link (transform_math.cc) in the windowless test.
void RotateAboutAxis(const double axis[3], double angle, const double v[3],
                     double out[3]) {
  const double c = std::cos(angle), s = std::sin(angle);
  const double dot = axis[0] * v[0] + axis[1] * v[1] + axis[2] * v[2];
  const double cross[3] = {axis[1] * v[2] - axis[2] * v[1],
                           axis[2] * v[0] - axis[0] * v[2],
                           axis[0] * v[1] - axis[1] * v[0]};
  for (int k = 0; k < 3; ++k) {
    out[k] = v[k] * c + cross[k] * s + axis[k] * dot * (1.0 - c);
  }
}

// A subtree mask: bodies whose parent chain passes through `root` (root itself
// included). Body ids are topologically ordered (parent id < child id).
std::vector<char> SubtreeMask(const mjModel* m, int root) {
  std::vector<char> in(m->nbody, 0);
  if (root < 0 || root >= m->nbody) return in;
  in[root] = 1;
  for (int b = root + 1; b < m->nbody; ++b) {
    const int par = m->body_parentid[b];
    if (par >= 0 && par < m->nbody && in[par]) in[b] = 1;
  }
  return in;
}

// mjvGeom decor hygiene: keep the geom out of pick/segmentation passes.
void MarkDecor(mjvGeom* g) {
  g->category = mjCAT_DECOR;
  g->objtype = mjOBJ_UNKNOWN;
  g->objid = -1;
  g->segid = -1;
}

}  // namespace

// --- Units ------------------------------------------------------------------ //

double JointDofToDisplay(int jnt_type, double compiled, bool angle_is_degree) {
  if (jnt_type == mjJNT_SLIDE) return compiled;  // metres, never converted
  return angle_is_degree ? compiled * 180.0 / kPi : compiled;
}

double JointDisplayToDof(int jnt_type, double display, bool angle_is_degree) {
  if (jnt_type == mjJNT_SLIDE) return display;
  return angle_is_degree ? display * kPi / 180.0 : display;
}

bool AngleIsDegree(const mj::Model& tree) {
  bool degree = true;  // MuJoCo default
  for (const auto& c : tree.compilers) {
    if (c && c->angle) degree = (*c->angle == mj::AngleUnit::degree);
  }
  return degree;
}

// --- Selection / scrub preview ---------------------------------------------- //

int JointIdForSerial(const mj::Binding& binding, std::uint64_t serial) {
  if (serial == 0) return -1;
  for (const mj::Binding::Entry& e : binding.entries()) {
    if (e.serial == serial && e.id >= 0 &&
        (e.etype == mj::ElementType::Joint ||
         e.etype == mj::ElementType::FreeJoint)) {
      return e.id;
    }
  }
  return -1;
}

void BallQuatFromEuler(const double euler[3], double quat[4]) {
  // Intrinsic XYZ: rotate about body X, then the new Y, then the new Z. As
  // quaternions this is the plain left-to-right product qx ⊗ qy ⊗ qz (each factor
  // about the fixed body axis; the intrinsic sequencing IS this multiplication
  // order). Scalar-first, MuJoCo's qpos ball convention.
  const double ax[3] = {1, 0, 0}, ay[3] = {0, 1, 0}, az[3] = {0, 0, 1};
  double qx[4], qy[4], qz[4], t[4];
  mju_axisAngle2Quat(qx, ax, euler[0]);
  mju_axisAngle2Quat(qy, ay, euler[1]);
  mju_axisAngle2Quat(qz, az, euler[2]);
  mju_mulQuat(t, qx, qy);
  mju_mulQuat(quat, t, qz);
  mju_normalize4(quat);
}

// Reset ctx.sim_data to qpos0 then write every held overlay value on top and
// mj_forward, so scrubbed joints compose while unscrubbed ones stay at qpos0.
// active == the overlay is non-empty. Pure preview: no authored-state change.
void ApplyRigOverlay(EditorContext& ctx) {
  mjModel* m = ctx.compiled.model.get();
  mjData* d = ctx.sim_data;
  if (!m || !d) return;
  mj_resetData(m, d);  // qpos0 baseline; the overlay composes on top of it
  for (const auto& [serial, dof] : ctx.rig_preview.overlay) {
    const int jid = JointIdForSerial(ctx.compiled.binding, serial);
    if (jid < 0 || jid >= m->njnt) continue;
    const int adr = m->jnt_qposadr[jid];
    const int type = m->jnt_type[jid];
    if (type == mjJNT_BALL && dof.ball) {
      double delta[4], q0[4], quat[4];
      BallQuatFromEuler(dof.v, delta);
      for (int k = 0; k < 4; ++k) q0[k] = m->qpos0[adr + k];
      mju_mulQuat(quat, q0, delta);  // qpos0 orientation, then the local swing
      mju_normalize4(quat);
      for (int k = 0; k < 4; ++k) d->qpos[adr + k] = quat[k];
    } else if (type == mjJNT_HINGE || type == mjJNT_SLIDE) {
      d->qpos[adr] = dof.v[0];
    }
  }
  mj_forward(m, d);
  ctx.rig_preview.active = !ctx.rig_preview.overlay.empty();
}

bool SetJointPreview(EditorContext& ctx, std::uint64_t serial, double q) {
  mjModel* m = ctx.compiled.model.get();
  if (!m || !ctx.sim_data) return false;
  const int jid = JointIdForSerial(ctx.compiled.binding, serial);
  if (jid < 0 || jid >= m->njnt) return false;
  const int type = m->jnt_type[jid];
  if (type != mjJNT_HINGE && type != mjJNT_SLIDE) return false;  // ball: SetBall

  RigDof& dof = ctx.rig_preview.overlay[serial];
  dof.ball = false;
  dof.v[0] = q;
  ctx.rig_preview.serial = serial;
  ctx.rig_preview.q = q;
  ApplyRigOverlay(ctx);
  return true;
}

bool SetBallPreview(EditorContext& ctx, std::uint64_t serial,
                    const double euler[3]) {
  mjModel* m = ctx.compiled.model.get();
  if (!m || !ctx.sim_data) return false;
  const int jid = JointIdForSerial(ctx.compiled.binding, serial);
  if (jid < 0 || jid >= m->njnt) return false;
  if (m->jnt_type[jid] != mjJNT_BALL) return false;

  RigDof& dof = ctx.rig_preview.overlay[serial];
  dof.ball = true;
  for (int k = 0; k < 3; ++k) dof.v[k] = euler[k];
  ctx.rig_preview.serial = serial;
  ApplyRigOverlay(ctx);
  return true;
}

void RemoveJointPreview(EditorContext& ctx, std::uint64_t serial) {
  auto it = ctx.rig_preview.overlay.find(serial);
  if (it == ctx.rig_preview.overlay.end()) return;
  ctx.rig_preview.overlay.erase(it);
  if (ctx.rig_preview.serial == serial) ctx.rig_preview.serial = 0;
  ApplyRigOverlay(ctx);  // re-pose the survivors (or snap all back if now empty)
}

void ClearJointPreview(EditorContext& ctx) {
  ctx.rig_preview.overlay.clear();
  ctx.rig_preview.active = false;
  ctx.rig_preview.serial = 0;
  mjModel* m = ctx.compiled.model.get();
  mjData* d = ctx.sim_data;
  if (m && d) {
    mj_resetData(m, d);  // snap back to qpos0
    mj_forward(m, d);
  }
}

void BallSwingQpos(const mjModel* m, int jid, const double axis_local[3],
                   double angle, std::vector<double>& out) {
  out.assign(m->qpos0, m->qpos0 + m->nq);
  if (jid < 0 || jid >= m->njnt || m->jnt_type[jid] != mjJNT_BALL) return;
  const int adr = m->jnt_qposadr[jid];
  double delta[4], q0[4], quat[4];
  mju_axisAngle2Quat(delta, axis_local, angle);
  for (int k = 0; k < 4; ++k) q0[k] = m->qpos0[adr + k];
  mju_mulQuat(quat, q0, delta);
  mju_normalize4(quat);
  for (int k = 0; k < 4; ++k) out[adr + k] = quat[k];
}

// --- Limit ghosts ----------------------------------------------------------- //

std::vector<int> SubtreeGeoms(const mjModel* m, int jid) {
  std::vector<int> out;
  if (!m || jid < 0 || jid >= m->njnt) return out;
  const std::vector<char> in = SubtreeMask(m, m->jnt_bodyid[jid]);
  for (int g = 0; g < m->ngeom; ++g) {
    const int b = m->geom_bodyid[g];
    if (b >= 0 && b < m->nbody && in[b]) out.push_back(g);
  }
  return out;
}

void GhostQpos(const mjModel* m, int jid, double q_limit,
               std::vector<double>& out) {
  out.assign(m->qpos0, m->qpos0 + m->nq);
  if (jid >= 0 && jid < m->njnt) out[m->jnt_qposadr[jid]] = q_limit;
}

std::vector<mjvGeom> CollectGhostGeoms(const mjModel* m, const mjData* scratch_d,
                                       int jid, const float rgba[4],
                                       int max_geoms) {
  std::vector<mjvGeom> out;
  if (!m || !scratch_d) return out;
  for (int g : SubtreeGeoms(m, jid)) {
    const int gt = m->geom_type[g];
    // Exotic families skipped in P1 (plan §2b): plane/hfield/sdf.
    if (gt == mjGEOM_PLANE || gt == mjGEOM_HFIELD || gt == mjGEOM_SDF) continue;
    if (static_cast<int>(out.size()) >= max_geoms) break;
    mjvGeom geom;
    mjv_initGeom(&geom, gt, m->geom_size + 3 * g, scratch_d->geom_xpos + 3 * g,
                 scratch_d->geom_xmat + 9 * g, rgba);
    // Mesh dataid re-encoding: the visualizer keys the original mesh at 2*id
    // (2*id+1 is the convex hull); mjv_initGeom left dataid = geom_dataid.
    if (gt == mjGEOM_MESH && m->geom_dataid[g] >= 0) {
      geom.dataid = 2 * m->geom_dataid[g];
    }
    MarkDecor(&geom);
    out.push_back(geom);
  }
  return out;
}

// --- Depth-occluded glyph + arc contract ------------------------------------ //

void JointLimitChildPoint(int jnt_type, const double xanchor[3],
                          const double xaxis[3], double q_now, double q_limit,
                          const double p_ref[3], double out[3]) {
  const double dq = q_limit - q_now;
  if (jnt_type == mjJNT_SLIDE) {
    for (int k = 0; k < 3; ++k) out[k] = p_ref[k] + xaxis[k] * dq;
    return;
  }
  // Hinge (and, by rigid-rotation geometry, ball about a single axis): rotate the
  // reference offset about the world axis through the anchor by the dof delta.
  double rel[3];
  for (int k = 0; k < 3; ++k) rel[k] = p_ref[k] - xanchor[k];
  double rot[3];
  RotateAboutAxis(xaxis, dq, rel, rot);
  for (int k = 0; k < 3; ++k) out[k] = xanchor[k] + rot[k];
}

int PickArcReferenceGeom(const mjModel* m, const mjData* d, int jid) {
  if (!m || !d || jid < 0 || jid >= m->njnt) return -1;
  const double* anchor = d->xanchor + 3 * jid;
  const double* axis = d->xaxis + 3 * jid;
  int best = -1;
  double best_r2 = 0.0;
  for (int g : SubtreeGeoms(m, jid)) {
    double rel[3];
    for (int k = 0; k < 3; ++k) rel[k] = d->geom_xpos[3 * g + k] - anchor[k];
    const double along = rel[0] * axis[0] + rel[1] * axis[1] + rel[2] * axis[2];
    double perp2 = 0.0;
    for (int k = 0; k < 3; ++k) {
      const double p = rel[k] - along * axis[k];
      perp2 += p * p;
    }
    if (perp2 > best_r2) {
      best_r2 = perp2;
      best = g;
    }
  }
  // All subtree geoms sit on the axis (a slide's child, or a degenerate hinge):
  // radius is irrelevant for a slide (pure translation) and the fan collapses for
  // a hinge, but any subtree geom is still a valid contract reference -- take the
  // first so the pin holds and the travel line has an endpoint.
  if (best < 0) {
    const std::vector<int> geoms = SubtreeGeoms(m, jid);
    if (!geoms.empty()) best = geoms.front();
  }
  return best;
}

std::vector<mjvGeom> CollectJointGlyph(const mjModel* m, const mjData* d, int jid,
                                       int max_geoms) {
  std::vector<mjvGeom> out;
  if (!m || !d || jid < 0 || jid >= m->njnt) return out;
  const int type = m->jnt_type[jid];
  if (type == mjJNT_FREE) return out;  // nothing to rig

  double extent = m->stat.extent;
  if (!(extent > 1e-6)) extent = 1.0;
  const double len = 0.15 * extent;
  const double aw = 0.006 * extent;  // arrow shaft width
  const float col_axis[4] = {1.0f, 0.92f, 0.15f, 1.0f};   // selected-joint yellow
  const float col_arc[4] = {0.20f, 0.90f, 0.45f, 1.0f};   // hinge green
  const float col_slide[4] = {0.25f, 0.70f, 1.0f, 1.0f};  // slide blue

  const double* anchor = d->xanchor + 3 * jid;
  const double* axis = d->xaxis + 3 * jid;

  auto push = [&](const mjvGeom& g) {
    if (static_cast<int>(out.size()) < max_geoms) out.push_back(g);
  };

  // Anchor sphere.
  {
    const double r = 0.02 * extent;
    const double size[3] = {r, r, r};
    const double ident[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    mjvGeom s;
    mjv_initGeom(&s, mjGEOM_SPHERE, size, anchor, ident, col_axis);
    MarkDecor(&s);
    push(s);
  }

  // Axis arrow (hinge/slide/ball). Hinge/ball: bidirectional; slide: forward only.
  if (type == mjJNT_HINGE || type == mjJNT_SLIDE || type == mjJNT_BALL) {
    double tip[3], tail[3];
    for (int k = 0; k < 3; ++k) {
      tip[k] = anchor[k] + axis[k] * len;
      tail[k] = anchor[k] - axis[k] * len * (type == mjJNT_SLIDE ? 0.0 : 1.0);
    }
    mjvGeom a;
    mjv_initGeom(&a, mjGEOM_ARROW, nullptr, nullptr, nullptr, col_axis);
    mjv_connector(&a, mjGEOM_ARROW, aw, tail, tip);
    MarkDecor(&a);
    push(a);
  }

  // Hinge range ARC / slide travel LINE, pinned to the real child pose.
  if ((type == mjJNT_HINGE || type == mjJNT_SLIDE) && m->jnt_limited[jid]) {
    const double r0 = m->jnt_range[2 * jid + 0];
    const double r1 = m->jnt_range[2 * jid + 1];
    const double q_now = d->qpos[m->jnt_qposadr[jid]];
    const int refg = PickArcReferenceGeom(m, d, jid);
    double p_ref[3];
    if (refg >= 0) {
      for (int k = 0; k < 3; ++k) p_ref[k] = d->geom_xpos[3 * refg + k];
    } else {
      // No off-axis child geom: synthesize a reference at the current pose one
      // arm-length out along an arbitrary perpendicular (angular extent stays
      // correct even if the zero direction is then unpinned).
      double perp[3] = {0, 0, 1};
      if (std::fabs(axis[2]) > 0.9) {
        perp[0] = 1;
        perp[2] = 0;
      }
      double along = perp[0] * axis[0] + perp[1] * axis[1] + perp[2] * axis[2];
      double n2 = 0;
      for (int k = 0; k < 3; ++k) {
        perp[k] -= along * axis[k];
        n2 += perp[k] * perp[k];
      }
      const double n = std::sqrt(n2 > 1e-12 ? n2 : 1.0);
      for (int k = 0; k < 3; ++k) p_ref[k] = anchor[k] + perp[k] / n * len;
    }

    const float* lc = (type == mjJNT_HINGE) ? col_arc : col_slide;
    const float lw = 4.0f;  // mjGEOM_LINE width is in pixels
    if (type == mjJNT_HINGE) {
      // Fan of line segments tracing where p_ref sweeps between the two limits.
      constexpr int kSeg = 24;
      double prev[3];
      JointLimitChildPoint(type, anchor, axis, q_now, r0, p_ref, prev);
      for (int i = 1; i <= kSeg; ++i) {
        const double q = r0 + (r1 - r0) * i / kSeg;
        double cur[3];
        JointLimitChildPoint(type, anchor, axis, q_now, q, p_ref, cur);
        mjvGeom seg;
        mjv_initGeom(&seg, mjGEOM_LINE, nullptr, nullptr, nullptr, lc);
        mjv_connector(&seg, mjGEOM_LINE, lw, prev, cur);
        MarkDecor(&seg);
        push(seg);
        for (int k = 0; k < 3; ++k) prev[k] = cur[k];
      }
      // Radial spokes from anchor to each limit endpoint (which limit is which).
      double e0[3], e1[3];
      JointLimitChildPoint(type, anchor, axis, q_now, r0, p_ref, e0);
      JointLimitChildPoint(type, anchor, axis, q_now, r1, p_ref, e1);
      for (const double* e : {e0, e1}) {
        mjvGeom sp;
        mjv_initGeom(&sp, mjGEOM_LINE, nullptr, nullptr, nullptr, lc);
        mjv_connector(&sp, mjGEOM_LINE, lw, anchor, e);
        MarkDecor(&sp);
        push(sp);
      }
    } else {  // slide travel line between the two limit positions of p_ref
      double e0[3], e1[3];
      JointLimitChildPoint(type, anchor, axis, q_now, r0, p_ref, e0);
      JointLimitChildPoint(type, anchor, axis, q_now, r1, p_ref, e1);
      mjvGeom seg;
      mjv_initGeom(&seg, mjGEOM_LINE, nullptr, nullptr, nullptr, lc);
      mjv_connector(&seg, mjGEOM_LINE, lw, e0, e1);
      MarkDecor(&seg);
      push(seg);
    }
  }
  return out;
}

// --- Actuator / tendon awareness -------------------------------------------- //

namespace {

// The tree serial + compiled name bound to compiled (objtype, id), via the
// Binding's reverse entry list. serial 0 (name empty) when the element compiled
// away or was never in the tree (e.g. a macro-expanded actuator).
void ResolveBound(const mj::Binding& binding, int objtype, int id,
                  std::uint64_t* serial, std::string* name) {
  for (const mj::Binding::Entry& e : binding.entries()) {
    if (e.objtype == objtype && e.id == id) {
      *serial = e.serial;
      *name = e.name;
      return;
    }
  }
}

}  // namespace

std::vector<JointDriver> CollectJointDrivers(const mjModel* m,
                                             const mj::Binding& binding, int jid) {
  std::vector<JointDriver> out;
  if (!m || jid < 0 || jid >= m->njnt) return out;

  // Actuators whose transmission targets this joint (trnid[0] is the joint id for
  // JOINT / JOINTINPARENT transmissions; trnid[1] is unused for those).
  for (int a = 0; a < m->nu; ++a) {
    const int tt = m->actuator_trntype[a];
    if ((tt == mjTRN_JOINT || tt == mjTRN_JOINTINPARENT) &&
        m->actuator_trnid[2 * a] == jid) {
      JointDriver drv;
      drv.objtype = mjOBJ_ACTUATOR;
      drv.id = a;
      drv.is_tendon = false;
      ResolveBound(binding, mjOBJ_ACTUATOR, a, &drv.serial, &drv.name);
      out.push_back(std::move(drv));
    }
  }

  // Tendons that wrap this joint (a mjWRAP_JOINT wrap object naming it).
  for (int t = 0; t < m->ntendon; ++t) {
    const int adr = m->tendon_adr[t];
    const int num = m->tendon_num[t];
    bool wraps = false;
    for (int w = adr; w < adr + num && w < m->nwrap; ++w) {
      if (m->wrap_type[w] == mjWRAP_JOINT && m->wrap_objid[w] == jid) {
        wraps = true;
        break;
      }
    }
    if (wraps) {
      JointDriver drv;
      drv.objtype = mjOBJ_TENDON;
      drv.id = t;
      drv.is_tendon = true;
      ResolveBound(binding, mjOBJ_TENDON, t, &drv.serial, &drv.name);
      out.push_back(std::move(drv));
    }
  }
  return out;
}

}  // namespace ps::studio
