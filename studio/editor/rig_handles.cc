// ProtoSpec Studio: interactive joint-rig handles. See rig_handles.h.
//
// The file splits at PS_RIG_HANDLES_NO_CONTROLLER: above it the PURE cursor->dof
// mapping (self-contained vector math, no ImGui / gizmo_math link, so the test
// splices just this part); below it the ImGui controller that drives the mapping
// from the viewport. The rigger core (joint_rig.h) supplies the pinned joint
// geometry (JointLimitChildPoint / PickArcReferenceGeom) and the unit helpers.

#include "editor/rig_handles.h"

#include <cmath>

#include <mujoco/mujoco.h>

#include "editor/joint_rig.h"

namespace ps::studio {

namespace {

double Dot(const double a[3], const double b[3]) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
void Cross(const double a[3], const double b[3], double out[3]) {
  out[0] = a[1] * b[2] - a[2] * b[1];
  out[1] = a[2] * b[0] - a[0] * b[2];
  out[2] = a[0] * b[1] - a[1] * b[0];
}

// Intersect ray (o + t*d) with the plane through `po` normal `pn` (unit). Returns
// false when the ray is parallel. Unlike the gizmo's helper this accepts hits at
// any t (the joint plane can sit behind the cursor's near point during a drag).
bool RayPlaneHit(const double o[3], const double d[3], const double po[3],
                 const double pn[3], double hit[3]) {
  const double denom = Dot(d, pn);
  if (std::fabs(denom) < 1e-12) return false;
  double rel[3] = {po[0] - o[0], po[1] - o[1], po[2] - o[2]};
  const double t = Dot(rel, pn) / denom;
  for (int k = 0; k < 3; ++k) hit[k] = o[k] + d[k] * t;
  return true;
}

// Parameter s on the axis line (a0 + s*ad, ad unit) at the point closest to the
// ray (ro + t*rd). Standard two-line closest-approach; false when parallel.
bool ClosestAxisParam(const double a0[3], const double ad[3], const double ro[3],
                      const double rd[3], double* s_out) {
  double r[3] = {a0[0] - ro[0], a0[1] - ro[1], a0[2] - ro[2]};
  const double a = Dot(ad, ad);       // == 1 for a unit axis, kept general
  const double b = Dot(ad, rd);
  const double c = Dot(rd, rd);
  const double d = Dot(ad, r);
  const double e = Dot(rd, r);
  const double denom = a * c - b * b;
  if (std::fabs(denom) < 1e-12) return false;
  *s_out = (b * e - c * d) / denom;
  return true;
}

}  // namespace

// --- Pure cursor->dof mapping ----------------------------------------------- //

RangeEndpoints ComputeRangeEndpoints(const mjModel* m, const mjData* d, int jid) {
  RangeEndpoints re;
  if (!m || !d || jid < 0 || jid >= m->njnt) return re;
  const int type = m->jnt_type[jid];
  if ((type != mjJNT_HINGE && type != mjJNT_SLIDE) || !m->jnt_limited[jid]) {
    return re;
  }
  const double* anchor = d->xanchor + 3 * jid;
  const double* axis = d->xaxis + 3 * jid;
  const double q_now = d->qpos[m->jnt_qposadr[jid]];

  // p_ref: the arc/travel reference child point, exactly as CollectJointGlyph
  // chooses it (joint_rig.cc) so the handles sit on the drawn arc ends. Prefer
  // the farthest-off-axis subtree geom; synthesize one perpendicular arm out
  // when the subtree has no usable off-axis geom.
  const int refg = PickArcReferenceGeom(m, d, jid);
  double p_ref[3];
  if (refg >= 0) {
    for (int k = 0; k < 3; ++k) p_ref[k] = d->geom_xpos[3 * refg + k];
  } else {
    double extent = m->stat.extent;
    if (!(extent > 1e-6)) extent = 1.0;
    const double len = 0.15 * extent;
    double perp[3] = {0, 0, 1};
    if (std::fabs(axis[2]) > 0.9) {
      perp[0] = 1;
      perp[2] = 0;
    }
    const double along = Dot(perp, axis);
    double n2 = 0;
    for (int k = 0; k < 3; ++k) {
      perp[k] -= along * axis[k];
      n2 += perp[k] * perp[k];
    }
    const double n = std::sqrt(n2 > 1e-12 ? n2 : 1.0);
    for (int k = 0; k < 3; ++k) p_ref[k] = anchor[k] + perp[k] / n * len;
  }

  re.valid = true;
  re.jnt_type = type;
  re.q_now = q_now;
  for (int k = 0; k < 3; ++k) re.p_ref[k] = p_ref[k];
  for (int end = 0; end < 2; ++end) {
    JointLimitChildPoint(type, anchor, axis, q_now, m->jnt_range[2 * jid + end],
                         p_ref, re.e[end]);
  }
  return re;
}

double HingeDofFromRay(const double xanchor[3], const double xaxis[3],
                       const double p_ref[3], double q_now, const double ray_o[3],
                       const double ray_d[3]) {
  double hit[3];
  if (!RayPlaneHit(ray_o, ray_d, xanchor, xaxis, hit)) return q_now;
  // u = the p_ref direction projected into the plane (angle 0 == q_now); v =
  // xaxis x u so that RotateAboutAxis(xaxis, +theta, u) sweeps u toward v --
  // the same right-handed convention JointLimitChildPoint rotates by.
  double rel[3];
  for (int k = 0; k < 3; ++k) rel[k] = p_ref[k] - xanchor[k];
  const double along = Dot(rel, xaxis);
  double u[3];
  double n2 = 0;
  for (int k = 0; k < 3; ++k) {
    u[k] = rel[k] - along * xaxis[k];
    n2 += u[k] * u[k];
  }
  if (n2 < 1e-18) return q_now;  // p_ref on the axis: angle undefined
  const double n = std::sqrt(n2);
  for (int k = 0; k < 3; ++k) u[k] /= n;
  double v[3];
  Cross(xaxis, u, v);
  double h[3];
  for (int k = 0; k < 3; ++k) h[k] = hit[k] - xanchor[k];
  const double angle = std::atan2(Dot(h, v), Dot(h, u));
  return q_now + angle;
}

double SlideDofFromRay(const double xanchor[3], const double xaxis[3],
                       const double p_ref[3], double q_now, const double ray_o[3],
                       const double ray_d[3]) {
  double s;
  if (!ClosestAxisParam(xanchor, xaxis, ray_o, ray_d, &s)) return q_now;
  // Axial coord of p_ref (measured at q_now); the dof that puts p_ref's axial
  // coord at the cursor's s is q_now + (s - s0).
  double rel[3];
  for (int k = 0; k < 3; ++k) rel[k] = p_ref[k] - xanchor[k];
  const double s0 = Dot(rel, xaxis);
  return q_now + (s - s0);
}

double JointDofFromRay(int jnt_type, const double xanchor[3],
                       const double xaxis[3], const double p_ref[3], double q_now,
                       const double ray_o[3], const double ray_d[3]) {
  if (jnt_type == mjJNT_SLIDE) {
    return SlideDofFromRay(xanchor, xaxis, p_ref, q_now, ray_o, ray_d);
  }
  return HingeDofFromRay(xanchor, xaxis, p_ref, q_now, ray_o, ray_d);
}

double DofToAuthored(int jnt_type, double compiled, bool angle_is_degree) {
  // Authored range units == display units (compiler-angle for hinge, metres for
  // slide). The write direction of the P1 display helper: this is the ONE
  // conversion on the commit path (plan §1.6), round-tripped by the test.
  return JointDofToDisplay(jnt_type, compiled, angle_is_degree);
}

double SnapDof(int jnt_type, double q, bool on, double snap_rot_rad,
               double snap_trans_m) {
  if (!on) return q;
  const double step = (jnt_type == mjJNT_SLIDE) ? snap_trans_m : snap_rot_rad;
  if (!(step > 0)) return q;
  return std::round(q / step) * step;
}

}  // namespace ps::studio

#ifndef PS_RIG_HANDLES_NO_CONTROLLER

#include <algorithm>
#include <array>
#include <cstdio>
#include <memory>
#include <vector>

#include <imgui.h>

#include "editor/element_access.h"  // FindSerialAs
#include "editor/gizmo_math.h"      // BuildViewProj / WorldToScreen / ScreenToRay
#include "editor/layers.h"          // SerialInActiveLayer
#include "protospec/classes.h"      // sdk::EffectiveContext / sdk::Effective

namespace ps::studio {

namespace mj = ps::mjcf;
namespace sdk = ps::sdk;

namespace {

constexpr double kPiC = 3.14159265358979323846;
constexpr float kHandlePixels = 10.0f;  // endpoint-handle pick radius
constexpr float kEngageMove = 0.006f;   // normalized cursor move that starts a drag

struct VpMetrics {
  float x = 0, y = 0, w = 1, h = 1;
};
VpMetrics MainViewport() {
  const ImGuiViewport* v = ImGui::GetMainViewport();
  return {v->Pos.x, v->Pos.y, v->Size.x, v->Size.y};
}
struct Px {
  ImVec2 p;
  bool visible = false;
};
Px ToPix(const ViewProj& vp, const double world[3], const VpMetrics& m) {
  const ScreenPt s = WorldToScreen(vp, world);
  Px out;
  out.visible = s.visible;
  out.p = ImVec2(m.x + static_cast<float>(s.x) * m.w,
                 m.y + static_cast<float>(s.y) * m.h);
  return out;
}

// The subtree geom the cursor ray hits, or -1. Honours all geom groups (the rig
// has no host vis_option, matching the viewport pick's own group policy).
int PickSubtreeGeom(const mjModel* m, const mjData* d, int jid, const double o[3],
                    const double dir[3]) {
  int geom = -1;
  const mjtNum pnt[3] = {o[0], o[1], o[2]};
  const mjtNum vec[3] = {dir[0], dir[1], dir[2]};
  const mjtNum dist = mj_ray(m, d, pnt, vec, nullptr, 1, -1, &geom, nullptr);
  if (geom < 0 || dist < 0) return -1;
  const std::vector<int> sub = SubtreeGeoms(m, jid);
  return std::find(sub.begin(), sub.end(), geom) != sub.end() ? geom : -1;
}

}  // namespace

void RigHandleController::CancelEndpoint(EditorContext& ctx) {
  mjModel* m = ctx.compiled.model.get();
  if (m && grab_jid_ >= 0 && grab_jid_ < m->njnt) {
    m->jnt_range[2 * grab_jid_ + 0] = grab_range_[0];
    m->jnt_range[2 * grab_jid_ + 1] = grab_range_[1];
  }
  if (began_edit_) ctx.CancelEdit();
  began_edit_ = false;
}

void RigHandleController::CommitEndpoint(EditorContext& ctx) {
  mjModel* m = ctx.compiled.model.get();
  mj::Joint* j = ctx.tree ? FindSerialAs<mj::Joint>(*ctx.tree, drag_serial_)
                          : nullptr;
  if (!m || !j || grab_jid_ < 0 || grab_jid_ >= m->njnt) {
    if (began_edit_) ctx.CancelEdit();
    began_edit_ = false;
    return;
  }
  const int type = m->jnt_type[grab_jid_];
  const bool deg = AngleIsDegree(*ctx.tree);
  // Base authored range: the effective (class-inherited) authored value, so the
  // NON-dragged endpoint keeps its exact authored number (no float round-trip);
  // only the dragged endpoint is rewritten from its patched compiled value.
  const sdk::EffectiveContext ectx(*ctx.tree);
  std::unique_ptr<mj::Joint> eff = sdk::Effective(ectx, *j, true);
  std::array<double, 2> authored;
  if (eff->range) {
    authored = *eff->range;
  } else {
    authored[0] = DofToAuthored(type, m->jnt_range[2 * grab_jid_ + 0], deg);
    authored[1] = DofToAuthored(type, m->jnt_range[2 * grab_jid_ + 1], deg);
  }
  authored[grab_endpoint_] =
      DofToAuthored(type, m->jnt_range[2 * grab_jid_ + grab_endpoint_], deg);
  j->range = authored;
  ctx.CommitEdit("joint range");
  began_edit_ = false;
}

bool RigHandleController::HandleMouse(EditorContext& ctx, const ViewportInput& in) {
  const bool press = in.left_down && !prev_left_;
  const bool release = !in.left_down && prev_left_;
  prev_left_ = in.left_down;

  mjModel* m = ctx.compiled.model.get();
  mjData* d = ctx.sim_data;

  // An in-flight grab drives / finalizes each frame; a transient eligibility dip
  // (model reload, selection change) reverts it (mirrors the gizmo's Cancel).
  if (mode_ != Mode::None) {
    const int jid = (m && d)
                        ? JointIdForSerial(ctx.compiled.binding, drag_serial_)
                        : -1;
    const bool ok = ctx.CanEdit() && m && d && in.model == m &&
                    jid == grab_jid_ && ctx.selected_serial == drag_serial_;
    if (!ok) {
      if (mode_ == Mode::Endpoint) {
        CancelEndpoint(ctx);
      } else if (mode_ == Mode::Scrub && !ctx.rig_preview.hold) {
        ClearJointPreview(ctx);
      }
      mode_ = Mode::None;
      return false;
    }

    if (release) {
      const bool owned = active();
      if (mode_ == Mode::Endpoint) {
        if (began_edit_) CommitEndpoint(ctx);
      } else if (mode_ == Mode::Scrub && !ctx.rig_preview.hold) {
        ClearJointPreview(ctx);
      }
      mode_ = Mode::None;
      return owned;  // suppress selection only if a drag/scrub actually engaged
    }

    const ViewProj vp =
        BuildViewProj(in.model, in.data, in.camera, in.aspect_ratio);
    double ro[3], rd[3];
    ScreenToRay(vp, in.x, in.y, ro, rd);
    const int type = m->jnt_type[grab_jid_];
    const bool moved = (std::fabs(in.x - press_x_) + std::fabs(in.y - press_y_)) >
                       kEngageMove;

    if (mode_ == Mode::Endpoint) {
      double q = JointDofFromRay(type, d->xanchor + 3 * grab_jid_,
                                 d->xaxis + 3 * grab_jid_, scrub_geom_.p_ref,
                                 scrub_geom_.q_now, ro, rd);
      const GizmoSettings& g = ctx.gizmo;
      q = SnapDof(type, q, g.snap, g.snap_rotate_deg * kPiC / 180.0,
                  g.snap_translate);
      // Keep the dragged endpoint on its side of the other (no inverted range).
      if (grab_endpoint_ == 0) {
        q = std::min(q, grab_range_[1]);
      } else {
        q = std::max(q, grab_range_[0]);
      }
      if (!began_edit_ && moved) {
        ctx.BeginEdit();
        began_edit_ = true;
      }
      if (began_edit_) {
        m->jnt_range[2 * grab_jid_ + grab_endpoint_] = q;
        mapped_q_ = q;
      }
      return true;
    }

    // Limb-scrub: engage on first movement (a bare click stays a selection).
    if (mode_ == Mode::ArmScrub && moved) mode_ = Mode::Scrub;
    if (mode_ == Mode::Scrub) {
      const double q =
          JointDofFromRay(type, d->xanchor + 3 * grab_jid_,
                          d->xaxis + 3 * grab_jid_, scrub_geom_.p_ref,
                          scrub_geom_.q_now, ro, rd);
      mapped_q_ = q;
      SetJointPreview(ctx, drag_serial_, q);
      return true;
    }
    return false;  // ArmScrub, not yet moved -- let the click reach selection
  }

  // Idle: try to grab on press. (Reached only after the gizmo declined -- pick
  // priority gizmo > rig, enforced by the caller's ordering.)
  if (!press) return false;
  const int jid =
      (m && d) ? JointIdForSerial(ctx.compiled.binding, ctx.selected_serial) : -1;
  const bool joint_ok = m && jid >= 0 && jid < m->njnt &&
                        (m->jnt_type[jid] == mjJNT_HINGE ||
                         m->jnt_type[jid] == mjJNT_SLIDE);
  if (!ctx.CanEdit() || !d || in.model != m || !joint_ok) return false;

  const ViewProj vp = BuildViewProj(in.model, in.data, in.camera, in.aspect_ratio);
  const VpMetrics vm = MainViewport();
  const ImVec2 mouse(vm.x + in.x * vm.w, vm.y + in.y * vm.h);

  // 1) Range-endpoint handle (an authored edit -> gated on the active layer).
  if (m->jnt_limited[jid] && SerialInActiveLayer(ctx, ctx.selected_serial)) {
    const RangeEndpoints re = ComputeRangeEndpoints(m, d, jid);
    if (re.valid) {
      int best = -1;
      double best_d2 = kHandlePixels * kHandlePixels;
      for (int k = 0; k < 2; ++k) {
        const Px p = ToPix(vp, re.e[k], vm);
        if (!p.visible) continue;
        const double dx = p.p.x - mouse.x, dy = p.p.y - mouse.y;
        const double d2 = dx * dx + dy * dy;
        if (d2 < best_d2) {
          best_d2 = d2;
          best = k;
        }
      }
      if (best >= 0) {
        mode_ = Mode::Endpoint;
        drag_serial_ = ctx.selected_serial;
        grab_jid_ = jid;
        grab_endpoint_ = best;
        began_edit_ = false;
        grab_range_[0] = m->jnt_range[2 * jid + 0];
        grab_range_[1] = m->jnt_range[2 * jid + 1];
        scrub_geom_ = re;  // p_ref / q_now feed the mapping
        press_x_ = in.x;
        press_y_ = in.y;
        mapped_q_ = grab_range_[best];
        return true;  // consume immediately (a precise handle grab)
      }
    }
  }

  // 2) Limb-scrub: press over a subtree geom -> arm (engages on movement).
  double ro[3], rd[3];
  ScreenToRay(vp, in.x, in.y, ro, rd);
  const int geom = PickSubtreeGeom(m, d, jid, ro, rd);
  if (geom >= 0) {
    mode_ = Mode::ArmScrub;
    drag_serial_ = ctx.selected_serial;
    grab_jid_ = jid;
    scrub_geom_.jnt_type = m->jnt_type[jid];
    scrub_geom_.q_now = d->qpos[m->jnt_qposadr[jid]];
    for (int k = 0; k < 3; ++k) scrub_geom_.p_ref[k] = d->geom_xpos[3 * geom + k];
    press_x_ = in.x;
    press_y_ = in.y;
    mapped_q_ = scrub_geom_.q_now;
    return false;  // don't consume yet: a bare click still selects
  }
  return false;
}

void RigHandleController::Draw(EditorContext& ctx, const ViewportContext& vc) {
  // Esc cancels an in-flight endpoint drag / clears a limb-scrub (mirrors the
  // gizmo's Esc handling; the host's Esc only resets the camera, never exits Edit).
  if (mode_ == Mode::Endpoint && ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
    CancelEndpoint(ctx);
    mode_ = Mode::None;
    return;
  }
  if (mode_ == Mode::Scrub && ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
    if (!ctx.rig_preview.hold) ClearJointPreview(ctx);
    mode_ = Mode::None;
    return;
  }

  hot_ = false;
  const mjModel* m = ctx.compiled.model.get();
  if (!vc.edit_mode || !ctx.tree || ctx.selected_serial == 0 || !vc.data ||
      vc.model != m) {
    return;
  }
  const mjData* d = vc.data;
  const int jid = JointIdForSerial(ctx.compiled.binding, ctx.selected_serial);
  if (jid < 0 || jid >= m->njnt) return;
  const int type = m->jnt_type[jid];
  if (type != mjJNT_HINGE && type != mjJNT_SLIDE) return;

  const ViewProj vp = BuildViewProj(m, d, vc.camera, vc.aspect_ratio);
  const VpMetrics vm = MainViewport();
  ImDrawList* dl = ImGui::GetForegroundDrawList();

  // Endpoint handles: two spheres at the arc/travel ends. Suppressed while a
  // limb-scrub owns the mouse (the arc has moved under the scrub).
  if (m->jnt_limited[jid] && mode_ != Mode::Scrub && mode_ != Mode::ArmScrub) {
    const RangeEndpoints re = ComputeRangeEndpoints(m, d, jid);
    if (re.valid) {
      const ImVec2 mp = ImGui::GetIO().MousePos;
      for (int k = 0; k < 2; ++k) {
        const Px p = ToPix(vp, re.e[k], vm);
        if (!p.visible) continue;
        const bool dragging_this = mode_ == Mode::Endpoint && grab_endpoint_ == k;
        const double dx = p.p.x - mp.x, dy = p.p.y - mp.y;
        const bool over = (dx * dx + dy * dy) < kHandlePixels * kHandlePixels;
        if (over && mode_ == Mode::None) hot_ = true;
        const ImU32 col = (dragging_this || (over && mode_ == Mode::None))
                              ? IM_COL32(255, 235, 40, 255)
                          : k == 0 ? IM_COL32(80, 160, 255, 255)
                                   : IM_COL32(255, 150, 50, 255);
        const float r = dragging_this ? 7.0f : 5.5f;
        dl->AddCircleFilled(p.p, r, col);
        dl->AddCircle(p.p, r, IM_COL32(20, 20, 20, 200), 0, 1.5f);
      }
    }
  }

  // Readout near the cursor: current value + range in display units (plan §5
  // risk 8 -- screen-space text is acceptable; the plugin cannot draw occluded
  // text). Shown while a handle drag or a limb-scrub is live.
  if (active()) {
    const bool deg = AngleIsDegree(*ctx.tree);
    const char* unit = (type == mjJNT_SLIDE) ? "m" : (deg ? "deg" : "rad");
    const double vdisp = JointDofToDisplay(type, mapped_q_, deg);
    char buf[128];
    if (m->jnt_limited[jid]) {
      const double lo = JointDofToDisplay(type, m->jnt_range[2 * jid + 0], deg);
      const double hi = JointDofToDisplay(type, m->jnt_range[2 * jid + 1], deg);
      std::snprintf(buf, sizeof buf, "%.4g %s   [%.4g, %.4g]", vdisp, unit, lo,
                    hi);
    } else {
      std::snprintf(buf, sizeof buf, "%.4g %s", vdisp, unit);
    }
    const ImVec2 mp = ImGui::GetIO().MousePos;
    const ImVec2 at(mp.x + 16.0f, mp.y + 10.0f);
    const ImVec2 ts = ImGui::CalcTextSize(buf);
    dl->AddRectFilled(ImVec2(at.x - 5, at.y - 3),
                      ImVec2(at.x + ts.x + 5, at.y + ts.y + 3),
                      IM_COL32(15, 17, 20, 225), 4.0f);
    dl->AddText(at, IM_COL32(240, 240, 245, 255), buf);
  }
}

}  // namespace ps::studio

#endif  // PS_RIG_HANDLES_NO_CONTROLLER
