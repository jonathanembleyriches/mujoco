// ProtoSpec Studio: the transform gizmo. See gizmo.h.

#include "editor/gizmo.h"

#include <algorithm>
#include <cmath>

#include <mujoco/mujoco.h>

#include <imgui.h>

namespace ps::studio {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr float kGizmoPixels = 90.0f;   // on-screen radius of the gizmo
constexpr float kHitPixels = 9.0f;      // pick tolerance
constexpr int kRingSegments = 48;

struct Px {
  ImVec2 p;
  bool visible = false;
};

struct VpMetrics {
  float x = 0, y = 0, w = 1, h = 1;
};

VpMetrics MainViewport() {
  const ImGuiViewport* v = ImGui::GetMainViewport();
  return {v->Pos.x, v->Pos.y, v->Size.x, v->Size.y};
}

Px ToPix(const ViewProj& vp, const double world[3], const VpMetrics& m) {
  ScreenPt s = WorldToScreen(vp, world);
  Px out;
  out.visible = s.visible;
  out.p = ImVec2(m.x + static_cast<float>(s.x) * m.w,
                 m.y + static_cast<float>(s.y) * m.h);
  return out;
}

double Dist2(const ImVec2& a, const ImVec2& b) {
  const double dx = a.x - b.x, dy = a.y - b.y;
  return dx * dx + dy * dy;
}

void Add(const double a[3], const double b[3], double s, double out[3]) {
  out[0] = a[0] + b[0] * s;
  out[1] = a[1] + b[1] * s;
  out[2] = a[2] + b[2] * s;
}

// World axis directions the gizmo shows. `local` picks the element's own frame.
void AxisDirs(const DragFrame& f, bool local, double ax[3][3]) {
  const double e[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  for (int i = 0; i < 3; ++i) {
    if (local) {
      QuatRotate(f.world_quat, e[i], ax[i]);
    } else {
      ax[i][0] = e[i][0];
      ax[i][1] = e[i][1];
      ax[i][2] = e[i][2];
    }
  }
}

double Snap(double v, double step, bool on) {
  return (on && step > 0) ? std::round(v / step) * step : v;
}

ImU32 AxisColor(int axis, bool hot) {
  if (hot) return IM_COL32(255, 220, 40, 255);
  switch (axis) {
    case 0: return IM_COL32(230, 70, 70, 255);
    case 1: return IM_COL32(90, 210, 90, 255);
    default: return IM_COL32(80, 130, 240, 255);
  }
}

// A display frame for the current selection: either the generic spatial drag
// frame, or -- for a joint -- a synthetic frame anchored at the joint origin
// with world axes, plus the grab-time JointDragFrame the apply step needs.
struct DisplayFrame {
  DragFrame f;
  bool joint = false;
  JointDragFrame jf;
};
static DisplayFrame BuildDisplay(EditorContext& ctx, const mjModel* m,
                                 const mjData* d) {
  DisplayFrame out;
  if (ctx.tree && IsJointSerial(*ctx.tree, ctx.selected_serial)) {
    out.joint = true;
    out.jf = BuildJointDragFrame(m, d, ctx.compiled.binding, *ctx.tree,
                                 ctx.selected_serial);
    out.f.valid = out.jf.valid;
    out.f.type = mj::ElementType::Joint;
    out.f.parent = out.jf.parent;
    for (int k = 0; k < 3; ++k) out.f.anchor[k] = out.jf.world_anchor[k];
    out.f.world_quat[0] = 1;
    out.f.world_quat[1] = out.f.world_quat[2] = out.f.world_quat[3] = 0;
  } else {
    out.f = BuildDragFrame(m, d, ctx.compiled.binding, *ctx.tree,
                           ctx.selected_serial);
  }
  return out;
}

}  // namespace

// Hit-test the assembled gizmo at `mouse` (pixels). Priority favours the inner
// centre/plane/uniform handles over the axes they overlap.
static GizmoHandle HitTest(const DragFrame& f, const GizmoSettings& g,
                           const ViewProj& vp, const VpMetrics& m,
                           double size, const ImVec2& mouse) {
  GizmoHandle none;
  const double tol2 = static_cast<double>(kHitPixels) * kHitPixels;
  double ax[3][3];

  auto axis_hit = [&](HandleKind kind, double reach) -> GizmoHandle {
    GizmoHandle best;
    double best_d = tol2;
    for (int i = 0; i < 3; ++i) {
      double tip[3];
      Add(f.anchor, ax[i], size * reach, tip);
      Px a = ToPix(vp, f.anchor, m);
      Px b = ToPix(vp, tip, m);
      if (!a.visible || !b.visible) continue;
      const double d = PointSegmentDist(mouse.x, mouse.y, a.p.x, a.p.y, b.p.x,
                                        b.p.y);
      if (d * d < best_d) {
        best_d = d * d;
        best = {kind, i};
      }
    }
    return best;
  };

  switch (g.tool) {
    case GizmoTool::Translate: {
      AxisDirs(f, !g.world_space, ax);
      // Screen centre.
      Px c = ToPix(vp, f.anchor, m);
      if (c.visible && Dist2(mouse, c.p) < tol2 * 1.6)
        return {HandleKind::TransScreen, 0};
      // Planes (quad centre at 0.35 size along the two non-normal axes).
      for (int n = 0; n < 3; ++n) {
        const int i = (n + 1) % 3, j = (n + 2) % 3;
        double p[3];
        Add(f.anchor, ax[i], size * 0.35, p);
        Add(p, ax[j], size * 0.35, p);
        Px pp = ToPix(vp, p, m);
        if (pp.visible && Dist2(mouse, pp.p) < tol2 * 2.2)
          return {HandleKind::TransPlane, n};
      }
      GizmoHandle a = axis_hit(HandleKind::TransAxis, 1.0);
      if (a.kind != HandleKind::None) return a;
      return none;
    }
    case GizmoTool::Rotate: {
      AxisDirs(f, !g.world_space, ax);
      GizmoHandle best;
      double best_d = tol2;
      // Axis rings + the camera-facing screen ring.
      double screen_axis[3] = {vp.forward[0], vp.forward[1], vp.forward[2]};
      for (int ring = 0; ring < 4; ++ring) {
        double normal[3];
        double radius = size;
        if (ring < 3) {
          normal[0] = ax[ring][0]; normal[1] = ax[ring][1]; normal[2] = ax[ring][2];
        } else {
          normal[0] = screen_axis[0]; normal[1] = screen_axis[1];
          normal[2] = screen_axis[2];
          radius = size * 1.15;
        }
        // Two axes spanning the ring plane.
        double u[3], v[3];
        double seed[3] = {1, 0, 0};
        if (std::fabs(normal[0]) > 0.9) { seed[0] = 0; seed[1] = 1; }
        // u = normalize(seed x normal); v = normal x u
        u[0] = seed[1] * normal[2] - seed[2] * normal[1];
        u[1] = seed[2] * normal[0] - seed[0] * normal[2];
        u[2] = seed[0] * normal[1] - seed[1] * normal[0];
        double un = std::sqrt(u[0]*u[0]+u[1]*u[1]+u[2]*u[2]);
        if (un < 1e-9) continue;
        for (int k = 0; k < 3; ++k) u[k] /= un;
        v[0] = normal[1]*u[2]-normal[2]*u[1];
        v[1] = normal[2]*u[0]-normal[0]*u[2];
        v[2] = normal[0]*u[1]-normal[1]*u[0];
        ImVec2 prev; bool have_prev = false;
        for (int s = 0; s <= kRingSegments; ++s) {
          const double ang = 2 * kPi * s / kRingSegments;
          double pt[3];
          for (int k = 0; k < 3; ++k)
            pt[k] = f.anchor[k] + radius * (std::cos(ang) * u[k] +
                                            std::sin(ang) * v[k]);
          Px pp = ToPix(vp, pt, m);
          if (pp.visible && have_prev) {
            const double d = PointSegmentDist(mouse.x, mouse.y, prev.x, prev.y,
                                              pp.p.x, pp.p.y);
            if (d * d < best_d) {
              best_d = d * d;
              best = {ring < 3 ? HandleKind::RotAxis : HandleKind::RotScreen,
                      ring < 3 ? ring : 0};
            }
          }
          prev = pp.p;
          have_prev = pp.visible;
        }
      }
      return best;
    }
    case GizmoTool::Scale: {
      AxisDirs(f, /*local=*/true, ax);  // scale is always object-local
      Px c = ToPix(vp, f.anchor, m);
      if (c.visible && Dist2(mouse, c.p) < tol2 * 1.6)
        return {HandleKind::ScaleUniform, 0};
      return axis_hit(HandleKind::ScaleAxis, 1.0);
    }
    default:
      return none;
  }
}

void GizmoController::Begin(EditorContext& ctx, const ViewportInput& in,
                            const ViewProj& vp, const GizmoHandle& h) {
  ctx.BeginEdit();
  dragging_ = true;
  grabbed_ = h;
  drag_serial_ = ctx.selected_serial;
  {
    DisplayFrame df = BuildDisplay(ctx, in.model, in.data);
    frame_ = df.f;
    joint_mode_ = df.joint;
    joint_frame_ = df.jf;
  }
  const VpMetrics m = MainViewport();
  gizmo_size_ = WorldSizeForPixels(vp, frame_.anchor, kGizmoPixels, m.h);

  double ax[3][3];
  const bool local = (ctx.gizmo.tool == GizmoTool::Scale) || !ctx.gizmo.world_space;
  AxisDirs(frame_, local, ax);

  double ro[3], rd[3];
  ScreenToRay(vp, in.x, in.y, ro, rd);

  switch (h.kind) {
    case HandleKind::TransAxis:
    case HandleKind::ScaleAxis: {
      double a0[3] = {ax[h.axis][0], ax[h.axis][1], ax[h.axis][2]};
      ClosestPointOnLine(frame_.anchor, a0, ro, rd, &grab_axis_t_);
      break;
    }
    case HandleKind::TransPlane: {
      double t;
      RayPlaneIntersect(ro, rd, frame_.anchor, ax[h.axis], &t, grab_hit_);
      break;
    }
    case HandleKind::TransScreen: {
      double n[3] = {vp.forward[0], vp.forward[1], vp.forward[2]};
      double t;
      RayPlaneIntersect(ro, rd, frame_.anchor, n, &t, grab_hit_);
      break;
    }
    case HandleKind::RotAxis:
    case HandleKind::RotScreen: {
      double n[3];
      if (h.kind == HandleKind::RotAxis) {
        n[0] = ax[h.axis][0]; n[1] = ax[h.axis][1]; n[2] = ax[h.axis][2];
      } else {
        n[0] = vp.forward[0]; n[1] = vp.forward[1]; n[2] = vp.forward[2];
      }
      double hit[3], t;
      if (RayPlaneIntersect(ro, rd, frame_.anchor, n, &t, hit)) {
        // Reference angle in the ring plane.
        double u[3] = {1, 0, 0};
        if (std::fabs(n[0]) > 0.9) { u[0] = 0; u[1] = 1; u[2] = 0; }
        double tmp[3] = {u[1]*n[2]-u[2]*n[1], u[2]*n[0]-u[0]*n[2],
                         u[0]*n[1]-u[1]*n[0]};
        double tn = std::sqrt(tmp[0]*tmp[0]+tmp[1]*tmp[1]+tmp[2]*tmp[2]);
        for (int k = 0; k < 3; ++k) u[k] = tmp[k] / tn;
        double v[3] = {n[1]*u[2]-n[2]*u[1], n[2]*u[0]-n[0]*u[2],
                       n[0]*u[1]-n[1]*u[0]};
        double vec[3] = {hit[0]-frame_.anchor[0], hit[1]-frame_.anchor[1],
                         hit[2]-frame_.anchor[2]};
        grab_angle_ = std::atan2(vec[0]*v[0]+vec[1]*v[1]+vec[2]*v[2],
                                 vec[0]*u[0]+vec[1]*u[1]+vec[2]*u[2]);
      }
      break;
    }
    case HandleKind::ScaleUniform:
      grab_hit_[0] = in.x;
      grab_hit_[1] = in.y;
      break;
    default:
      break;
  }
  if (ctx.gizmo.tool == GizmoTool::Scale) scale_base_ = BuildScaleBase(*ctx.tree, drag_serial_);

  // Capture the pose-patch for the drag fast path (deliverable 1): a pure
  // Translate/Rotate drag patches the live mjModel each frame instead of
  // recompiling; the release does one real Compile. Bodies/geoms/sites/cameras/
  // lights (incl. fromto geoms/sites, handled specially in UpdateDrag) get a
  // PosePatch; joints are patched via jnt_pos/jnt_axis (UpdateDrag, no PosePatch).
  // Only a light being ROTATED falls back to recompile (ApplyPosePatch moves
  // light_pos, not light_dir), as does any element with no captured PosePatch.
  pose_patch_.reset();
  const GizmoTool tool = ctx.gizmo.tool;
  const bool pose_tool =
      tool == GizmoTool::Translate || tool == GizmoTool::Rotate;
  const bool light_rotate =
      frame_.type == mj::ElementType::Light && tool == GizmoTool::Rotate;
  if (pose_tool && !joint_mode_ && !light_rotate && ctx.tree) {
    if (SpatialRef ref = FindSpatial(*ctx.tree, drag_serial_)) {
      pose_patch_ = ctx.compiled.binding.PosePatchFor(ref.ptr);
    }
  }
}

void GizmoController::UpdateDrag(EditorContext& ctx, const ViewportInput& in,
                                 const ViewProj& vp) {
  if (drag_serial_ != ctx.selected_serial) return;
  const GizmoSettings& g = ctx.gizmo;
  double ax[3][3];
  const bool local = (g.tool == GizmoTool::Scale) || !g.world_space;
  AxisDirs(frame_, local, ax);
  double ro[3], rd[3];
  ScreenToRay(vp, in.x, in.y, ro, rd);

  switch (grabbed_.kind) {
    case HandleKind::TransAxis: {
      double a0[3] = {ax[grabbed_.axis][0], ax[grabbed_.axis][1],
                      ax[grabbed_.axis][2]};
      double t;
      if (!ClosestPointOnLine(frame_.anchor, a0, ro, rd, &t)) return;
      double dist = Snap(t - grab_axis_t_, g.snap_translate, g.snap);
      double wd[3] = {a0[0] * dist, a0[1] * dist, a0[2] * dist};
      if (joint_mode_)
        ApplyJointTranslate(*ctx.tree, drag_serial_, joint_frame_, wd);
      else
        ApplyTranslate(*ctx.tree, drag_serial_, frame_, wd);
      break;
    }
    case HandleKind::TransPlane:
    case HandleKind::TransScreen: {
      double n[3];
      if (grabbed_.kind == HandleKind::TransScreen) {
        n[0] = vp.forward[0]; n[1] = vp.forward[1]; n[2] = vp.forward[2];
      } else {
        n[0] = ax[grabbed_.axis][0]; n[1] = ax[grabbed_.axis][1];
        n[2] = ax[grabbed_.axis][2];
      }
      double hit[3], t;
      if (!RayPlaneIntersect(ro, rd, frame_.anchor, n, &t, hit)) return;
      double wd[3] = {hit[0] - grab_hit_[0], hit[1] - grab_hit_[1],
                      hit[2] - grab_hit_[2]};
      if (g.snap && grabbed_.kind == HandleKind::TransPlane) {
        const int i = (grabbed_.axis + 1) % 3, j = (grabbed_.axis + 2) % 3;
        double di = wd[0]*ax[i][0]+wd[1]*ax[i][1]+wd[2]*ax[i][2];
        double dj = wd[0]*ax[j][0]+wd[1]*ax[j][1]+wd[2]*ax[j][2];
        di = Snap(di, g.snap_translate, true);
        dj = Snap(dj, g.snap_translate, true);
        for (int k = 0; k < 3; ++k) wd[k] = di*ax[i][k]+dj*ax[j][k];
      }
      if (joint_mode_)
        ApplyJointTranslate(*ctx.tree, drag_serial_, joint_frame_, wd);
      else
        ApplyTranslate(*ctx.tree, drag_serial_, frame_, wd);
      break;
    }
    case HandleKind::RotAxis:
    case HandleKind::RotScreen: {
      double n[3];
      if (grabbed_.kind == HandleKind::RotAxis) {
        n[0] = ax[grabbed_.axis][0]; n[1] = ax[grabbed_.axis][1];
        n[2] = ax[grabbed_.axis][2];
      } else {
        n[0] = vp.forward[0]; n[1] = vp.forward[1]; n[2] = vp.forward[2];
      }
      double hit[3], t;
      if (!RayPlaneIntersect(ro, rd, frame_.anchor, n, &t, hit)) return;
      double u[3] = {1, 0, 0};
      if (std::fabs(n[0]) > 0.9) { u[0] = 0; u[1] = 1; u[2] = 0; }
      double tmp[3] = {u[1]*n[2]-u[2]*n[1], u[2]*n[0]-u[0]*n[2],
                       u[0]*n[1]-u[1]*n[0]};
      double tn = std::sqrt(tmp[0]*tmp[0]+tmp[1]*tmp[1]+tmp[2]*tmp[2]);
      for (int k = 0; k < 3; ++k) u[k] = tmp[k] / tn;
      double v[3] = {n[1]*u[2]-n[2]*u[1], n[2]*u[0]-n[0]*u[2],
                     n[0]*u[1]-n[1]*u[0]};
      double vec[3] = {hit[0]-frame_.anchor[0], hit[1]-frame_.anchor[1],
                       hit[2]-frame_.anchor[2]};
      double ang = std::atan2(vec[0]*v[0]+vec[1]*v[1]+vec[2]*v[2],
                              vec[0]*u[0]+vec[1]*u[1]+vec[2]*u[2]);
      double delta = ang - grab_angle_;
      while (delta > kPi) delta -= 2 * kPi;
      while (delta < -kPi) delta += 2 * kPi;
      if (joint_mode_) {
        // Reorient the joint axis by the continuous angle; snapping (when on)
        // snaps the resulting axis to the nearest cardinal, not the angle.
        ApplyJointAxisRotate(*ctx.tree, drag_serial_, joint_frame_, n, delta,
                             g.snap);
        ctx.status_toast.Post(
            g.snap ? "joint: axis reoriented (snapped to X/Y/Z)"
                   : "joint: axis reoriented",
            StatusToast::Kind::Info, ImGui::GetTime());
      } else {
        double snapped = Snap(delta, g.snap_rotate_deg * kPi / 180.0, g.snap);
        ApplyRotate(*ctx.tree, drag_serial_, frame_, n, snapped);
        ctx.status_toast.Post("gizmo: rotation materialised as quat",
                              StatusToast::Kind::Info, ImGui::GetTime());
      }
      break;
    }
    case HandleKind::ScaleAxis: {
      double a0[3] = {ax[grabbed_.axis][0], ax[grabbed_.axis][1],
                      ax[grabbed_.axis][2]};
      double t;
      if (!ClosestPointOnLine(frame_.anchor, a0, ro, rd, &t)) return;
      double f = 1.0 + (t - grab_axis_t_) / (gizmo_size_ > 1e-6 ? gizmo_size_ : 1);
      f = Snap(f, g.snap_scale, g.snap);
      if (f < 1e-3) f = 1e-3;
      double factor[3] = {1, 1, 1};
      factor[grabbed_.axis] = f;
      ApplyScale(*ctx.tree, drag_serial_, scale_base_, factor);
      if (scale_base_.is_mesh)
        ctx.status_toast.Post("gizmo: mesh scale affects ALL users of the mesh",
                              StatusToast::Kind::Warning, ImGui::GetTime());
      break;
    }
    case HandleKind::ScaleUniform: {
      double f = 1.0 + ((in.x - grab_hit_[0]) - (in.y - grab_hit_[1])) * 3.0;
      f = Snap(f, g.snap_scale, g.snap);
      if (f < 1e-3) f = 1e-3;
      double factor[3] = {f, f, f};
      ApplyScale(*ctx.tree, drag_serial_, scale_base_, factor);
      if (scale_base_.is_mesh)
        ctx.status_toast.Post("gizmo: mesh scale affects ALL users of the mesh",
                              StatusToast::Kind::Warning, ImGui::GetTime());
      break;
    }
    default:
      break;
  }
  ctx.dirty = true;

  // Fast path: patch the live mjModel + mj_kinematics, no recompile. The authored
  // tree was already edited above (DR-S1), so the on-release Compile reconciles
  // from ground truth. Any patch failure falls back to a recompile so behaviour is
  // never worse than the debounced-recompile preview.
  if (LivePatch(ctx, in)) return;
  ctx.RequestRecompile();
}

// Patch the live compiled model to preview the current drag frame without
// recompiling. Returns false when the element is not live-patchable (framed
// joint, unbound, ...), in which case the caller recompiles. Covers:
//   * joints        -- write jnt_pos / jnt_axis from the authored anchor/axis
//                      mapped through the enclosing <frame> chain;
//   * fromto geoms  -- the compiled pose is the endpoint-derived (midpoint +
//                      z->axis) pose, so feed ApplyPosePatch  derived ∘ B^-1  to
//                      land  A ∘ derived  (B is the grab-time derived residual);
//   * everything else -- the authored pos/quat local pose.
bool GizmoController::LivePatch(EditorContext& ctx, const ViewportInput& in) {
  mjModel* m = ctx.compiled.model.get();
  mjData* d = const_cast<mjData*>(in.data);  // host-owned; const only in the input
  if (m == nullptr || d == nullptr || m != in.model || !ctx.tree) return false;

  if (joint_mode_) return LivePatchJoint(ctx, m, d);
  if (!pose_patch_) return false;

  Rigid L;
  if (frame_.is_fromto) {
    // Rebuild the display frame to read the fresh endpoint-derived local pose.
    DragFrame f =
        BuildDragFrame(m, d, ctx.compiled.binding, *ctx.tree, drag_serial_);
    if (!f.valid) return false;
    Rigid B;  // the captured residual suffix == the grab-time derived pose
    for (int i = 0; i < 3; ++i) B.pos[i] = pose_patch_->suffix.pos[i];
    for (int i = 0; i < 4; ++i) B.quat[i] = pose_patch_->suffix.quat[i];
    L = Compose(f.local, Invert(B));  // A ∘ L ∘ B == A ∘ derived_new
  } else {
    L = EffectiveLocalPose(*ctx.tree, drag_serial_);
  }
  mj::RigidPose Lnew;
  for (int i = 0; i < 3; ++i) Lnew.pos[i] = L.pos[i];
  for (int i = 0; i < 4; ++i) Lnew.quat[i] = L.quat[i];
  if (!mj::ApplyPosePatch(m, *pose_patch_, Lnew)) return false;
  // A free/ball body's rest pose lives in qpos0 (just reseeded): copy it into
  // qpos so mj_kinematics moves the body from the new rest pose.
  if (pose_patch_->reseed_qposadr >= 0) mju_copy(d->qpos, m->qpos0, m->nq);
  mj_kinematics(m, d);
  return true;
}

// Patch a joint's compiled anchor (jnt_pos) and axis (jnt_axis) from the authored
// values, so a joint drag previews without recompiling. jnt_pos/jnt_axis live in
// the body-local frame (frames flattened), so the authored parent-frame values
// map through the enclosing <frame> chain prefix. Returns false (recompile) for a
// free joint (nothing to drag) or an unbound joint.
bool GizmoController::LivePatchJoint(EditorContext& ctx, mjModel* m, mjData* d) {
  int jid = -1;
  for (const mj::Binding::Entry& e : ctx.compiled.binding.entries()) {
    if (e.serial == drag_serial_ && e.id >= 0 &&
        (e.etype == mj::ElementType::Joint ||
         e.etype == mj::ElementType::FreeJoint)) {
      jid = e.id;
      break;
    }
  }
  if (jid < 0 || jid >= m->njnt) return false;
  if (m->jnt_type[jid] == mjJNT_FREE) return true;  // no anchor/axis to drag; no-op

  const JointDragFrame f =
      BuildJointDragFrame(m, d, ctx.compiled.binding, *ctx.tree, drag_serial_);
  if (!f.valid) return false;
  const Rigid A = FrameChainPrefix(*ctx.tree, drag_serial_);

  // Compiled jnt_pos = A applied to the authored anchor (a point).
  double jp[3];
  QuatRotate(A.quat, f.pos, jp);
  for (int i = 0; i < 3; ++i) m->jnt_pos[3 * jid + i] = jp[i] + A.pos[i];
  // Compiled jnt_axis = A.quat . authored axis (a direction), renormalized.
  if (f.has_axis) {
    double ja[3];
    QuatRotate(A.quat, f.axis, ja);
    double n = std::sqrt(ja[0] * ja[0] + ja[1] * ja[1] + ja[2] * ja[2]);
    if (n > 1e-12)
      for (int i = 0; i < 3; ++i) ja[i] /= n;
    for (int i = 0; i < 3; ++i) m->jnt_axis[3 * jid + i] = ja[i];
  }
  mj_kinematics(m, d);
  return true;
}

void GizmoController::Cancel(EditorContext& ctx) {
  if (!dragging_) return;
  ctx.CancelEdit();
  dragging_ = false;
  grabbed_ = {};
}

bool GizmoController::HandleMouse(EditorContext& ctx, const ViewportInput& in) {
  ctx.gizmo_active = dragging_;
  const bool eligible = ctx.CanEdit() && ctx.tree &&
                        in.model == ctx.compiled.model.get() && in.data &&
                        ctx.selected_serial != 0 &&
                        ctx.gizmo.tool != GizmoTool::Select;
  if (!eligible) {
    if (dragging_) Cancel(ctx);  // selection/tool changed mid-drag
    prev_left_ = in.left_down;
    ctx.gizmo_active = dragging_;
    return false;
  }

  const ViewProj vp = BuildViewProj(in.model, in.data, in.camera, in.aspect_ratio);
  const bool press = in.left_down && !prev_left_;
  const bool release = !in.left_down && prev_left_;
  prev_left_ = in.left_down;

  if (dragging_) {
    if (release) {
      const char* label = ctx.gizmo.tool == GizmoTool::Translate ? "translate"
                          : ctx.gizmo.tool == GizmoTool::Rotate   ? "rotate"
                                                                  : "scale";
      ctx.CommitEdit(label);
      dragging_ = false;
      grabbed_ = {};
      ctx.gizmo_active = false;
      return true;
    }
    UpdateDrag(ctx, in, vp);
    ctx.gizmo_active = true;
    return true;
  }

  if (press) {
    const VpMetrics m = MainViewport();
    DisplayFrame df = BuildDisplay(ctx, in.model, in.data);
    DragFrame f = df.f;
    // Joints have no scale handle; the scale tool shows nothing for them.
    if (f.valid && df.joint && ctx.gizmo.tool == GizmoTool::Scale) return false;
    if (f.valid) {
      const double size = WorldSizeForPixels(vp, f.anchor, kGizmoPixels, m.h);
      const ImVec2 mouse(m.x + in.x * m.w, m.y + in.y * m.h);
      GizmoHandle h = HitTest(f, ctx.gizmo, vp, m, size, mouse);
      if (h.kind != HandleKind::None) {
        Begin(ctx, in, vp, h);
        ctx.gizmo_active = true;
        return true;  // consume: suppress camera
      }
    }
  }
  return false;
}

namespace {

// A note pill centred near the bottom of the viewport, drawn at `alpha` opacity
// with a severity-coloured border. Shared by the fading StatusToast and the
// steady "tool doesn't apply here" hint.
void DrawViewportNote(const char* msg, StatusToast::Kind kind, float alpha) {
  if (alpha <= 0.0f) return;
  const VpMetrics m = MainViewport();
  const ImVec2 ts = ImGui::CalcTextSize(msg);
  const ImVec2 pad(12.0f, 7.0f);
  const ImVec2 mid(m.x + m.w * 0.5f, m.y + m.h - 52.0f);
  const ImVec2 tl(mid.x - ts.x * 0.5f - pad.x, mid.y - ts.y * 0.5f - pad.y);
  const ImVec2 br(mid.x + ts.x * 0.5f + pad.x, mid.y + ts.y * 0.5f + pad.y);
  const ImVec4 accent = kind == StatusToast::Kind::Error
                            ? ImVec4(0.95f, 0.45f, 0.42f, 1.0f)
                        : kind == StatusToast::Kind::Warning
                            ? ImVec4(0.95f, 0.78f, 0.35f, 1.0f)
                            : ImVec4(0.65f, 0.75f, 0.95f, 1.0f);
  ImDrawList* dl = ImGui::GetForegroundDrawList();
  dl->AddRectFilled(
      tl, br, ImGui::GetColorU32(ImVec4(0.09f, 0.10f, 0.12f, 0.86f * alpha)),
      6.0f);
  dl->AddRect(
      tl, br,
      ImGui::GetColorU32(ImVec4(accent.x, accent.y, accent.z, 0.9f * alpha)),
      6.0f, 0, 1.5f);
  dl->AddText(ImVec2(mid.x - ts.x * 0.5f, mid.y - ts.y * 0.5f),
              ImGui::GetColorU32(ImVec4(0.94f, 0.94f, 0.96f, alpha)), msg);
}

// Info/Warning notes fade out; Error notes stay until replaced (see StatusToast).
// Drawn every edit frame, independent of the gizmo, so any tool can post it.
void DrawStatusToast(EditorContext& ctx) {
  DrawViewportNote(ctx.status_toast.message.c_str(), ctx.status_toast.kind,
                   ctx.status_toast.Alpha(ImGui::GetTime()));
}

}  // namespace

void GizmoController::Draw(EditorContext& ctx, const ViewportGuiPlugin::Context& vc) {
  if (vc.edit_mode) DrawStatusToast(ctx);
  if (!vc.edit_mode || !ctx.tree || ctx.selected_serial == 0 || !vc.data ||
      vc.model != ctx.compiled.model.get() ||
      ctx.gizmo.tool == GizmoTool::Select) {
    return;
  }
  // The Scale tool only means something for a sized element (geom / site). For a
  // body, light, camera, etc. it would draw handles that do nothing, so hide it
  // and say why instead of failing silently.
  if (ctx.gizmo.tool == GizmoTool::Scale && !dragging_ &&
      !BuildScaleBase(*ctx.tree, ctx.selected_serial).valid) {
    DrawViewportNote("Scale applies to a geom or site — this element has no size",
                     StatusToast::Kind::Warning, 1.0f);
    return;
  }
  if (dragging_ && ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
    Cancel(ctx);
    return;
  }

  const ViewProj vp = BuildViewProj(vc.model, vc.data, vc.camera, vc.aspect_ratio);
  const VpMetrics m = MainViewport();
  // The gizmo tracks the selection live -- during a drag too, so its handles
  // follow the object under the cursor instead of staying frozen at the grab
  // point (which reads as a disconnected, unsmooth drag). The world anchor
  // recomputes each frame as parent_pose . the freshly authored local pose, so it
  // follows both the fast-patch and recompile drag paths. The drag MATH still uses
  // the immutable grab-time frame_; only the drawn frame follows.
  bool joint_now = joint_mode_;
  DisplayFrame df = BuildDisplay(ctx, vc.model, vc.data);
  DragFrame f = df.f;
  joint_now = df.joint;
  if (dragging_ && !f.valid) f = frame_;  // keep the grab frame if it can't resolve
  if (!f.valid) return;
  // Joints expose translate + reorient handles only, never scale.
  if (joint_now && ctx.gizmo.tool == GizmoTool::Scale) return;
  const double size = dragging_
                          ? gizmo_size_
                          : WorldSizeForPixels(vp, f.anchor, kGizmoPixels, m.h);

  const ImVec2 mp = ImGui::GetIO().MousePos;
  GizmoHandle hover =
      dragging_ ? grabbed_ : HitTest(f, ctx.gizmo, vp, m, size, mp);

  ImDrawList* dl = ImGui::GetForegroundDrawList();
  double ax[3][3];
  const GizmoSettings& g = ctx.gizmo;
  Px center = ToPix(vp, f.anchor, m);

  auto hot = [&](HandleKind k, int axis) {
    return hover.kind == k && hover.axis == axis;
  };

  if (g.tool == GizmoTool::Translate || g.tool == GizmoTool::Scale) {
    const bool local = (g.tool == GizmoTool::Scale) || !g.world_space;
    AxisDirs(f, local, ax);
    // Axes.
    for (int i = 0; i < 3; ++i) {
      double tip[3];
      Add(f.anchor, ax[i], size, tip);
      Px t = ToPix(vp, tip, m);
      if (!center.visible || !t.visible) continue;
      const bool h = g.tool == GizmoTool::Translate ? hot(HandleKind::TransAxis, i)
                                                    : hot(HandleKind::ScaleAxis, i);
      const ImU32 col = AxisColor(i, h);
      dl->AddLine(center.p, t.p, col, 2.5f);
      if (g.tool == GizmoTool::Scale) {
        dl->AddRectFilled(ImVec2(t.p.x - 5, t.p.y - 5),
                          ImVec2(t.p.x + 5, t.p.y + 5), col);
      } else {
        dl->AddCircleFilled(t.p, 5.0f, col);
      }
    }
    if (g.tool == GizmoTool::Translate) {
      // Plane quads: a square spanning [0.2, 0.5]*size along the two in-plane axes.
      for (int n = 0; n < 3; ++n) {
        const int i = (n + 1) % 3, j = (n + 2) % 3;
        double c0[3], c1[3], c2[3], c3[3];
        Add(f.anchor, ax[i], size * 0.2, c0);
        Add(c0, ax[j], size * 0.2, c0);
        Add(f.anchor, ax[i], size * 0.5, c1); Add(c1, ax[j], size * 0.2, c1);
        Add(f.anchor, ax[i], size * 0.5, c2); Add(c2, ax[j], size * 0.5, c2);
        Add(f.anchor, ax[i], size * 0.2, c3); Add(c3, ax[j], size * 0.5, c3);
        Px q0 = ToPix(vp, c0, m), q1 = ToPix(vp, c1, m), q2 = ToPix(vp, c2, m),
           q3 = ToPix(vp, c3, m);
        if (q0.visible && q1.visible && q2.visible && q3.visible) {
          const bool h = hot(HandleKind::TransPlane, n);
          ImU32 col = h ? IM_COL32(255, 220, 40, 110) : AxisColor(n, false);
          if (!h) col = (col & 0x00FFFFFF) | (70u << 24);
          ImVec2 pts[4] = {q0.p, q1.p, q2.p, q3.p};
          dl->AddConvexPolyFilled(pts, 4, col);
        }
      }
      // Screen centre.
      if (center.visible) {
        const bool h = hot(HandleKind::TransScreen, 0);
        dl->AddCircleFilled(center.p, 4.5f,
                            h ? IM_COL32(255, 220, 40, 255)
                              : IM_COL32(220, 220, 220, 255));
      }
    } else if (center.visible) {  // scale uniform centre
      const bool h = hot(HandleKind::ScaleUniform, 0);
      dl->AddRectFilled(ImVec2(center.p.x - 5, center.p.y - 5),
                        ImVec2(center.p.x + 5, center.p.y + 5),
                        h ? IM_COL32(255, 220, 40, 255)
                          : IM_COL32(220, 220, 220, 255));
    }
  } else if (g.tool == GizmoTool::Rotate) {
    AxisDirs(f, !g.world_space, ax);
    double screen_axis[3] = {vp.forward[0], vp.forward[1], vp.forward[2]};
    for (int ring = 0; ring < 4; ++ring) {
      double normal[3];
      double radius = size;
      const bool is_screen = ring == 3;
      if (!is_screen) {
        normal[0] = ax[ring][0]; normal[1] = ax[ring][1]; normal[2] = ax[ring][2];
      } else {
        normal[0] = screen_axis[0]; normal[1] = screen_axis[1];
        normal[2] = screen_axis[2];
        radius = size * 1.15;
      }
      double seed[3] = {1, 0, 0};
      if (std::fabs(normal[0]) > 0.9) { seed[0] = 0; seed[1] = 1; }
      double u[3] = {seed[1]*normal[2]-seed[2]*normal[1],
                     seed[2]*normal[0]-seed[0]*normal[2],
                     seed[0]*normal[1]-seed[1]*normal[0]};
      double un = std::sqrt(u[0]*u[0]+u[1]*u[1]+u[2]*u[2]);
      if (un < 1e-9) continue;
      for (int k = 0; k < 3; ++k) u[k] /= un;
      double v[3] = {normal[1]*u[2]-normal[2]*u[1], normal[2]*u[0]-normal[0]*u[2],
                     normal[0]*u[1]-normal[1]*u[0]};
      const bool h = is_screen ? hot(HandleKind::RotScreen, 0)
                               : hot(HandleKind::RotAxis, ring);
      const ImU32 col = is_screen
                            ? (h ? IM_COL32(255, 220, 40, 255)
                                 : IM_COL32(210, 210, 210, 255))
                            : AxisColor(ring, h);
      ImVec2 prev; bool have = false;
      for (int s = 0; s <= kRingSegments; ++s) {
        const double ang = 2 * kPi * s / kRingSegments;
        double pt[3];
        for (int k = 0; k < 3; ++k)
          pt[k] = f.anchor[k] + radius * (std::cos(ang) * u[k] +
                                          std::sin(ang) * v[k]);
        Px pp = ToPix(vp, pt, m);
        if (have && pp.visible) dl->AddLine(prev, pp.p, col, 2.2f);
        prev = pp.p;
        have = pp.visible;
      }
    }
  }
}

}  // namespace ps::studio
