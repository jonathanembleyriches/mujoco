// Pose-patch capture + apply (pose.h). Inside the MuJoCo quarantine zone.
//
// PosePatchFor reconstructs the two baked frames of  field = A ∘ L ∘ B  from the
// compiled model + the authored tree, so it is compile-path agnostic (native or
// XML): A is the enclosing <frame> chain (body-local, from the tree); L is the
// element's effective authored local pose (SDK Effective, so a class-inherited
// pose resolves exactly as the compiler saw it); B is captured as the residual
// B = (A ∘ L)^-1 ∘ field read back from mjModel. Capturing B as a residual (a
// single inverse, done ONCE at capture, never during a drag) folds in whatever
// suffix the compiler baked -- mesh recentering, mesh fit, or a body's free-
// joint inertial fold -- without this file re-deriving MuJoCo's mesh convention.

#include "pose.h"

#include <cmath>
#include <optional>
#include <vector>

#include <mujoco/mujoco.h>

#include "binding.h"
#include "protospec/classes.h"    // sdk::Effective
#include "protospec/traversal.h"  // sdk::ParentMap
#include "reflect.h"
#include "types.h"

namespace ps::mjcf {
namespace {

namespace sdk = ps::sdk;

void Normalize(double* v, int n) {
  double s = 0;
  for (int i = 0; i < n; ++i) s += v[i] * v[i];
  s = std::sqrt(s);
  if (s > 1e-12)
    for (int i = 0; i < n; ++i) v[i] /= s;
}

void QuatMul(const double p[4], const double q[4], double out[4]) {
  double t[4] = {
      p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],
      p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2],
      p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1],
      p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0]};
  Normalize(t, 4);
  for (int i = 0; i < 4; ++i) out[i] = t[i];
}

void QuatRotate(const double q[4], const double v[3], double out[3]) {
  const double w = q[0], x = q[1], y = q[2], z = q[3];
  const double m[9] = {1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
                       2 * (x * z + y * w),     2 * (x * y + z * w),
                       1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                       2 * (x * z - y * w),     2 * (y * z + x * w),
                       1 - 2 * (x * x + y * y)};
  double t[3] = {m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
                 m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
                 m[6] * v[0] + m[7] * v[1] + m[8] * v[2]};
  for (int i = 0; i < 3; ++i) out[i] = t[i];
}

// The canonical orientation quat of an element field, or identity when absent.
void QuatOf(const ps::opt<std::array<double, 4>>& q, double out[4]) {
  if (q) {
    for (int i = 0; i < 4; ++i) out[i] = (*q)[i];
    Normalize(out, 4);
  } else {
    out[0] = 1;
    out[1] = out[2] = out[3] = 0;
  }
}

// The effective authored local pose L of a spatial element (Effective resolves a
// class-inherited pos/quat exactly as the compiler saw it). Light has pos but no
// orientation field, so its L rotation is identity. `ctx` is the caller's
// per-capture lookup context (one build serves the whole capture).
template <class E>
RigidPose EffectiveLocal(const sdk::EffectiveContext& ctx, const E& e) {
  RigidPose L;
  std::unique_ptr<E> eff = sdk::Effective(ctx, e, /*apply_idl_defaults=*/true);
  if (eff->pos)
    for (int i = 0; i < 3; ++i) L.pos[i] = (*eff->pos)[i];
  if constexpr (requires { eff->quat; }) QuatOf(eff->quat, L.quat);
  return L;
}

// A = the enclosing <frame> chain of `elem`, composed in body-local coordinates
// (mjModel flattens frames away, folding them into the element's pos/quat as a
// left prefix). Raw field reads are exact here: MJCF defines no <frame> entry
// in <default>, so a frame's own pos/quat is always authored, never
// class-inherited.
RigidPose ReconstructPrefix(const sdk::ParentMap& pm, const void* elem) {
  std::vector<const Frame*> frames;  // innermost first
  const sdk::ParentMap::Node* self = pm.Lookup(elem);
  const void* p = self ? self->parent : nullptr;
  while (p) {
    const sdk::ParentMap::Node* n = pm.Lookup(p);
    if (!n) break;
    if (n->type == ElementType::Frame) {
      frames.push_back(static_cast<const Frame*>(p));
      p = n->parent;
    } else {
      break;  // a Body / world / non-spatial container: A is body-local, stop
    }
  }
  RigidPose A;  // identity
  for (auto it = frames.rbegin(); it != frames.rend(); ++it) {  // outermost first
    const Frame* f = *it;
    RigidPose fr;
    if (f->pos)
      for (int i = 0; i < 3; ++i) fr.pos[i] = (*f->pos)[i];
    QuatOf(f->quat, fr.quat);
    A = Compose(A, fr);
  }
  return A;
}

// The compiled pose field of an object as a RigidPose. Light has no quat field
// (orientation is a direction vector), so its rotation reads as identity.
RigidPose ReadField(const mjModel* m, int objtype, int id) {
  RigidPose F;
  const mjtNum* pos = nullptr;
  const mjtNum* quat = nullptr;
  switch (objtype) {
    case mjOBJ_BODY:   pos = m->body_pos + 3 * id; quat = m->body_quat + 4 * id; break;
    case mjOBJ_GEOM:   pos = m->geom_pos + 3 * id; quat = m->geom_quat + 4 * id; break;
    case mjOBJ_SITE:   pos = m->site_pos + 3 * id; quat = m->site_quat + 4 * id; break;
    case mjOBJ_CAMERA: pos = m->cam_pos  + 3 * id; quat = m->cam_quat  + 4 * id; break;
    case mjOBJ_LIGHT:  pos = m->light_pos + 3 * id; break;  // dir, not quat
    default: break;
  }
  if (pos)
    for (int i = 0; i < 3; ++i) F.pos[i] = pos[i];
  if (quat)
    for (int i = 0; i < 4; ++i) F.quat[i] = quat[i];
  return F;
}

RigidPose LocalOf(const sdk::EffectiveContext& ctx, ElementType etype,
                  const void* elem) {
  switch (etype) {
    case ElementType::Body:
      return EffectiveLocal(ctx, *static_cast<const Body*>(elem));
    case ElementType::Geom:
      return EffectiveLocal(ctx, *static_cast<const Geom*>(elem));
    case ElementType::Site:
      return EffectiveLocal(ctx, *static_cast<const Site*>(elem));
    case ElementType::Camera:
      return EffectiveLocal(ctx, *static_cast<const Camera*>(elem));
    case ElementType::Light:
      return EffectiveLocal(ctx, *static_cast<const Light*>(elem));
    default:
      return {};
  }
}

}  // namespace

// --- Public rigid-transform algebra (declared in pose.h) ------------------ //

RigidPose Compose(const RigidPose& a, const RigidPose& b) {
  RigidPose out;
  QuatMul(a.quat, b.quat, out.quat);
  double rb[3];
  QuatRotate(a.quat, b.pos, rb);
  for (int i = 0; i < 3; ++i) out.pos[i] = rb[i] + a.pos[i];
  return out;
}

RigidPose Invert(const RigidPose& a) {
  RigidPose out;
  out.quat[0] = a.quat[0];
  for (int i = 1; i < 4; ++i) out.quat[i] = -a.quat[i];
  double rp[3];
  QuatRotate(out.quat, a.pos, rp);
  for (int i = 0; i < 3; ++i) out.pos[i] = -rp[i];
  return out;
}

// --- Capture -------------------------------------------------------------- //

std::optional<PosePatch> Binding::PosePatchFor(const void* elem) const {
  if (m_ == nullptr || model_ == nullptr) return std::nullopt;
  auto it = by_ptr_.find(elem);
  if (it == by_ptr_.end()) return std::nullopt;
  const Entry& e = entries_[it->second];
  if (e.id < 0) return std::nullopt;

  switch (e.objtype) {
    case mjOBJ_BODY:
    case mjOBJ_GEOM:
    case mjOBJ_SITE:
    case mjOBJ_CAMERA:
    case mjOBJ_LIGHT:
      break;
    default:
      return std::nullopt;  // not a patchable spatial family
  }

  PosePatch p;
  p.objtype = e.objtype;
  p.id = e.id;
  // One lookup build (parent map + class index) serves the whole capture.
  const sdk::EffectiveContext ectx(*model_);
  p.prefix = ReconstructPrefix(ectx.parents(), elem);
  const RigidPose L = LocalOf(ectx, e.etype, elem);

  // A FREE-jointed body's rest pose is driven by qpos, not body_pos: MuJoCo
  // seeds qpos0 from the body's pose (ComputeReference, user_model.cc) and
  // mj_kinematics ignores body_pos/body_quat when the single joint is free. So
  // record the 7-wide qpos slice ApplyPosePatch must reseed, and read the
  // compiled field FROM qpos0. A BALL joint is NOT special: its qpos0 quat is
  // identity by construction and kinematics composes body_quat * quat(qpos),
  // so the rest pose lives entirely in body_pos/body_quat -- reseeding qpos0
  // there would apply the rotation twice and clobber the authored body_quat.
  RigidPose F = ReadField(m_, e.objtype, e.id);
  if (e.objtype == mjOBJ_BODY) {
    const int nj = m_->body_jntnum[e.id];
    const int adr = m_->body_jntadr[e.id];
    for (int k = 0; k < nj; ++k) {
      const int j = adr + k;
      if (m_->jnt_type[j] == mjJNT_FREE) {
        p.reseed_qposadr = m_->jnt_qposadr[j];
        p.reseed_width = 7;
        const mjtNum* q0 = m_->qpos0 + p.reseed_qposadr;
        for (int i = 0; i < 3; ++i) F.pos[i] = q0[i];
        for (int i = 0; i < 4; ++i) F.quat[i] = q0[3 + i];
        break;
      }
    }
  }

  // B = (A ∘ L)^-1 ∘ F -- the residual suffix. Computed once here; a drag only
  // ever composes A ∘ L_new ∘ B forward (never inverts B), which is exactly
  // correct for mesh geoms (DR-S6).
  p.suffix = Compose(Invert(Compose(p.prefix, L)), F);
  return p;
}

// --- Apply ---------------------------------------------------------------- //

bool ApplyPosePatch(mjModel* m, const PosePatch& p, const RigidPose& L_new) {
  if (m == nullptr || p.id < 0) return false;
  const RigidPose F = Compose(Compose(p.prefix, L_new), p.suffix);

  mjtNum* pos = nullptr;
  mjtNum* quat = nullptr;
  switch (p.objtype) {
    case mjOBJ_BODY:
      pos = m->body_pos + 3 * p.id;
      quat = m->body_quat + 4 * p.id;
      m->body_sameframe[p.id] = 0;  // force mj_kinematics to recompose from body_pos
      break;
    case mjOBJ_GEOM:
      pos = m->geom_pos + 3 * p.id;
      quat = m->geom_quat + 4 * p.id;
      m->geom_sameframe[p.id] = 0;  // clear the frame-coincidence shortcut
      break;
    case mjOBJ_SITE:
      pos = m->site_pos + 3 * p.id;
      quat = m->site_quat + 4 * p.id;
      m->site_sameframe[p.id] = 0;
      break;
    case mjOBJ_CAMERA:
      pos = m->cam_pos + 3 * p.id;
      quat = m->cam_quat + 4 * p.id;
      break;
    case mjOBJ_LIGHT:
      pos = m->light_pos + 3 * p.id;  // dir kept as-is
      break;
    default:
      return false;
  }
  if (pos)
    for (int i = 0; i < 3; ++i) pos[i] = F.pos[i];
  if (quat)
    for (int i = 0; i < 4; ++i) quat[i] = F.quat[i];

  if (p.reseed_qposadr >= 0 && p.reseed_width == 7 && m->qpos0 != nullptr) {
    mjtNum* q0 = m->qpos0 + p.reseed_qposadr;
    for (int i = 0; i < 3; ++i) q0[i] = F.pos[i];
    for (int i = 0; i < 4; ++i) q0[3 + i] = F.quat[i];
  }
  return true;
}

}  // namespace ps::mjcf
