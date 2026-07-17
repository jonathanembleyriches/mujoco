// ProtoSpec Studio: gizmo transform math -- THE delta rule (DR-S6). See header.

#include "editor/transform_math.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include <mujoco/mujoco.h>

#include "protospec/classes.h"    // sdk::Effective, HasDefaultFamily
#include "protospec/detail.h"     // WalkModelAll, NameOf
#include "protospec/traversal.h"  // ParentMap, Find
#include "reflect.h"
#include "visit.h"

namespace ps::studio {

namespace sdk = ps::sdk;
namespace sdkd = ps::sdk::detail;

// --- Quaternion / rigid-transform primitives ------------------------------ //

static double Norm(double* v, int n) {
  double s = 0;
  for (int i = 0; i < n; ++i) s += v[i] * v[i];
  s = std::sqrt(s);
  if (s > 1e-12) {
    for (int i = 0; i < n; ++i) v[i] /= s;
  }
  return s;
}

void QuatMul(const double p[4], const double q[4], double out[4]) {
  double t[4] = {
      p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],
      p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2],
      p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1],
      p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0]};
  Norm(t, 4);
  for (int i = 0; i < 4; ++i) out[i] = t[i];
}

static void QuatConj(const double q[4], double out[4]) {
  out[0] = q[0];
  out[1] = -q[1];
  out[2] = -q[2];
  out[3] = -q[3];
}

void QuatRotate(const double q[4], const double v[3], double out[3]) {
  // out = R(q) * v, with R from the unit quaternion.
  const double w = q[0], x = q[1], y = q[2], z = q[3];
  const double m[9] = {1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
                       2 * (x * z + y * w),     2 * (x * y + z * w),
                       1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                       2 * (x * z - y * w),     2 * (y * z + x * w),
                       1 - 2 * (x * x + y * y)};
  double t[3] = {m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
                 m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
                 m[6] * v[0] + m[7] * v[1] + m[8] * v[2]};
  out[0] = t[0];
  out[1] = t[1];
  out[2] = t[2];
}

// Unit quat rotating +z onto `vec` (mjuu_z2quat / build.cc fromto path), so the
// gizmo's local axes align with the derived fromto capsule axis.
static void Z2Quat(double quat[4], const double vec[3]) {
  double v[3] = {vec[0], vec[1], vec[2]};
  double axis[3] = {-v[1], v[0], 0};  // z x v
  const double s = Norm(axis, 3);
  const double ang = std::atan2(s, v[2]);
  if (s < 1e-10) {
    axis[0] = 1;
    axis[1] = axis[2] = 0;
  }
  QuatFromAxisAngle(axis, ang, quat);
}

void QuatFromAxisAngle(const double axis[3], double angle, double out[4]) {
  double a[3] = {axis[0], axis[1], axis[2]};
  if (Norm(a, 3) < 1e-12) {
    out[0] = 1;
    out[1] = out[2] = out[3] = 0;
    return;
  }
  const double s = std::sin(angle / 2);
  out[0] = std::cos(angle / 2);
  out[1] = s * a[0];
  out[2] = s * a[1];
  out[3] = s * a[2];
}

Rigid Compose(const Rigid& a, const Rigid& b) {
  Rigid out;
  QuatMul(a.quat, b.quat, out.quat);
  double rb[3];
  QuatRotate(a.quat, b.pos, rb);
  out.pos[0] = rb[0] + a.pos[0];
  out.pos[1] = rb[1] + a.pos[1];
  out.pos[2] = rb[2] + a.pos[2];
  return out;
}

Rigid Invert(const Rigid& a) {
  Rigid out;
  QuatConj(a.quat, out.quat);
  double rp[3];
  QuatRotate(out.quat, a.pos, rp);
  out.pos[0] = -rp[0];
  out.pos[1] = -rp[1];
  out.pos[2] = -rp[2];
  return out;
}

// --- Orientation (canonical quat; Q-ORIENT) ------------------------------- //

OrientContext ReadOrientContext(const mj::Model& model) {
  OrientContext oc;
  for (const auto& c : model.compilers) {
    if (!c) continue;
    if (c->angle) oc.degree = (*c->angle == mj::AngleUnit::degree);
    if (c->eulerseq) oc.eulerseq = *c->eulerseq;
  }
  return oc;
}

void QuatOf(const ps::opt<std::array<double, 4>>& quat, double out[4]) {
  if (quat) {
    out[0] = (*quat)[0];
    out[1] = (*quat)[1];
    out[2] = (*quat)[2];
    out[3] = (*quat)[3];
    Norm(out, 4);
  } else {
    out[0] = 1;
    out[1] = out[2] = out[3] = 0;
  }
}

// --- Spatial element resolution ------------------------------------------- //

static bool IsSpatial(mj::ElementType t) {
  return t == mj::ElementType::Body || t == mj::ElementType::Geom ||
         t == mj::ElementType::Site || t == mj::ElementType::Camera ||
         t == mj::ElementType::Light || t == mj::ElementType::Frame;
}

SpatialRef FindSpatial(mj::Model& tree, std::uint64_t serial) {
  SpatialRef out;
  if (serial == 0) return out;
  sdkd::WalkModelAll(tree, [&](auto& e) {
    using E = std::decay_t<decltype(e)>;
    if (out.ptr) return;
    if constexpr (requires { e.serial; }) {
      if constexpr (!std::is_same_v<E, mj::Model>) {
        const mj::ElementType t = mj::element_type_of<E>::value;
        if (e.serial == serial && IsSpatial(t)) {
          out.ptr = &e;
          out.type = t;
        }
      }
    }
  });
  return out;
}

// The world pose of a body id at qpos0 (data->xpos/xquat).
static Rigid BodyWorldPose(const mjData* data, int body_id) {
  Rigid r;
  if (body_id < 0) return r;
  for (int i = 0; i < 3; ++i) r.pos[i] = data->xpos[3 * body_id + i];
  for (int i = 0; i < 4; ++i) r.quat[i] = data->xquat[4 * body_id + i];
  Norm(r.quat, 4);
  return r;
}

// Compose the parent world pose P: nearest body ancestor's compiled world pose
// composed with the authored transforms of the <frame> nodes between that body
// and the element (outermost frame first). Frames are absent from mjModel, so
// their poses come from the tree.
static Rigid ComputeParentPose(const mjData* data,
                               const mj::Binding& binding,
                               const sdk::ParentMap& pm, const OrientContext& oc,
                               const void* elem_ptr) {
  std::vector<const mj::Frame*> frames;  // innermost first
  const mj::Body* body = nullptr;

  const sdk::ParentMap::Node* self = pm.Lookup(elem_ptr);
  const void* p = self ? self->parent : nullptr;
  while (p) {
    const sdk::ParentMap::Node* n = pm.Lookup(p);
    if (!n) break;
    if (n->type == mj::ElementType::Frame) {
      frames.push_back(static_cast<const mj::Frame*>(p));
      p = n->parent;
    } else if (n->type == mj::ElementType::Body) {
      body = static_cast<const mj::Body*>(p);
      break;
    } else {
      break;  // worldbody (Model root) or a non-spatial container -> world frame
    }
  }

  Rigid P;  // identity == worldbody
  if (body) {
    if (std::optional<int> id = binding.Id(*body)) {
      P = BodyWorldPose(data, *id);
    }
  }
  // Compose frames from outermost (nearest the body) to innermost.
  for (auto it = frames.rbegin(); it != frames.rend(); ++it) {
    const mj::Frame* f = *it;
    Rigid fr;
    if (f->pos) {
      fr.pos[0] = (*f->pos)[0];
      fr.pos[1] = (*f->pos)[1];
      fr.pos[2] = (*f->pos)[2];
    }
    (void)oc;
    QuatOf(f->quat, fr.quat);
    P = Compose(P, fr);
  }
  return P;
}

// Materialise an element's authored local pose (Effective fills a class-inherited
// pos/orient for the default families; Body/Frame default to origin/identity;
// Light carries pos but no orient, so its local rotation is identity).
//
// pos and quat resolve INDEPENDENTLY: authored wins, else the class-effective
// value. They must not be coupled -- the old form consulted Effective only when
// pos was unset, so the first translate of an element with a class-inherited
// orientation (pos becomes authored) silently dropped the class quat to
// identity, and the live drag preview snapped by the class rotation. This is
// also the convention PosePatchFor's B capture uses (pose.cc EffectiveLocal);
// the two sides composing A . L . B must agree on what L is.
template <class E>
static Rigid MaterializeLocal(mj::Model& tree, E& e, const OrientContext& oc) {
  Rigid L;
  (void)oc;
  std::array<double, 3> pos{0, 0, 0};
  ps::opt<std::array<double, 4>> quat;
  if constexpr (requires { e.quat; }) quat = e.quat;
  if (e.pos) pos = *e.pos;
  if constexpr (sdk::HasDefaultFamily<E>::value) {
    bool need_eff = !e.pos;
    if constexpr (requires { e.quat; }) need_eff = need_eff || !quat;
    if (need_eff) {
      std::unique_ptr<E> eff = sdk::Effective(tree, e, true);
      if (!e.pos && eff->pos) pos = *eff->pos;
      if constexpr (requires { e.quat; }) {
        if (!quat) quat = eff->quat;
      }
    }
  }
  L.pos[0] = pos[0];
  L.pos[1] = pos[1];
  L.pos[2] = pos[2];
  QuatOf(quat, L.quat);
  return L;
}

// The effective fromto endpoints of a geom/site (via Effective, so a
// class-inherited shape resolves), or false when the element is not
// fromto-authored. Only Geom/Site carry the `shape` variant.
template <class E>
static bool EffectiveFromTo(mj::Model& tree, E& e, double from[3],
                            double to[3]) {
  if constexpr (requires { e.shape; }) {
    ps::opt<mj::GeomShape> shape;
    if (e.shape) {
      shape = e.shape;
    } else if constexpr (sdk::HasDefaultFamily<E>::value) {
      std::unique_ptr<E> eff = sdk::Effective(tree, e, true);
      shape = eff->shape;
    }
    if (shape) {
      if (const mj::FromTo* ft = std::get_if<mj::FromTo>(&*shape)) {
        for (int i = 0; i < 3; ++i) {
          from[i] = ft->fromto[i];
          to[i] = ft->fromto[i + 3];
        }
        return true;
      }
    }
  }
  return false;
}

template <class E>
static DragFrame BuildFor(mj::Model& tree, const mjModel* model,
                          const mjData* data, const mj::Binding& binding,
                          const sdk::ParentMap& pm, const OrientContext& oc,
                          E& e) {
  DragFrame f;
  f.valid = true;
  f.type = mj::element_type_of<E>::value;
  f.parent = ComputeParentPose(data, binding, pm, oc, &e);
  f.local = MaterializeLocal(tree, e, oc);

  // fromto override: the compiled pose is the endpoint midpoint + the z->axis
  // rotation, NOT the authored pos/orient. Anchor the gizmo on the capsule
  // centre and align world_quat with the derived axis so local-axis modes and
  // the scale gizmo act on the limb, not on an ignored authored frame.
  if (EffectiveFromTo(tree, e, f.from, f.to)) {
    f.is_fromto = true;
    double vec[3] = {f.from[0] - f.to[0], f.from[1] - f.to[1],
                     f.from[2] - f.to[2]};
    for (int i = 0; i < 3; ++i) f.local.pos[i] = 0.5 * (f.from[i] + f.to[i]);
    Z2Quat(f.local.quat, vec);
  } else {
    // B, the baked residual suffix (mesh/fit recentering). The element's
    // visible centre is P . L . B; anchoring on P . L put every unposed mesh
    // part's gizmo at the body origin. fromto keeps its own convention: there
    // the LivePatch residual trick already owns the suffix.
    if (auto patch = binding.PosePatchFor(e)) {
      for (int i = 0; i < 4; ++i) f.suffix.quat[i] = patch->suffix.quat[i];
      for (int i = 0; i < 3; ++i) f.suffix.pos[i] = patch->suffix.pos[i];
    }
    if constexpr (std::is_same_v<E, mj::Geom>) {
      if (model) {
        if (auto id = binding.Id(e)) {
          f.is_mesh = model->geom_type[*id] == mjGEOM_MESH;
        }
      }
    }
  }

  // anchor = P . L . B (the visible centre);  world_quat follows the same
  // composition so local-axis modes act on the axes the user can see.
  const Rigid lb = f.is_fromto ? f.local : Compose(f.local, f.suffix);
  double rl[3];
  QuatRotate(f.parent.quat, lb.pos, rl);
  for (int i = 0; i < 3; ++i) f.anchor[i] = f.parent.pos[i] + rl[i];
  QuatMul(f.parent.quat, lb.quat, f.world_quat);
  return f;
}

DragFrame BuildDragFrame(const mjModel* model, const mjData* data,
                         const mj::Binding& binding, mj::Model& tree,
                         std::uint64_t serial) {
  DragFrame f;
  SpatialRef ref = FindSpatial(tree, serial);
  if (!ref || !data) return f;
  const OrientContext oc = ReadOrientContext(tree);
  sdk::ParentMap pm(tree);
  switch (ref.type) {
    case mj::ElementType::Body:
      return BuildFor(tree, model, data, binding, pm, oc,
                      *static_cast<mj::Body*>(ref.ptr));
    case mj::ElementType::Geom:
      return BuildFor(tree, model, data, binding, pm, oc,
                      *static_cast<mj::Geom*>(ref.ptr));
    case mj::ElementType::Site:
      return BuildFor(tree, model, data, binding, pm, oc,
                      *static_cast<mj::Site*>(ref.ptr));
    case mj::ElementType::Camera:
      return BuildFor(tree, model, data, binding, pm, oc,
                      *static_cast<mj::Camera*>(ref.ptr));
    case mj::ElementType::Light:
      return BuildFor(tree, model, data, binding, pm, oc,
                      *static_cast<mj::Light*>(ref.ptr));
    case mj::ElementType::Frame:
      return BuildFor(tree, model, data, binding, pm, oc,
                      *static_cast<mj::Frame*>(ref.ptr));
    default:
      return f;
  }
}

Rigid FrameChainPrefix(mj::Model& tree, std::uint64_t serial) {
  Rigid A;  // identity
  if (serial == 0) return A;
  const void* elem = nullptr;
  sdkd::WalkModelAll(tree, [&](auto& e) {
    using E = std::decay_t<decltype(e)>;
    if constexpr (!std::is_same_v<E, mj::Model>) {
      if constexpr (requires { e.serial; }) {
        if (!elem && e.serial == serial) elem = &e;
      }
    }
  });
  if (!elem) return A;

  sdk::ParentMap pm(tree);
  std::vector<const mj::Frame*> frames;  // innermost first
  const sdk::ParentMap::Node* self = pm.Lookup(elem);
  const void* p = self ? self->parent : nullptr;
  while (p) {
    const sdk::ParentMap::Node* n = pm.Lookup(p);
    if (!n) break;
    if (n->type == mj::ElementType::Frame) {
      frames.push_back(static_cast<const mj::Frame*>(p));
      p = n->parent;
    } else {
      break;  // a Body / world / non-spatial container: A is body-local, stop
    }
  }
  for (auto it = frames.rbegin(); it != frames.rend(); ++it) {  // outermost first
    const mj::Frame* f = *it;
    Rigid fr;
    if (f->pos) {
      fr.pos[0] = (*f->pos)[0];
      fr.pos[1] = (*f->pos)[1];
      fr.pos[2] = (*f->pos)[2];
    }
    QuatOf(f->quat, fr.quat);
    A = Compose(A, fr);
  }
  return A;
}

Rigid EffectiveLocalPose(mj::Model& tree, std::uint64_t serial) {
  SpatialRef ref = FindSpatial(tree, serial);
  if (!ref) return {};
  const OrientContext oc = ReadOrientContext(tree);
  switch (ref.type) {
    case mj::ElementType::Body:
      return MaterializeLocal(tree, *static_cast<mj::Body*>(ref.ptr), oc);
    case mj::ElementType::Geom:
      return MaterializeLocal(tree, *static_cast<mj::Geom*>(ref.ptr), oc);
    case mj::ElementType::Site:
      return MaterializeLocal(tree, *static_cast<mj::Site*>(ref.ptr), oc);
    case mj::ElementType::Camera:
      return MaterializeLocal(tree, *static_cast<mj::Camera*>(ref.ptr), oc);
    case mj::ElementType::Light:
      return MaterializeLocal(tree, *static_cast<mj::Light*>(ref.ptr), oc);
    case mj::ElementType::Frame:
      return MaterializeLocal(tree, *static_cast<mj::Frame*>(ref.ptr), oc);
    default:
      return {};
  }
}

// --- Delta application ---------------------------------------------------- //

template <class E>
static void WritePos(E& e, const double pos[3]) {
  e.pos = std::array<double, 3>{pos[0], pos[1], pos[2]};
}
template <class E>
static void WriteQuat(E& e, const double q[4]) {
  double n[4] = {q[0], q[1], q[2], q[3]};
  Norm(n, 4);
  // Q-ORIENT: authored orientation IS the canonical quat now (no variant).
  e.quat = std::array<double, 4>{n[0], n[1], n[2], n[3]};
}

template <class E>
static void WriteFromTo(E& e, const double from[3], const double to[3]) {
  e.shape = mj::GeomShape(mj::FromTo{
      {from[0], from[1], from[2], to[0], to[1], to[2]}});
}

template <class E>
static void TranslateElem(E& e, const DragFrame& f, const double world_delta[3]) {
  // Parent-frame delta: inv(P).quat * world_delta.
  double pq[4];
  QuatConj(f.parent.quat, pq);
  double dl[3];
  QuatRotate(pq, world_delta, dl);
  if constexpr (requires { e.shape; }) {
    // fromto: translate BOTH endpoints by the parent-frame delta (the midpoint,
    // hence the compiled pose, follows; the axis is unchanged). Writing pos here
    // would be silently ignored by the compiler.
    if (f.is_fromto) {
      double nf[3], nt[3];
      for (int i = 0; i < 3; ++i) {
        nf[i] = f.from[i] + dl[i];
        nt[i] = f.to[i] + dl[i];
      }
      WriteFromTo(e, nf, nt);
      return;
    }
  }
  // L_new.pos = L_base.pos + inv(P).quat * world_delta.
  double np[3] = {f.local.pos[0] + dl[0], f.local.pos[1] + dl[1],
                  f.local.pos[2] + dl[2]};
  WritePos(e, np);
}

template <class E>
static void RotateElem(E& e, const DragFrame& f, const double axis[3],
                       double angle) {
  double qd[4];
  QuatFromAxisAngle(axis, angle, qd);
  double pq[4];
  QuatConj(f.parent.quat, pq);
  if constexpr (requires { e.shape; }) {
    // fromto: rotate BOTH endpoints about their midpoint by the parent-frame
    // rotation qL = conj(P).quat . qD . P.quat. Radius (size) and the derived
    // pos/quat are untouched. A rotation about an axis parallel to the fromto
    // direction is a NO-OP (the endpoint offsets lie along that axis and are
    // fixed by the rotation) -- fromto capsules/cylinders are axisymmetric, so
    // twist about the limb axis is not representable and produces no visible
    // change, which is the correct behaviour.
    if (f.is_fromto) {
      double t2[4], qL[4];
      QuatMul(qd, f.parent.quat, t2);
      QuatMul(pq, t2, qL);
      double m[3];
      for (int i = 0; i < 3; ++i) m[i] = 0.5 * (f.from[i] + f.to[i]);
      double vf[3] = {f.from[0] - m[0], f.from[1] - m[1], f.from[2] - m[2]};
      double vt[3] = {f.to[0] - m[0], f.to[1] - m[1], f.to[2] - m[2]};
      double rf[3], rt[3];
      QuatRotate(qL, vf, rf);
      QuatRotate(qL, vt, rt);
      double nf[3], nt[3];
      for (int i = 0; i < 3; ++i) {
        nf[i] = m[i] + rf[i];
        nt[i] = m[i] + rt[i];
      }
      WriteFromTo(e, nf, nt);
      return;
    }
  }
  if constexpr (requires { e.quat; }) {
    // L_new.quat = conj(P.quat) . qD . P.quat . L_base.quat.
    double t1[4], t2[4], t3[4];
    QuatMul(f.parent.quat, f.local.quat, t1);
    QuatMul(qd, t1, t2);
    QuatMul(pq, t2, t3);
    WriteQuat(e, t3);

    // Pivot on the VISIBLE centre (L . B), not the authored frame origin. For a
    // suffix-free element the two coincide and pos stays untouched (the old
    // guarantee). For a mesh geom the compiler bakes a recentering offset B, so
    // holding pos while rotating L makes the mesh ORBIT its authored origin;
    // keeping the anchor fixed instead means solving
    //   L_new.pos + R(L_new.quat) . B.pos  ==  L.pos + R(L.quat) . B.pos.
    const bool has_suffix =
        std::abs(f.suffix.pos[0]) + std::abs(f.suffix.pos[1]) +
            std::abs(f.suffix.pos[2]) + std::abs(1.0 - f.suffix.quat[0]) >
        1e-12;
    if (has_suffix) {
      double c[3], rb_old[3], rb_new[3];
      QuatRotate(f.local.quat, f.suffix.pos, rb_old);
      for (int i = 0; i < 3; ++i) c[i] = f.local.pos[i] + rb_old[i];
      QuatRotate(t3, f.suffix.pos, rb_new);
      double np[3] = {c[0] - rb_new[0], c[1] - rb_new[1], c[2] - rb_new[2]};
      WritePos(e, np);
    }
  } else if constexpr (requires { e.dir; }) {
    // A light has no orientation; rotate its authored direction vector by the
    // conjugated world delta (same delta rule, applied to a direction).
    std::array<double, 3> d = e.dir ? *e.dir : std::array<double, 3>{0, 0, -1};
    double wd[3], rwd[3], nd[3];
    QuatRotate(f.parent.quat, d.data(), wd);
    QuatRotate(qd, wd, rwd);
    QuatRotate(pq, rwd, nd);
    e.dir = std::array<double, 3>{nd[0], nd[1], nd[2]};
  }
}

void ApplyTranslate(mj::Model& tree, std::uint64_t serial, const DragFrame& f,
                    const double world_delta[3]) {
  SpatialRef ref = FindSpatial(tree, serial);
  if (!ref) return;
  switch (ref.type) {
    case mj::ElementType::Body:
      TranslateElem(*static_cast<mj::Body*>(ref.ptr), f, world_delta); break;
    case mj::ElementType::Geom:
      TranslateElem(*static_cast<mj::Geom*>(ref.ptr), f, world_delta); break;
    case mj::ElementType::Site:
      TranslateElem(*static_cast<mj::Site*>(ref.ptr), f, world_delta); break;
    case mj::ElementType::Camera:
      TranslateElem(*static_cast<mj::Camera*>(ref.ptr), f, world_delta); break;
    case mj::ElementType::Light:
      TranslateElem(*static_cast<mj::Light*>(ref.ptr), f, world_delta); break;
    case mj::ElementType::Frame:
      TranslateElem(*static_cast<mj::Frame*>(ref.ptr), f, world_delta); break;
    default: break;
  }
}

void ApplyRotate(mj::Model& tree, std::uint64_t serial, const DragFrame& f,
                 const double axis[3], double angle) {
  SpatialRef ref = FindSpatial(tree, serial);
  if (!ref) return;
  switch (ref.type) {
    case mj::ElementType::Body:
      RotateElem(*static_cast<mj::Body*>(ref.ptr), f, axis, angle); break;
    case mj::ElementType::Geom:
      RotateElem(*static_cast<mj::Geom*>(ref.ptr), f, axis, angle); break;
    case mj::ElementType::Site:
      RotateElem(*static_cast<mj::Site*>(ref.ptr), f, axis, angle); break;
    case mj::ElementType::Camera:
      RotateElem(*static_cast<mj::Camera*>(ref.ptr), f, axis, angle); break;
    case mj::ElementType::Light:
      RotateElem(*static_cast<mj::Light*>(ref.ptr), f, axis, angle); break;
    case mj::ElementType::Frame:
      RotateElem(*static_cast<mj::Frame*>(ref.ptr), f, axis, angle); break;
    default: break;
  }
}

// --- Scale ---------------------------------------------------------------- //

static int SizeDofsFor(mj::GeomType t) {
  switch (t) {
    case mj::GeomType::sphere: return 1;
    case mj::GeomType::capsule:
    case mj::GeomType::cylinder:
    case mj::GeomType::plane: return 2;
    case mj::GeomType::box:
    case mj::GeomType::ellipsoid: return 3;
    default: return 3;
  }
}

template <class E>
static ScaleBase BuildScaleFor(mj::Model& tree, E& e) {
  ScaleBase b;
  b.valid = true;
  std::unique_ptr<E> eff = sdk::Effective(tree, e, true);
  const mj::GeomType type =
      eff->type ? *eff->type : mj::GeomType::sphere;
  if (type == mj::GeomType::mesh) {
    b.is_mesh = true;
    if constexpr (requires { e.mesh; }) {
      if (e.mesh) {
        if (const mj::Mesh* m = sdk::Find<mj::Mesh>(tree, e.mesh->name)) {
          b.mesh_serial = m->serial;
          if (m->scale) {
            b.mesh_scale[0] = (*m->scale)[0];
            b.mesh_scale[1] = (*m->scale)[1];
            b.mesh_scale[2] = (*m->scale)[2];
          }
        }
      }
    }
    return b;
  }
  b.size_dofs = SizeDofsFor(type);
  if (eff->size) {
    for (std::size_t i = 0; i < eff->size->size() && i < 3; ++i) {
      b.size[i] = (*eff->size)[i];
    }
  }
  // fromto: the long axis (half-length) is derived from the endpoints, so the
  // scale gizmo drives the endpoint separation for the axis component and the
  // `size` radius for the radial components. size[0] holds the grab-time radius.
  if constexpr (requires { e.shape; }) {
    if (EffectiveFromTo(tree, e, b.from, b.to)) b.is_fromto = true;
  }
  return b;
}

ScaleBase BuildScaleBase(mj::Model& tree, std::uint64_t serial) {
  SpatialRef ref = FindSpatial(tree, serial);
  if (!ref) return {};
  if (ref.type == mj::ElementType::Geom)
    return BuildScaleFor(tree, *static_cast<mj::Geom*>(ref.ptr));
  if (ref.type == mj::ElementType::Site)
    return BuildScaleFor(tree, *static_cast<mj::Site*>(ref.ptr));
  return {};
}

template <class E>
static void ApplyScaleFromTo(E& e, const ScaleBase& base, const double factor[3]) {
  // Local z is the fromto axis (Z2Quat), so factor[2] drives the half-length by
  // moving the endpoints about their midpoint; factor[0] drives the radius.
  const double axf = factor[2];
  const double rf = factor[0];
  double m[3], nf[3], nt[3];
  for (int i = 0; i < 3; ++i) {
    m[i] = 0.5 * (base.from[i] + base.to[i]);
    nf[i] = m[i] + axf * (base.from[i] - m[i]);
    nt[i] = m[i] + axf * (base.to[i] - m[i]);
  }
  WriteFromTo(e, nf, nt);
  ps::InlineVec<double, 3> size;   // fromto derives the axis size; only the
  size.push_back(base.size[0] * rf);  // radius (size[0]) is authored.
  e.size = size;
}

template <class E>
static void ApplyScaleSize(E& e, const ScaleBase& base, const double factor[3]) {
  ps::InlineVec<double, 3> size;
  const int n = base.size_dofs > 0 ? base.size_dofs : 3;
  for (int i = 0; i < n; ++i) {
    const double fx = (i < 3) ? factor[i] : factor[0];
    size.push_back(base.size[i] * fx);
  }
  e.size = size;
}

void ApplyScale(mj::Model& tree, std::uint64_t serial, const ScaleBase& base,
                const double factor[3]) {
  if (!base.valid) return;
  if (base.is_mesh) {
    if (mj::Mesh* m = [&]() -> mj::Mesh* {
          mj::Mesh* found = nullptr;
          sdkd::WalkModelAll(tree, [&](auto& x) {
            using X = std::decay_t<decltype(x)>;
            if constexpr (std::is_same_v<X, mj::Mesh>) {
              if (!found && x.serial == base.mesh_serial) found = &x;
            }
          });
          return found;
        }()) {
      m->scale = std::array<double, 3>{base.mesh_scale[0] * factor[0],
                                       base.mesh_scale[1] * factor[1],
                                       base.mesh_scale[2] * factor[2]};
    }
    return;
  }
  SpatialRef ref = FindSpatial(tree, serial);
  if (!ref) return;
  if (ref.type == mj::ElementType::Geom) {
    if (base.is_fromto)
      ApplyScaleFromTo(*static_cast<mj::Geom*>(ref.ptr), base, factor);
    else
      ApplyScaleSize(*static_cast<mj::Geom*>(ref.ptr), base, factor);
  } else if (ref.type == mj::ElementType::Site) {
    if (base.is_fromto)
      ApplyScaleFromTo(*static_cast<mj::Site*>(ref.ptr), base, factor);
    else
      ApplyScaleSize(*static_cast<mj::Site*>(ref.ptr), base, factor);
  }
}

void ApplyScaleMeshUniform(mj::Model& tree, std::uint64_t serial,
                           const ScaleBase& base, const DragFrame& f,
                           double factor) {
  if (!base.valid || !base.is_mesh) return;
  if (factor < 1e-3) factor = 1e-3;

  // The mesh asset: every axis of the grab-time scale multiplies by the same
  // uniform factor (a non-uniform authored scale keeps its proportions).
  mj::Mesh* mesh = nullptr;
  sdkd::WalkModelAll(tree, [&](auto& x) {
    using X = std::decay_t<decltype(x)>;
    if constexpr (std::is_same_v<X, mj::Mesh>) {
      if (!mesh && x.serial == base.mesh_serial) mesh = &x;
    }
  });
  if (!mesh) return;
  mesh->scale = std::array<double, 3>{base.mesh_scale[0] * factor,
                                      base.mesh_scale[1] * factor,
                                      base.mesh_scale[2] * factor};

  // Hold the visible centre: see the header. Only when a recentering suffix
  // exists -- for a mesh already centred at its own origin there is nothing to
  // compensate, and writing pos would needlessly author the field.
  const bool has_suffix =
      std::abs(f.suffix.pos[0]) + std::abs(f.suffix.pos[1]) +
          std::abs(f.suffix.pos[2]) >
      1e-12;
  if (!has_suffix) return;
  SpatialRef ref = FindSpatial(tree, serial);
  if (!ref || ref.type != mj::ElementType::Geom) return;
  double rb0[3], rbf[3], c0[3], np[3];
  QuatRotate(f.local.quat, f.suffix.pos, rb0);
  double scaled[3] = {f.suffix.pos[0] * factor, f.suffix.pos[1] * factor,
                      f.suffix.pos[2] * factor};
  QuatRotate(f.local.quat, scaled, rbf);
  for (int i = 0; i < 3; ++i) {
    c0[i] = f.local.pos[i] + rb0[i];
    np[i] = c0[i] - rbf[i];
  }
  WritePos(*static_cast<mj::Geom*>(ref.ptr), np);
}

void ApplyScaleMeshNonUniform(mj::Model& tree, std::uint64_t serial,
                              const ScaleBase& base, const DragFrame& f,
                              const mj::Binding& binding,
                              const double factor[3]) {
  if (!base.valid || !base.is_mesh) return;
  double fx[3];
  for (int i = 0; i < 3; ++i) fx[i] = factor[i] < 1e-3 ? 1e-3 : factor[i];

  mj::Mesh* mesh = nullptr;
  sdkd::WalkModelAll(tree, [&](auto& x) {
    using X = std::decay_t<decltype(x)>;
    if constexpr (std::is_same_v<X, mj::Mesh>) {
      if (!mesh && x.serial == base.mesh_serial) mesh = &x;
    }
  });
  if (!mesh) return;
  mesh->scale = std::array<double, 3>{base.mesh_scale[0] * fx[0],
                                      base.mesh_scale[1] * fx[1],
                                      base.mesh_scale[2] * fx[2]};

  // Hold the grab-time visible centre using the MEASURED current B (see the
  // header). Falls back to no compensation when the element is not patchable
  // (e.g. a pruned-layer compile clone owns the binding pointers).
  SpatialRef ref = FindSpatial(tree, serial);
  if (!ref || ref.type != mj::ElementType::Geom) return;
  auto patch = binding.PosePatchFor(ref.ptr);
  if (!patch) return;
  double c0[3], rb0[3], rbc[3], np[3];
  QuatRotate(f.local.quat, f.suffix.pos, rb0);
  const double bcur[3] = {patch->suffix.pos[0], patch->suffix.pos[1],
                          patch->suffix.pos[2]};
  QuatRotate(f.local.quat, bcur, rbc);
  for (int i = 0; i < 3; ++i) {
    c0[i] = f.local.pos[i] + rb0[i];
    np[i] = c0[i] - rbc[i];
  }
  WritePos(*static_cast<mj::Geom*>(ref.ptr), np);
}

// ========================================================================== //
// Joint rigging (SE4). Deliberately kept in its own block so a later upstream
// transform_math sync merges cleanly: none of the generic FindSpatial /
// BuildDragFrame / Translate / Rotate / Scale code above is touched. Joints
// carry `pos` (the hinge/slide anchor, in the body frame) and, for hinge/slide,
// an `axis` unit vector; both are edited by the same delta rule (DR-S6).
// ========================================================================== //

// The Joint element with `serial`, or nullptr (also nullptr for a FreeJoint).
static mj::Joint* FindJoint(mj::Model& tree, std::uint64_t serial) {
  mj::Joint* out = nullptr;
  sdkd::WalkModelAll(tree, [&](auto& e) {
    using E = std::decay_t<decltype(e)>;
    if constexpr (std::is_same_v<E, mj::Joint>) {
      if (!out && e.serial == serial) out = &e;
    }
  });
  return out;
}

// The FreeJoint element pointer with `serial`, or nullptr.
static const void* FindFreeJoint(mj::Model& tree, std::uint64_t serial) {
  const void* out = nullptr;
  sdkd::WalkModelAll(tree, [&](auto& e) {
    using E = std::decay_t<decltype(e)>;
    if constexpr (std::is_same_v<E, mj::FreeJoint>) {
      if (!out && e.serial == serial) out = &e;
    }
  });
  return out;
}

bool IsJointSerial(mj::Model& tree, std::uint64_t serial) {
  if (serial == 0) return false;
  return FindJoint(tree, serial) != nullptr || FindFreeJoint(tree, serial);
}

JointDragFrame BuildJointDragFrame(const mjModel* model, const mjData* data,
                                   const mj::Binding& binding,
                                   mj::Model& tree, std::uint64_t serial) {
  (void)model;
  JointDragFrame f;
  if (!data || serial == 0) return f;
  const OrientContext oc = ReadOrientContext(tree);
  sdk::ParentMap pm(tree);

  const void* elem = nullptr;
  if (mj::Joint* j = FindJoint(tree, serial)) {
    elem = j;
    f.type = j->type ? *j->type : mj::JointType::hinge;
    std::array<double, 3> pos{0, 0, 0};
    std::array<double, 3> axis{0, 0, 1};
    std::unique_ptr<mj::Joint> eff = sdk::Effective(tree, *j, true);
    if (eff->pos) pos = *eff->pos;
    if (eff->axis) axis = *eff->axis;
    for (int i = 0; i < 3; ++i) {
      f.pos[i] = pos[i];
      f.axis[i] = axis[i];
    }
    Norm(f.axis, 3);
    f.has_axis =
        (f.type == mj::JointType::hinge || f.type == mj::JointType::slide);
  } else if (const void* fj = FindFreeJoint(tree, serial)) {
    elem = fj;
    f.type = mj::JointType::free;
    f.has_axis = false;  // free = 6 DOF at the body origin, no anchor/axis edit
  } else {
    return f;
  }

  f.valid = true;
  f.parent = ComputeParentPose(data, binding, pm, oc, elem);
  double rp[3];
  QuatRotate(f.parent.quat, f.pos, rp);
  for (int i = 0; i < 3; ++i) f.world_anchor[i] = f.parent.pos[i] + rp[i];
  QuatRotate(f.parent.quat, f.axis, f.world_axis);
  Norm(f.world_axis, 3);
  return f;
}

void ApplyJointTranslate(mj::Model& tree, std::uint64_t serial,
                         const JointDragFrame& f, const double world_delta[3]) {
  mj::Joint* j = FindJoint(tree, serial);
  if (!j) return;  // FreeJoint has no anchor to move.
  // Parent-frame delta: inv(P).quat * world_delta, added to the authored anchor.
  double pq[4];
  QuatConj(f.parent.quat, pq);
  double dl[3];
  QuatRotate(pq, world_delta, dl);
  j->pos = std::array<double, 3>{f.pos[0] + dl[0], f.pos[1] + dl[1],
                                 f.pos[2] + dl[2]};
}

void ApplyJointAxisRotate(mj::Model& tree, std::uint64_t serial,
                          const JointDragFrame& f, const double world_axis[3],
                          double angle, bool snap_axis) {
  mj::Joint* j = FindJoint(tree, serial);
  if (!j || !f.has_axis) return;
  // Grab-time axis in world, rotated by the world delta, brought back to the
  // parent frame: a_new = conj(P).quat * qD * P.quat * a_grab.
  double qd[4];
  QuatFromAxisAngle(world_axis, angle, qd);
  double w[3];
  QuatRotate(f.parent.quat, f.axis, w);
  double wr[3];
  QuatRotate(qd, w, wr);
  double pq[4];
  QuatConj(f.parent.quat, pq);
  double la[3];
  QuatRotate(pq, wr, la);
  Norm(la, 3);
  if (snap_axis) {
    int best = 0;
    double bestv = std::fabs(la[0]);
    for (int i = 1; i < 3; ++i) {
      if (std::fabs(la[i]) > bestv) {
        bestv = std::fabs(la[i]);
        best = i;
      }
    }
    const double s = la[best] >= 0 ? 1.0 : -1.0;
    la[0] = la[1] = la[2] = 0;
    la[best] = s;
  }
  j->axis = std::array<double, 3>{la[0], la[1], la[2]};
}

}  // namespace ps::studio
