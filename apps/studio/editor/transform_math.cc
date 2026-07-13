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

// --- Orientation resolution (mirrors compile/build.cc ResolveQuat) -------- //

static void Cross(const double a[3], const double b[3], double out[3]) {
  out[0] = a[1] * b[2] - a[2] * b[1];
  out[1] = a[2] * b[0] - a[0] * b[2];
  out[2] = a[0] * b[1] - a[1] * b[0];
}
static double Dot(const double a[3], const double b[3]) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// Quaternion from an orthonormal frame whose columns are x, y, z (mjuu_frame2quat).
static void Frame2Quat(double q[4], const double x[3], const double y[3],
                       const double z[3]) {
  const double* m[3] = {x, y, z};  // m[col][row]
  if (m[0][0] + m[1][1] + m[2][2] > 0) {
    q[0] = 0.5 * std::sqrt(1 + m[0][0] + m[1][1] + m[2][2]);
    q[1] = 0.25 * (m[1][2] - m[2][1]) / q[0];
    q[2] = 0.25 * (m[2][0] - m[0][2]) / q[0];
    q[3] = 0.25 * (m[0][1] - m[1][0]) / q[0];
  } else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
    q[1] = 0.5 * std::sqrt(1 + m[0][0] - m[1][1] - m[2][2]);
    q[0] = 0.25 * (m[1][2] - m[2][1]) / q[1];
    q[2] = 0.25 * (m[1][0] + m[0][1]) / q[1];
    q[3] = 0.25 * (m[2][0] + m[0][2]) / q[1];
  } else if (m[1][1] > m[2][2]) {
    q[2] = 0.5 * std::sqrt(1 - m[0][0] + m[1][1] - m[2][2]);
    q[0] = 0.25 * (m[2][0] - m[0][2]) / q[2];
    q[1] = 0.25 * (m[1][0] + m[0][1]) / q[2];
    q[3] = 0.25 * (m[2][1] + m[1][2]) / q[2];
  } else {
    q[3] = 0.5 * std::sqrt(1 - m[0][0] - m[1][1] + m[2][2]);
    q[0] = 0.25 * (m[0][1] - m[1][0]) / q[3];
    q[1] = 0.25 * (m[2][0] + m[0][2]) / q[3];
    q[2] = 0.25 * (m[2][1] + m[1][2]) / q[3];
  }
  Norm(q, 4);
}

OrientContext ReadOrientContext(const mj::Model& model) {
  OrientContext oc;
  for (const auto& c : model.compilers) {
    if (!c) continue;
    if (c->angle) oc.degree = (*c->angle == mj::AngleUnit::degree);
    if (c->eulerseq) oc.eulerseq = *c->eulerseq;
  }
  return oc;
}

void OrientationToQuat(const ps::opt<mj::Orientation>& orient,
                       const OrientContext& oc, double quat[4]) {
  quat[0] = 1;
  quat[1] = quat[2] = quat[3] = 0;
  if (!orient) return;

  if (const mj::Quat* q = std::get_if<mj::Quat>(&*orient)) {
    quat[0] = q->w;
    quat[1] = q->x;
    quat[2] = q->y;
    quat[3] = q->z;
    Norm(quat, 4);
  } else if (const mj::AxisAngle* aa = std::get_if<mj::AxisAngle>(&*orient)) {
    double ax[3] = {aa->axis[0], aa->axis[1], aa->axis[2]};
    double ang = aa->angle;
    if (oc.degree) ang = ang / 180.0 * mjPI;
    if (Norm(ax, 3) < 1e-9) return;
    QuatFromAxisAngle(ax, ang, quat);
  } else if (const mj::XYAxes* xy = std::get_if<mj::XYAxes>(&*orient)) {
    double a[3] = {xy->xyaxes[0], xy->xyaxes[1], xy->xyaxes[2]};
    double b[3] = {xy->xyaxes[3], xy->xyaxes[4], xy->xyaxes[5]};
    if (Norm(a, 3) < 1e-9) return;
    const double d = Dot(a, b);
    b[0] -= a[0] * d;
    b[1] -= a[1] * d;
    b[2] -= a[2] * d;
    if (Norm(b, 3) < 1e-9) return;
    double z[3];
    Cross(a, b, z);
    if (Norm(z, 3) < 1e-9) return;
    Frame2Quat(quat, a, b, z);
  } else if (const mj::ZAxis* za = std::get_if<mj::ZAxis>(&*orient)) {
    double z[3] = {za->zaxis[0], za->zaxis[1], za->zaxis[2]};
    if (Norm(z, 3) < 1e-9) return;
    double axis[3];
    const double zref[3] = {0, 0, 1};
    Cross(zref, z, axis);
    const double s = std::sqrt(axis[0] * axis[0] + axis[1] * axis[1] +
                               axis[2] * axis[2]);
    const double ang = std::atan2(s, z[2]);
    if (s < 1e-10) {
      axis[0] = 1;
      axis[1] = axis[2] = 0;
    }
    QuatFromAxisAngle(axis, ang, quat);
  } else if (const mj::Euler* eu = std::get_if<mj::Euler>(&*orient)) {
    double e[3] = {eu->angles[0], eu->angles[1], eu->angles[2]};
    if (oc.degree)
      for (double& v : e) v = v / 180.0 * mjPI;
    quat[0] = 1;
    quat[1] = quat[2] = quat[3] = 0;
    for (int i = 0; i < 3 && i < static_cast<int>(oc.eulerseq.size()); ++i) {
      const char c = oc.eulerseq[i];
      double qrot[4] = {std::cos(e[i] / 2), 0, 0, 0};
      const double sa = std::sin(e[i] / 2);
      const bool intrinsic = (c == 'x' || c == 'y' || c == 'z');
      const char lc = (c >= 'A' && c <= 'Z') ? c - 'A' + 'a' : c;
      if (lc == 'x') qrot[1] = sa;
      else if (lc == 'y') qrot[2] = sa;
      else if (lc == 'z') qrot[3] = sa;
      double tmp[4];
      if (intrinsic) QuatMul(quat, qrot, tmp);
      else QuatMul(qrot, quat, tmp);
      for (int k = 0; k < 4; ++k) quat[k] = tmp[k];
    }
    Norm(quat, 4);
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
                               const mj::bridge::Binding& binding,
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
    OrientationToQuat(f->orient, oc, fr.quat);
    P = Compose(P, fr);
  }
  return P;
}

// Materialise an element's authored local pose (Effective fills a class-inherited
// pos/orient for the default families; Body/Frame default to origin/identity;
// Light carries pos but no orient, so its local rotation is identity).
template <class E>
static Rigid MaterializeLocal(mj::Model& tree, E& e, const OrientContext& oc) {
  Rigid L;
  std::array<double, 3> pos{0, 0, 0};
  ps::opt<mj::Orientation> orient;
  if constexpr (requires { e.orient; }) orient = e.orient;
  if (e.pos) {
    pos = *e.pos;
  } else if constexpr (sdk::HasDefaultFamily<E>::value) {
    std::unique_ptr<E> eff = sdk::Effective(tree, e, true);
    if (eff->pos) pos = *eff->pos;
    if constexpr (requires { e.orient; }) {
      if (!orient) orient = eff->orient;
    }
  }
  L.pos[0] = pos[0];
  L.pos[1] = pos[1];
  L.pos[2] = pos[2];
  OrientationToQuat(orient, oc, L.quat);
  return L;
}

template <class E>
static DragFrame BuildFor(mj::Model& tree, const mjData* data,
                          const mj::bridge::Binding& binding,
                          const sdk::ParentMap& pm, const OrientContext& oc,
                          E& e) {
  DragFrame f;
  f.valid = true;
  f.type = mj::element_type_of<E>::value;
  f.parent = ComputeParentPose(data, binding, pm, oc, &e);
  f.local = MaterializeLocal(tree, e, oc);

  // anchor = P . L (world frame origin);  world_quat = P.quat . L.quat.
  double rl[3];
  QuatRotate(f.parent.quat, f.local.pos, rl);
  for (int i = 0; i < 3; ++i) f.anchor[i] = f.parent.pos[i] + rl[i];
  QuatMul(f.parent.quat, f.local.quat, f.world_quat);
  return f;
}

DragFrame BuildDragFrame(const mjModel* model, const mjData* data,
                         const mj::bridge::Binding& binding, mj::Model& tree,
                         std::uint64_t serial) {
  (void)model;
  DragFrame f;
  SpatialRef ref = FindSpatial(tree, serial);
  if (!ref || !data) return f;
  const OrientContext oc = ReadOrientContext(tree);
  sdk::ParentMap pm(tree);
  switch (ref.type) {
    case mj::ElementType::Body:
      return BuildFor(tree, data, binding, pm, oc,
                      *static_cast<mj::Body*>(ref.ptr));
    case mj::ElementType::Geom:
      return BuildFor(tree, data, binding, pm, oc,
                      *static_cast<mj::Geom*>(ref.ptr));
    case mj::ElementType::Site:
      return BuildFor(tree, data, binding, pm, oc,
                      *static_cast<mj::Site*>(ref.ptr));
    case mj::ElementType::Camera:
      return BuildFor(tree, data, binding, pm, oc,
                      *static_cast<mj::Camera*>(ref.ptr));
    case mj::ElementType::Light:
      return BuildFor(tree, data, binding, pm, oc,
                      *static_cast<mj::Light*>(ref.ptr));
    case mj::ElementType::Frame:
      return BuildFor(tree, data, binding, pm, oc,
                      *static_cast<mj::Frame*>(ref.ptr));
    default:
      return f;
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
  e.orient = mj::Orientation(mj::Quat{n[0], n[1], n[2], n[3]});
}

template <class E>
static void TranslateElem(E& e, const DragFrame& f, const double world_delta[3]) {
  // L_new.pos = L_base.pos + inv(P).quat * world_delta.
  double pq[4];
  QuatConj(f.parent.quat, pq);
  double dl[3];
  QuatRotate(pq, world_delta, dl);
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
  if constexpr (requires { e.orient; }) {
    // L_new.quat = conj(P.quat) . qD . P.quat . L_base.quat  (pos unchanged: the
    // pivot is the element's own frame origin).
    double t1[4], t2[4], t3[4];
    QuatMul(f.parent.quat, f.local.quat, t1);
    QuatMul(qd, t1, t2);
    QuatMul(pq, t2, t3);
    WriteQuat(e, t3);
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
  if (ref.type == mj::ElementType::Geom)
    ApplyScaleSize(*static_cast<mj::Geom*>(ref.ptr), base, factor);
  else if (ref.type == mj::ElementType::Site)
    ApplyScaleSize(*static_cast<mj::Site*>(ref.ptr), base, factor);
}

}  // namespace ps::studio
