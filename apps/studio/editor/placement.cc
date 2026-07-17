// ProtoSpec Studio: placement / surface-snapping math. See placement.h.

#include "editor/placement.h"

#include <cmath>
#include <cstdint>

#include <mujoco/mujoco.h>

namespace ps::studio {
namespace {

// Unit-normalize in place; returns the pre-normalization length.
double Norm3(double v[3]) {
  const double n = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  if (n > 1e-12) {
    v[0] /= n;
    v[1] /= n;
    v[2] /= n;
  }
  return n;
}

double Dot3(const double a[3], const double b[3]) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

void Cross3(const double a[3], const double b[3], double out[3]) {
  out[0] = a[1] * b[2] - a[2] * b[1];
  out[1] = a[2] * b[0] - a[0] * b[2];
  out[2] = a[0] * b[1] - a[1] * b[0];
}

// Append geom `g`'s compiled AABB (centre + half extents, geom frame) as a
// SupportBox relative to `anchor`.
void AppendGeomBox(const mjModel* m, const mjData* d, int g,
                   const double anchor[3], std::vector<SupportBox>& out) {
  SupportBox b;
  const double* R = d->geom_xmat + 9 * g;      // row-major world<-geom
  const double* c = m->geom_aabb + 6 * g;      // aabb centre, geom frame
  for (int i = 0; i < 9; ++i) b.R[i] = R[i];
  for (int i = 0; i < 3; ++i) b.half[i] = m->geom_aabb[6 * g + 3 + i];
  for (int r = 0; r < 3; ++r) {
    b.rel[r] = d->geom_xpos[3 * g + r] + R[3 * r + 0] * c[0] +
               R[3 * r + 1] * c[1] + R[3 * r + 2] * c[2] - anchor[r];
  }
  out.push_back(b);
}

}  // namespace

bool BodyInSubtree(const mjModel* m, int body, int root) {
  if (root < 0 || body < 0) return false;
  while (body > 0) {
    if (body == root) return true;
    body = m->body_parentid[body];
  }
  return body == root;  // root == 0 (world) contains everything
}

SupportCache BuildSupportCache(const mjModel* m, const mjData* d,
                               const mj::Binding& binding, std::uint64_t serial,
                               const double anchor[3],
                               const double world_quat[4]) {
  SupportCache c;
  if (!m || !d || serial == 0) return c;
  const double z[3] = {0, 0, 1};
  QuatRotate(world_quat, z, c.world_z);

  for (const mj::Binding::Entry& e : binding.entries()) {
    if (e.serial != serial || e.id < 0) continue;
    switch (e.etype) {
      case mj::ElementType::Body: {
        if (e.id == 0) break;  // the worldbody is not placeable
        c.exclude_body = e.id;
        for (int g = 0; g < m->ngeom; ++g) {
          if (BodyInSubtree(m, m->geom_bodyid[g], e.id)) {
            AppendGeomBox(m, d, g, anchor, c.boxes);
          }
        }
        c.valid = true;
        return c;
      }
      case mj::ElementType::Geom: {
        const int body = m->geom_bodyid[e.id];
        c.exclude_body = body > 0 ? body : -1;  // never exclude the worldbody
        c.exclude_geom = e.id;
        AppendGeomBox(m, d, e.id, anchor, c.boxes);
        c.valid = true;
        return c;
      }
      case mj::ElementType::Site:
        c.exclude_body = m->site_bodyid[e.id] > 0 ? m->site_bodyid[e.id] : -1;
        c.valid = true;  // a point: empty footprint
        return c;
      case mj::ElementType::Camera:
        c.exclude_body = m->cam_bodyid[e.id] > 0 ? m->cam_bodyid[e.id] : -1;
        c.valid = true;
        return c;
      case mj::ElementType::Light:
        c.exclude_body = m->light_bodyid[e.id] > 0 ? m->light_bodyid[e.id] : -1;
        c.valid = true;
        return c;
      default:
        break;
    }
  }
  return c;
}

double SupportOffset(const SupportCache& c, const double n[3],
                     const double align_quat[4]) {
  double best = 0;
  bool first = true;
  for (const SupportBox& b : c.boxes) {
    // Optionally pre-rotate the cached configuration about the anchor.
    double rel[3] = {b.rel[0], b.rel[1], b.rel[2]};
    double cols[3][3];  // world direction of each geom-frame axis
    for (int i = 0; i < 3; ++i) {
      cols[i][0] = b.R[0 + i];
      cols[i][1] = b.R[3 + i];
      cols[i][2] = b.R[6 + i];
    }
    if (align_quat) {
      double t[3];
      QuatRotate(align_quat, rel, t);
      for (int k = 0; k < 3; ++k) rel[k] = t[k];
      for (int i = 0; i < 3; ++i) {
        QuatRotate(align_quat, cols[i], t);
        for (int k = 0; k < 3; ++k) cols[i][k] = t[k];
      }
    }
    double ext = 0;
    for (int i = 0; i < 3; ++i) ext += b.half[i] * std::fabs(Dot3(cols[i], n));
    const double v = Dot3(rel, n) - ext;
    if (first || v < best) {
      best = v;
      first = false;
    }
  }
  return first ? 0.0 : best;
}

void SurfaceTargetAnchor(const double hit[3], const double n[3],
                         double support_offset, double out[3]) {
  for (int k = 0; k < 3; ++k) out[k] = hit[k] - support_offset * n[k];
}

void SurfaceNormalForGeom(const mjModel* m, const mjData* d, int geom,
                          const double hit[3], const double ray_dir[3],
                          double out[3]) {
  out[0] = 0;
  out[1] = 0;
  out[2] = 1;
  if (m && d && geom >= 0 && geom < m->ngeom) {
    const double* R = d->geom_xmat + 9 * geom;
    if (m->geom_type[geom] == mjGEOM_PLANE) {
      out[0] = R[2];
      out[1] = R[5];
      out[2] = R[8];
    } else if (m->geom_type[geom] == mjGEOM_BOX) {
      // Hit point in the geom frame; the face with the largest normalized
      // |coordinate| is the face the ray struck; its outward axis is the normal.
      double lp[3] = {0, 0, 0};
      for (int i = 0; i < 3; ++i) {
        for (int r = 0; r < 3; ++r) {
          lp[i] += R[3 * r + i] * (hit[r] - d->geom_xpos[3 * geom + r]);
        }
      }
      int face = 2;
      double best = -1;
      for (int i = 0; i < 3; ++i) {
        const double s = m->geom_size[3 * geom + i];
        const double score = std::fabs(lp[i]) / (s > 1e-9 ? s : 1e-9);
        if (score > best) {
          best = score;
          face = i;
        }
      }
      const double sign = lp[face] >= 0 ? 1.0 : -1.0;
      out[0] = sign * R[0 + face];
      out[1] = sign * R[3 + face];
      out[2] = sign * R[6 + face];
    }
    // Every other type: world +Z fallback (documented v1 limitation).
  }
  // Face the viewer: a normal pointing along the ray would rest the element on
  // the far side of the surface.
  if (Dot3(out, ray_dir) > 0) {
    out[0] = -out[0];
    out[1] = -out[1];
    out[2] = -out[2];
  }
}

SurfaceHit RaycastPlacementSurface(const mjModel* m, const mjData* d,
                                   const mjvOption* opt, const double ro[3],
                                   const double rd[3], int exclude_body,
                                   int exclude_geom) {
  SurfaceHit out;
  if (!m || !d) return out;
  const mjtByte* group = opt ? opt->geomgroup : nullptr;
  const mjtByte flg_static = opt ? opt->flags[mjVIS_STATIC] : 1;
  mjtNum pnt[3] = {ro[0], ro[1], ro[2]};
  const mjtNum vec[3] = {rd[0], rd[1], rd[2]};
  double travelled = 0;
  // bodyexclude skips the dragged body itself; descendants of a dragged body
  // and a world-parented dragged geom are skipped by re-casting just past them
  // (PickAlongRay's advance trick).
  for (int guard = 0; guard < 32; ++guard) {
    int geom = -1;
    const mjtNum dist = mj_ray(m, d, pnt, vec, group, flg_static,
                               exclude_body > 0 ? exclude_body : -1, &geom,
                               nullptr);
    if (geom < 0 || dist < 0) return out;
    travelled += dist;
    if (geom == exclude_geom ||
        (exclude_body > 0 &&
         BodyInSubtree(m, m->geom_bodyid[geom], exclude_body))) {
      for (int i = 0; i < 3; ++i) pnt[i] += vec[i] * (dist + 1e-6);
      travelled += 1e-6;
      continue;
    }
    out.valid = true;
    out.geom = geom;
    for (int i = 0; i < 3; ++i) out.pos[i] = ro[i] + rd[i] * travelled;
    SurfaceNormalForGeom(m, d, geom, out.pos, rd, out.normal);
    return out;
  }
  return out;
}

bool AlignZRotation(const double world_z[3], const double n[3],
                    double axis_out[3], double* angle_out) {
  double z[3] = {world_z[0], world_z[1], world_z[2]};
  double nn[3] = {n[0], n[1], n[2]};
  if (Norm3(z) < 1e-9 || Norm3(nn) < 1e-9) return false;
  double axis[3];
  Cross3(z, nn, axis);
  const double s = Norm3(axis);
  const double c = Dot3(z, nn);
  if (s < 1e-9) {
    if (c > 0) return false;  // already aligned
    // Anti-parallel: rotate half a turn about any axis perpendicular to n.
    double seed[3] = {1, 0, 0};
    if (std::fabs(nn[0]) > 0.9) {
      seed[0] = 0;
      seed[1] = 1;
    }
    Cross3(seed, nn, axis);
    if (Norm3(axis) < 1e-9) return false;
    for (int k = 0; k < 3; ++k) axis_out[k] = axis[k];
    *angle_out = 3.14159265358979323846;
    return true;
  }
  for (int k = 0; k < 3; ++k) axis_out[k] = axis[k];
  *angle_out = std::atan2(s, c);
  return *angle_out > 1e-9;
}

void AbsoluteGridDelta(const DragFrame& f, double step, double wd[3]) {
  if (!(step > 0)) return;
  // Parent-frame delta: conj(P.quat) * wd.
  const double pq[4] = {f.parent.quat[0], -f.parent.quat[1], -f.parent.quat[2],
                        -f.parent.quat[3]};
  double dl[3];
  QuatRotate(pq, wd, dl);
  // Round the RESULTING authored parent-frame position to the grid.
  double back[3];
  for (int i = 0; i < 3; ++i) {
    back[i] = SnapIncrement(f.local.pos[i] + dl[i], step) - f.local.pos[i];
  }
  QuatRotate(f.parent.quat, back, wd);
}

bool DropToGroundOp(EditorContext& ctx, const mjData* d, const mjvOption* opt) {
  const mjModel* m = ctx.compiled.model.get();
  if (!ctx.CanEdit() || !ctx.tree || !m || !d || ctx.selected_serial == 0) {
    return false;
  }
  if (IsJointSerial(*ctx.tree, ctx.selected_serial)) return false;
  DragFrame f =
      BuildDragFrame(m, d, ctx.compiled.binding, *ctx.tree, ctx.selected_serial);
  if (!f.valid) return false;
  SupportCache cache = BuildSupportCache(m, d, ctx.compiled.binding,
                                         ctx.selected_serial, f.anchor,
                                         f.world_quat);
  if (!cache.valid) return false;

  const double up[3] = {0, 0, 1};
  const double so = SupportOffset(cache, up, nullptr);  // <= 0: bottom - anchor
  const double bottom_z = f.anchor[2] + so;

  // Straight down from the anchor (a cast from the exact aabb bottom would
  // start inside any surface the element already touches or penetrates); the
  // element's own subtree is excluded so it never rests on itself.
  const double down[3] = {0, 0, -1};
  SurfaceHit sh = RaycastPlacementSurface(m, d, opt, f.anchor, down,
                                          cache.exclude_body,
                                          cache.exclude_geom);
  const double surface_z = sh.valid ? sh.pos[2] : 0.0;  // ground-plane fallback

  const double dz = surface_z - bottom_z;
  if (std::fabs(dz) < 1e-12) return false;  // already resting
  ctx.BeginEdit();
  const double wd[3] = {0, 0, dz};
  ApplyTranslate(*ctx.tree, ctx.selected_serial, f, wd);
  ctx.CommitEdit("drop to ground");
  return true;
}

}  // namespace ps::studio
