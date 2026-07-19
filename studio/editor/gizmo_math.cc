// ProtoSpec Studio: gizmo projection + hit-testing math. See header.

#include "editor/gizmo_math.h"

#include <cmath>

#include <mujoco/mujoco.h>

namespace ps::studio {

static double Dot(const double a[3], const double b[3]) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
static void Sub(const double a[3], const double b[3], double o[3]) {
  o[0] = a[0] - b[0];
  o[1] = a[1] - b[1];
  o[2] = a[2] - b[2];
}

ViewProj BuildViewProj(const mjModel* m, const mjData* d, const mjvCamera* cam,
                       double aspect) {
  ViewProj vp;
  vp.aspect = aspect;
  vp.orthographic = cam && cam->orthographic;
  mjtNum head[3], fwd[3], up[3], right[3];
  mjv_cameraFrame(head, fwd, up, right, d, cam);
  float zver[2], zhor[2], zclip[2] = {0, 0};
  mjv_cameraFrustum(zver, zhor, zclip, m, cam);
  for (int i = 0; i < 3; ++i) {
    vp.eye[i] = head[i];
    vp.forward[i] = fwd[i];
    vp.up[i] = up[i];
    vp.right[i] = right[i];
  }
  vp.zver[0] = zver[0];
  vp.zver[1] = zver[1];
  vp.zhor[0] = zhor[0];
  vp.zhor[1] = zhor[1];
  vp.zclip[0] = zclip[0];
  vp.zclip[1] = zclip[1];
  return vp;
}

ScreenPt WorldToScreen(const ViewProj& vp, const double world[3]) {
  ScreenPt out;
  double v[3];
  Sub(world, vp.eye, v);
  const double fdist = Dot(v, vp.forward);
  const double u = Dot(v, vp.up);
  const double r = Dot(v, vp.right);

  const double span_v = vp.zver[0] + vp.zver[1];
  const double half_width = 0.5 * vp.aspect * span_v;
  const double frustum_center = (vp.zhor[1] - vp.zhor[0]) / 2;

  double d_up, d_right;
  if (vp.orthographic) {
    d_up = u;
    d_right = r;
    out.visible = true;
  } else {
    if (fdist <= 1e-6) {
      out.visible = false;
      return out;
    }
    const double s = vp.zclip[0] / fdist;
    d_up = u * s;
    d_right = r * s;
    out.visible = true;
  }
  const double rely = span_v != 0 ? (d_up + vp.zver[0]) / span_v : 0.5;
  const double relx =
      half_width != 0 ? ((d_right - frustum_center) / half_width + 1) / 2 : 0.5;
  out.x = relx;
  out.y = 1.0 - rely;
  out.depth = fdist;
  return out;
}

void ScreenToRay(const ViewProj& vp, double sx, double sy, double origin[3],
                 double dir[3]) {
  const double relx = sx;
  const double rely = 1.0 - sy;
  const double span_v = vp.zver[0] + vp.zver[1];
  const double half_width = 0.5 * vp.aspect * span_v;
  const double frustum_center = (vp.zhor[1] - vp.zhor[0]) / 2;
  const double d_up = -vp.zver[0] + rely * span_v;
  const double d_right = frustum_center + (2 * relx - 1) * half_width;

  if (vp.orthographic) {
    for (int i = 0; i < 3; ++i) {
      dir[i] = vp.forward[i];
      origin[i] = vp.eye[i] + vp.up[i] * d_up + vp.right[i] * d_right;
    }
  } else {
    double n = 0;
    for (int i = 0; i < 3; ++i) {
      dir[i] = vp.forward[i] * vp.zclip[0] + vp.up[i] * d_up +
               vp.right[i] * d_right;
      n += dir[i] * dir[i];
    }
    n = std::sqrt(n);
    if (n > 1e-12)
      for (int i = 0; i < 3; ++i) dir[i] /= n;
    for (int i = 0; i < 3; ++i) origin[i] = vp.eye[i];
  }
}

double WorldSizeForPixels(const ViewProj& vp, const double world_anchor[3],
                          double pixels, double viewport_height_px) {
  // Project the anchor and a point offset one world unit along the camera-up
  // axis; the screen delta gives pixels-per-world-unit at that depth.
  ScreenPt a = WorldToScreen(vp, world_anchor);
  double off[3] = {world_anchor[0] + vp.up[0], world_anchor[1] + vp.up[1],
                   world_anchor[2] + vp.up[2]};
  ScreenPt b = WorldToScreen(vp, off);
  const double dy = (b.y - a.y) * viewport_height_px;
  const double dx = (b.x - a.x) * viewport_height_px * vp.aspect;
  const double px_per_world = std::sqrt(dx * dx + dy * dy);
  if (px_per_world < 1e-6) return pixels;  // degenerate; avoid blow-up
  return pixels / px_per_world;
}

double PointSegmentDist(double px, double py, double ax, double ay, double bx,
                        double by) {
  const double vx = bx - ax, vy = by - ay;
  const double wx = px - ax, wy = py - ay;
  const double len2 = vx * vx + vy * vy;
  double t = len2 > 1e-12 ? (wx * vx + wy * vy) / len2 : 0.0;
  if (t < 0) t = 0;
  if (t > 1) t = 1;
  const double cx = ax + t * vx, cy = ay + t * vy;
  const double dx = px - cx, dy = py - cy;
  return std::sqrt(dx * dx + dy * dy);
}

bool ClosestPointOnLine(const double p0[3], const double pd[3],
                        const double q0[3], const double qd[3], double* t_out) {
  const double a = Dot(pd, pd);
  const double b = Dot(pd, qd);
  const double c = Dot(qd, qd);
  double w0[3];
  Sub(p0, q0, w0);
  const double d = Dot(pd, w0);
  const double e = Dot(qd, w0);
  const double denom = a * c - b * b;
  if (std::fabs(denom) < 1e-12) return false;
  *t_out = (b * e - c * d) / denom;
  return true;
}

bool RayPlaneIntersect(const double ro[3], const double rd[3],
                       const double po[3], const double pn[3], double* t_out,
                       double hit[3]) {
  const double denom = Dot(rd, pn);
  if (std::fabs(denom) < 1e-9) return false;
  double diff[3];
  Sub(po, ro, diff);
  const double t = Dot(diff, pn) / denom;
  if (t < 0) return false;
  *t_out = t;
  if (hit) {
    hit[0] = ro[0] + t * rd[0];
    hit[1] = ro[1] + t * rd[1];
    hit[2] = ro[2] + t * rd[2];
  }
  return true;
}

}  // namespace ps::studio
