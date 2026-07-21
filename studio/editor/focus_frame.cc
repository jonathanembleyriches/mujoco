// ProtoSpec Studio "F to focus" frame math (ps::studio, ours). See focus_frame.h.

#include "editor/focus_frame.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace ps::studio {

namespace mj = ps::mjcf;

namespace {

// Copy a 3-vector into the frame centre.
void SetCenter(FocusFrame& f, const mjtNum* src) {
  for (int k = 0; k < 3; ++k) f.center[k] = static_cast<double>(src[k]);
}

// The bindable entry for `serial` with a live compiled id, or nullptr.
const mj::Binding::Entry* ResolveEntry(const mj::Binding& binding,
                                       std::uint64_t serial) {
  for (const mj::Binding::Entry& e : binding.entries()) {
    if (e.serial == serial && e.id >= 0) return &e;
  }
  return nullptr;
}

}  // namespace

FocusFrame ComputeFocusFrame(const mjModel* m, const mjData* d,
                             const mj::Binding& binding, std::uint64_t serial) {
  FocusFrame f;
  if (!m || !d || serial == 0) return f;
  const mj::Binding::Entry* e = ResolveEntry(binding, serial);
  if (!e) return f;

  const double extent = m->stat.extent > 0 ? m->stat.extent : 1.0;
  const double point_r = 0.05 * extent;  // default radius for point-like elements
  const int id = e->id;

  switch (e->etype) {
    case mj::ElementType::Geom: {
      if (id >= m->ngeom) return f;
      SetCenter(f, d->geom_xpos + 3 * id);
      const double rb = m->geom_rbound[id];
      f.radius = rb > 0 ? rb : point_r;
      break;
    }
    case mj::ElementType::Body: {
      if (id >= m->nbody) return f;
      // Kinematic-subtree mask: bodies whose parent chain passes through `id`
      // (id included). Body ids are topologically ordered (parent < child).
      std::vector<char> in(m->nbody, 0);
      in[id] = 1;
      for (int b = id + 1; b < m->nbody; ++b) {
        const int par = m->body_parentid[b];
        if (par >= 0 && par < m->nbody && in[par]) in[b] = 1;
      }
      // Axis-aligned union of the subtree geoms' bounding spheres.
      double lo[3] = {0, 0, 0}, hi[3] = {0, 0, 0};
      bool any = false;
      for (int g = 0; g < m->ngeom; ++g) {
        const int gb = m->geom_bodyid[g];
        if (gb < 0 || gb >= m->nbody || !in[gb]) continue;
        const double r = m->geom_rbound[g];
        const mjtNum* c = d->geom_xpos + 3 * g;
        for (int k = 0; k < 3; ++k) {
          const double lok = c[k] - r, hik = c[k] + r;
          if (!any) {
            lo[k] = lok;
            hi[k] = hik;
          } else {
            lo[k] = std::min(lo[k], lok);
            hi[k] = std::max(hi[k], hik);
          }
        }
        any = true;
      }
      if (!any) {  // a bodiless / geomless body: frame its origin
        SetCenter(f, d->xpos + 3 * id);
        f.radius = point_r;
        break;
      }
      for (int k = 0; k < 3; ++k) f.center[k] = 0.5 * (lo[k] + hi[k]);
      // Enclosing radius: the farthest geom surface from the box centre.
      double rad = 0.0;
      for (int g = 0; g < m->ngeom; ++g) {
        const int gb = m->geom_bodyid[g];
        if (gb < 0 || gb >= m->nbody || !in[gb]) continue;
        const mjtNum* c = d->geom_xpos + 3 * g;
        double dd = 0.0;
        for (int k = 0; k < 3; ++k) {
          const double t = static_cast<double>(c[k]) - f.center[k];
          dd += t * t;
        }
        rad = std::max(rad, std::sqrt(dd) + m->geom_rbound[g]);
      }
      f.radius = rad > 0 ? rad : point_r;
      break;
    }
    case mj::ElementType::Site:
      if (id >= m->nsite) return f;
      SetCenter(f, d->site_xpos + 3 * id);
      f.radius = point_r;
      break;
    case mj::ElementType::Camera:
      if (id >= m->ncam) return f;
      SetCenter(f, d->cam_xpos + 3 * id);
      f.radius = point_r;
      break;
    case mj::ElementType::Light:
      if (id >= m->nlight) return f;
      SetCenter(f, d->light_xpos + 3 * id);
      f.radius = point_r;
      break;
    case mj::ElementType::Joint:
    case mj::ElementType::FreeJoint:
      if (id >= m->njnt) return f;
      SetCenter(f, d->xanchor + 3 * id);
      f.radius = point_r;
      break;
    default:
      return f;  // unframable family: leave ok == false
  }

  f.distance = std::max(f.radius * 2.5, 0.15 * extent);
  f.ok = true;
  return f;
}

}  // namespace ps::studio
