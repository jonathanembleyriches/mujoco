// ProtoSpec Studio: placement / surface-snapping math (ps::studio), windowless.
//
// The Unity/Unreal placement idiom over the compiled snapshot: while a translate
// drag is in surface-snap mode the dragged element glides on the surface under
// the cursor (ray through the cursor, excluding the dragged body, support-point
// resting on the hit surface), and the End key drops the selection onto whatever
// is beneath it. Everything here is pure geometry over mjModel/mjData + the
// authored tree -- no ImGui, no SDL -- so the probe/test batteries drive the
// exact code the gizmo runs.
//
// v1 surface-normal support (SurfaceNormalForGeom): an exact normal for PLANE
// geoms (the plane's +z axis) and BOX geoms (the outward normal of the hit
// face); every other hit geom type (sphere, capsule, mesh, ...) falls back to
// world +Z. mj_ray does not return a contact normal and the general gradient
// estimate is out of scope for v1 -- gliding across a curved or mesh surface
// therefore behaves like gliding on flat ground at the hit height.

#ifndef PS_STUDIO_EDITOR_PLACEMENT_H_
#define PS_STUDIO_EDITOR_PLACEMENT_H_

#include <cmath>
#include <cstdint>
#include <vector>

#include "binding.h"
#include "editor/editor_context.h"
#include "editor/transform_math.h"

struct mjModel_;
struct mjData_;
struct mjvOption_;
typedef struct mjModel_ mjModel;
typedef struct mjData_ mjData;
typedef struct mjvOption_ mjvOption;

namespace ps::studio {

// The one increment-snapping formula (shared with the gizmo's Snap so the
// windowless battery exercises the exact rounding the drag applies): round `v`
// to the nearest multiple of `step`. Caller guarantees step > 0.
inline double SnapIncrement(double v, double step) {
  return std::round(v / step) * step;
}

// One compiled-geom AABB of the dragged element, captured at grab relative to
// the drag anchor: `rel` is the world offset of the aabb centre from the
// anchor, `R` the geom's world rotation (row-major, world<-geom), `half` the
// aabb half extents in the geom frame (mjModel geom_aabb).
struct SupportBox {
  double rel[3] = {0, 0, 0};
  double R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  double half[3] = {0, 0, 0};
};

// The dragged element's support footprint, captured once at grab. Pure
// translation leaves the relative configuration invariant, so per-frame surface
// placement only re-projects it onto the current surface normal.
struct SupportCache {
  bool valid = false;
  int exclude_body = -1;  // compiled body id whose subtree raycasts must skip
                          // (-1 when the owner is the worldbody: excluding body
                          // 0 would exclude every static geom, floor included)
  int exclude_geom = -1;  // the dragged geom itself (world-parented geoms have
                          // no excludable body, so they are skipped by id)
  std::vector<SupportBox> boxes;   // empty => treat the element as a point
  double world_z[3] = {0, 0, 1};   // element world +Z at grab (align-to-surface)
};

// True when `body` equals `root` or descends from it (body_parentid walk).
bool BodyInSubtree(const mjModel* m, int body, int root);

// Capture the support footprint for the element bound to `serial`:
//   Body   -> every compiled geom of the body's kinematic subtree
//             (a body drag moves the subtree, so children rest too);
//   Geom   -> that geom alone (its owning body is the raycast exclusion);
//   Site / Camera / Light -> a point (no aabb), owning body excluded.
// Invalid when the serial is unbound or the element carries no body.
SupportCache BuildSupportCache(const mjModel* m, const mjData* d,
                               const mj::Binding& binding, std::uint64_t serial,
                               const double anchor[3],
                               const double world_quat[4]);

// Signed offset (along unit `n`) of the element's support point -- the lowest
// point of the cached footprint -- relative to the anchor:
//   min over boxes of  rel.n - sum_i half_i |col_i(R) . n|,   0 when empty.
// `align_quat` (nullable, [w,x,y,z]) pre-rotates the cached configuration about
// the anchor -- the align-to-surface rotation applied before placement.
double SupportOffset(const SupportCache& c, const double n[3],
                     const double align_quat[4]);

// The anchor position that rests the support point exactly on the surface:
//   out = hit + (-support_offset) * n.
void SurfaceTargetAnchor(const double hit[3], const double n[3],
                         double support_offset, double out[3]);

// v1 surface normal at `hit` on compiled geom `geom` (see the header comment):
// plane +z / box hit-face / world +Z fallback, flipped to face the ray origin.
void SurfaceNormalForGeom(const mjModel* m, const mjData* d, int geom,
                          const double hit[3], const double ray_dir[3],
                          double out[3]);

// A placement raycast hit: world point, v1 surface normal, and the hit geom.
struct SurfaceHit {
  bool valid = false;
  int geom = -1;
  double pos[3] = {0, 0, 0};
  double normal[3] = {0, 0, 1};
};

// Cast ro+t*rd through the scene (mj_ray, honouring the viewer's geom groups),
// skipping every geom of `exclude_body`'s subtree plus `exclude_geom` itself
// (mj_ray's bodyexclude covers the body; descendants and the lone geom are
// skipped by re-casting past them). Pass -1 to disable either exclusion.
// Misses yield !valid -- callers decide their own ground-plane fallback.
SurfaceHit RaycastPlacementSurface(const mjModel* m, const mjData* d,
                                   const mjvOption* opt, const double ro[3],
                                   const double rd[3], int exclude_body,
                                   int exclude_geom);

// The world rotation taking the cached element +Z (`world_z`) onto surface
// normal `n`: writes a unit axis + angle (radians). False when already aligned
// (nothing to compose). The 180-degree flip picks an arbitrary perpendicular.
bool AlignZRotation(const double world_z[3], const double n[3],
                    double axis_out[3], double* angle_out);

// Absolute-grid translate snapping: adjust the cumulative world delta `wd` in
// place so the RESULTING authored parent-frame position (f.local.pos + the
// conjugated delta) lands on multiples of `step` -- versus the default
// relative mode, which rounds the drag distance and preserves any off-grid
// authored offset. No-op when step <= 0.
void AbsoluteGridDelta(const DragFrame& f, double step, double wd[3]);

// Drop the current selection to the ground (End key): cast straight down from
// the element's support bottom (own subtree excluded) and translate it so the
// support point rests on the first surface beneath -- or on the world ground
// plane z=0 when nothing is hit. One undo entry ("drop to ground"). Returns
// false when the selection is not a droppable spatial element (or cannot be
// edited); `opt` may be null (no geom-group filtering then).
bool DropToGroundOp(EditorContext& ctx, const mjData* d, const mjvOption* opt);

}  // namespace ps::studio

#endif  // PS_STUDIO_EDITOR_PLACEMENT_H_
