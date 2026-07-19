// Vendored from MuJoCo Studio, adapted for ProtoSpec Studio (ps::studio).
//
// Upstream: src/experimental/platform/ux/interaction.h @ mujoco 67a1ea6d
// Adaptation: kept the pick-ray math (Pick/PickResult), the public camera-motion
// helper (MoveCamera) and the camera selector (SetCamera). Dropped the mjvPerturb
// wiring (InitPerturb/MovePerturb): Edit mode computes tree edits, not perturb
// drags (SE2), and Play-mode perturb lands in SE4.

#ifndef PS_STUDIO_PLATFORM_UX_INTERACTION_H_
#define PS_STUDIO_PLATFORM_UX_INTERACTION_H_

#include <mujoco/mujoco.h>

namespace ps::studio {

// The result of a pick operation.
struct PickResult {
  mjtNum point[3] = {0, 0, 0};  // World coordinates
  mjtNum dist = -1;             // Distance from the camera (< 0 == miss)
  int body = -1;
  int geom = -1;
  int flex = -1;
  int skin = -1;
};

// Returns information about the object (if any) under the mouse cursor.
// x, y are normalized viewport coordinates in [0, 1] (y measured from the top).
PickResult Pick(const mjModel* m, const mjData* d, const mjvCamera* camera,
                float x, float y, float aspect_ratio,
                const mjvOption* vis_options);

// Indices for cameras that are not defined in the model.
static constexpr int kTumbleCameraIdx = -3;
static constexpr int kFreeCameraIdx = -2;
static constexpr int kTrackingCameraIdx = -1;

// Updates the camera according to the requested index; returns the (possibly
// clamped) index actually applied.
int SetCamera(const mjModel* m, mjvCamera* camera, int request_idx);

// Camera-relative and target-relative motions (see upstream nomenclature).
enum class CameraMotion {
  ZOOM,
  ORBIT,
  TRUCK_PEDESTAL,
  TRUCK_DOLLY,
  PAN_TILT,
  PLANAR_MOVE_H,
  PLANAR_MOVE_V,
};

void MoveCamera(const mjModel* m, const mjData* d, mjvCamera* cam,
                CameraMotion motion, mjtNum dx, mjtNum dy);

}  // namespace ps::studio

#endif  // PS_STUDIO_PLATFORM_UX_INTERACTION_H_
