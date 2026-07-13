// ProtoSpec Studio: gizmo projection + hit-testing math (ps::studio), windowless.
//
// The gizmo is drawn in screen space over the mjr-rendered scene (an ImGuizmo-
// style ImGui drawlist), so it needs to (a) project world points to the exact
// same screen coordinates MuJoCo rasterises to, and (b) turn screen clicks back
// into world rays. Both are derived as the forward / inverse of MuJoCo's own
// pick-ray construction (interaction.cc MakePickRay), so the overlay lines up
// with the scene and with `Pick` to the pixel.
//
// Everything here is pure geometry over a captured camera frustum; the ImGui
// rendering and interaction state machine live in gizmo.cc. The projection and
// the ray/axis/plane/ring intersection helpers are unit-tested against fixed
// camera fixtures with no window.

#ifndef PS_STUDIO_EDITOR_GIZMO_MATH_H_
#define PS_STUDIO_EDITOR_GIZMO_MATH_H_

struct mjModel_;
struct mjData_;
struct mjvCamera_;
typedef struct mjModel_ mjModel;
typedef struct mjData_ mjData;
typedef struct mjvCamera_ mjvCamera;

namespace ps::studio {

// A captured camera: orthonormal eye frame + the frustum planes, enough to both
// project and unproject. Matches MuJoCo's mjv_cameraFrame / mjv_cameraFrustum.
struct ViewProj {
  double eye[3] = {0, 0, 0};
  double forward[3] = {0, 0, -1};
  double up[3] = {0, 1, 0};
  double right[3] = {1, 0, 0};
  double zver[2] = {-1, 1};   // near-plane vertical extents (bottom, top)
  double zhor[2] = {-1, 1};   // near-plane horizontal extents (left, right)
  double zclip[2] = {1, 100};  // near, far
  double aspect = 1.0;
  bool orthographic = false;
};

// Build a ViewProj for the live camera (data must be forwarded). aspect is
// width/height of the viewport.
ViewProj BuildViewProj(const mjModel* m, const mjData* d, const mjvCamera* cam,
                       double aspect);

// A projected point: normalized viewport coords x,y in [0,1] with y measured from
// the TOP (ImGui convention), forward depth, and whether it is in front of the eye.
struct ScreenPt {
  double x = 0, y = 0;
  double depth = 0;
  bool visible = false;
};
ScreenPt WorldToScreen(const ViewProj& vp, const double world[3]);

// Ray through a normalized screen point (y from top). Origin/dir in world space;
// dir is normalized for perspective, the forward axis for orthographic.
void ScreenToRay(const ViewProj& vp, double sx, double sy, double origin[3],
                 double dir[3]);

// The world length that projects to `pixels` screen pixels at `world_anchor`,
// given the viewport height in pixels -- used to keep the gizmo a constant
// on-screen size regardless of camera distance.
double WorldSizeForPixels(const ViewProj& vp, const double world_anchor[3],
                          double pixels, double viewport_height_px);

// --- Pure 2D / 3D intersection helpers (shared by the hit tests) ---------- //

// Distance from point p to segment [a,b], all in the same 2D units.
double PointSegmentDist(double px, double py, double ax, double ay, double bx,
                        double by);

// Parameter t on line (p0 + t*pd) at the point closest to line (q0 + s*qd).
// Returns false when the lines are parallel.
bool ClosestPointOnLine(const double p0[3], const double pd[3],
                        const double q0[3], const double qd[3], double* t_out);

// Intersect ray (ro + t*rd) with plane through po with normal pn. Returns false
// when parallel or behind the origin.
bool RayPlaneIntersect(const double ro[3], const double rd[3],
                       const double po[3], const double pn[3], double* t_out,
                       double hit[3]);

}  // namespace ps::studio

#endif  // PS_STUDIO_EDITOR_GIZMO_MATH_H_
