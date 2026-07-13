// ProtoSpec Studio: gizmo transform math -- THE delta rule (DR-S6), windowless.
//
// Compiled poses are NOT invertible into authored fields (the compiler bakes the
// mesh frame into geom_pos/geom_quat), so gizmos never reconstruct absolute
// authored poses. Instead they apply the drag DELTA conjugated into the parent
// frame onto the existing authored value:
//
//   W       = P . L_authored . M          (M = compiler-baked suffix, or identity)
//   W_new   = D . W                        (D = world-space gizmo delta)
//   => L_new = (inv(P) . D . P) . L_authored     (M cancels; P is never inverted
//                                                 into authored fields, only used
//                                                 as a rigid conjugator)
//
// `P` is the parent world pose at qpos0: the nearest body ancestor's COMPILED
// world pose (mjData xpos/xquat) composed with the AUTHORED transforms of any
// <frame> nodes between that body and the element (frames do not exist in
// mjModel, so they must be composed in from the tree). `L_authored` comes from
// the tree, materialised via sdk::Effective when class-inherited.
//
// Everything here is pure algebra over the tree + a compiled snapshot; no ImGui,
// no SDL. The gizmo interaction (gizmo.cc) and the delta-math tests both drive
// this module, so the load-bearing math is exercised without a window.

#ifndef PS_STUDIO_EDITOR_TRANSFORM_MATH_H_
#define PS_STUDIO_EDITOR_TRANSFORM_MATH_H_

#include <cstdint>
#include <string>

#include "binding.h"
#include "types.h"

struct mjModel_;
struct mjData_;
typedef struct mjModel_ mjModel;
typedef struct mjData_ mjData;

namespace ps::studio {

namespace mj = ps::mjcf;

// A rigid transform acting as T(x) = quat * x + pos. quat is a unit quaternion
// stored [w, x, y, z]; composition and inversion keep it a rigid motion.
struct Rigid {
  double quat[4] = {1, 0, 0, 0};
  double pos[3] = {0, 0, 0};
};

// out = a . b  (b applied first, then a).
Rigid Compose(const Rigid& a, const Rigid& b);
// out = a^-1.
Rigid Invert(const Rigid& a);
// out = quat * v  (rotate a vector).
void QuatRotate(const double quat[4], const double v[3], double out[3]);
// Hamilton product p * q, both [w,x,y,z], normalised on write is caller's job.
void QuatMul(const double p[4], const double q[4], double out[4]);
// Right-handed rotation of `angle` (radians) about unit `axis`, as [w,x,y,z].
void QuatFromAxisAngle(const double axis[3], double angle, double out[4]);

// Compiler settings that govern how an authored Orientation resolves to a quat.
struct OrientContext {
  bool degree = true;         // <compiler angle="degree"> (MuJoCo default)
  std::string eulerseq = "xyz";
};
OrientContext ReadOrientContext(const mj::Model& model);

// Authored Orientation variant (any of the five forms) -> unit quat [w,x,y,z].
// An absent orientation resolves to identity. Mirrors the compiler's resolver so
// the gizmo's notion of L_authored matches what MuJoCo compiles.
void OrientationToQuat(const ps::opt<mj::Orientation>& orient,
                       const OrientContext& oc, double quat[4]);

// The spatial element a serial resolves to. Only Body/Geom/Site/Camera/Light/
// Frame carry a pos/orient the gizmo can drive; anything else yields a null ptr.
struct SpatialRef {
  void* ptr = nullptr;
  mj::ElementType type = mj::ElementType::Model;
  explicit operator bool() const { return ptr != nullptr; }
};
SpatialRef FindSpatial(mj::Model& tree, std::uint64_t serial);

// Immutable frame captured at grab time: the parent world pose, the materialised
// authored local pose, and derived world-space anchors the gizmo draws against.
struct DragFrame {
  bool valid = false;
  mj::ElementType type = mj::ElementType::Model;
  Rigid parent;                 // P: parent world pose at qpos0
  Rigid local;                  // L_authored (materialised) at grab
  double anchor[3] = {0, 0, 0}; // world pos of the element's frame origin
  double world_quat[4] = {1, 0, 0, 0};  // element world orientation (for local-axis modes)

  // fromto-authored geom/site (capsule/cylinder/box/ellipsoid limbs): when a
  // shape variant carries `fromto`, the compiler DERIVES pos/quat from the two
  // endpoints and IGNORES any authored pos/quat (build.cc geom/site Compile).
  // Editing pos would therefore be a no-op; the gizmo must edit the endpoints
  // instead. These are the grab-time endpoints in the element's PARENT frame
  // (the frame `fromto` is authored in, i.e. the frame P conjugates into).
  bool is_fromto = false;
  double from[3] = {0, 0, 0};
  double to[3] = {0, 0, 0};
};

// Build the drag frame for `serial`: resolves P from the compiled model+data (at
// qpos0) and the authored frame chain, and L_authored from the tree (Effective
// for class-inherited pos/orient). `data` must be reset to qpos0 and forwarded.
DragFrame BuildDragFrame(const mjModel* model, const mjData* data,
                         const mj::bridge::Binding& binding, mj::Model& tree,
                         std::uint64_t serial);

// Apply a cumulative world-space translation `world_delta` (since grab) onto the
// element's authored pos, per the delta rule. Writes (materialises) pos; leaves
// orient untouched (its authored form is preserved).
void ApplyTranslate(mj::Model& tree, std::uint64_t serial, const DragFrame& f,
                    const double world_delta[3]);

// Apply a cumulative world-space rotation of `angle` (radians) about unit world
// `axis`, pivoting on the element's frame origin. Writes orient as a quat
// (resolved decision #1); pos is provably unchanged (the pivot is the origin) so
// it is left as authored.
void ApplyRotate(mj::Model& tree, std::uint64_t serial, const DragFrame& f,
                 const double axis[3], double angle);

// --- Scale ---------------------------------------------------------------- //
// Scale maps to size, never to a transform delta. `factor` are per-axis
// multipliers relative to the grab-time size; components the geom/site type does
// not expose are ignored (sphere is uniform, capsule/cylinder use r + half-len).

// Grab-time size snapshot for a geom/site (materialised via Effective).
struct ScaleBase {
  bool valid = false;
  bool is_mesh = false;         // geom type=mesh -> scale maps to the mesh asset
  int size_dofs = 0;            // independent size components the type exposes
  double size[3] = {0, 0, 0};   // grab-time geom/site size
  double mesh_scale[3] = {1, 1, 1};  // grab-time mesh asset scale (mesh geoms)
  std::uint64_t mesh_serial = 0;     // the referenced Mesh element (mesh geoms)

  // fromto-authored geom/site: the half-length (long axis) is DERIVED from the
  // endpoint separation, not from `size`, so a long-axis scale must move the
  // endpoints apart/together about their midpoint; only the radius (size[0])
  // maps to the `size` field. size[0] holds the grab-time radius here.
  bool is_fromto = false;
  double from[3] = {0, 0, 0};
  double to[3] = {0, 0, 0};
};
ScaleBase BuildScaleBase(mj::Model& tree, std::uint64_t serial);

// Apply cumulative per-axis `factor` to the grab-time size (or, for a mesh geom,
// to the referenced mesh asset's scale -- a model-wide change the caller should
// warn about). Writes the geom/site size (or mesh scale).
void ApplyScale(mj::Model& tree, std::uint64_t serial, const ScaleBase& base,
                const double factor[3]);

}  // namespace ps::studio

#endif  // PS_STUDIO_EDITOR_TRANSFORM_MATH_H_
