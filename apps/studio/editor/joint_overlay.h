// ProtoSpec Studio: joint visualization + pick support (ps::studio, ours).
//
// The pure part of the joint-rigging overlay (deliverable 3a/3b): given a
// compiled model + data + Binding + the current selection, decide which joints
// to draw and expose each joint's world anchor/axis/type/range/highlight. The
// mjvScene geom emission (arrows, ball markers, free triads, range arcs) and the
// screen-space pick live in viewport_plugin.cc; this collect step is windowless
// so it can be unit-tested against a compiled fixture.

#ifndef PS_STUDIO_EDITOR_JOINT_OVERLAY_H_
#define PS_STUDIO_EDITOR_JOINT_OVERLAY_H_

#include <cstdint>
#include <vector>

#include "binding.h"

struct mjModel_;
struct mjData_;
typedef struct mjModel_ mjModel;
typedef struct mjData_ mjData;

namespace ps::studio {

// One joint to draw, with everything the overlay + pick need in world space.
struct JointVis {
  std::uint64_t serial = 0;
  int jnt_id = -1;
  int type = 0;              // mjtJoint (mjJNT_FREE/BALL/SLIDE/HINGE)
  double anchor[3] = {0, 0, 0};
  double axis[3] = {0, 0, 1};
  bool has_range = false;
  double range[2] = {0, 0};  // radians (hinge) / metres (slide)
  bool selected = false;     // this joint is the current selection
};

// The joints to visualize for the current selection. When `show_all`, every
// joint in the model is returned; otherwise only the joints on the body that
// owns the selection (the selected body, or the parent body of a selected
// joint/geom/site/camera/light). `selected_serial` marks the highlighted joint.
// Empty when nothing relevant is selected and `show_all` is false.
std::vector<JointVis> CollectJointVis(const mjModel* m, const mjData* d,
                                      const ps::mjcf::bridge::Binding& binding,
                                      std::uint64_t selected_serial,
                                      bool show_all);

}  // namespace ps::studio

#endif  // PS_STUDIO_EDITOR_JOINT_OVERLAY_H_
