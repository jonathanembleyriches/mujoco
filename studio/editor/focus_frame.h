// ProtoSpec Studio "F to focus" frame math (ps::studio, ours). Pure over a
// compiled mjModel / forwarded mjData + the compile Binding, so the camera-frame
// computation is unit-tested windowless against a real fixture (geom / body /
// point-like element), and the viewport's ServiceFocus only writes the camera.

#ifndef PS_STUDIO_EDITOR_FOCUS_FRAME_H_
#define PS_STUDIO_EDITOR_FOCUS_FRAME_H_

#include <cstdint>

#include <mujoco/mujoco.h>

#include "binding.h"

namespace ps::studio {

// Where the camera should look and how far back to sit to frame one element.
struct FocusFrame {
  bool ok = false;         // the serial resolved to a framable element
  double center[3] = {0, 0, 0};
  double radius = 0.0;     // enclosing radius of the element (world units)
  double distance = 0.0;   // camera distance that frames it (radius-derived)
};

// Frame the element bound to `serial`, measured off the forwarded preview `d`:
//   geom  -> geom_xpos + geom_rbound
//   body  -> the union bounding sphere of its kinematic-subtree geoms
//            (falls back to the body origin + a default radius when it has none)
//   site / camera / light / joint -> the world anchor + a small default radius
// distance = max(radius * 2.5, a floor derived from m->stat.extent) so a tiny
// element does not drive the camera onto/into it. `ok` is false (and the frame
// left zero) when `serial` binds nothing or an unframable family.
FocusFrame ComputeFocusFrame(const mjModel* m, const mjData* d,
                             const ps::mjcf::Binding& binding,
                             std::uint64_t serial);

}  // namespace ps::studio

#endif  // PS_STUDIO_EDITOR_FOCUS_FRAME_H_
