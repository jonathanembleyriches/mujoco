// ProtoSpec Studio: editor-owned viewport input/context structs (ps::studio).
//
// R1 re-target: the editor no longer receives viewport mouse/draw state from
// host ViewportPlugin/ViewportGuiPlugin seams. Instead a single GuiPlugin
// reconstructs these plain-data snapshots from ImGui::GetIO() plus the editor's
// own compiled model/data and the cached host camera, and threads them through
// the gizmo/pick code unchanged. These are the exact field shapes the gizmo was
// always written against (formerly ps_plugin_ext.h's ViewportInput and
// ViewportGuiPlugin::Context), now owned editor-side.

#ifndef PS_STUDIO_EDITOR_VIEWPORT_INPUT_H_
#define PS_STUDIO_EDITOR_VIEWPORT_INPUT_H_

struct mjModel_;
struct mjData_;
struct mjvCamera_;
struct mjvOption_;
typedef struct mjModel_ mjModel;
typedef struct mjData_ mjData;
typedef struct mjvCamera_ mjvCamera;
typedef struct mjvOption_ mjvOption;

namespace ps::studio {

// Snapshot of viewport mouse state for a frame. Positions are normalized to the
// viewport in [0, 1] with y measured from the top; deltas are in the same units.
struct ViewportInput {
  const mjModel* model = nullptr;
  const mjData* data = nullptr;
  const mjvCamera* camera = nullptr;
  const mjvOption* vis_option = nullptr;
  float x = 0, y = 0;
  float dx = 0, dy = 0;
  float scroll = 0;
  float aspect_ratio = 1.0f;
  bool left_down = false, right_down = false, middle_down = false;
  bool left_double = false, right_double = false;
  bool ctrl = false, shift = false, alt = false;
};

// Per-frame draw context for the gizmo/overlay: the live camera + viewport
// metrics so the drawlist projects against the rendered scene. `edit_mode` lets
// an overlay draw only when the editor is in Edit mode (physics frozen at reset).
struct ViewportContext {
  const mjModel* model = nullptr;
  const mjData* data = nullptr;
  mjvCamera* camera = nullptr;  // mutable: an overlay may service an F-focus
  float aspect_ratio = 1.0f;
  bool edit_mode = true;
};

}  // namespace ps::studio

#endif  // PS_STUDIO_EDITOR_VIEWPORT_INPUT_H_
