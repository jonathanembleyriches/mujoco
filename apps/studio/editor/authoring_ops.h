// ProtoSpec Studio SE3 authoring operations (ps::studio, ours): the windowless
// core of the add / duplicate / reparent / new-model milestone.
//
// Every op mutates the authored ps::Model through the SDK builders, wrapped in
// the editor's BeginEdit/CommitEdit undo contract, produces a compilable model,
// auto-selects the new element, and keeps names unique. Like editor_ops.* this
// TU is pure (no SDL/ImGui/plugin-registry), so the ctest targets drive it
// directly.
//
// PERF (the SE1a lesson): the SDK builders are templated on the concrete parent /
// element type, and several ops embed whole-model walks. Dispatching those
// templates through a generic 142-type walk instantiates ~142x142 combinations
// (the historical 12-minute compile). Every op here confines template
// instantiation to the concrete container/element set it actually needs (a small
// runtime switch, or the generated Visit hook whose instantiations already
// exist), and does its name/ref bookkeeping through ONE generic-lambda walk
// rather than a per-type templated SDK op.

#ifndef PS_STUDIO_EDITOR_AUTHORING_OPS_H_
#define PS_STUDIO_EDITOR_AUTHORING_OPS_H_

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "editor/editor_context.h"
#include "types.h"

namespace ps::studio {

namespace mj = ps::mjcf;

// --- Add: body-tree primitives -------------------------------------------- //
//
// Each adds one child element under the body-context container identified by
// `parent_serial` (a Body, a Frame, or -- when parent_serial == 0 -- the world
// body). The new element is uniquely named, sensible authoring defaults are
// stamped where a bare element would not compile (geom sizes), the op is one
// labelled undo entry, and the new element is auto-selected. Returns the new
// element's creation serial, or 0 when the parent does not resolve to a valid
// body-context container.

std::uint64_t AddBodyOp(EditorContext& ctx, std::uint64_t parent_serial);
std::uint64_t AddGeomOp(EditorContext& ctx, std::uint64_t parent_serial,
                        mj::GeomType type);

// Batch add: `count` geoms of `type` under `parent_serial`, laid out in a row
// (each offset a fixed step further along +X) so they do not overlap. ONE undo
// entry for the whole batch; the last geom is selected. `count <= 1` is exactly
// AddGeomOp. Returns the last geom's serial, or 0 when the parent is invalid.
std::uint64_t AddGeomsOp(EditorContext& ctx, std::uint64_t parent_serial,
                         mj::GeomType type, int count);
// JointType::free adds a <freejoint>; the other three add a typed <joint>.
std::uint64_t AddJointOp(EditorContext& ctx, std::uint64_t parent_serial,
                         mj::JointType type);
// Quick-rig add: a hinge/slide joint with `axis` (a body-frame unit vector)
// baked in, so the context-menu presets ("hinge X/Y/Z", "slide X/Y/Z") land a
// ready-to-use joint. One undo entry; the new joint is selected.
std::uint64_t AddJointAxisOp(EditorContext& ctx, std::uint64_t parent_serial,
                             mj::JointType type, const double axis[3]);
std::uint64_t AddSiteOp(EditorContext& ctx, std::uint64_t parent_serial);
std::uint64_t AddCameraOp(EditorContext& ctx, std::uint64_t parent_serial);
std::uint64_t AddLightOp(EditorContext& ctx, std::uint64_t parent_serial);
std::uint64_t AddFrameOp(EditorContext& ctx, std::uint64_t parent_serial);

// Viewport "drop" add: a new world-parented body carrying one geom, positioned
// at `world_point` (the ray hit on existing geometry or the ground plane). The
// body is selected. Returns the new body's serial.
std::uint64_t AddDropBodyGeomOp(EditorContext& ctx, mj::GeomType type,
                                const double world_point[3]);

// --- Add: model-level elements -------------------------------------------- //

// The actuator spellings surfaced by the Add menu.
enum class ActuatorSpelling {
  Motor, Position, Velocity, IntVelocity, Damper, Cylinder, Muscle, Adhesion,
  DcMotor, General
};
// A subset of sensor spellings (the common, target-aware ones).
enum class SensorSpelling {
  Jointpos, Jointvel, Actuatorpos, Actuatorvel, Framepos, Framequat,
  Gyro, Accelerometer, Velocimeter, Force, Torque, Touch
};

// Add an actuator of `spelling`. When the spelling has a joint transmission and
// `target_joint_serial` names a Joint, it is wired as the target. One undo
// entry; selects the new actuator; returns its serial.
std::uint64_t AddActuatorOp(EditorContext& ctx, ActuatorSpelling spelling,
                            std::uint64_t target_joint_serial);
// Add a sensor of `spelling`. `target_serial` (0 == none) wires the sensor's
// site (gyro/accel/...), joint (jointpos/vel), or frame object (framepos/quat)
// where the type has one and the target resolves to a compatible element.
std::uint64_t AddSensorOp(EditorContext& ctx, SensorSpelling spelling,
                          std::uint64_t target_serial);

// --- Assets: materials & textures ----------------------------------------- //

// A new material's authored appearance. Only fields the New Material dialog
// exposes; everything else stays unset (inherits IDL/class defaults, DR-1).
struct MaterialSpec {
  std::string name;
  std::array<float, 4> rgba{0.8f, 0.8f, 0.8f, 1.0f};
  float specular = 0.5f;
  float shininess = 0.5f;
  float reflectance = 0.0f;
  float metallic = 0.0f;
  float roughness = 1.0f;
  // When non-empty, a MaterialLayer{role=rgb} is added referencing this texture.
  std::string texture_rgb;
};

// A new texture. `builtin` selects a procedural pattern (checker/gradient/flat)
// with rgb1/rgb2/markrgb colours and a width/height; otherwise `file` names an
// on-disk image the compiler loads.
struct TextureSpec {
  std::string name;
  bool builtin = true;
  mj::TextureBuiltin builtin_type = mj::TextureBuiltin::checker;
  mj::TextureType type = mj::TextureType::twod;
  std::array<double, 3> rgb1{0.8, 0.8, 0.8};
  std::array<double, 3> rgb2{0.2, 0.2, 0.2};
  std::array<double, 3> markrgb{0.0, 0.0, 0.0};
  int width = 100;
  int height = 100;
  std::string file;  // when !builtin
};

// Create a fully-specified material / texture asset from a dialog spec (name +
// appearance in one shot, the "New Material" / "New Texture" flow). One undo
// entry each; the new asset is selected; returns its creation serial.
std::uint64_t CreateMaterialOp(EditorContext& ctx, const MaterialSpec& spec);
std::uint64_t CreateTextureOp(EditorContext& ctx, const TextureSpec& spec);

// Dialog-commit validation for the New Material / New Texture flows. The create
// button is gated on these: a create needs a non-empty name, and a builtin
// texture needs positive dimensions while a file texture needs a path. Pure so
// the dialog gate and the tests share one definition of "ready to create".
inline bool MaterialSpecValid(const MaterialSpec& spec) {
  return !spec.name.empty();
}
inline bool TextureSpecValid(const TextureSpec& spec) {
  if (spec.name.empty()) return false;
  return spec.builtin ? (spec.width >= 1 && spec.height >= 1)
                      : !spec.file.empty();
}

// Assign (or clear, when `material_name` is empty) the material of the geom with
// `geom_serial`. One undo entry; the geom stays selected. Returns false when the
// serial does not resolve to a geom.
bool AssignGeomMaterialOp(EditorContext& ctx, std::uint64_t geom_serial,
                          const std::string& material_name);

// Add a fixed tendon, a weld equality, a contact pair, an exclude, a keyframe,
// or a new default class nested under `main`. One undo entry each; the new
// element is selected; returns its creation serial. (Tendon/equality/pair/
// exclude are created empty -- they need targets wired in Details before they
// compile; keyframe and default class compile as-is.)
std::uint64_t AddTendonOp(EditorContext& ctx);
std::uint64_t AddEqualityWeldOp(EditorContext& ctx);
std::uint64_t AddPairOp(EditorContext& ctx);
std::uint64_t AddExcludeOp(EditorContext& ctx);
std::uint64_t AddKeyframeOp(EditorContext& ctx);
std::uint64_t AddDefaultClassOp(EditorContext& ctx, const std::string& name);

// --- Duplicate ------------------------------------------------------------ //

// Deep-clone the element with `serial` and its subtree as the NEXT SIBLING in the
// same container. The clone carries FRESH serials (the generated Clone mints
// them), every named element in the clone is uniquely renamed (suffix _1, _2,
// ...), and typed references INTERNAL to the clone are remapped to the new names
// while references pointing OUTSIDE the clone are preserved verbatim. One undo
// entry; the copy is selected. Returns the copy's root serial, or 0 when the
// serial does not resolve to a duplicatable element.
std::uint64_t DuplicateOp(EditorContext& ctx, std::uint64_t serial);

// --- Reparent ------------------------------------------------------------- //

struct ReparentResult {
  bool ok = false;
  std::string error;           // why it was rejected (empty on success)
  std::uint64_t serial = 0;    // the moved element (unchanged identity)
};

// Move the body-context element `elem_serial` out of its current parent and into
// the body-context container `new_parent_serial` (a Body / Frame / the world
// body) as its last child. Rejected (nothing moved) when the target is not a
// valid container, when the move would form a cycle (dropping a body into its own
// subtree), or when the element is not a movable body-context child. With
// `keep_world_pose` the element's authored pos/orient are recomputed so its world
// pose at qpos0 is invariant (the SE2 parent-frame math, frames included). One
// undo entry.
ReparentResult ReparentOp(EditorContext& ctx, std::uint64_t elem_serial,
                          std::uint64_t new_parent_serial, bool keep_world_pose);

// --- New model ------------------------------------------------------------ //

// A minimal starter model: worldbody + an infinite ground-plane geom + one
// light (the MuJoCo starter scene). Ownership transfers to the caller.
std::unique_ptr<mj::Model> NewStarterModel();

// Install a fresh starter model as the live tree: clears history / selection /
// source path, compiles it, and marks the model dirty (unsaved). Returns true
// when the starter compiled and was adopted.
bool NewModelOp(EditorContext& ctx);

// --- Shared helpers (exposed for tests) ----------------------------------- //

// A name unique among the elements whose type is in `category` (Joint and
// FreeJoint share one category; every other element type is its own): returns
// `base` when free, else `base_1`, `base_2`, ...
std::string UniqueName(mj::Model& model,
                       const std::vector<mj::ElementType>& category,
                       const std::string& base);

}  // namespace ps::studio

#endif  // PS_STUDIO_EDITOR_AUTHORING_OPS_H_
