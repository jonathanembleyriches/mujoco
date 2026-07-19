// ProtoSpec Studio SE3 authoring operations (ps::studio, ours). See the header
// for the contract and the per-op instantiation-cost rationale.

#include "editor/authoring_ops.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <variant>
#include <vector>

#include <mujoco/mujoco.h>

#include "compile.h"
#include "editor/editor_ops.h"
#include "editor/element_access.h"
#include "editor/layers.h"
#include "editor/transform_math.h"
#include "protospec/attach.h"  // sdk::Duplicate / sdk::Reparent (public verbs)
#include "protospec/builders.h"
#include "protospec/refs.h"
#include "protospec/traversal.h"
#include "reflect.h"
#include "types.h"
#include "visit.h"

namespace ps::studio {

namespace mj = ps::mjcf;
namespace sdk = ps::sdk;
namespace reflect = ps::mjcf::reflect;

namespace {

// A body-context container: a Body, a Frame, or the world body (for a 0 serial).
struct Container {
  mj::Body* body = nullptr;
  mj::Frame* frame = nullptr;
  bool valid() const { return body || frame; }
  std::vector<mj::BodyChildAny>& subtree() {
    return body ? body->subtree : frame->subtree;
  }
  std::uint64_t serial() const { return body ? body->serial : frame->serial; }
  const void* ptr() const {
    return body ? static_cast<const void*>(body)
                : static_cast<const void*>(frame);
  }
};

Container FindContainer(mj::Model& tree, std::uint64_t serial) {
  Container c;
  if (serial == 0) {
    c.body = &sdk::World(tree);
    return c;
  }
  auto [ptr, type] = FindSerialWithType(tree, serial);
  if (type == mj::ElementType::Body) {
    c.body = static_cast<mj::Body*>(ptr);
  } else if (type == mj::ElementType::Frame) {
    c.frame = static_cast<mj::Frame*>(ptr);
  }
  return c;
}

std::string GeomLabel(mj::GeomType t) {
  switch (t) {
    case mj::GeomType::plane: return "plane";
    case mj::GeomType::hfield: return "hfield";
    case mj::GeomType::sphere: return "sphere";
    case mj::GeomType::capsule: return "capsule";
    case mj::GeomType::ellipsoid: return "ellipsoid";
    case mj::GeomType::cylinder: return "cylinder";
    case mj::GeomType::box: return "box";
    case mj::GeomType::mesh: return "mesh";
    case mj::GeomType::sdf: return "sdf";
  }
  return "geom";
}

// The BeginEdit/mutate/CommitEdit/select boilerplate every add op shares. `fn`
// mutates the tree and returns the new element's serial, or 0 to abort (which
// rolls the pristine snapshot back, recording no undo entry).
template <class Fn>
std::uint64_t DoAdd(EditorContext& ctx, const std::string& label, Fn&& fn) {
  if (!ctx.tree) return 0;
  ctx.BeginEdit();
  const std::uint64_t s = fn(*ctx.tree);
  if (s == 0) {
    ctx.CancelEdit();
    return 0;
  }
  ctx.CommitEdit(label);
  SelectBySerial(ctx, s);
  return s;
}

struct TargetRef {
  mj::ElementType type = mj::ElementType::Model;
  std::string name;
};

// The (type, name) of the element carrying `serial`, over the SDK's single
// find-by-serial walk. `type` stays Model / `name` stays empty when unfound or
// nameless.
TargetRef TargetBySerial(mj::Model& tree, std::uint64_t serial) {
  TargetRef t;
  const sdk::Located loc = sdk::FindBySerialTyped(tree, serial);
  if (loc) {
    t.type = loc.type;
    t.name = sdk::NameOfSerial(tree, serial);
  }
  return t;
}

// True when `t` is a spelling of the ActuatorAny union (classified off the
// schema, so it tracks the IDL rather than a hand-kept list).
inline bool IsActuatorType(mj::ElementType t) {
  const reflect::UnionDescriptor& u = reflect::DescribeUnion("ActuatorAny");
  for (std::size_t i = 0; i < u.member_count; ++i) {
    if (u.members[i] == t) return true;
  }
  return false;
}

// Wire a sensor's target field from a resolved (type, name) selection, using
// whichever transmission the sensor spelling exposes (joint / site / actuator /
// frame object). No-op when the target is empty or incompatible.
template <class S>
void WireSensorTarget(S& s, mj::ElementType ttype, const std::string& tname) {
  if (tname.empty()) return;
  if constexpr (requires { s.joint; }) {
    if (ttype == mj::ElementType::Joint || ttype == mj::ElementType::FreeJoint) {
      s.joint = ps::Ref<mj::Joint>(tname);
      return;
    }
  }
  if constexpr (requires { s.site; }) {
    if (ttype == mj::ElementType::Site) {
      s.site = ps::Ref<mj::Site>(tname);
      return;
    }
  }
  if constexpr (requires { s.actuator; }) {
    if (IsActuatorType(ttype)) {
      s.actuator = ps::Ref<mj::ActuatorAny>(tname);
      return;
    }
  }
  if constexpr (requires { s.objtype; } && requires { s.objname; }) {
    s.objtype = std::string(reflect::Describe(ttype).xml);
    s.objname = tname;
    return;
  }
}

}  // namespace

// --- Public: name uniquing ------------------------------------------------ //

std::string UniqueName(mj::Model& model,
                       const std::vector<mj::ElementType>& category,
                       const std::string& base) {
  // Thin adapter over the promoted `sdk::UniqueName(model, type, base)`. The SDK
  // verb namespaces by MuJoCo name-category (detail::NameCategory), which already
  // folds the two joint spellings (and each actuator / sensor / tendon / equality
  // union) into one namespace -- so passing the first requested type reproduces
  // the old vector semantics for every category the editor uses.
  if (category.empty()) return sdk::UniqueName(model, mj::ElementType::Body, base);
  return sdk::UniqueName(model, category.front(), base);
}

// --- Public: body-tree primitive adds ------------------------------------- //

std::uint64_t AddBodyOp(EditorContext& ctx, std::uint64_t parent_serial) {
  return DoAdd(ctx, "Add body", [&](mj::Model& tree) -> std::uint64_t {
    Container c = FindContainer(tree, parent_serial);
    if (!c.valid()) return 0;
    const std::string n = sdk::UniqueName(tree, mj::ElementType::Body, "body");
    return c.body ? sdk::AddBody(*c.body, n).serial
                  : sdk::AddBody(*c.frame, n).serial;
  });
}

std::uint64_t AddGeomOp(EditorContext& ctx, std::uint64_t parent_serial,
                        mj::GeomType type) {
  return DoAdd(ctx, "Add " + GeomLabel(type),
               [&](mj::Model& tree) -> std::uint64_t {
    Container c = FindContainer(tree, parent_serial);
    if (!c.valid()) return 0;
    const std::string n = sdk::UniqueName(tree, mj::ElementType::Geom, "geom");
    mj::Geom& g =
        c.body ? sdk::AddGeom(*c.body, type, n) : sdk::AddGeom(*c.frame, type, n);
    sdk::SeedPrimitiveSize(g);
    return g.serial;
  });
}

std::uint64_t AddGeomsOp(EditorContext& ctx, std::uint64_t parent_serial,
                         mj::GeomType type, int count) {
  const int n = count < 1 ? 1 : count;
  const std::string label =
      n == 1 ? ("Add " + GeomLabel(type))
             : ("Add " + std::to_string(n) + " " + GeomLabel(type) + "s");
  return DoAdd(ctx, label, [&](mj::Model& tree) -> std::uint64_t {
    Container c = FindContainer(tree, parent_serial);
    if (!c.valid()) return 0;
    constexpr double kStep = 0.3;  // metres between adjacent geoms in the row
    std::uint64_t last = 0;
    for (int i = 0; i < n; ++i) {
      const std::string gn =
          sdk::UniqueName(tree, mj::ElementType::Geom, "geom");
      mj::Geom& g = c.body ? sdk::AddGeom(*c.body, type, gn)
                           : sdk::AddGeom(*c.frame, type, gn);
      sdk::SeedPrimitiveSize(g);
      if (i > 0) g.pos = std::array<double, 3>{i * kStep, 0.0, 0.0};
      last = g.serial;
    }
    return last;
  });
}

std::uint64_t AddJointOp(EditorContext& ctx, std::uint64_t parent_serial,
                         mj::JointType type) {
  const bool freejoint = (type == mj::JointType::free);
  return DoAdd(ctx, freejoint ? "Add freejoint" : "Add joint",
               [&](mj::Model& tree) -> std::uint64_t {
    Container c = FindContainer(tree, parent_serial);
    if (!c.valid()) return 0;
    if (freejoint) {
      const std::string n =
          sdk::UniqueName(tree, mj::ElementType::FreeJoint, "freejoint");
      return c.body ? sdk::AddFreeJoint(*c.body, n).serial
                    : sdk::AddFreeJoint(*c.frame, n).serial;
    }
    const std::string n = sdk::UniqueName(tree, mj::ElementType::Joint, "joint");
    return c.body ? sdk::AddJoint(*c.body, type, n).serial
                  : sdk::AddJoint(*c.frame, type, n).serial;
  });
}

std::uint64_t AddJointAxisOp(EditorContext& ctx, std::uint64_t parent_serial,
                             mj::JointType type, const double axis[3]) {
  const std::array<double, 3> ax{axis[0], axis[1], axis[2]};
  return DoAdd(ctx, "Add joint", [&](mj::Model& tree) -> std::uint64_t {
    Container c = FindContainer(tree, parent_serial);
    if (!c.valid()) return 0;
    const std::string n = sdk::UniqueName(tree, mj::ElementType::Joint, "joint");
    mj::Joint& j =
        c.body ? sdk::AddJoint(*c.body, type, n) : sdk::AddJoint(*c.frame, type, n);
    j.axis = ax;
    return j.serial;
  });
}

std::uint64_t AddSiteOp(EditorContext& ctx, std::uint64_t parent_serial) {
  return DoAdd(ctx, "Add site", [&](mj::Model& tree) -> std::uint64_t {
    Container c = FindContainer(tree, parent_serial);
    if (!c.valid()) return 0;
    const std::string n = sdk::UniqueName(tree, mj::ElementType::Site, "site");
    return c.body ? sdk::AddSite(*c.body, n).serial
                  : sdk::AddSite(*c.frame, n).serial;
  });
}

std::uint64_t AddCameraOp(EditorContext& ctx, std::uint64_t parent_serial) {
  return DoAdd(ctx, "Add camera", [&](mj::Model& tree) -> std::uint64_t {
    Container c = FindContainer(tree, parent_serial);
    if (!c.valid()) return 0;
    const std::string n = sdk::UniqueName(tree, mj::ElementType::Camera, "camera");
    return c.body ? sdk::AddCamera(*c.body, n).serial
                  : sdk::AddCamera(*c.frame, n).serial;
  });
}

std::uint64_t AddLightOp(EditorContext& ctx, std::uint64_t parent_serial) {
  return DoAdd(ctx, "Add light", [&](mj::Model& tree) -> std::uint64_t {
    Container c = FindContainer(tree, parent_serial);
    if (!c.valid()) return 0;
    const std::string n = sdk::UniqueName(tree, mj::ElementType::Light, "light");
    return c.body ? sdk::AddLight(*c.body, n).serial
                  : sdk::AddLight(*c.frame, n).serial;
  });
}

std::uint64_t AddFrameOp(EditorContext& ctx, std::uint64_t parent_serial) {
  return DoAdd(ctx, "Add frame", [&](mj::Model& tree) -> std::uint64_t {
    Container c = FindContainer(tree, parent_serial);
    if (!c.valid()) return 0;
    const std::string n = sdk::UniqueName(tree, mj::ElementType::Frame, "frame");
    return c.body ? sdk::AddFrame(*c.body, n).serial
                  : sdk::AddFrame(*c.frame, n).serial;
  });
}

std::uint64_t AddDropBodyGeomOp(EditorContext& ctx, mj::GeomType type,
                                const double world_point[3]) {
  return DoAdd(ctx, "Add " + GeomLabel(type),
               [&](mj::Model& tree) -> std::uint64_t {
    mj::Body& w = sdk::World(tree);
    const std::string bn = sdk::UniqueName(tree, mj::ElementType::Body, "body");
    mj::Body& b = sdk::AddBody(w, bn);
    if (world_point) {
      b.pos = std::array<double, 3>{world_point[0], world_point[1],
                                    world_point[2]};
    }
    const std::string gn = sdk::UniqueName(tree, mj::ElementType::Geom, "geom");
    mj::Geom& g = sdk::AddGeom(b, type, gn);
    sdk::SeedPrimitiveSize(g);
    return b.serial;
  });
}

// --- Public: model-level adds --------------------------------------------- //

std::uint64_t AddActuatorOp(EditorContext& ctx, ActuatorSpelling spelling,
                            std::uint64_t target_joint_serial) {
  return DoAdd(ctx, "Add actuator", [&](mj::Model& tree) -> std::uint64_t {
    const std::string jn = sdk::NameOfSerial(tree, target_joint_serial);
    switch (spelling) {
      case ActuatorSpelling::Motor:
        return sdk::AddActuator<mj::Motor>(
                   tree, jn, sdk::UniqueName(tree, mj::ElementType::Motor, "motor"))
            .serial;
      case ActuatorSpelling::Position:
        return sdk::AddActuator<mj::Position>(
                   tree, jn,
                   sdk::UniqueName(tree, mj::ElementType::Position, "position"))
            .serial;
      case ActuatorSpelling::Velocity:
        return sdk::AddActuator<mj::Velocity>(
                   tree, jn,
                   sdk::UniqueName(tree, mj::ElementType::Velocity, "velocity"))
            .serial;
      case ActuatorSpelling::IntVelocity:
        return sdk::AddActuator<mj::IntVelocity>(
                   tree, jn,
                   sdk::UniqueName(tree, mj::ElementType::IntVelocity, "intvelocity"))
            .serial;
      case ActuatorSpelling::Damper:
        return sdk::AddActuator<mj::Damper>(
                   tree, jn,
                   sdk::UniqueName(tree, mj::ElementType::Damper, "damper"))
            .serial;
      case ActuatorSpelling::Cylinder:
        return sdk::AddActuator<mj::Cylinder>(
                   tree, jn,
                   sdk::UniqueName(tree, mj::ElementType::Cylinder, "cylinder"))
            .serial;
      case ActuatorSpelling::Muscle:
        return sdk::AddActuator<mj::Muscle>(
                   tree, jn,
                   sdk::UniqueName(tree, mj::ElementType::Muscle, "muscle"))
            .serial;
      case ActuatorSpelling::Adhesion:
        return sdk::AddActuator<mj::Adhesion>(
                   tree, "",
                   sdk::UniqueName(tree, mj::ElementType::Adhesion, "adhesion"))
            .serial;
      case ActuatorSpelling::DcMotor:
        return sdk::AddActuator<mj::DcMotor>(
                   tree, jn,
                   sdk::UniqueName(tree, mj::ElementType::DcMotor, "dcmotor"))
            .serial;
      case ActuatorSpelling::General:
        return sdk::AddActuator<mj::ActuatorGeneral>(
                   tree, jn,
                   sdk::UniqueName(tree, mj::ElementType::ActuatorGeneral, "general"))
            .serial;
    }
    return 0;
  });
}

std::uint64_t AddSensorOp(EditorContext& ctx, SensorSpelling spelling,
                          std::uint64_t target_serial) {
  return DoAdd(ctx, "Add sensor", [&](mj::Model& tree) -> std::uint64_t {
    const TargetRef t = TargetBySerial(tree, target_serial);
    auto make = [&](auto& s) -> std::uint64_t {
      WireSensorTarget(s, t.type, t.name);
      return s.serial;
    };
    switch (spelling) {
      case SensorSpelling::Jointpos:
        return make(sdk::AddSensor<mj::Jointpos>(
            tree, sdk::UniqueName(tree, mj::ElementType::Jointpos, "jointpos")));
      case SensorSpelling::Jointvel:
        return make(sdk::AddSensor<mj::Jointvel>(
            tree, sdk::UniqueName(tree, mj::ElementType::Jointvel, "jointvel")));
      case SensorSpelling::Actuatorpos:
        return make(sdk::AddSensor<mj::Actuatorpos>(
            tree,
            sdk::UniqueName(tree, mj::ElementType::Actuatorpos, "actuatorpos")));
      case SensorSpelling::Actuatorvel:
        return make(sdk::AddSensor<mj::Actuatorvel>(
            tree,
            sdk::UniqueName(tree, mj::ElementType::Actuatorvel, "actuatorvel")));
      case SensorSpelling::Framepos:
        return make(sdk::AddSensor<mj::Framepos>(
            tree, sdk::UniqueName(tree, mj::ElementType::Framepos, "framepos")));
      case SensorSpelling::Framequat:
        return make(sdk::AddSensor<mj::Framequat>(
            tree, sdk::UniqueName(tree, mj::ElementType::Framequat, "framequat")));
      case SensorSpelling::Gyro:
        return make(sdk::AddSensor<mj::Gyro>(
            tree, sdk::UniqueName(tree, mj::ElementType::Gyro, "gyro")));
      case SensorSpelling::Accelerometer:
        return make(sdk::AddSensor<mj::Accelerometer>(
            tree, sdk::UniqueName(tree, mj::ElementType::Accelerometer,
                                "accelerometer")));
      case SensorSpelling::Velocimeter:
        return make(sdk::AddSensor<mj::Velocimeter>(
            tree,
            sdk::UniqueName(tree, mj::ElementType::Velocimeter, "velocimeter")));
      case SensorSpelling::Force:
        return make(sdk::AddSensor<mj::Force>(
            tree, sdk::UniqueName(tree, mj::ElementType::Force, "force")));
      case SensorSpelling::Torque:
        return make(sdk::AddSensor<mj::Torque>(
            tree, sdk::UniqueName(tree, mj::ElementType::Torque, "torque")));
      case SensorSpelling::Touch:
        return make(sdk::AddSensor<mj::Touch>(
            tree, sdk::UniqueName(tree, mj::ElementType::Touch, "touch")));
    }
    return 0;
  });
}

std::uint64_t CreateMaterialOp(EditorContext& ctx, const MaterialSpec& spec) {
  return DoAdd(ctx, "Add material", [&](mj::Model& tree) -> std::uint64_t {
    const std::string n = sdk::UniqueName(
        tree, mj::ElementType::Material,
        spec.name.empty() ? "material" : spec.name);
    mj::Material& m = sdk::AddMaterial(tree, n);
    m.rgba = spec.rgba;
    m.specular = spec.specular;
    m.shininess = spec.shininess;
    m.reflectance = spec.reflectance;
    m.metallic = spec.metallic;
    m.roughness = spec.roughness;
    if (!spec.texture_rgb.empty()) {
      auto layer = std::make_unique<mj::MaterialLayer>();
      layer->texture = ps::Ref<mj::Texture>(spec.texture_rgb);
      layer->role = mj::TexRole::rgb;
      m.layers.push_back(std::move(layer));
    }
    return m.serial;
  });
}

std::uint64_t CreateTextureOp(EditorContext& ctx, const TextureSpec& spec) {
  return DoAdd(ctx, "Add texture", [&](mj::Model& tree) -> std::uint64_t {
    const std::string n = sdk::UniqueName(
        tree, mj::ElementType::Texture,
        spec.name.empty() ? "texture" : spec.name);
    mj::Texture& t = sdk::AddTexture(tree, n);
    t.type = spec.type;
    if (spec.builtin) {
      t.source = mj::TextureSource{spec.builtin_type};  // builtin arm
      t.rgb1 = spec.rgb1;
      t.rgb2 = spec.rgb2;
      t.markrgb = spec.markrgb;
      t.width = spec.width;
      t.height = spec.height;
    } else {
      t.source = mj::TextureSource{mj::TexFile{spec.file}};
    }
    return t.serial;
  });
}

bool AssignGeomMaterialOp(EditorContext& ctx, std::uint64_t geom_serial,
                          const std::string& material_name) {
  if (!ctx.tree || geom_serial == 0) return false;
  mj::Geom* target = FindSerialAs<mj::Geom>(*ctx.tree, geom_serial);
  if (!target) return false;
  ctx.BeginEdit();
  if (material_name.empty()) {
    target->material.reset();
  } else {
    target->material = ps::Ref<mj::Material>(material_name);
  }
  ctx.CommitEdit(material_name.empty() ? "Clear geom material"
                                       : "Assign geom material");
  SelectBySerial(ctx, geom_serial);
  return true;
}

std::uint64_t AddTendonOp(EditorContext& ctx) {
  return DoAdd(ctx, "Add tendon", [&](mj::Model& tree) -> std::uint64_t {
    return sdk::AddTendon<mj::Fixed>(
               tree, sdk::UniqueName(tree, mj::ElementType::Fixed, "tendon"))
        .serial;
  });
}

std::uint64_t AddEqualityWeldOp(EditorContext& ctx) {
  return DoAdd(ctx, "Add equality", [&](mj::Model& tree) -> std::uint64_t {
    return sdk::AddEquality<mj::Weld>(
               tree, sdk::UniqueName(tree, mj::ElementType::Weld, "weld"))
        .serial;
  });
}

std::uint64_t AddPairOp(EditorContext& ctx) {
  return DoAdd(ctx, "Add pair", [&](mj::Model& tree) -> std::uint64_t {
    return sdk::AddPair(tree, sdk::UniqueName(tree, mj::ElementType::Pair, "pair"))
        .serial;
  });
}

std::uint64_t AddExcludeOp(EditorContext& ctx) {
  return DoAdd(ctx, "Add exclude", [&](mj::Model& tree) -> std::uint64_t {
    return sdk::AddExclude(
               tree, sdk::UniqueName(tree, mj::ElementType::Exclude, "exclude"))
        .serial;
  });
}

std::uint64_t AddKeyframeOp(EditorContext& ctx) {
  return DoAdd(ctx, "Add keyframe", [&](mj::Model& tree) -> std::uint64_t {
    return sdk::AddKey(tree, sdk::UniqueName(tree, mj::ElementType::Key, "key"))
        .serial;
  });
}

std::uint64_t AddDefaultClassOp(EditorContext& ctx, const std::string& name) {
  return DoAdd(ctx, "Add default class", [&](mj::Model& tree) -> std::uint64_t {
    const std::string n = sdk::UniqueName(tree, mj::ElementType::Default,
                                        name.empty() ? "class" : name);
    return sdk::AddDefault(tree, n).serial;
  });
}

// --- Public: Duplicate ---------------------------------------------------- //

std::uint64_t DuplicateOp(EditorContext& ctx, std::uint64_t serial) {
  if (!ctx.tree || serial == 0) return 0;
  const void* target = sdk::FindBySerial(*ctx.tree, serial);
  if (!target) return 0;

  ctx.BeginEdit();
  // The public runtime-pointer SDK verb: deep-clone as the next sibling with fresh
  // serials, re-unique names, and remap the clone's internal refs. sdk::SerialOf
  // recovers the clone's serial (the identity the editor selects by).
  const sdk::DuplicateResult dup = sdk::Duplicate(*ctx.tree, target);
  const std::uint64_t new_serial =
      dup.ok ? sdk::SerialOf(*ctx.tree, dup.clone) : 0;
  if (new_serial == 0) {
    ctx.CancelEdit();
    return 0;
  }
  ctx.CommitEdit("Duplicate");
  SelectBySerial(ctx, new_serial);
  return new_serial;
}

// --- Public: Reparent ----------------------------------------------------- //

namespace {

// Write the authored local pose L (pos + quat) onto the element at `elem` (still
// valid after sdk::Reparent's move, which relocates the owning unique_ptr but not
// the element object). The keep-world-pose fixup the SDK deliberately leaves to
// the compile-aware caller.
void SetElemLocalPose(mj::Model& tree, const void* elem, const Rigid& L) {
  sdk::WalkModel(tree, [&](auto& e) {
    if (static_cast<const void*>(&e) != elem) return;
    if constexpr (requires { e.pos; }) {
      e.pos = std::array<double, 3>{L.pos[0], L.pos[1], L.pos[2]};
    }
    if constexpr (requires { e.quat; }) {
      e.quat = std::array<double, 4>{L.quat[0], L.quat[1], L.quat[2], L.quat[3]};
    }
  });
}

}  // namespace

ReparentResult ReparentOp(EditorContext& ctx, std::uint64_t elem_serial,
                          std::uint64_t new_parent_serial,
                          bool keep_world_pose) {
  ReparentResult r;
  if (!ctx.tree) {
    r.error = "no model";
    return r;
  }
  const void* elem = sdk::FindBySerial(*ctx.tree, elem_serial);
  if (!elem) {
    r.error = "element not found";
    return r;
  }
  Container pc = FindContainer(*ctx.tree, new_parent_serial);
  if (!pc.valid()) {
    r.error = "target is not a body or frame";
    return r;
  }
  void* new_parent = pc.body ? static_cast<void*>(pc.body)
                             : static_cast<void*>(pc.frame);

  // Capture the pre-move authored world pose of the element and of the new parent
  // (a forwarded qpos0 mjData off the last good compile). Both come from the tree
  // side of the delta rule -- no compiled mesh-baked suffix is ever inverted.
  DragFrame we, wp;
  bool have_pose = false;
  if (keep_world_pose && ctx.compiled.model) {
    const mjModel* m = ctx.compiled.model.get();
    if (mjData* dd = mj_makeData(m)) {
      mj_resetData(m, dd);
      mj_forward(m, dd);
      we = BuildDragFrame(m, dd, ctx.compiled.binding, *ctx.tree, elem_serial);
      wp = BuildDragFrame(m, dd, ctx.compiled.binding, *ctx.tree, pc.serial());
      mj_deleteData(dd);
      have_pose = we.valid;  // false for non-spatial (e.g. joint): nothing to keep
    }
  }

  ctx.BeginEdit();
  // The public runtime-pointer SDK verb: pure-tree move (element keeps its
  // authored local pose; the SDK rejects cycles / non-container targets).
  sdk::ReparentResult sr = sdk::Reparent(*ctx.tree, elem, new_parent);
  if (!sr.ok) {
    ctx.CancelEdit();
    r.error = sr.reason;
    return r;
  }

  if (keep_world_pose && have_pose) {
    Rigid W;
    for (int i = 0; i < 4; ++i) W.quat[i] = we.world_quat[i];
    for (int i = 0; i < 3; ++i) W.pos[i] = we.anchor[i];
    Rigid Pn;  // identity when the new parent is the world body
    if (wp.valid) {
      for (int i = 0; i < 4; ++i) Pn.quat[i] = wp.world_quat[i];
      for (int i = 0; i < 3; ++i) Pn.pos[i] = wp.anchor[i];
    }
    const Rigid L = Compose(Invert(Pn), W);
    SetElemLocalPose(*ctx.tree, elem, L);
  }

  ctx.CommitEdit("Reparent");
  SelectBySerial(ctx, elem_serial);
  r.ok = true;
  r.serial = elem_serial;
  return r;
}

// --- Public: New model ---------------------------------------------------- //

std::unique_ptr<mj::Model> NewStarterModel() {
  auto m = std::make_unique<mj::Model>();
  m->model = "untitled";
  mj::Body& w = sdk::World(*m);
  mj::Geom& g = sdk::AddGeom(w, mj::GeomType::plane, "ground");
  g.size = ps::InlineVec<double, 3>{0, 0, 0.1};  // infinite ground, 0.1 spacing
  mj::Light& l = sdk::AddLight(w, "light");
  l.pos = std::array<double, 3>{0, 0, 3};
  l.dir = std::array<double, 3>{0, 0, -1};
  return m;
}

bool NewModelOp(EditorContext& ctx) {
  std::unique_ptr<mj::Model> prev_tree = std::move(ctx.tree);
  mj::Compiled prev = std::move(ctx.compiled);
  std::unique_ptr<mj::Model> prev_compile_tree = std::move(ctx.compile_tree);
  std::vector<Layer> prev_layers = std::move(ctx.layers);
  LayerGraph prev_graph = std::move(ctx.layer_graph);
  const int prev_active = ctx.active_layer;
  ctx.tree = NewStarterModel();
  ctx.base_dir.clear();
  ctx.vfs_assets.clear();
  // A new scene starts as one authored layer with no backing file: the starter
  // elements carry no provenance, so the split stamps them "layer://base".
  SplitLayersFromTree(ctx, "layer://base", "base");
  if (!RecompileTree(ctx)) {
    ctx.tree = std::move(prev_tree);
    ctx.compiled = std::move(prev);
    ctx.compile_tree = std::move(prev_compile_tree);
    ctx.layers = std::move(prev_layers);
    ctx.layer_graph = std::move(prev_graph);
    ctx.active_layer = prev_active;
    return false;
  }
  ctx.source_path.clear();
  ctx.history.Clear();
  // The HOST adopts a model only through the model-source poll, and the poll
  // only hands one over when a recompile is serviced. The trial compile above
  // proved the starter tree good, but clearing the flag here left the host on
  // its previous model -- or, from an empty start, on no model at all (a
  // permanently blank viewport after File > New). Leave the request up so the
  // next poll recompiles and the host adopts.
  ctx.recompile_requested = true;
  ctx.fresh_load = true;
  ctx.selected_serial = 0;
  ctx.selected_desc.clear();
  ctx.dirty = true;
  return true;
}

}  // namespace ps::studio
