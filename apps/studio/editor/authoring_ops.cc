// ProtoSpec Studio SE3 authoring operations (ps::studio, ours). See the header
// for the contract and the per-op instantiation-cost rationale.

#include "editor/authoring_ops.h"

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
#include "editor/transform_math.h"
#include "protospec/builders.h"
#include "protospec/detail.h"
#include "protospec/refs.h"
#include "protospec/traversal.h"
#include "reflect.h"
#include "types.h"
#include "visit.h"

namespace ps::studio {

namespace mj = ps::mjcf;
namespace sdk = ps::sdk;
namespace sdk_detail = ps::sdk::detail;
namespace bridge = ps::mjcf::bridge;
namespace reflect = ps::mjcf::reflect;

namespace {

// MuJoCo names are unique within an object category; every ProtoSpec element type
// is its own category except the two joint spellings, which share one. (Actuator
// and sensor spellings also share a category in MuJoCo, but each Add op derives a
// distinct base name per spelling -- "motor" vs "position", "jointpos" vs "gyro"
// -- so per-type uniquing never produces a cross-spelling clash there.)
int CategoryId(mj::ElementType t) {
  if (t == mj::ElementType::Joint || t == mj::ElementType::FreeJoint) return -1;
  return static_cast<int>(t);
}

std::vector<mj::ElementType> CatOf(mj::ElementType t) {
  if (t == mj::ElementType::Joint || t == mj::ElementType::FreeJoint)
    return {mj::ElementType::Joint, mj::ElementType::FreeJoint};
  return {t};
}

std::string UniqueNameFor(mj::Model& model, mj::ElementType type,
                          const std::string& base) {
  return UniqueName(model, CatOf(type), base);
}

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
  sdk_detail::WalkModelAll(tree, [&](auto& e) {
    using E = std::decay_t<decltype(e)>;
    if (c.valid()) return;
    if constexpr (std::is_same_v<E, mj::Body>) {
      if (e.serial == serial) c.body = &e;
    } else if constexpr (std::is_same_v<E, mj::Frame>) {
      if (e.serial == serial) c.frame = &e;
    }
  });
  return c;
}

// Sensible authoring sizes so a bare primitive geom compiles (a size-0 geom is a
// compile error for every non-mesh type). Only `size` is stamped -- DR-1: nothing
// else of the IDL default set is written.
void StampGeomSize(mj::Geom& g, mj::GeomType t) {
  using V = ps::InlineVec<double, 3>;
  switch (t) {
    case mj::GeomType::sphere: g.size = V{0.1}; break;
    case mj::GeomType::capsule: g.size = V{0.05, 0.1}; break;
    case mj::GeomType::cylinder: g.size = V{0.05, 0.1}; break;
    case mj::GeomType::ellipsoid: g.size = V{0.1, 0.1, 0.1}; break;
    case mj::GeomType::box: g.size = V{0.1, 0.1, 0.1}; break;
    case mj::GeomType::plane: g.size = V{1, 1, 0.1}; break;
    default: break;  // mesh / hfield / sdf: geometry comes from the asset
  }
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

std::string JointNameBySerial(mj::Model& tree, std::uint64_t serial) {
  std::string out;
  if (serial == 0) return out;
  sdk_detail::WalkModelAll(tree, [&](auto& e) {
    using E = std::decay_t<decltype(e)>;
    if constexpr (std::is_same_v<E, mj::Joint> ||
                  std::is_same_v<E, mj::FreeJoint>) {
      if (out.empty() && e.serial == serial) {
        if (const std::string* nm = sdk_detail::NameOf(e)) out = *nm;
      }
    }
  });
  return out;
}

struct TargetRef {
  mj::ElementType type = mj::ElementType::Model;
  std::string name;
};

TargetRef TargetBySerial(mj::Model& tree, std::uint64_t serial) {
  TargetRef t;
  if (serial == 0) return t;
  sdk_detail::WalkModelAll(tree, [&](auto& e) {
    using E = std::decay_t<decltype(e)>;
    if constexpr (!std::is_same_v<E, mj::Model>) {
      if constexpr (requires { e.serial; }) {
        if (t.name.empty() && e.serial == serial) {
          if (const std::string* nm = sdk_detail::NameOf(e)) {
            t.type = mj::element_type_of<E>::value;
            t.name = *nm;
          }
        }
      }
    }
  });
  return t;
}

// Wire a sensor's target field from a resolved (type, name) selection, using
// whichever transmission the sensor spelling exposes (joint / site / frame
// object). No-op when the target is empty or incompatible.
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
  std::unordered_set<std::string> used;
  sdk_detail::WalkModelAll(model, [&](auto& e) {
    using E = std::decay_t<decltype(e)>;
    if constexpr (!std::is_same_v<E, mj::Model>) {
      if (sdk_detail::Contains(category, mj::element_type_of<E>::value)) {
        if (const std::string* nm = sdk_detail::NameOf(e)) used.insert(*nm);
      }
    }
  });
  if (!used.count(base)) return base;
  for (int k = 1;; ++k) {
    std::string c = base + "_" + std::to_string(k);
    if (!used.count(c)) return c;
  }
}

// --- Public: body-tree primitive adds ------------------------------------- //

std::uint64_t AddBodyOp(EditorContext& ctx, std::uint64_t parent_serial) {
  return DoAdd(ctx, "Add body", [&](mj::Model& tree) -> std::uint64_t {
    Container c = FindContainer(tree, parent_serial);
    if (!c.valid()) return 0;
    const std::string n = UniqueNameFor(tree, mj::ElementType::Body, "body");
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
    const std::string n = UniqueNameFor(tree, mj::ElementType::Geom, "geom");
    mj::Geom& g =
        c.body ? sdk::AddGeom(*c.body, type, n) : sdk::AddGeom(*c.frame, type, n);
    StampGeomSize(g, type);
    return g.serial;
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
          UniqueNameFor(tree, mj::ElementType::FreeJoint, "freejoint");
      return c.body ? sdk::AddFreeJoint(*c.body, n).serial
                    : sdk::AddFreeJoint(*c.frame, n).serial;
    }
    const std::string n = UniqueNameFor(tree, mj::ElementType::Joint, "joint");
    return c.body ? sdk::AddJoint(*c.body, type, n).serial
                  : sdk::AddJoint(*c.frame, type, n).serial;
  });
}

std::uint64_t AddSiteOp(EditorContext& ctx, std::uint64_t parent_serial) {
  return DoAdd(ctx, "Add site", [&](mj::Model& tree) -> std::uint64_t {
    Container c = FindContainer(tree, parent_serial);
    if (!c.valid()) return 0;
    const std::string n = UniqueNameFor(tree, mj::ElementType::Site, "site");
    return c.body ? sdk::AddSite(*c.body, n).serial
                  : sdk::AddSite(*c.frame, n).serial;
  });
}

std::uint64_t AddCameraOp(EditorContext& ctx, std::uint64_t parent_serial) {
  return DoAdd(ctx, "Add camera", [&](mj::Model& tree) -> std::uint64_t {
    Container c = FindContainer(tree, parent_serial);
    if (!c.valid()) return 0;
    const std::string n = UniqueNameFor(tree, mj::ElementType::Camera, "camera");
    return c.body ? sdk::AddCamera(*c.body, n).serial
                  : sdk::AddCamera(*c.frame, n).serial;
  });
}

std::uint64_t AddLightOp(EditorContext& ctx, std::uint64_t parent_serial) {
  return DoAdd(ctx, "Add light", [&](mj::Model& tree) -> std::uint64_t {
    Container c = FindContainer(tree, parent_serial);
    if (!c.valid()) return 0;
    const std::string n = UniqueNameFor(tree, mj::ElementType::Light, "light");
    return c.body ? sdk::AddLight(*c.body, n).serial
                  : sdk::AddLight(*c.frame, n).serial;
  });
}

std::uint64_t AddFrameOp(EditorContext& ctx, std::uint64_t parent_serial) {
  return DoAdd(ctx, "Add frame", [&](mj::Model& tree) -> std::uint64_t {
    Container c = FindContainer(tree, parent_serial);
    if (!c.valid()) return 0;
    const std::string n = UniqueNameFor(tree, mj::ElementType::Frame, "frame");
    return c.body ? sdk::AddFrame(*c.body, n).serial
                  : sdk::AddFrame(*c.frame, n).serial;
  });
}

std::uint64_t AddDropBodyGeomOp(EditorContext& ctx, mj::GeomType type,
                                const double world_point[3]) {
  return DoAdd(ctx, "Add " + GeomLabel(type),
               [&](mj::Model& tree) -> std::uint64_t {
    mj::Body& w = sdk::World(tree);
    const std::string bn = UniqueNameFor(tree, mj::ElementType::Body, "body");
    mj::Body& b = sdk::AddBody(w, bn);
    if (world_point) {
      b.pos = std::array<double, 3>{world_point[0], world_point[1],
                                    world_point[2]};
    }
    const std::string gn = UniqueNameFor(tree, mj::ElementType::Geom, "geom");
    mj::Geom& g = sdk::AddGeom(b, type, gn);
    StampGeomSize(g, type);
    return b.serial;
  });
}

// --- Public: model-level adds --------------------------------------------- //

std::uint64_t AddActuatorOp(EditorContext& ctx, ActuatorSpelling spelling,
                            std::uint64_t target_joint_serial) {
  return DoAdd(ctx, "Add actuator", [&](mj::Model& tree) -> std::uint64_t {
    const std::string jn = JointNameBySerial(tree, target_joint_serial);
    switch (spelling) {
      case ActuatorSpelling::Motor:
        return sdk::AddActuator<mj::Motor>(
                   tree, jn, UniqueNameFor(tree, mj::ElementType::Motor, "motor"))
            .serial;
      case ActuatorSpelling::Position:
        return sdk::AddActuator<mj::Position>(
                   tree, jn,
                   UniqueNameFor(tree, mj::ElementType::Position, "position"))
            .serial;
      case ActuatorSpelling::Velocity:
        return sdk::AddActuator<mj::Velocity>(
                   tree, jn,
                   UniqueNameFor(tree, mj::ElementType::Velocity, "velocity"))
            .serial;
      case ActuatorSpelling::IntVelocity:
        return sdk::AddActuator<mj::IntVelocity>(
                   tree, jn,
                   UniqueNameFor(tree, mj::ElementType::IntVelocity, "intvelocity"))
            .serial;
      case ActuatorSpelling::Damper:
        return sdk::AddActuator<mj::Damper>(
                   tree, jn,
                   UniqueNameFor(tree, mj::ElementType::Damper, "damper"))
            .serial;
      case ActuatorSpelling::Cylinder:
        return sdk::AddActuator<mj::Cylinder>(
                   tree, jn,
                   UniqueNameFor(tree, mj::ElementType::Cylinder, "cylinder"))
            .serial;
      case ActuatorSpelling::Muscle:
        return sdk::AddActuator<mj::Muscle>(
                   tree, jn,
                   UniqueNameFor(tree, mj::ElementType::Muscle, "muscle"))
            .serial;
      case ActuatorSpelling::Adhesion:
        return sdk::AddActuator<mj::Adhesion>(
                   tree, "",
                   UniqueNameFor(tree, mj::ElementType::Adhesion, "adhesion"))
            .serial;
      case ActuatorSpelling::DcMotor:
        return sdk::AddActuator<mj::DcMotor>(
                   tree, jn,
                   UniqueNameFor(tree, mj::ElementType::DcMotor, "dcmotor"))
            .serial;
      case ActuatorSpelling::General:
        return sdk::AddActuator<mj::ActuatorGeneral>(
                   tree, jn,
                   UniqueNameFor(tree, mj::ElementType::ActuatorGeneral, "general"))
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
            tree, UniqueNameFor(tree, mj::ElementType::Jointpos, "jointpos")));
      case SensorSpelling::Jointvel:
        return make(sdk::AddSensor<mj::Jointvel>(
            tree, UniqueNameFor(tree, mj::ElementType::Jointvel, "jointvel")));
      case SensorSpelling::Framepos:
        return make(sdk::AddSensor<mj::Framepos>(
            tree, UniqueNameFor(tree, mj::ElementType::Framepos, "framepos")));
      case SensorSpelling::Framequat:
        return make(sdk::AddSensor<mj::Framequat>(
            tree, UniqueNameFor(tree, mj::ElementType::Framequat, "framequat")));
      case SensorSpelling::Gyro:
        return make(sdk::AddSensor<mj::Gyro>(
            tree, UniqueNameFor(tree, mj::ElementType::Gyro, "gyro")));
      case SensorSpelling::Accelerometer:
        return make(sdk::AddSensor<mj::Accelerometer>(
            tree, UniqueNameFor(tree, mj::ElementType::Accelerometer,
                                "accelerometer")));
      case SensorSpelling::Velocimeter:
        return make(sdk::AddSensor<mj::Velocimeter>(
            tree,
            UniqueNameFor(tree, mj::ElementType::Velocimeter, "velocimeter")));
      case SensorSpelling::Force:
        return make(sdk::AddSensor<mj::Force>(
            tree, UniqueNameFor(tree, mj::ElementType::Force, "force")));
      case SensorSpelling::Torque:
        return make(sdk::AddSensor<mj::Torque>(
            tree, UniqueNameFor(tree, mj::ElementType::Torque, "torque")));
      case SensorSpelling::Touch:
        return make(sdk::AddSensor<mj::Touch>(
            tree, UniqueNameFor(tree, mj::ElementType::Touch, "touch")));
    }
    return 0;
  });
}

std::uint64_t AddTendonOp(EditorContext& ctx) {
  return DoAdd(ctx, "Add tendon", [&](mj::Model& tree) -> std::uint64_t {
    return sdk::AddTendon<mj::Fixed>(
               tree, UniqueNameFor(tree, mj::ElementType::Fixed, "tendon"))
        .serial;
  });
}

std::uint64_t AddEqualityWeldOp(EditorContext& ctx) {
  return DoAdd(ctx, "Add equality", [&](mj::Model& tree) -> std::uint64_t {
    // sdk::AddEquality names the section vector `equalitys`, but the Equality
    // element's union list is `equalities`; push the weld directly to avoid that
    // stale builder (cpp/sdk is out of scope to edit here).
    mj::Equality& section = sdk::EnsureEqualitySection(tree);
    auto weld = std::make_unique<mj::Weld>();
    sdk_detail::SetName(*weld,
                        UniqueNameFor(tree, mj::ElementType::Weld, "weld"));
    return sdk::detail::PushUnion<mj::Weld>(section.equalities, std::move(weld))
        .serial;
  });
}

std::uint64_t AddPairOp(EditorContext& ctx) {
  return DoAdd(ctx, "Add pair", [&](mj::Model& tree) -> std::uint64_t {
    return sdk::AddPair(tree, UniqueNameFor(tree, mj::ElementType::Pair, "pair"))
        .serial;
  });
}

std::uint64_t AddExcludeOp(EditorContext& ctx) {
  return DoAdd(ctx, "Add exclude", [&](mj::Model& tree) -> std::uint64_t {
    return sdk::AddExclude(
               tree, UniqueNameFor(tree, mj::ElementType::Exclude, "exclude"))
        .serial;
  });
}

std::uint64_t AddKeyframeOp(EditorContext& ctx) {
  return DoAdd(ctx, "Add keyframe", [&](mj::Model& tree) -> std::uint64_t {
    return sdk::AddKey(tree, UniqueNameFor(tree, mj::ElementType::Key, "key"))
        .serial;
  });
}

std::uint64_t AddDefaultClassOp(EditorContext& ctx, const std::string& name) {
  return DoAdd(ctx, "Add default class", [&](mj::Model& tree) -> std::uint64_t {
    const std::string n = UniqueNameFor(tree, mj::ElementType::Default,
                                        name.empty() ? "class" : name);
    return sdk::AddDefault(tree, n).serial;
  });
}

// --- Public: Duplicate ---------------------------------------------------- //

namespace {

// Insert a fresh-serial deep clone of the element at `target` immediately after
// it in its owning list. The concrete element type is recovered inside the Visit
// hook (whose instantiations already exist), so mj::Clone is the only per-type
// code -- no whole-model walk is instantiated per type.
struct Duplicator {
  const void* target;
  std::uint64_t* new_serial;
  bool* done;

  template <class U>
  void field(int, const char*, U&) {}

  template <class C>
  void child(int, const char*, C& list) {
    if (*done) return;
    for (std::size_t i = 0; i < list.size(); ++i) {
      if (list[i] && static_cast<const void*>(list[i].get()) == target) {
        auto clone = mj::Clone(*list[i]);
        *new_serial = clone->serial;
        list.insert(list.begin() + i + 1, std::move(clone));
        *done = true;
        return;
      }
    }
    for (auto& p : list) {
      if (*done) return;
      if (p) {
        Duplicator d{target, new_serial, done};
        mj::Visit(*p, d);
      }
    }
  }

  template <class C>
  void union_child(int, const char*, C& list) {
    if (*done) return;
    for (std::size_t i = 0; i < list.size(); ++i) {
      bool match = false;
      std::visit(
          [&](auto& p) {
            if (p && static_cast<const void*>(p.get()) == target) match = true;
          },
          list[i].node);
      if (match) {
        typename C::value_type node;
        std::visit(
            [&](auto& p) {
              auto clone = mj::Clone(*p);
              *new_serial = clone->serial;
              node.node = std::move(clone);
            },
            list[i].node);
        list.insert(list.begin() + i + 1, std::move(node));
        *done = true;
        return;
      }
    }
    for (auto& item : list) {
      if (*done) return;
      std::visit(
          [&](auto& p) {
            if (p) {
              Duplicator d{target, new_serial, done};
              mj::Visit(*p, d);
            }
          },
          item.node);
    }
  }
};

const void* FindPtrBySerial(mj::Model& model, std::uint64_t serial) {
  const void* out = nullptr;
  sdk_detail::WalkModelAll(model, [&](auto& e) {
    using E = std::decay_t<decltype(e)>;
    if constexpr (!std::is_same_v<E, mj::Model>) {
      if constexpr (requires { e.serial; }) {
        if (!out && e.serial == serial) out = &e;
      }
    }
  });
  return out;
}

// Uniquely rename every named element of the freshly-cloned subtree rooted at
// `root_serial` and remap the clone's INTERNAL typed references to the new names;
// references pointing outside the clone are untouched. All bookkeeping runs
// through generic-lambda walks (one Walker instantiation each), never a per-type
// templated SDK op with an embedded walk.
void RemapClonedSubtree(mj::Model& model, std::uint64_t root_serial) {
  const void* root = FindPtrBySerial(model, root_serial);
  if (!root) return;

  sdk::ParentMap pm(model);
  auto in_sub = [&](const void* p) -> bool {
    for (const void* q = p; q;) {
      if (q == root) return true;
      const sdk::ParentMap::Node* n = pm.Lookup(q);
      if (!n) break;
      q = n->parent;
    }
    return false;
  };

  struct SubElem {
    mj::ElementType type;
    std::string name;
    std::function<void(const std::string&)> set;
  };
  std::vector<SubElem> sub;
  std::map<int, std::unordered_set<std::string>> reserved;  // outside names
  sdk_detail::WalkModelAll(model, [&](auto& e) {
    using E = std::decay_t<decltype(e)>;
    if constexpr (!std::is_same_v<E, mj::Model>) {
      const std::string* nm = sdk_detail::NameOf(e);
      if (!nm) return;
      const mj::ElementType t = mj::element_type_of<E>::value;
      if (in_sub(&e)) {
        auto* ep = &e;
        sub.push_back({t, *nm, [ep](const std::string& s) {
                         sdk_detail::SetName(*ep, s);
                       }});
      } else {
        reserved[CategoryId(t)].insert(*nm);
      }
    }
  });

  struct Ren {
    mj::ElementType type;
    std::string oldn;
    std::string newn;
  };
  std::vector<Ren> renamed;
  for (auto& se : sub) {
    const int c = CategoryId(se.type);
    std::string cand = se.name;
    for (int k = 1; reserved[c].count(cand); ++k)
      cand = se.name + "_" + std::to_string(k);
    reserved[c].insert(cand);
    if (cand != se.name) {
      renamed.push_back({se.type, se.name, cand});
      se.set(cand);
    }
  }
  if (renamed.empty()) return;

  sdk_detail::WalkModelAll(model, [&](auto& e) {
    using E = std::decay_t<decltype(e)>;
    if constexpr (!std::is_same_v<E, mj::Model>) {
      if (!in_sub(&e)) return;
      sdk_detail::ScanRefs(
          e, [&](int, const char*, std::string& rn,
                 const std::vector<mj::ElementType>& tgts) {
            for (const Ren& r : renamed) {
              if (r.oldn == rn && sdk_detail::Contains(tgts, r.type)) {
                rn = r.newn;
                break;
              }
            }
          });
    }
  });
}

}  // namespace

std::uint64_t DuplicateOp(EditorContext& ctx, std::uint64_t serial) {
  if (!ctx.tree || serial == 0) return 0;
  const void* target = FindPtrBySerial(*ctx.tree, serial);
  if (!target) return 0;

  ctx.BeginEdit();
  std::uint64_t new_serial = 0;
  bool done = false;
  Duplicator d{target, &new_serial, &done};
  mj::Visit(*ctx.tree, d);
  if (!done || new_serial == 0) {
    ctx.CancelEdit();
    return 0;
  }
  RemapClonedSubtree(*ctx.tree, new_serial);
  ctx.CommitEdit("Duplicate");
  SelectBySerial(ctx, new_serial);
  return new_serial;
}

// --- Public: Reparent ----------------------------------------------------- //

namespace {

bool FindBodyChild(std::vector<mj::BodyChildAny>& sub, const void* target,
                   std::vector<mj::BodyChildAny>** out_list, std::size_t* out_i) {
  for (std::size_t i = 0; i < sub.size(); ++i) {
    bool match = false;
    std::visit(
        [&](auto& up) {
          if (up && static_cast<const void*>(up.get()) == target) match = true;
        },
        sub[i].node);
    if (match) {
      *out_list = &sub;
      *out_i = i;
      return true;
    }
    bool rec = false;
    std::visit(
        [&](auto& up) {
          using T = std::decay_t<decltype(*up)>;
          if constexpr (std::is_same_v<T, mj::Body> ||
                        std::is_same_v<T, mj::Frame>) {
            if (up && FindBodyChild(up->subtree, target, out_list, out_i))
              rec = true;
          }
        },
        sub[i].node);
    if (rec) return true;
  }
  return false;
}

bool FindBodyChildInModel(mj::Model& tree, const void* target,
                          std::vector<mj::BodyChildAny>** out_list,
                          std::size_t* out_i) {
  for (auto& w : tree.worldbody) {
    if (w && FindBodyChild(w->subtree, target, out_list, out_i)) return true;
  }
  return false;
}

void CollectSubtreePtrs(mj::BodyChildAny& node,
                        std::unordered_set<const void*>& out) {
  std::visit(
      [&](auto& up) {
        if (up)
          sdk_detail::WalkTree(*up, [&](auto& e) {
            out.insert(static_cast<const void*>(&e));
          });
      },
      node.node);
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
  const void* elem = FindPtrBySerial(*ctx.tree, elem_serial);
  if (!elem) {
    r.error = "element not found";
    return r;
  }
  Container pc = FindContainer(*ctx.tree, new_parent_serial);
  if (!pc.valid()) {
    r.error = "target is not a body or frame";
    return r;
  }
  std::vector<mj::BodyChildAny>* list = nullptr;
  std::size_t idx = 0;
  if (!FindBodyChildInModel(*ctx.tree, elem, &list, &idx)) {
    r.error = "element is not a movable body-context child";
    return r;
  }

  // Cycle rejection: the target container must not be the moved element itself
  // nor anything inside its subtree.
  {
    std::unordered_set<const void*> subptrs;
    CollectSubtreePtrs((*list)[idx], subptrs);
    if (subptrs.count(pc.ptr())) {
      r.error = "cannot reparent into its own subtree";
      return r;
    }
  }

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
  mj::BodyChildAny moved = std::move((*list)[idx]);
  list->erase(list->begin() + idx);
  std::vector<mj::BodyChildAny>& dst = pc.subtree();
  dst.push_back(std::move(moved));
  mj::BodyChildAny& reattached = dst.back();

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
    std::visit(
        [&](auto& up) {
          if (!up) return;
          if constexpr (requires { up->pos; }) {
            up->pos = std::array<double, 3>{L.pos[0], L.pos[1], L.pos[2]};
          }
          if constexpr (requires { up->quat; }) {
            up->quat = std::array<double, 4>{L.quat[0], L.quat[1], L.quat[2],
                                             L.quat[3]};
          }
        },
        reattached.node);
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
  bridge::Compiled prev = std::move(ctx.compiled);
  ctx.tree = NewStarterModel();
  ctx.base_dir.clear();
  ctx.vfs_assets.clear();
  if (!RecompileTree(ctx)) {
    ctx.tree = std::move(prev_tree);
    ctx.compiled = std::move(prev);
    return false;
  }
  ctx.source_name = "untitled";
  ctx.source_path.clear();
  ctx.history.Clear();
  ctx.recompile_requested = false;
  ctx.fresh_load = true;
  ctx.selected_serial = 0;
  ctx.selected_desc.clear();
  ctx.dirty = true;
  return true;
}

}  // namespace ps::studio
