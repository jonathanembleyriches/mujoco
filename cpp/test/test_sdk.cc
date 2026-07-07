// Property tests for the ProtoSpec SDK (milestone 6).
//
// Builds the milestone-exit robot programmatically through the typed builders,
// round-trips it through the MJCF writer/reader as a fixpoint check, then
// exercises the ergonomic layer: traversal + parent map + path, typed-reference
// resolve / find-referrers / rename / recursive-delete, the default-class
// operations (Effective / FlattenDefaults / ExtractClass), and namespaced
// attach. MuJoCo-free: correctness is asserted against the reader fixpoint and
// against the generated value-equality, not against a live compile.

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "mjcf.h"
#include "protospec/sdk.h"
#include "types.h"

using namespace ps::mjcf;
namespace sdk = ps::sdk;

static int g_failed = 0;
static int g_checks = 0;

#define CHECK(cond)                                                     \
  do {                                                                  \
    ++g_checks;                                                         \
    if (!(cond)) {                                                      \
      ++g_failed;                                                       \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);      \
    }                                                                   \
  } while (0)

// --- The milestone-6 exit robot, built entirely through the builders ------ //
//
// world -> torso (freejoint + geom)
//            -> thigh  (childclass "leg"): hinge "hip",  geom "thigh_geom"
//                 -> shin (childclass inherited): hinge "knee", geom "shin_geom"
// assets:   material "steel"
// defaults: class "leg" -> geom{size,rgba}, joint{damping}
// actuators: position "act_hip" -> joint "hip"; position "act_knee" -> "knee"
// sensors:   jointpos "sens_hip" -> "hip"; jointvel "sens_knee" -> "knee"
static std::unique_ptr<Model> BuildRobot() {
  auto m = std::make_unique<Model>();
  m->model = "robot";

  // Default class "leg": shared geom size/rgba + joint damping.
  Default& leg = sdk::AddDefault(*m, "leg");
  {
    auto g = std::make_unique<Geom>();
    g->size = ps::InlineVec<double, 3>{0.05};
    g->rgba = std::array<float, 4>{{1.0f, 0.0f, 0.0f, 1.0f}};
    leg.geom.push_back(std::move(g));
    auto j = std::make_unique<Joint>();
    j->damping = ps::InlineVec<double, 3>{0.5};
    leg.joint.push_back(std::move(j));
  }

  sdk::AddMaterial(*m, "steel");

  Body& world = sdk::World(*m);
  Body& torso = sdk::AddBody(world, "torso");
  sdk::AddFreeJoint(torso);
  sdk::AddGeom(torso, GeomType::capsule, "torso_geom");

  Body& thigh = sdk::AddBody(torso, "thigh");
  thigh.childclass = ps::Ref<Default>("leg");
  sdk::AddJoint(thigh, JointType::hinge, "hip");
  Geom& thigh_geom = sdk::AddGeom(thigh, GeomType::capsule, "thigh_geom");
  thigh_geom.material = ps::Ref<Material>("steel");

  Body& shin = sdk::AddBody(thigh, "shin");
  sdk::AddJoint(shin, JointType::hinge, "knee");
  sdk::AddGeom(shin, GeomType::capsule, "shin_geom");

  Position& act_hip = sdk::AddActuator<Position>(*m, "hip", "act_hip");
  act_hip.kp = 100.0;
  Position& act_knee = sdk::AddActuator<Position>(*m, "knee", "act_knee");
  act_knee.kp = 80.0;

  Jointpos& s_hip = sdk::AddSensor<Jointpos>(*m, "sens_hip");
  s_hip.joint = ps::Ref<Joint>("hip");
  Jointvel& s_knee = sdk::AddSensor<Jointvel>(*m, "sens_knee");
  s_knee.joint = ps::Ref<Joint>("knee");

  return m;
}

// --- builders link elements into the right lists -------------------------- //
static void TestBuilders() {
  auto m = BuildRobot();

  // Body tree: world holds torso, torso holds a freejoint, a geom, and thigh.
  Body& world = sdk::World(*m);
  CHECK(world.subtree.size() == 1);  // torso
  Body* torso = sdk::Find<Body>(*m, "torso");
  CHECK(torso != nullptr);
  // torso subtree: FreeJoint, Geom, Body(thigh) in authored order.
  CHECK(torso->subtree.size() == 3);
  CHECK(torso->subtree[0].kind() == BodyChildAny::Kind::FreeJoint);
  CHECK(torso->subtree[1].kind() == BodyChildAny::Kind::Geom);
  CHECK(torso->subtree[2].kind() == BodyChildAny::Kind::Body);

  // Actuators + sensors landed in their single ordered union sections.
  CHECK(m->actuators.size() == 1);
  CHECK(m->actuators.front()->actuators.size() == 2);
  CHECK(m->actuators.front()->actuators[0].kind() ==
        ActuatorAny::Kind::Position);
  CHECK(m->sensors.front()->sensors.size() == 2);

  // Asset + default class present.
  CHECK(m->assets.front()->materials.size() == 1);
  CHECK(!m->defaults.empty());

  // Structural default mirrors the IDL: AddJoint default type is hinge.
  Body dummy_parent;
  Joint& j = sdk::AddJoint(dummy_parent);
  CHECK(j.type.has_value() && j.type.value() == JointType::hinge);
  // ...but no other IDL default is silently stamped on (DR-1).
  CHECK(!j.damping.has_value());
}

// --- writer/reader fixpoint ----------------------------------------------- //
static void TestFixpoint() {
  auto m = BuildRobot();
  std::string s1 = io::WriteMjcf(*m);
  io::ParseResult parsed = io::ParseMjcfString(s1, "<built>");
  if (!parsed.ok()) {
    for (const auto& e : parsed.errors) std::printf("  parse: %s\n",
                                                     e.Render().c_str());
  }
  CHECK(parsed.ok());
  if (parsed.ok()) {
    std::string s2 = io::WriteMjcf(*parsed.model);
    CHECK(s1 == s2);  // deterministic write -> read -> write fixpoint
  }
}

// --- traversal: find, parent map, path, ForEach --------------------------- //
static void TestTraversal() {
  auto m = BuildRobot();

  Joint* hip = sdk::Find<Joint>(*m, "hip");
  CHECK(hip != nullptr);
  CHECK(sdk::Find<Joint>(*m, "nope") == nullptr);

  sdk::ParentMap pm(*m);
  const void* parent = pm.ParentOf(*hip);
  CHECK(parent != nullptr);
  const sdk::ParentMap::Node* pn = pm.Lookup(parent);
  CHECK(pn && pn->type == ElementType::Body && pn->name == "thigh");

  std::string path = pm.PathTo(*hip);
  CHECK(path.find("thigh") != std::string::npos);
  CHECK(path.find("[hip]") != std::string::npos);

  // Recursive geom count under the world body: torso/thigh/shin geoms = 3.
  Body& world = sdk::World(*m);
  int geoms = 0;
  sdk::ForEachGeom(world, /*recursive=*/true, [&](Geom&) { ++geoms; });
  CHECK(geoms == 3);
  // Shallow: the world body directly holds no geoms (they are in torso).
  int shallow = 0;
  sdk::ForEachGeom(world, /*recursive=*/false, [&](Geom&) { ++shallow; });
  CHECK(shallow == 0);
  // torso directly holds exactly one geom, shallow.
  Body* torso = sdk::Find<Body>(*m, "torso");
  int torso_geoms = 0;
  sdk::ForEachGeom(*torso, false, [&](Geom&) { ++torso_geoms; });
  CHECK(torso_geoms == 1);
}

// --- refs: resolve + find referrers --------------------------------------- //
static void TestResolveAndReferrers() {
  auto m = BuildRobot();

  // Resolve the actuator's joint transmission to the actual joint.
  Position* act = sdk::Find<Position>(*m, "act_hip");
  CHECK(act && act->joint.has_value());
  Joint* hip = sdk::ResolveTo(*m, *act->joint);
  CHECK(hip == sdk::Find<Joint>(*m, "hip"));

  // Referrers of "hip": the position actuator and the jointpos sensor.
  Joint* hipJ = sdk::Find<Joint>(*m, "hip");
  auto refs = sdk::FindReferrers(*m, *hipJ);
  CHECK(refs.size() == 2);
  bool from_act = false, from_sensor = false;
  for (const auto& r : refs) {
    CHECK(r.refname == "hip");
    CHECK(r.field == "joint");
    if (r.element.type == ElementType::Position) from_act = true;
    if (r.element.type == ElementType::Jointpos) from_sensor = true;
  }
  CHECK(from_act && from_sensor);

  // A material referrer: the thigh geom.
  Material* steel = sdk::Find<Material>(*m, "steel");
  auto mrefs = sdk::FindReferrers(*m, *steel);
  CHECK(mrefs.size() == 1);
  CHECK(mrefs.front().element.type == ElementType::Geom);
}

// --- rename updates every referrer ---------------------------------------- //
static void TestRename() {
  auto m = BuildRobot();
  Joint* hip = sdk::Find<Joint>(*m, "hip");

  int updated = sdk::Rename(*m, *hip, "hip_renamed");
  CHECK(updated == 2);  // actuator + sensor
  CHECK(hip->name.value() == "hip_renamed");

  // Old name resolves to nothing; new name resolves; referrers followed it.
  CHECK(sdk::Find<Joint>(*m, "hip") == nullptr);
  CHECK(sdk::Find<Joint>(*m, "hip_renamed") == hip);
  Position* act = sdk::Find<Position>(*m, "act_hip");
  CHECK(act->joint->name == "hip_renamed");
  CHECK(sdk::FindReferrers(*m, "hip", ElementType::Joint).empty());
  CHECK(sdk::FindReferrers(*m, "hip_renamed", ElementType::Joint).size() == 2);

  // Renaming to the same name is a no-op.
  CHECK(sdk::Rename(*m, *hip, "hip_renamed") == 0);
}

// --- delete reports (and can cascade) dangling referrers ------------------ //
static void TestDelete() {
  auto m = BuildRobot();

  // Delete the thigh subtree: removes hip + shin + knee and their geoms. The
  // hip/knee actuators and sensors are left dangling.
  Body* thigh = sdk::Find<Body>(*m, "thigh");
  CHECK(thigh != nullptr);
  auto report = sdk::DeleteRecursive(*m, *thigh, /*cascade=*/false);
  CHECK(report.removed);
  CHECK(!report.cascaded);
  // 4 dangling refs: act_hip, act_knee, sens_hip, sens_knee.
  CHECK(report.dangling.size() == 4);
  CHECK(sdk::Find<Body>(*m, "thigh") == nullptr);
  CHECK(sdk::Find<Joint>(*m, "knee") == nullptr);
  // Not cascaded: the actuator still names the now-absent joint.
  Position* act = sdk::Find<Position>(*m, "act_hip");
  CHECK(act && act->joint.has_value());

  // Now delete an already-removed pointer target: a fresh model, cascade mode.
  auto m2 = BuildRobot();
  Body* thigh2 = sdk::Find<Body>(*m2, "thigh");
  auto rep2 = sdk::DeleteRecursive(*m2, *thigh2, /*cascade=*/true);
  CHECK(rep2.removed && rep2.cascaded);
  CHECK(rep2.dangling.size() == 4);
  // Cascade cleared the dangling transmissions.
  Position* act2 = sdk::Find<Position>(*m2, "act_hip");
  CHECK(act2 && !act2->joint.has_value());
  CHECK(sdk::FindReferrers(*m2, "hip", ElementType::Joint).empty());
}

// --- Effective / FlattenDefaults round-trip ------------------------------- //
static void TestEffectiveAndFlatten() {
  auto m = BuildRobot();

  // The thigh geom authors neither size nor rgba; they come from class "leg".
  Geom* tg = sdk::Find<Geom>(*m, "thigh_geom");
  CHECK(tg && !tg->size.has_value() && !tg->rgba.has_value());
  auto eff = sdk::Effective(*m, *tg);
  CHECK(eff->size.has_value() && eff->size->size() == 1 &&
        (*eff->size)[0] == 0.05);
  CHECK(eff->rgba.has_value() && (*eff->rgba)[0] == 1.0f);
  // material stayed authored on the element itself.
  CHECK(eff->material.has_value() && eff->material->name == "steel");

  // The hip joint inherits damping from class "leg" via the thigh childclass.
  Joint* hip = sdk::Find<Joint>(*m, "hip");
  auto heff = sdk::Effective(*m, *hip);
  CHECK(heff->damping.has_value() && (*heff->damping)[0] == 0.5);

  // Snapshot every live geom/joint's effective value, flatten, recompute:
  // effective values must be preserved (semantic equality) even though the
  // authored attributes moved from class to element and classes are gone.
  Body& world = sdk::World(*m);
  std::vector<Geom*> geoms;
  std::vector<Joint*> joints;
  sdk::ForEachGeom(world, true, [&](Geom& g) { geoms.push_back(&g); });
  sdk::ForEachJoint(world, true, [&](Joint& j) { joints.push_back(&j); });

  std::vector<std::unique_ptr<Geom>> before_g;
  for (Geom* g : geoms) before_g.push_back(sdk::Effective(*m, *g));
  std::vector<std::unique_ptr<Joint>> before_j;
  for (Joint* j : joints) before_j.push_back(sdk::Effective(*m, *j));

  sdk::FlattenDefaults(*m);

  // Classes and childclass references are gone; the values are baked in.
  CHECK(m->defaults.empty());
  Body* thigh = sdk::Find<Body>(*m, "thigh");
  CHECK(thigh && !thigh->childclass.has_value());
  CHECK(tg->size.has_value() && (*tg->size)[0] == 0.05);   // baked from class
  CHECK(tg->rgba.has_value() && (*tg->rgba)[0] == 1.0f);
  CHECK(hip->damping.has_value() && (*hip->damping)[0] == 0.5);

  // Effective values are unchanged across the flatten (semantic equality).
  for (std::size_t i = 0; i < geoms.size(); ++i) {
    auto after = sdk::Effective(*m, *geoms[i]);
    CHECK(*after == *before_g[i]);
  }
  for (std::size_t i = 0; i < joints.size(); ++i) {
    auto after = sdk::Effective(*m, *joints[i]);
    CHECK(*after == *before_j[i]);
  }
}

// --- ExtractClass factors shared authored values ------------------------- //
static void TestExtractClass() {
  auto m = std::make_unique<Model>();
  m->model = "extract";
  Body& world = sdk::World(*m);
  Body& b = sdk::AddBody(world, "b");

  // Three geoms sharing rgba (blue) and contype, but with distinct names/sizes.
  std::vector<Geom*> gs;
  for (int i = 0; i < 3; ++i) {
    Geom& g = sdk::AddGeom(b, GeomType::box, "g" + std::to_string(i));
    g.rgba = std::array<float, 4>{{0.0f, 0.0f, 1.0f, 1.0f}};
    g.contype = 2;
    g.size = ps::InlineVec<double, 3>{static_cast<double>(i) + 1.0};  // differs
    gs.push_back(&g);
  }

  Default* cls = sdk::ExtractClass(*m, gs, "blue");
  CHECK(cls != nullptr);
  // Shared fields moved to the class and cleared on each element.
  for (Geom* g : gs) {
    CHECK(!g->rgba.has_value());
    CHECK(!g->contype.has_value());
    CHECK(g->size.has_value());  // distinct value stayed on the element
    CHECK(g->dclass.has_value() && g->dclass->name == "blue");
  }
  CHECK(!cls->geom.empty());
  CHECK(cls->geom.front()->rgba.has_value() &&
        (*cls->geom.front()->rgba)[2] == 1.0f);
  CHECK(cls->geom.front()->contype.has_value() &&
        cls->geom.front()->contype.value() == 2);

  // Effective restores the extracted values through the class.
  auto eff = sdk::Effective(*m, *gs[0]);
  CHECK(eff->rgba.has_value() && (*eff->rgba)[2] == 1.0f);
  CHECK(eff->contype.has_value() && eff->contype.value() == 2);
  CHECK(eff->size.has_value() && (*eff->size)[0] == 1.0);
}

// --- attach: namespaced deep-clone splice --------------------------------- //
static void TestAttach() {
  auto m = BuildRobot();

  // Build a small standalone gripper subtree to graft onto the shin.
  Body gripper;
  gripper.name = "palm";
  sdk::AddJoint(gripper, JointType::slide, "slide");
  Geom& gg = sdk::AddGeom(gripper, GeomType::box, "pad");
  (void)gg;
  Body& finger = sdk::AddBody(gripper, "finger");
  sdk::AddGeom(finger, GeomType::box, "tip");

  Body* shin = sdk::Find<Body>(*m, "shin");
  CHECK(shin != nullptr);
  auto result = sdk::Attach(*m, *shin, gripper, "g1_");
  CHECK(result.ok);
  CHECK(result.collisions.empty());
  CHECK(result.attached != nullptr);

  // Every name in the graft is prefixed; the source is untouched (deep clone).
  CHECK(sdk::Find<Body>(*m, "g1_palm") != nullptr);
  CHECK(sdk::Find<Body>(*m, "g1_finger") != nullptr);
  CHECK(sdk::Find<Geom>(*m, "g1_pad") != nullptr);
  CHECK(sdk::Find<Geom>(*m, "g1_tip") != nullptr);
  CHECK(sdk::Find<Joint>(*m, "g1_slide") != nullptr);
  CHECK(gripper.name.value() == "palm");  // source unchanged

  // Attaching the same subtree again with the same prefix collides.
  auto dup = sdk::Attach(*m, *shin, gripper, "g1_");
  CHECK(!dup.ok);
  CHECK(!dup.collisions.empty());
  // A different prefix succeeds.
  auto ok2 = sdk::Attach(*m, *shin, gripper, "g2_");
  CHECK(ok2.ok);
  CHECK(sdk::Find<Joint>(*m, "g2_slide") != nullptr);

  // An internal reference inside a cloned subtree is namespaced too: give the
  // gripper geom a material ref and attach; the clone's ref is prefixed.
  Body refbody;
  refbody.name = "rb";
  Geom& rg = sdk::AddGeom(refbody, GeomType::sphere, "rgeom");
  rg.material = ps::Ref<Material>("mat");
  auto r3 = sdk::Attach(*m, *shin, refbody, "p_");
  CHECK(r3.ok);
  Geom* cloned = sdk::Find<Geom>(*m, "p_rgeom");
  CHECK(cloned && cloned->material.has_value() &&
        cloned->material->name == "p_mat");
}

int main() {
  TestBuilders();
  TestFixpoint();
  TestTraversal();
  TestResolveAndReferrers();
  TestRename();
  TestDelete();
  TestEffectiveAndFlatten();
  TestExtractClass();
  TestAttach();

  std::printf("%d checks, %d failures\n", g_checks, g_failed);
  return g_failed == 0 ? 0 : 1;
}
