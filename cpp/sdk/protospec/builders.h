// ProtoSpec SDK: typed builders.
//
// Ergonomic Add* helpers that construct an element, link it into the correct
// child list or ordered union subtree (Section 6 interleave), and return a
// reference to the live element for further authoring. Structural parameters
// (a geom's type, a joint's type) default to the same value the IDL records as
// the compiler default, so `AddGeom(body)` yields a sphere. In keeping with
// DR-1 ("defaults are never silently written into models"), builders do NOT
// stamp the full IDL default set onto an element -- only what you pass is
// authored. Call `ps::mjcf::ApplyDefault(elem)` explicitly to seed the rest.
//
// The `name` argument is always optional; unnamed elements are still bindable
// through their creation serial (DR-10), so a name is only for your own refs.
#ifndef PROTOSPEC_SDK_BUILDERS_H
#define PROTOSPEC_SDK_BUILDERS_H

#include <memory>
#include <string>
#include <vector>

#include "protospec/detail.h"
#include "types.h"

namespace ps::sdk {

namespace mj = ps::mjcf;

namespace detail {

// Wrap an owned child in its union node and append it, returning the live child.
template <class Child, class Any>
Child& PushUnion(std::vector<Any>& list, std::unique_ptr<Child> c) {
  Child& ref = *c;
  Any node;
  node.node = std::move(c);
  list.push_back(std::move(node));
  return ref;
}

template <class T>
T& PushOwned(std::vector<std::unique_ptr<T>>& list, std::unique_ptr<T> c) {
  T& ref = *c;
  list.push_back(std::move(c));
  return ref;
}

}  // namespace detail

// --- Body tree ------------------------------------------------------------ //

// The single world body, created (unnamed) if the model has none.
inline mj::Body& World(mj::Model& model) {
  if (model.worldbody.empty() || !model.worldbody.front())
    model.worldbody.insert(model.worldbody.begin(),
                           std::make_unique<mj::Body>());
  return *model.worldbody.front();
}

// `Parent` is any body-context element carrying a `subtree` (Body, Frame,
// Replicate). Each helper appends into that ordered union list.
template <class Parent>
mj::Body& AddBody(Parent& parent, const std::string& name = "") {
  auto b = std::make_unique<mj::Body>();
  if (!name.empty()) b->name = name;
  return detail::PushUnion<mj::Body>(parent.subtree, std::move(b));
}

template <class Parent>
mj::Geom& AddGeom(Parent& parent, mj::GeomType type = mj::GeomType::sphere,
                  const std::string& name = "") {
  auto g = std::make_unique<mj::Geom>();
  g->type = type;
  if (!name.empty()) g->name = name;
  return detail::PushUnion<mj::Geom>(parent.subtree, std::move(g));
}

template <class Parent>
mj::Joint& AddJoint(Parent& parent, mj::JointType type = mj::JointType::hinge,
                    const std::string& name = "") {
  auto j = std::make_unique<mj::Joint>();
  j->type = type;
  if (!name.empty()) j->name = name;
  return detail::PushUnion<mj::Joint>(parent.subtree, std::move(j));
}

template <class Parent>
mj::FreeJoint& AddFreeJoint(Parent& parent, const std::string& name = "") {
  auto j = std::make_unique<mj::FreeJoint>();
  if (!name.empty()) j->name = name;
  return detail::PushUnion<mj::FreeJoint>(parent.subtree, std::move(j));
}

template <class Parent>
mj::Site& AddSite(Parent& parent, const std::string& name = "") {
  auto s = std::make_unique<mj::Site>();
  if (!name.empty()) s->name = name;
  return detail::PushUnion<mj::Site>(parent.subtree, std::move(s));
}

template <class Parent>
mj::Camera& AddCamera(Parent& parent, const std::string& name = "") {
  auto c = std::make_unique<mj::Camera>();
  if (!name.empty()) c->name = name;
  return detail::PushUnion<mj::Camera>(parent.subtree, std::move(c));
}

template <class Parent>
mj::Light& AddLight(Parent& parent, const std::string& name = "") {
  auto l = std::make_unique<mj::Light>();
  if (!name.empty()) l->name = name;
  return detail::PushUnion<mj::Light>(parent.subtree, std::move(l));
}

template <class Parent>
mj::Frame& AddFrame(Parent& parent, const std::string& name = "") {
  auto f = std::make_unique<mj::Frame>();
  if (!name.empty()) f->name = name;
  return detail::PushUnion<mj::Frame>(parent.subtree, std::move(f));
}

// The inertial element is a distinct (non-union) child of a body/frame.
template <class Parent>
mj::Inertial& AddInertial(Parent& parent) {
  return detail::PushOwned(parent.inertial, std::make_unique<mj::Inertial>());
}

// --- Assets --------------------------------------------------------------- //

inline mj::Asset& EnsureAsset(mj::Model& model) {
  if (model.assets.empty() || !model.assets.front())
    model.assets.insert(model.assets.begin(), std::make_unique<mj::Asset>());
  return *model.assets.front();
}

inline mj::Mesh& AddMesh(mj::Model& model, const std::string& name = "",
                         const std::string& file = "") {
  auto m = std::make_unique<mj::Mesh>();
  if (!name.empty()) m->name = name;
  if (!file.empty()) m->file = file;
  return detail::PushOwned(EnsureAsset(model).meshs, std::move(m));
}

inline mj::Material& AddMaterial(mj::Model& model, const std::string& name = "") {
  auto m = std::make_unique<mj::Material>();
  if (!name.empty()) m->name = name;
  return detail::PushOwned(EnsureAsset(model).materials, std::move(m));
}

inline mj::Texture& AddTexture(mj::Model& model, const std::string& name = "") {
  auto t = std::make_unique<mj::Texture>();
  if (!name.empty()) t->name = name;
  return detail::PushOwned(EnsureAsset(model).textures, std::move(t));
}

inline mj::Hfield& AddHfield(mj::Model& model, const std::string& name = "") {
  auto h = std::make_unique<mj::Hfield>();
  if (!name.empty()) h->name = name;
  return detail::PushOwned(EnsureAsset(model).hfields, std::move(h));
}

inline mj::Skin& AddSkin(mj::Model& model, const std::string& name = "") {
  auto s = std::make_unique<mj::Skin>();
  if (!name.empty()) s->name = name;
  return detail::PushOwned(EnsureAsset(model).skins, std::move(s));
}

// --- Actuators ------------------------------------------------------------ //

inline mj::Actuator& EnsureActuatorSection(mj::Model& model) {
  if (model.actuators.empty() || !model.actuators.front())
    model.actuators.insert(model.actuators.begin(),
                           std::make_unique<mj::Actuator>());
  return *model.actuators.front();
}

// Add an actuator of a specific spelling (Position, Motor, Velocity, ...). When
// the spelling has a joint transmission and `joint` is non-empty it is set as
// the target; other transmissions (site/tendon/body) are set on the returned
// reference.
template <class A>
A& AddActuator(mj::Model& model, const std::string& joint = "",
               const std::string& name = "") {
  auto a = std::make_unique<A>();
  if (!name.empty()) detail::SetName(*a, name);
  if constexpr (requires { a->joint; }) {
    if (!joint.empty()) a->joint = ps::Ref<mj::Joint>(joint);
  }
  return detail::PushUnion<A>(EnsureActuatorSection(model).actuators,
                              std::move(a));
}

// --- Sensors -------------------------------------------------------------- //

inline mj::Sensor& EnsureSensorSection(mj::Model& model) {
  if (model.sensors.empty() || !model.sensors.front())
    model.sensors.insert(model.sensors.begin(),
                         std::make_unique<mj::Sensor>());
  return *model.sensors.front();
}

template <class S>
S& AddSensor(mj::Model& model, const std::string& name = "") {
  auto s = std::make_unique<S>();
  if (!name.empty()) detail::SetName(*s, name);
  return detail::PushUnion<S>(EnsureSensorSection(model).sensors, std::move(s));
}

// --- Contact / equality / tendon ------------------------------------------ //

inline mj::Contact& EnsureContact(mj::Model& model) {
  if (model.contacts.empty() || !model.contacts.front())
    model.contacts.insert(model.contacts.begin(),
                          std::make_unique<mj::Contact>());
  return *model.contacts.front();
}

inline mj::Pair& AddPair(mj::Model& model, const std::string& name = "") {
  auto p = std::make_unique<mj::Pair>();
  if (!name.empty()) p->name = name;
  return detail::PushOwned(EnsureContact(model).pairs, std::move(p));
}

inline mj::Exclude& AddExclude(mj::Model& model, const std::string& name = "") {
  auto e = std::make_unique<mj::Exclude>();
  if (!name.empty()) e->name = name;
  return detail::PushOwned(EnsureContact(model).excludes, std::move(e));
}

inline mj::Equality& EnsureEqualitySection(mj::Model& model) {
  if (model.equalitys.empty() || !model.equalitys.front())
    model.equalitys.insert(model.equalitys.begin(),
                           std::make_unique<mj::Equality>());
  return *model.equalitys.front();
}

template <class Q>
Q& AddEquality(mj::Model& model, const std::string& name = "") {
  auto q = std::make_unique<Q>();
  if (!name.empty()) detail::SetName(*q, name);
  return detail::PushUnion<Q>(EnsureEqualitySection(model).equalitys,
                              std::move(q));
}

inline mj::Tendon& EnsureTendonSection(mj::Model& model) {
  if (model.tendons.empty() || !model.tendons.front())
    model.tendons.insert(model.tendons.begin(),
                         std::make_unique<mj::Tendon>());
  return *model.tendons.front();
}

template <class TN>
TN& AddTendon(mj::Model& model, const std::string& name = "") {
  auto t = std::make_unique<TN>();
  if (!name.empty()) detail::SetName(*t, name);
  return detail::PushUnion<TN>(EnsureTendonSection(model).tendons,
                               std::move(t));
}

// --- Keyframes ------------------------------------------------------------ //

inline mj::Keyframe& EnsureKeyframe(mj::Model& model) {
  if (model.keyframes.empty() || !model.keyframes.front())
    model.keyframes.insert(model.keyframes.begin(),
                           std::make_unique<mj::Keyframe>());
  return *model.keyframes.front();
}

inline mj::Key& AddKey(mj::Model& model, const std::string& name = "") {
  auto k = std::make_unique<mj::Key>();
  if (!name.empty()) k->name = name;
  return detail::PushOwned(EnsureKeyframe(model).keys, std::move(k));
}

// --- Defaults ------------------------------------------------------------- //

// The root `main` default, created if absent.
inline mj::Default& RootDefault(mj::Model& model) {
  for (auto& d : model.defaults) {
    if (!d) continue;
    std::string n = d->dclass ? *d->dclass : std::string();
    if (n.empty() || n == "main") return *d;
  }
  auto d = std::make_unique<mj::Default>();
  d->dclass = "main";
  return detail::PushOwned(model.defaults, std::move(d));
}

// A new named class nested under `main`.
inline mj::Default& AddDefault(mj::Model& model, const std::string& name) {
  auto d = std::make_unique<mj::Default>();
  d->dclass = name;
  return detail::PushOwned(RootDefault(model).subclasses, std::move(d));
}

}  // namespace ps::sdk

#endif  // PROTOSPEC_SDK_BUILDERS_H
