// ProtoSpec Studio Hierarchy row icons (ps::studio, ours). See hierarchy_icons.h.

#include "editor/hierarchy_icons.h"

#include <cstddef>

#include "reflect.h"
#include "types.h"

namespace ps::studio {

namespace mj = ps::mjcf;
namespace reflect = ps::mjcf::reflect;

namespace {

// True when `t` is a member of the named generated union (ActuatorAny /
// SensorAny / TendonAny / EqualityAny / PathItemAny). This lets the leaf
// spellings (49 sensors, 11 actuators, ...) classify off the schema itself
// rather than a hand-maintained list that would drift as the IDL grows.
bool InUnion(const char* union_name, mj::ElementType t) {
  const reflect::UnionDescriptor& u = reflect::DescribeUnion(union_name);
  for (std::size_t i = 0; i < u.member_count; ++i) {
    if (u.members[i] == t) return true;
  }
  return false;
}

}  // namespace

IconFamily FamilyOf(mj::ElementType t) {
  using ET = mj::ElementType;

  // Union membership first: these cover the many leaf spellings and are pairwise
  // disjoint (path items are a tendon sub-part, so they fold into Tendon).
  if (InUnion("ActuatorAny", t)) return IconFamily::Actuator;
  if (InUnion("SensorAny", t)) return IconFamily::Sensor;
  if (InUnion("EqualityAny", t)) return IconFamily::Equality;
  if (InUnion("TendonAny", t)) return IconFamily::Tendon;
  if (InUnion("PathItemAny", t)) return IconFamily::Tendon;

  switch (t) {
    case ET::Body:
      return IconFamily::Body;

    case ET::Geom:
    case ET::CompositeGeom:
    case ET::Inertial:
      return IconFamily::Geom;

    case ET::Joint:
    case ET::FreeJoint:
    case ET::FixedJoint:
    case ET::CompositeJoint:
      return IconFamily::Joint;

    case ET::Site:
    case ET::CompositeSite:
      return IconFamily::Site;

    case ET::Camera:
      return IconFamily::Camera;

    case ET::Light:
      return IconFamily::Light;

    case ET::Frame:
      return IconFamily::Frame;

    case ET::Actuator:
      return IconFamily::Actuator;

    case ET::Sensor:
      return IconFamily::Sensor;

    case ET::Tendon:
      return IconFamily::Tendon;

    case ET::Equality:
      return IconFamily::Equality;

    case ET::Pair:
    case ET::Exclude:
    case ET::Contact:
      return IconFamily::Contact;

    case ET::Mesh:
    case ET::Hfield:
    case ET::Skin:
    case ET::SkinBone:
    case ET::Texture:
    case ET::Material:
    case ET::MaterialLayer:
    case ET::ModelAsset:
    case ET::Asset:
      return IconFamily::Asset;

    case ET::Default:
    case ET::EqualityDefault:
    case ET::TendonDefault:
      return IconFamily::Default;

    case ET::Keyframe:
    case ET::Key:
      return IconFamily::Keyframe;

    case ET::Numeric:
    case ET::Text:
    case ET::Tuple:
    case ET::TupleElement:
    case ET::Custom:
      return IconFamily::Custom;

    default:
      return IconFamily::Other;
  }
}

const char* IconForFamily(IconFamily family) {
  switch (family) {
    case IconFamily::Body:     return "\xEF\x86\xB2";  // U+F1B2 cube
    case IconFamily::Geom:     return "\xEF\x86\xB3";  // U+F1B3 cubes
    case IconFamily::Joint:    return "\xEF\x82\x85";  // U+F085 cogs
    case IconFamily::Site:     return "\xEF\x81\x9B";  // U+F05B crosshairs
    case IconFamily::Camera:   return "\xEF\x80\xB0";  // U+F030 camera
    case IconFamily::Light:    return "\xEF\x83\xAB";  // U+F0EB lightbulb
    case IconFamily::Frame:    return "\xEF\x81\x87";  // U+F047 arrows
    case IconFamily::Actuator: return "\xEF\x83\xA7";  // U+F0E7 bolt
    case IconFamily::Sensor:   return "\xEF\x81\xAE";  // U+F06E eye
    case IconFamily::Tendon:   return "\xEF\x83\x81";  // U+F0C1 link
    case IconFamily::Equality: return "\xEF\x83\xAC";  // U+F0EC exchange
    case IconFamily::Contact:  return "\xEF\x86\x92";  // U+F192 dot-circle
    case IconFamily::Asset:    return "\xEF\x80\xBE";  // U+F03E image
    case IconFamily::Default:  return "\xEF\x87\x9E";  // U+F1DE sliders
    case IconFamily::Keyframe: return "\xEF\x80\xA4";  // U+F024 flag
    case IconFamily::Custom:   return "\xEF\x80\xAB";  // U+F02B tag
    case IconFamily::Other:    return "\xEF\x84\x91";  // U+F111 circle
  }
  return "\xEF\x84\x91";  // U+F111 circle (unreachable; keeps the return total)
}

const char* IconForElementType(mj::ElementType type) {
  return IconForFamily(FamilyOf(type));
}

}  // namespace ps::studio
