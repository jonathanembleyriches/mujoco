// ProtoSpec Studio Hierarchy row icons (ps::studio, ours). Maps every element
// type to a FontAwesome glyph via a small set of visual families. Windowless and
// total (FamilyOf covers every ElementType; the glyph lookups never return
// empty), so the ctest target can assert the mapping's totality with no window.

#ifndef PS_STUDIO_EDITOR_HIERARCHY_ICONS_H_
#define PS_STUDIO_EDITOR_HIERARCHY_ICONS_H_

#include "types.h"

namespace ps::studio {

// The visual family a tree row belongs to. Every ElementType maps to exactly one
// family; `Other` is the deliberate catch-all for element kinds that never
// surface as their own hierarchy row (options, statistics, visual settings, ...).
enum class IconFamily {
  Body,
  Geom,
  Joint,
  Site,
  Camera,
  Light,
  Frame,
  Actuator,
  Sensor,
  Tendon,
  Equality,
  Contact,
  Asset,
  Default,
  Keyframe,
  Custom,
  Other,
};

// Classifies an element type into its icon family. Total over ElementType.
IconFamily FamilyOf(ps::mjcf::ElementType type);

// The FontAwesome glyph (UTF-8) for a family / element type. Both always return
// a non-empty glyph within the host's merged icon range (U+F000..U+F3FF).
const char* IconForFamily(IconFamily family);
const char* IconForElementType(ps::mjcf::ElementType type);

}  // namespace ps::studio

#endif  // PS_STUDIO_EDITOR_HIERARCHY_ICONS_H_
