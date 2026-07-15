// Native-compiler feature gate (CDR-2): the manifest of features the native
// path can compile today. The gate walks a Model, collects every element
// family (and, later, finer per-feature keys) it uses, and checks each against
// this supported set. Any unsupported feature routes the WHOLE model to the XML
// fallback with a FallbackReason -- no half-native models.
//
// Today the supported set is EMPTY: the native path is pure scaffolding, so
// every non-empty model reports at least one unsupported feature and falls
// back. The set grows one milestone at a time (NC1 flips on blocks/defaults/
// body-tree/primitive-geom features); growing it IS moving the ratchet.
//
// Feature keys are strings (e.g. "geom", "joint") so the manifest can become
// finer than element families without an ABI change -- "geom.mesh_ref" can be a
// distinct key from "geom" when the mesh pipeline lands (NC3 before which a
// primitive-geom model compiles natively but a mesh-geom one does not).
#ifndef PROTOSPEC_COMPILE_NATIVE_SUPPORTED_H
#define PROTOSPEC_COMPILE_NATIVE_SUPPORTED_H

#include <string_view>

namespace ps::mjcf::compile {

// True when `feature_key` is compilable on the native path. NC1 grows this set
// wave by wave; a model is routed native only when EVERY feature key it uses is
// admitted here (native.cc gate). Keys are MJCF-tag granular today (finer keys,
// e.g. "geom.mesh", can be added without an ABI change).
inline bool IsFeatureSupported(std::string_view feature_key) {
  // Header/config blocks (option/size/statistic/visual/compiler) and the
  // default-class machinery. These carry no per-element geometry; their authored
  // overrides are applied in the fill/finalize stages.
  if (feature_key == "compiler" || feature_key == "option" ||
      feature_key == "flag" || feature_key == "size" ||
      feature_key == "statistic" || feature_key == "visual" ||
      feature_key == "global" || feature_key == "quality" ||
      feature_key == "headlight" || feature_key == "map" ||
      feature_key == "scale" || feature_key == "rgba" ||
      feature_key == "default") {
    return true;
  }
  // Rigid-body tree families landed so far in NC1. Note: "geom" here admits the
  // family; a geom that references a mesh/hfield/material or is of mesh/sdf type
  // is rejected by the finer scan in CollectUnsupportedFeatures (native.cc).
  if (feature_key == "body" || feature_key == "inertial" ||
      feature_key == "geom" || feature_key == "joint" ||
      feature_key == "freejoint") {
    return true;
  }
  // NC1b: frames-as-containers, sites/cameras/lights, contact pairs/excludes,
  // keyframes. A site referencing a material or a light referencing a texture is
  // rejected by the finer scan in CollectUnsupportedFeatures (native.cc), since
  // those assets are not native yet.
  if (feature_key == "frame" || feature_key == "site" ||
      feature_key == "camera" || feature_key == "light" ||
      feature_key == "contact" || feature_key == "pair" ||
      feature_key == "exclude" || feature_key == "keyframe" ||
      feature_key == "key") {
    return true;
  }
  // NC4 custom fields: the <custom> container and its numeric/text/tuple
  // elements. A tuple referencing an object type outside the resolvable set
  // (body/xbody/geom/site/joint/camera/tendon/actuator) routes to the XML
  // fallback via the finer scan (tuple.objtype).
  if (feature_key == "custom" || feature_key == "numeric" ||
      feature_key == "text" || feature_key == "tuple" ||
      feature_key == "element") {
    return true;
  }
  // NC4 replicate: the <replicate> macro, expanded natively at collect time as a
  // tree-clone (count copies through an accumulating offset/euler pose + zero-
  // padded name suffix). A replicate carrying a childclass routes to the XML
  // fallback via the finer scan (replicate.childclass) until childclass-into-
  // clone propagation lands.
  if (feature_key == "replicate") {
    return true;
  }
  // NC2 equality: connect/weld (body|site) and joint/tendon polycoef equalities.
  // The container tag is "equality"; "connect"/"weld" are the element tags.
  // EqualityJoint shares the "joint" tag (admitted above) and EqualityTendon the
  // "tendon" tag (admitted with the tendon family); flex equalities keep the
  // unsupported "flex" tag and route to fallback.
  if (feature_key == "equality" || feature_key == "connect" ||
      feature_key == "weld") {
    return true;
  }
  // NC5 flex: the <deformable> container and direct <flex> elements (young=0,
  // non-interpolated edge-only path). The "flex" key is shared by the Flex
  // element and the EqualityFlex spelling; EqualityFlex, node/dof interpolation,
  // young>0 elasticity, and elastic2d bending route to the XML fallback via the
  // finer scan (native.cc). flexcomp / flexvert / flexstrain stay unlisted.
  if (feature_key == "deformable" || feature_key == "flex" ||
      feature_key == "edge" || feature_key == "elasticity") {
    return true;
  }
  // NC5 Wave 2/3/4 flexcomp: the procedural grid/box/square family and the
  // direct type (inline point/element geometry, Wave 4) expand natively (non-
  // interpolated, dof=full) into per-vertex slider bodies and a synthesized
  // <flex> (+ optional edge equality), and Wave 3 compiles young>0 linear
  // elasticity (Stencil2D/3D stiffness + elastic2d bending). The "pin" tag is the
  // <pin> child (flexcomp-only). Interpolated dof, mesh/gmsh file types, and
  // vert/strain equalities route to the XML fallback via the finer scan
  // (flexcomp.* keys in native.cc).
  if (feature_key == "flexcomp" || feature_key == "pin") {
    return true;
  }
  // NC3 assets: the <asset> container and materials. A material referencing a
  // texture (legacy attr or <layer>) is rejected by the finer scan
  // (material.texture) until textures land; geoms/sites/tendons referencing a
  // material now resolve its id. mesh/texture/hfield/skin element tags stay
  // unlisted (their families are not native yet).
  if (feature_key == "asset" || feature_key == "material" ||
      feature_key == "layer" || feature_key == "texture" ||
      feature_key == "hfield" || feature_key == "mesh") {
    return true;
  }
  // NC6 skins: <skin> in <asset> or <deformable>, inline geometry + per-bone
  // <bone> child, or a .skn binary file. bindpos/bindquat/vertid/vertweight
  // packing, weight normalization, and body/material id resolution are native.
  if (feature_key == "skin" || feature_key == "bone") {
    return true;
  }
  // NC2 tendons: spatial (site/geom/pulley wraps) and fixed (joint) tendons.
  // A tendon material ref or non-zero armature routes to fallback (finer scan).
  if (feature_key == "tendon" || feature_key == "spatial" ||
      feature_key == "fixed" || feature_key == "pulley") {
    return true;
  }
  // NC2 actuators: general/motor/position/velocity/intvelocity/damper/cylinder/
  // adhesion with joint/jointinparent/tendon/body transmission. muscle (needs
  // mj_setLengthRange), dcmotor, plugin actuators, site/refsite/slidercrank
  // transmission, and delay buffers route to the XML fallback (finer scan or
  // unlisted tag).
  if (feature_key == "actuator" || feature_key == "general" ||
      feature_key == "motor" || feature_key == "position" ||
      feature_key == "velocity" || feature_key == "intvelocity" ||
      feature_key == "damper" || feature_key == "cylinder" ||
      feature_key == "adhesion") {
    return true;
  }
  // NC2 sensors: fixed-dim, single-typed-target (site/joint/tendon/actuator/
  // body) and frame sensors, plus the scalar energy/clock sensors. Gated:
  // plugin/user (dim/needstage from callbacks), tactile/contact/rangefinder
  // (variable dim + intprm), camprojection/insidesite (extra refs), and the
  // geom-distance sensors (distance/normal/fromto) -- their tags stay unlisted.
  if (feature_key == "sensor" || feature_key == "touch" ||
      feature_key == "accelerometer" || feature_key == "velocimeter" ||
      feature_key == "gyro" || feature_key == "force" ||
      feature_key == "torque" || feature_key == "magnetometer" ||
      feature_key == "jointpos" || feature_key == "jointvel" ||
      feature_key == "jointactuatorfrc" || feature_key == "ballquat" ||
      feature_key == "ballangvel" || feature_key == "jointlimitpos" ||
      feature_key == "jointlimitvel" || feature_key == "jointlimitfrc" ||
      feature_key == "tendonpos" || feature_key == "tendonvel" ||
      feature_key == "tendonactuatorfrc" || feature_key == "tendonlimitpos" ||
      feature_key == "tendonlimitvel" || feature_key == "tendonlimitfrc" ||
      feature_key == "actuatorpos" || feature_key == "actuatorvel" ||
      feature_key == "actuatorfrc" || feature_key == "subtreecom" ||
      feature_key == "subtreelinvel" || feature_key == "subtreeangmom" ||
      feature_key == "framepos" || feature_key == "framequat" ||
      feature_key == "framexaxis" || feature_key == "frameyaxis" ||
      feature_key == "framezaxis" || feature_key == "framelinvel" ||
      feature_key == "frameangvel" || feature_key == "framelinacc" ||
      feature_key == "frameangacc" || feature_key == "clock" ||
      feature_key == "e_potential" || feature_key == "e_kinetic") {
    return true;
  }
  return false;
}

}  // namespace ps::mjcf::compile

#endif  // PROTOSPEC_COMPILE_NATIVE_SUPPORTED_H
