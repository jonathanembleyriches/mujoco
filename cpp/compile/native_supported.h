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
  return false;
}

}  // namespace ps::mjcf::compile

#endif  // PROTOSPEC_COMPILE_NATIVE_SUPPORTED_H
