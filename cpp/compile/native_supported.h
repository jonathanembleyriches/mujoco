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

// True when `feature_key` is compilable on the native path. Empty set today.
inline bool IsFeatureSupported(std::string_view feature_key) {
  (void)feature_key;
  // No features are natively supported yet. NC1+ adds keys here (or promotes
  // this to a generated table sourced from native_supported.json).
  return false;
}

}  // namespace ps::mjcf::compile

#endif  // PROTOSPEC_COMPILE_NATIVE_SUPPORTED_H
