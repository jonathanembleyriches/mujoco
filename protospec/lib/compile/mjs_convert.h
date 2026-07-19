// ProtoSpec -> mjSpec conversion helpers (handwritten; the small surface the
// generated mjs_binding.cc calls that is not purely mechanical). MuJoCo-linked:
// compiled only by the compile target, never the Tier-0 protospec library.
//
// Three kinds of helper live here and nowhere else:
//   * opaque-handle setters (mjString vector / numeric vectors) that convert the
//     ProtoSpec container element type to the mjs handle's element type;
//   * the two variant aliasing groups (GeomShape, TextureSource) whose arm choice
//     needs std::get_if dispatch rather than a field write;
//   * the TextureBuiltin -> mjtBuiltin map (an enum with no generated ToMjt).
// Everything else is emitted by protospec_gen.emit_mjs.
#ifndef PROTOSPEC_COMPILE_MJS_CONVERT_H
#define PROTOSPEC_COMPILE_MJS_CONVERT_H

#include <cstddef>
#include <string>
#include <variant>
#include <vector>

#include <mujoco/mujoco.h>  // mjSpec structs + mjs_set* setter surface

#include "types.h"

namespace ps::mjcf::compile {

// --- Opaque numeric-handle setters (element-type converting) --------------- //
template <class T>
inline void SetDouble(mjDoubleVec* dest, const std::vector<T>& v) {
  std::vector<double> tmp(v.begin(), v.end());
  mjs_setDouble(dest, tmp.data(), static_cast<int>(tmp.size()));
}

template <class T>
inline void SetInt(mjIntVec* dest, const std::vector<T>& v) {
  std::vector<int> tmp(v.begin(), v.end());
  mjs_setInt(dest, tmp.data(), static_cast<int>(tmp.size()));
}

template <class T>
inline void SetFloat(mjFloatVec* dest, const std::vector<T>& v) {
  std::vector<float> tmp(v.begin(), v.end());
  mjs_setFloat(dest, tmp.data(), static_cast<int>(tmp.size()));
}

// --- Reference-name vector -> mjStringVec ---------------------------------- //
template <class T>
inline void SetRefVec(mjStringVec* dest, const std::vector<ps::Ref<T>>& refs) {
  for (const auto& r : refs) mjs_appendString(dest, r.name.c_str());
}

// --- Shape variant (Q-SHAPE): explicit size vs fromto endpoints ------------ //
// The explicit arm fills the type-specific size (partial: unauthored trailing
// entries keep the mjs default); the fromto arm fills the 6-vector endpoints.
template <class MjsGeomLike>
inline void ApplyShape(const GeomShape& shape, MjsGeomLike* out) {
  if (const Explicit* ex = std::get_if<Explicit>(&shape)) {
    for (std::size_t i = 0; i < ex->size.size(); ++i) out->size[i] = ex->size[i];
  } else if (const FromTo* ft = std::get_if<FromTo>(&shape)) {
    for (int i = 0; i < 6; ++i) out->fromto[i] = ft->fromto[i];
  }
}

// --- Texture source variant: builtin generator vs image file --------------- //
inline int TextureBuiltinToMjt(TextureBuiltin v) {
  switch (v) {
    case TextureBuiltin::none: return mjBUILTIN_NONE;
    case TextureBuiltin::gradient: return mjBUILTIN_GRADIENT;
    case TextureBuiltin::checker: return mjBUILTIN_CHECKER;
    case TextureBuiltin::flat: return mjBUILTIN_FLAT;
  }
  return mjBUILTIN_NONE;
}

inline void ApplyTextureSource(const TextureSource& source, mjsTexture* out) {
  if (const TextureBuiltin* b = std::get_if<TextureBuiltin>(&source)) {
    out->builtin = TextureBuiltinToMjt(*b);
  } else if (const TexFile* f = std::get_if<TexFile>(&source)) {
    mjs_setString(out->file, f->file.c_str());
  }
}

}  // namespace ps::mjcf::compile

#endif  // PROTOSPEC_COMPILE_MJS_CONVERT_H
