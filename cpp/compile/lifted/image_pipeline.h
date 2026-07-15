// Lifted from MuJoCo src/user/user_objects.cc: the PNG image decode path used by
// file textures and file height fields (PNGImage::Load, class-local at :57-136).
// lodepng is the decoder MuJoCo itself links; the wrapper lives in the lifted TU
// (protospec_lifted) so build.cc drives file-texture/hfield loading without
// pulling lodepng into the conformance-mode compiler TU. Provenance + registry:
// snapshots/lifted_code.json (entry image_png_decode). Symbols live in
// ps::mjcf::compile::lifted.
#ifndef PROTOSPEC_COMPILE_LIFTED_IMAGE_PIPELINE_H
#define PROTOSPEC_COMPILE_LIFTED_IMAGE_PIPELINE_H

#include <cstddef>
#include <string>
#include <vector>

namespace ps::mjcf::compile::lifted {

// Decode a PNG buffer to a tightly-packed 8-bit raster with `nchannel` channels
// (1 = grey, 3 = RGB, 4 = RGBA), mirroring PNGImage::Load + mjCTexture::LoadPNG.
// On success fills `out` (size = width*height*nchannel), `width`, `height`, and
// `is_srgb` (the PNG's sRGB chunk presence, which drives colorspace=AUTO), and
// returns true. On any error returns false with a message in `err` matching
// MuJoCo's wording.
bool DecodePNG(const unsigned char* buffer, int nbuffer, int nchannel,
               std::vector<unsigned char>& out, int& width, int& height,
               bool& is_srgb, std::string& err);

}  // namespace ps::mjcf::compile::lifted

#endif  // PROTOSPEC_COMPILE_LIFTED_IMAGE_PIPELINE_H
