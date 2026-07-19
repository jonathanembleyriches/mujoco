// Lifted from MuJoCo src/user/user_objects.cc (PNGImage::Load, :81-136; the
// nchannel->LodePNGColorType dispatch from mjCTexture::LoadPNG, :5221-5242).
// The algorithm and lodepng invocation are verbatim; the plumbing is retargeted
// off mjCBase/mjResource/mjCError to a plain byte buffer + std::string error.
// Registry: snapshots/lifted_code.json (image_png_decode). Pin mjVERSION 3010000.
//
// SPDX-License-Identifier: Apache-2.0
// Portions Copyright 2021 DeepMind Technologies Limited (MuJoCo, Apache-2.0);
// see NOTICE.
#include "image_pipeline.h"

#include <cstdlib>
#include <memory>
#include <sstream>

#include "lodepng.h"

namespace ps::mjcf::compile::lifted {

bool DecodePNG(const unsigned char* buffer, int nbuffer, int nchannel,
               std::vector<unsigned char>& out, int& width, int& height,
               bool& is_srgb, std::string& err) {
  // nchannel -> lodepng raw color type (mjCTexture::LoadPNG).
  LodePNGColorType color_type;
  if (nchannel == 4) {
    color_type = LCT_RGBA;
  } else if (nchannel == 3) {
    color_type = LCT_RGB;
  } else if (nchannel == 1) {
    color_type = LCT_GREY;
  } else {
    err = "Unsupported number of channels: " + std::to_string(nchannel);
    return false;
  }

  if (nbuffer < 0) {
    err = "could not read PNG file";
    return false;
  }
  if (!nbuffer) {
    err = "empty PNG file";
    return false;
  }

  // decode PNG from buffer (PNGImage::Load).
  unsigned int w, h;
  lodepng::State state;
  state.info_raw.colortype = color_type;
  state.info_raw.bitdepth = 8;
  unsigned char* data_ptr = nullptr;
  unsigned e = lodepng_decode(&data_ptr, &w, &h, &state, buffer,
                              static_cast<std::size_t>(nbuffer));
  struct free_delete {
    void operator()(unsigned char* ptr) const { std::free(ptr); }
  };
  std::unique_ptr<unsigned char, free_delete> data{data_ptr};

  if (e) {
    std::stringstream ss;
    ss << "error decoding PNG file: " << lodepng_error_text(e);
    err = ss.str();
    return false;
  }

  out.clear();
  if (data) {
    std::size_t buffersize = lodepng_get_raw_size(w, h, &state.info_raw);
    out.insert(out.end(), data.get(), data.get() + buffersize);
  }

  width = static_cast<int>(w);
  height = static_cast<int>(h);
  is_srgb = (state.info_png.srgb_defined == 1);

  if (width <= 0 || height < 0) {
    err = "error decoding PNG file: dimensions are invalid";
    return false;
  }
  return true;
}

}  // namespace ps::mjcf::compile::lifted
