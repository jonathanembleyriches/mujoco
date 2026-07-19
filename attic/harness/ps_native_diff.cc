// ps_native_diff: the three-way differential driver (impl-plan T0.2 / design
// Section 4.1). Given one corpus MJCF file it produces, in-process:
//
//   leg B = bridge Compile(model, XmlPath)     -- ProtoSpec -> XML -> mj_loadXML
//   leg C = bridge Compile(model, NativePath)  -- the native compiler
//
// and emits a JSON verdict the ratchet test keys off. Leg B is the permanent
// oracle (test_differential.py independently keeps A==B green, so B stands in
// for the original). When the native path claims a model, leg C is diffed
// against leg B with model_diff_lib and must be bit-identical.
//
// Today the native path claims nothing (native_supported == false everywhere),
// so every file reports the XML fallback with its fallback_reasons and the
// ratchet expects zero native files.
//
// Contract (the Python harness invokes this exactly so):
//   ps_native_diff <in.xml> [--base-dir DIR]
//   exit 0  -> pipeline ran; JSON verdict on stdout
//   exit 3  -> file uses elements outside the supported IO families (skip);
//              unsupported tags on stderr (same signal as ps_compile)
//   exit 2  -> native CLAIMED the model but leg C diverged from leg B, OR leg B
//              failed to compile (a real problem); JSON on stdout, detail stderr
//   exit 1  -> any other error (malformed input, missing file, bad usage)
//
// JSON verdict shape (append-only):
//   { "file", "native_supported", "path_taken", "identical",
//     "first_divergence", "fallback_reasons":[...], "xml_ok" }

#include <cstdio>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>

#include "compile.h"
#include "mjcf.h"
#include "model_diff_lib.h"
#include "plugin_registry.h"

namespace {

std::string JsonEscape(const std::string& s) {
  std::string o;
  for (char c : s) {
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\t': o += "\\t"; break;
      default: o += c;
    }
  }
  return o;
}

std::string PathName(ps::mjcf::CompilePath p) {
  using P = ps::mjcf::CompilePath;
  switch (p) {
    case P::Auto: return "auto";
    case P::XmlPath: return "xml";
    case P::NativePath: return "native";
  }
  return "?";
}

}  // namespace

int main(int argc, char** argv) {
  const char* in = nullptr;
  std::string base_dir;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--base-dir" && i + 1 < argc) {
      base_dir = argv[++i];
    } else if (!in) {
      in = argv[i];
    } else {
      std::fprintf(stderr, "unexpected argument: %s\n", a.c_str());
      return 1;
    }
  }
  if (!in) {
    std::fprintf(stderr, "usage: ps_native_diff <in.xml> [--base-dir DIR]\n");
    return 1;
  }

  // Register the first-party engine plugins so both compile legs resolve
  // plugin-bearing models (elasticity/sdf/sensor/pid).
  ps::plugin::RegisterFirstPartyPlugins();

  ps::mjcf::io::ParseResult parsed = ps::mjcf::io::ParseMjcfFile(in);
  if (!parsed.ok()) {
    if (parsed.unsupported_only()) {
      for (const auto& d : parsed.errors)
        std::fprintf(stderr, "%s\n", d.Render().c_str());
      return 3;
    }
    for (const auto& d : parsed.errors)
      std::fprintf(stderr, "%s\n", d.Render().c_str());
    return 1;
  }


  ps::mjcf::CompileOptions opt_xml;
  opt_xml.path = ps::mjcf::CompilePath::XmlPath;
  opt_xml.base_dir = base_dir;
  ps::mjcf::Compiled leg_b = ps::mjcf::Compile(*parsed.model, opt_xml);

  ps::mjcf::CompileOptions opt_native;
  opt_native.path = ps::mjcf::CompilePath::NativePath;
  opt_native.base_dir = base_dir;
  ps::mjcf::Compiled leg_c = ps::mjcf::Compile(*parsed.model, opt_native);

  const bool xml_ok = leg_b.model != nullptr && leg_b.report.ok();
  const bool native_supported = leg_c.model != nullptr && leg_c.report.ok();

  bool identical = false;
  std::string first_divergence;
  std::string diff_err;
  if (native_supported && xml_ok) {
    ps::harness::Tol tol;
    ps::harness::DiffReport report = ps::harness::DiffModels(
        leg_b.model.get(), leg_c.model.get(), tol, 8, diff_err);
    identical = diff_err.empty() && !report.Differs();
    first_divergence = report.FirstDivergence();
  }

  // Emit the verdict.
  std::printf("{\n");
  std::printf("  \"file\": \"%s\",\n", JsonEscape(in).c_str());
  std::printf("  \"native_supported\": %s,\n",
              native_supported ? "true" : "false");
  std::printf("  \"path_taken\": \"%s\",\n",
              PathName(leg_c.report.taken).c_str());
  std::printf("  \"xml_ok\": %s,\n", xml_ok ? "true" : "false");
  std::printf("  \"identical\": %s,\n", identical ? "true" : "false");
  std::printf("  \"first_divergence\": \"%s\",\n",
              JsonEscape(first_divergence).c_str());
  std::printf("  \"fallback_reasons\": [");
  for (size_t i = 0; i < leg_c.report.fallback_reasons.size(); ++i)
    std::printf("%s\"%s\"", i ? ", " : "",
                JsonEscape(leg_c.report.fallback_reasons[i].feature).c_str());
  std::printf("]\n}\n");

  // Exit signalling. A native-claimed model that diverges, or a failed XML leg,
  // is a hard problem; everything else (including the expected all-fallback
  // case today) is a clean run.
  if (!xml_ok) {
    std::fprintf(stderr, "leg B (XML path) failed to compile\n");
    for (const auto& d : leg_b.report.errors)
      std::fprintf(stderr, "%s\n", d.Render().c_str());
    return 2;
  }
  if (native_supported && !identical) {
    std::fprintf(stderr, "native leg diverged from XML leg: %s (%s)\n",
                 first_divergence.c_str(), diff_err.c_str());
    return 2;
  }
  return 0;
}
