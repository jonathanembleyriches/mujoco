// ProtoSpec Studio entry point (ps::studio, ours).
//
// Plain argv parsing (no absl): an optional positional model path plus
// --smoke-frames N (the headless CI hook: render N frames then exit 0). Builds
// the host, registers the ProtoSpec editor plugin cluster, feeds the load slot,
// and runs the main loop.

#include <cstdlib>
#include <cstring>
#include <string>

#include "editor/editor_context.h"
#include "editor/plugins.h"
#include "host/app.h"

namespace {

struct Args {
  std::string model_path;
  int smoke_frames = 0;
};

Args ParseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--smoke-frames") {
      if (i + 1 < argc) {
        args.smoke_frames = std::atoi(argv[++i]);
      }
    } else if (a.rfind("--smoke-frames=", 0) == 0) {
      args.smoke_frames = std::atoi(a.c_str() + std::strlen("--smoke-frames="));
    } else if (!a.empty() && a[0] != '-') {
      if (args.model_path.empty()) {
        args.model_path = a;
      }
    }
    // Unknown flags are ignored.
  }
  return args;
}

}  // namespace

int main(int argc, char** argv) {
  const Args args = ParseArgs(argc, argv);
  const bool smoke = args.smoke_frames > 0;

  ps::studio::App::Config config;
  config.title = "ProtoSpec Studio";
  config.hidden = smoke;

  ps::studio::App app(config);

  // The editor cluster (ProtoSpec authority + panels + pick logger) shares one
  // context that must outlive the loop.
  ps::studio::EditorContext editor_ctx;
  ps::studio::RegisterEditorPlugins(editor_ctx);

  if (!args.model_path.empty()) {
    app.RequestLoad(args.model_path);
  }

  if (smoke) {
    return app.SmokeRun(args.smoke_frames);
  }

  while (app.Update()) {
    app.BuildGui();
    app.Render();
  }
  return 0;
}
