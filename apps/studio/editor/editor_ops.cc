// ProtoSpec Studio editor operations (ps::studio, ours). See editor_ops.h.

#include "editor/editor_ops.h"

#include <filesystem>
#include <string>

#include <mujoco/mujoco.h>

#include "binding.h"
#include "compile.h"
#include "mjcf.h"
#include "reflect.h"
#include "report.h"
#include "types.h"
#include "validate.h"

namespace ps::studio {

namespace io = ps::mjcf::io;
namespace validate = ps::mjcf::validate;
namespace bridge = ps::mjcf::bridge;
namespace reflect = ps::mjcf::reflect;

static const char* CompilePathName(bridge::CompilePath p) {
  switch (p) {
    case bridge::CompilePath::XmlPath:
      return "xml";
    case bridge::CompilePath::NativePath:
      return "native";
    case bridge::CompilePath::Auto:
      return "auto";
  }
  return "?";
}

bool LoadModel(EditorContext& ctx, const std::string& path) {
  ctx.Log("load: " + path);

  io::ParseResult parsed = io::ParseMjcfFile(path);
  for (const io::Diagnostic& w : parsed.warnings) {
    ctx.Log("  [parse warn] " + w.Render());
  }
  if (!parsed.ok()) {
    ctx.Log("  parse FAILED:");
    for (const io::Diagnostic& e : parsed.errors) {
      ctx.Log("    " + e.Render());
    }
    ctx.status_line = "parse failed";
    return false;
  }

  // Validate tiers 1-2 (mandatory before compile); log every finding.
  const std::vector<validate::Diagnostic> diags = validate::Validate(
      *parsed.model, validate::kTierStructural | validate::kTierReferential);
  int validate_errors = 0;
  for (const validate::Diagnostic& d : diags) {
    ctx.Log("  [validate] " + d.Render());
    if (d.severity == validate::Severity::Error) {
      ++validate_errors;
    }
  }

  // Compile through the bridge (Auto path).
  bridge::CompileOptions opts;
  opts.path = bridge::CompilePath::Auto;
  std::error_code ec;
  const std::filesystem::path fpath(path);
  const std::filesystem::path parent = fpath.parent_path();
  if (!parent.empty()) {
    opts.base_dir = parent.string();
  }

  bridge::Compiled compiled = bridge::Compile(*parsed.model, opts);
  for (const bridge::Diagnostic& w : compiled.report.warnings) {
    ctx.Log("  [compile warn] " + w.Render());
  }
  if (!compiled.ok()) {
    ctx.Log("  compile FAILED:");
    for (const bridge::Diagnostic& e : compiled.report.errors) {
      ctx.Log("    " + e.Render());
    }
    ctx.status_line = "compile failed";
    return false;  // last good ctx.compiled left untouched (DR-S1)
  }

  // Adopt the new tree + artifact.
  ctx.tree = std::move(parsed.model);
  ctx.compiled = std::move(compiled);
  ctx.model_ready = true;
  ctx.source_name = fpath.stem().string();
  (void)ec;

  const mjModel* m = ctx.compiled.model.get();
  ctx.status_line =
      "compiled ok  path=" + std::string(CompilePathName(ctx.compiled.report.taken)) +
      "  nq=" + std::to_string(m->nq) + "  nbody=" + std::to_string(m->nbody) +
      (validate_errors ? "  (validate errors: " + std::to_string(validate_errors) + ")"
                       : "");
  ctx.Log("  " + ctx.status_line);
  return true;
}

PickResolution ResolvePick(EditorContext& ctx, int geom_id, int body_id) {
  PickResolution r;
  const bridge::Binding& b = ctx.compiled.binding;

  const ps::mjcf::Geom* g = geom_id >= 0 ? b.GeomAt(geom_id) : nullptr;
  const ps::mjcf::Body* bd = nullptr;
  if (g) {
    r.hit = true;
    r.type = std::string(reflect::Describe(ps::mjcf::ElementType::Geom).name);
    r.name = g->name ? *g->name : "<unnamed>";
    r.serial = g->serial;
  } else {
    bd = body_id >= 0 ? b.BodyAt(body_id) : nullptr;
    if (bd) {
      r.hit = true;
      r.type = std::string(reflect::Describe(ps::mjcf::ElementType::Body).name);
      r.name = bd->name ? *bd->name : "<unnamed>";
      r.serial = bd->serial;
    }
  }

  if (r.hit) {
    ctx.selected_serial = r.serial;
    ctx.selected_desc = r.type + " '" + r.name + "' (serial " +
                        std::to_string(r.serial) + ")";
    ctx.Log("pick -> " + ctx.selected_desc);
  } else {
    ctx.Log("pick -> no bindable element at that id");
  }
  return r;
}

}  // namespace ps::studio
