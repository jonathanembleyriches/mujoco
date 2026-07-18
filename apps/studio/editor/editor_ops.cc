// ProtoSpec Studio editor operations (ps::studio, ours). See editor_ops.h.

#include "editor/editor_ops.h"
#include "editor/element_access.h"
#include "editor/layers.h"
#include "editor/transform_math.h"

#include <filesystem>
#include <fstream>
#include <functional>
#include <optional>
#include <string>
#include <type_traits>

#include <mujoco/mujoco.h>

#include "binding.h"
#include "compile.h"
#include "mjcf.h"
#include "protospec/detail.h"
#include "protospec/refs.h"
#include "reflect.h"
#include "report.h"
#include "types.h"
#include "validate.h"

namespace ps::studio {

namespace io = ps::mjcf::io;
namespace validate = ps::mjcf::validate;
namespace reflect = ps::mjcf::reflect;
namespace mj = ps::mjcf;
namespace sdk = ps::sdk;
namespace sdk_detail = ps::sdk::detail;

// A type-erased element pointer by creation serial in an arbitrary model (the
// live tree or a probe clone), or nullptr. FindBySerial(ctx, ...) resolves the
// live tree; this resolves any model, as PreviewDeleteReferrers needs on its
// serial-preserving clone.
static const void* FindPtrInModel(const mj::Model& model, std::uint64_t serial) {
  return FindSerial(model, serial);
}

// See editor_ops.h. Only the trailing path segment (the offending element) is
// consulted; unnamed elements ("tag[#idx]") return none.
std::optional<std::uint64_t> SerialForValidatePath(const mj::Model& tree,
                                                   const std::string& path) {
  const std::size_t seg_start = path.find_last_of('/');
  const std::string seg =
      seg_start == std::string::npos ? path : path.substr(seg_start + 1);
  const std::size_t lb = seg.find('[');
  const std::size_t rb = seg.find(']');
  if (lb == std::string::npos || rb == std::string::npos || rb <= lb + 1) {
    return std::nullopt;
  }
  const std::string tag = seg.substr(0, lb);
  const std::string name = seg.substr(lb + 1, rb - lb - 1);
  if (name.empty() || name[0] == '#') {
    return std::nullopt;  // unnamed element -> path uses an index, not a name
  }

  std::optional<std::uint64_t> found;
  sdk_detail::WalkModelAll(tree, [&](const auto& e) {
    using E = std::decay_t<decltype(e)>;
    if (found) {
      return;
    }
    if constexpr (requires { e.serial; }) {
      const std::string* nm = sdk_detail::NameOf(e);
      if (nm && *nm == name &&
          reflect::Describe(mj::element_type_of<E>::value).xml == tag) {
        found = e.serial;
      }
    }
  });
  return found;
}

// Fold a bridge compile Diagnostic into a Diagnostics-panel entry: severity
// mapped, the pass tag kept inline, SourceLoc carried through when known.
static DiagEntry FromCompileDiagnostic(const mj::Diagnostic& d) {
  DiagEntry e;
  e.severity = d.severity == mj::Diagnostic::Severity::Error
                   ? DiagEntry::Severity::Error
                   : DiagEntry::Severity::Warning;
  e.message = d.pass.empty() ? d.message : ("[" + d.pass + "] " + d.message);
  if (!d.loc.file.empty()) {
    e.loc = d.loc;
  }
  return e;
}

static const char* CompilePathName(mj::CompilePath p) {
  switch (p) {
    case mj::CompilePath::XmlPath:
      return "xml";
    case mj::CompilePath::NativePath:
      return "native";
    case mj::CompilePath::Auto:
      return "auto";
  }
  return "?";
}

// Compile ctx.tree (Auto path) with the model's base_dir and adopt the artifact
// on success. Shared by LoadModel and RecompileTree. Logs warnings/errors and
// leaves the prior good artifact untouched on failure.
static bool CompileCurrent(EditorContext& ctx, const char* what) {
  mj::CompileOptions opts;
  opts.path = mj::CompilePath::Auto;
  if (!ctx.base_dir.empty()) {
    opts.base_dir = ctx.base_dir;
  }
  // In-memory assets (imported meshes not yet saved to disk) resolve through the
  // compile VFS on every recompile until Save externalizes them.
  opts.vfs_assets = ctx.vfs_assets;

  // The compile input: ctx.tree itself when every layer is enabled, else a
  // serial-preserving clone with the disabled layers' elements pruned. The
  // pruned clone is adopted into ctx.compile_tree only on SUCCESS, because the
  // last good artifact's Binding keys on element pointers into whatever model
  // it was compiled from -- that model must stay alive until replaced.
  std::unique_ptr<mj::Model> pruned;
  mj::Model* to_compile = BuildCompileModel(ctx, &pruned);
  if (!to_compile) {
    ctx.status_line = std::string(what) + " skipped (no model)";
    return false;
  }
  // Pruning a disabled layer can orphan references into it (a lock only sees
  // declared refs; force-disabling or editing around it can still dangle).
  // The native compiler tolerates some of those silently, so surface them
  // honestly here: referential validation on the ACTUAL compile input.
  if (pruned) {
    for (const validate::Diagnostic& d :
         validate::Validate(*to_compile, validate::kTierReferential)) {
      DiagEntry e;
      e.severity = d.severity == validate::Severity::Error
                       ? DiagEntry::Severity::Error
                       : DiagEntry::Severity::Warning;
      e.message = "[layers] pruned-input validation: " + d.message +
                  (d.path.empty() ? std::string() : ("  (" + d.path + ")"));
      if (!d.loc.file.empty()) e.loc = d.loc;
      ctx.Diagnose(std::move(e));
    }
  }
  mj::Compiled compiled = mj::Compile(*to_compile, opts);
  for (const mj::Diagnostic& w : compiled.report.warnings) {
    ctx.Diagnose(FromCompileDiagnostic(w));
  }
  if (!compiled.ok()) {
    ctx.Log(std::string(what) + " FAILED:");
    for (const mj::Diagnostic& e : compiled.report.errors) {
      ctx.Diagnose(FromCompileDiagnostic(e));
    }
    ctx.status_line = std::string(what) + " failed (last good model kept)";
    return false;
  }

  ctx.compiled = std::move(compiled);
  ctx.compile_tree = std::move(pruned);  // null when ctx.tree compiled directly
  ctx.model_ready = true;

  const mjModel* m = ctx.compiled.model.get();
  ctx.status_line = std::string(what) + " ok  path=" +
                    CompilePathName(ctx.compiled.report.taken) +
                    "  nq=" + std::to_string(m->nq) +
                    "  nbody=" + std::to_string(m->nbody) +
                    (ctx.dirty ? "  *" : "");
  ctx.Log("  " + ctx.status_line);
  return true;
}

bool LoadModel(EditorContext& ctx, const std::string& path) {
  ctx.Log("load: " + path);

  io::ParseResult parsed = io::ParseMjcfFile(path);
  for (const io::Diagnostic& w : parsed.warnings) {
    DiagEntry e{DiagEntry::Severity::Warning, "[parse] " + w.message, {}, {}};
    if (!w.loc.file.empty()) e.loc = w.loc;
    ctx.Diagnose(std::move(e));
  }
  if (!parsed.ok()) {
    ctx.Log("parse FAILED:");
    for (const io::Diagnostic& e : parsed.errors) {
      DiagEntry d{DiagEntry::Severity::Error, "[parse] " + e.message, {}, {}};
      if (!e.loc.file.empty()) d.loc = e.loc;
      ctx.Diagnose(std::move(d));
    }
    ctx.status_line = "parse failed";
    return false;
  }

  const std::vector<validate::Diagnostic> diags = validate::Validate(
      *parsed.model, validate::kTierStructural | validate::kTierReferential);
  for (const validate::Diagnostic& d : diags) {
    DiagEntry e;
    e.severity = d.severity == validate::Severity::Error
                     ? DiagEntry::Severity::Error
                     : DiagEntry::Severity::Warning;
    e.message = "[validate] " + d.message +
                (d.path.empty() ? std::string() : ("  (" + d.path + ")"));
    if (!d.loc.file.empty()) e.loc = d.loc;
    // Route the row back to the offending element where the path names one.
    e.serial = SerialForValidatePath(*parsed.model, d.path);
    ctx.Diagnose(std::move(e));
  }

  const std::filesystem::path fpath(path);
  const std::filesystem::path parent = fpath.parent_path();

  // Adopt the tree first so CompileCurrent can use the recorded base_dir.
  std::unique_ptr<mj::Model> prev_tree = std::move(ctx.tree);
  mj::Compiled prev_compiled = std::move(ctx.compiled);
  std::unique_ptr<mj::Model> prev_compile_tree = std::move(ctx.compile_tree);
  std::vector<Layer> prev_layers = std::move(ctx.layers);
  LayerGraph prev_graph = std::move(ctx.layer_graph);
  const int prev_active = ctx.active_layer;
  ctx.tree = std::move(parsed.model);
  ctx.base_dir = parent.empty() ? std::string() : parent.string();
  ctx.dirty = false;

  // A load replaces the whole stack: partition the fresh tree by per-element
  // provenance (loc.file), one layer per distinct file, all enabled. Must
  // precede the compile, whose pruning consults `layers`.
  SplitLayersFromTree(ctx, fpath.string(), fpath.stem().string());

  if (!CompileCurrent(ctx, "compiled")) {
    // Roll back to the previous good state (DR-S1: a failed load changes nothing).
    ctx.tree = std::move(prev_tree);
    ctx.compiled = std::move(prev_compiled);
    ctx.compile_tree = std::move(prev_compile_tree);
    ctx.layers = std::move(prev_layers);
    ctx.layer_graph = std::move(prev_graph);
    ctx.active_layer = prev_active;
    return false;
  }

  ctx.source_path = fpath.string();
  ctx.history.Clear();
  ctx.recompile_requested = false;
  ctx.fresh_load = true;  // host resets the free camera on this adopt
  ctx.selected_serial = 0;
  ctx.selected_desc.clear();
  return true;
}

bool RecompileTree(EditorContext& ctx) {
  if (!ctx.tree) {
    return false;
  }
  return CompileCurrent(ctx, "recompiled");
}

bool ShouldServiceRecompile(const EditorContext& ctx) {
  return (ctx.recompile_requested || ctx.apply_edits) &&
         (ctx.apply_edits || ctx.mode == EditorMode::Edit);
}

void ServiceMeshScaleFixup(EditorContext& ctx) {
  if (ctx.mesh_fix_serial == 0 || !ctx.tree || !ctx.compiled.ok()) return;
  SpatialRef ref = FindSpatial(*ctx.tree, ctx.mesh_fix_serial);
  if (!ref || ref.type != mj::ElementType::Geom) {
    ctx.mesh_fix_serial = 0;
    return;
  }
  auto patch = ctx.compiled.binding.PosePatchFor(ref.ptr);
  if (!patch) {  // e.g. a pruned-layer clone owns the binding pointers
    ctx.mesh_fix_serial = 0;
    return;
  }
  double rbc[3], np[3];
  const double bcur[3] = {patch->suffix.pos[0], patch->suffix.pos[1],
                          patch->suffix.pos[2]};
  QuatRotate(ctx.mesh_fix_lquat, bcur, rbc);
  for (int i = 0; i < 3; ++i) np[i] = ctx.mesh_fix_centre[i] - rbc[i];

  mj::Geom* g = static_cast<mj::Geom*>(ref.ptr);
  const std::array<double, 3> cur =
      g->pos ? *g->pos : std::array<double, 3>{0, 0, 0};
  const double delta = std::abs(np[0] - cur[0]) + std::abs(np[1] - cur[1]) +
                       std::abs(np[2] - cur[2]);
  if (delta > 1e-12) {
    g->pos = std::array<double, 3>{np[0], np[1], np[2]};
    ctx.RequestRecompile();
  } else {
    ctx.mesh_fix_serial = 0;  // converged
  }
}

bool SaveModel(EditorContext& ctx, const std::string& path) {
  if (!ctx.tree) {
    return false;
  }
  // A multi-layer model saves as its stack -- one file per layer, written back
  // to each layer's own location -- never as a flattened single document (that
  // silently destroyed the include structure the stack was loaded from).
  if (ctx.layers.size() > 1) {
    std::string err;
    if (!SaveLayeredMjcf(ctx, path, &err)) {
      ctx.Log("save FAILED: " + err);
      ctx.status_line = "save failed";
      return false;
    }
    ctx.source_path = path;
    ctx.dirty = false;
    return true;
  }
  const std::string mjcf = io::WriteMjcf(*ctx.tree);
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    ctx.Log("save FAILED: cannot open " + path);
    ctx.status_line = "save failed";
    return false;
  }
  out.write(mjcf.data(), static_cast<std::streamsize>(mjcf.size()));
  out.close();
  if (!out) {
    ctx.Log("save FAILED: write error " + path);
    ctx.status_line = "save failed";
    return false;
  }
  ctx.source_path = path;
  ctx.dirty = false;
  ctx.Log("saved: " + path);
  ctx.status_line = "saved " + path;
  return true;
}

PickResolution ResolvePick(EditorContext& ctx, int geom_id, int body_id) {
  PickResolution r;
  const mj::Binding& b = ctx.compiled.binding;

  const mj::Geom* g = geom_id >= 0 ? b.GeomAt(geom_id) : nullptr;
  if (g) {
    r.hit = true;
    r.type = std::string(reflect::Describe(mj::ElementType::Geom).name);
    r.name = g->name ? *g->name : "<unnamed>";
    r.serial = g->serial;
  } else {
    const mj::Body* bd = body_id >= 0 ? b.BodyAt(body_id) : nullptr;
    if (bd) {
      r.hit = true;
      r.type = std::string(reflect::Describe(mj::ElementType::Body).name);
      r.name = bd->name ? *bd->name : "<unnamed>";
      r.serial = bd->serial;
    }
  }

  if (r.hit) {
    ctx.selected_serial = r.serial;
    ctx.selected_desc = r.type + " '" + r.name + "' (serial " +
                        std::to_string(r.serial) + ")";
    ctx.Diagnose(DiagEntry{DiagEntry::Severity::Info, "pick -> " + ctx.selected_desc,
                           r.serial, {}});
  } else {
    ctx.Log("pick -> no bindable element at that id");
  }
  return r;
}

ElementRef FindBySerial(EditorContext& ctx, std::uint64_t serial) {
  ElementRef out;
  if (!ctx.tree) return out;
  auto [ptr, type] = FindSerialWithType(*ctx.tree, serial);
  if (ptr) {
    out.ptr = ptr;
    out.type = type;
    out.serial = serial;
  }
  return out;
}

bool SelectBySerial(EditorContext& ctx, std::uint64_t serial) {
  bool found = false;
  std::string desc;
  if (ctx.tree && serial != 0) {
    sdk_detail::WalkModelAll(*ctx.tree, [&](const auto& e) {
      using E = std::decay_t<decltype(e)>;
      if (found) {
        return;
      }
      if constexpr (requires { e.serial; }) {
        if (e.serial == serial) {
          found = true;
          const std::string type =
              std::string(reflect::Describe(mj::element_type_of<E>::value).name);
          const std::string* nm = sdk_detail::NameOf(e);
          desc = type + " '" + (nm ? *nm : std::string("<unnamed>")) +
                 "' (serial " + std::to_string(serial) + ")";
        }
      }
    });
  }
  if (!found) {
    ctx.selected_serial = 0;
    ctx.selected_desc.clear();
    return false;
  }
  ctx.selected_serial = serial;
  ctx.selected_desc = std::move(desc);
  return true;
}

int RenameBySerial(EditorContext& ctx, std::uint64_t serial,
                   const std::string& new_name) {
  if (!ctx.tree) {
    return -1;
  }
  ElementRef ref = FindBySerial(ctx, serial);
  if (!ref) {
    return -1;
  }
  // The public runtime-pointer SDK verb: rename + rewrite every typed referrer.
  return sdk::Rename(*ctx.tree, ref.ptr, new_name);
}

namespace {

// Render sdk::DeleteSubtree's dangling referrers as the confirm modal's
// "path.field -> 'name'" lines (the shape the Hierarchy panel and tests expect).
std::vector<std::string> RenderDangling(
    const std::vector<sdk::Referrer>& dangling) {
  std::vector<std::string> out;
  out.reserve(dangling.size());
  for (const sdk::Referrer& r : dangling) {
    out.push_back(r.path + "." + r.field + " -> '" + r.refname + "'");
  }
  return out;
}

}  // namespace

DeleteResult DeleteBySerial(EditorContext& ctx, std::uint64_t serial,
                            bool cascade) {
  DeleteResult out;
  if (!ctx.tree) {
    return out;
  }
  ElementRef ref = FindBySerial(ctx, serial);
  out.found = static_cast<bool>(ref);
  if (!ref) {
    return out;
  }
  // The public runtime-pointer SDK verb: remove the subtree, report/clear refs.
  sdk::DeleteReport rep = sdk::DeleteSubtree(*ctx.tree, ref.ptr, cascade);
  out.removed = rep.removed;
  out.dangling = RenderDangling(rep.dangling);
  return out;
}

std::vector<std::string> PreviewDeleteReferrers(EditorContext& ctx,
                                                std::uint64_t serial) {
  if (!ctx.tree) {
    return {};
  }
  // Serial-preserving clone so `serial` resolves identically; the real tree is
  // never touched (no recompile, no undo entry).
  std::unique_ptr<mj::Model> probe = CloneWithSerials(*ctx.tree);
  const void* target = FindPtrInModel(*probe, serial);
  if (!target) {
    return {};
  }
  return RenderDangling(
      sdk::DeleteSubtree(*probe, target, /*cascade=*/false).dangling);
}

bool Undo(EditorContext& ctx) {
  if (!ctx.history.can_undo() || !ctx.tree) {
    return false;
  }
  std::string label;
  ctx.tree = ctx.history.Undo(std::move(ctx.tree), &label);
  ctx.dirty = true;
  ctx.RequestRecompile();
  SelectBySerial(ctx, ctx.selected_serial);
  ctx.Diagnose(DiagEntry{DiagEntry::Severity::Info,
                         "undo: " + (label.empty() ? std::string("edit") : label),
                         ctx.selected_serial ? std::optional(ctx.selected_serial)
                                             : std::nullopt,
                         {}});
  return true;
}

bool Redo(EditorContext& ctx) {
  if (!ctx.history.can_redo() || !ctx.tree) {
    return false;
  }
  std::string label;
  ctx.tree = ctx.history.Redo(std::move(ctx.tree), &label);
  ctx.dirty = true;
  ctx.RequestRecompile();
  SelectBySerial(ctx, ctx.selected_serial);
  ctx.Diagnose(DiagEntry{DiagEntry::Severity::Info,
                         "redo: " + (label.empty() ? std::string("edit") : label),
                         ctx.selected_serial ? std::optional(ctx.selected_serial)
                                             : std::nullopt,
                         {}});
  return true;
}

}  // namespace ps::studio
