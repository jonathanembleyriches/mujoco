// ProtoSpec Studio editor operations (ps::studio, ours). See editor_ops.h.

#include "editor/editor_ops.h"

#include <filesystem>
#include <fstream>
#include <functional>
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
namespace bridge = ps::mjcf::bridge;
namespace reflect = ps::mjcf::reflect;
namespace mj = ps::mjcf;
namespace sdk_detail = ps::sdk::detail;

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

// Compile ctx.tree (Auto path) with the model's base_dir and adopt the artifact
// on success. Shared by LoadModel and RecompileTree. Logs warnings/errors and
// leaves the prior good artifact untouched on failure.
static bool CompileCurrent(EditorContext& ctx, const char* what) {
  bridge::CompileOptions opts;
  opts.path = bridge::CompilePath::Auto;
  if (!ctx.base_dir.empty()) {
    opts.base_dir = ctx.base_dir;
  }

  bridge::Compiled compiled = bridge::Compile(*ctx.tree, opts);
  for (const bridge::Diagnostic& w : compiled.report.warnings) {
    ctx.Log("  [compile warn] " + w.Render());
  }
  if (!compiled.ok()) {
    ctx.Log(std::string("  ") + what + " FAILED:");
    for (const bridge::Diagnostic& e : compiled.report.errors) {
      ctx.Log("    " + e.Render());
    }
    ctx.status_line = std::string(what) + " failed (last good model kept)";
    return false;
  }

  ctx.compiled = std::move(compiled);
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

  const std::vector<validate::Diagnostic> diags = validate::Validate(
      *parsed.model, validate::kTierStructural | validate::kTierReferential);
  for (const validate::Diagnostic& d : diags) {
    ctx.Log("  [validate] " + d.Render());
  }

  const std::filesystem::path fpath(path);
  const std::filesystem::path parent = fpath.parent_path();

  // Adopt the tree first so CompileCurrent can use the recorded base_dir.
  std::unique_ptr<mj::Model> prev_tree = std::move(ctx.tree);
  bridge::Compiled prev_compiled = std::move(ctx.compiled);
  ctx.tree = std::move(parsed.model);
  ctx.base_dir = parent.empty() ? std::string() : parent.string();
  ctx.dirty = false;

  if (!CompileCurrent(ctx, "compiled")) {
    // Roll back to the previous good state (DR-S1: a failed load changes nothing).
    ctx.tree = std::move(prev_tree);
    ctx.compiled = std::move(prev_compiled);
    return false;
  }

  ctx.source_name = fpath.stem().string();
  ctx.source_path = fpath.string();
  ctx.history.Clear();
  ctx.recompile_requested = false;
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

bool SaveModel(EditorContext& ctx, const std::string& path) {
  if (!ctx.tree) {
    return false;
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
  const bridge::Binding& b = ctx.compiled.binding;

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
    ctx.Log("pick -> " + ctx.selected_desc);
  } else {
    ctx.Log("pick -> no bindable element at that id");
  }
  return r;
}

ElementRef FindBySerial(EditorContext& ctx, std::uint64_t serial) {
  ElementRef out;
  if (!ctx.tree || serial == 0) {
    return out;
  }
  sdk_detail::WalkModelAll(*ctx.tree, [&](auto& e) {
    using E = std::decay_t<decltype(e)>;
    if (out.ptr) {
      return;
    }
    if constexpr (requires { e.serial; }) {
      if (e.serial == serial) {
        out.ptr = &e;
        out.type = mj::element_type_of<E>::value;
        out.serial = serial;
      }
    }
  });
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

// NOTE ON COMPILE COST. The SDK's sdk::Rename / sdk::DeleteRecursive are
// templated on the element's static type, and each embeds a whole-model walk. A
// serial is only known at runtime, so dispatching those templates through the
// generic tree walk would instantiate them for every one of the ~142 element
// types, and each instantiation re-instantiates the model walk -- a ~142x142
// fan-out that made this TU take minutes. The rename/delete referrer work is
// really keyed on the target's runtime ElementType and name, not its static
// type, so we drive it directly off the SDK's *runtime* helpers (RemoveByPtr,
// ScanRefs, ClearRefs, ParentMap) plus one trivial per-type SetName. Same
// behavior, a handful of walk instantiations instead of hundreds.

int RenameBySerial(EditorContext& ctx, std::uint64_t serial,
                   const std::string& new_name) {
  if (!ctx.tree) {
    return -1;
  }
  bool found = false;
  std::string old_name;
  mj::ElementType etype = mj::ElementType::Model;
  std::function<void()> set_name;  // trivial SetName<E> for the matched type only
  sdk_detail::WalkModelAll(*ctx.tree, [&](auto& e) {
    using E = std::decay_t<decltype(e)>;
    if (found) {
      return;
    }
    if constexpr (requires { e.serial; }) {
      if (e.serial == serial) {
        found = true;
        etype = mj::element_type_of<E>::value;
        if (const std::string* nm = sdk_detail::NameOf(e)) {
          old_name = *nm;
        }
        auto* ep = &e;
        set_name = [ep, &new_name] { sdk_detail::SetName(*ep, new_name); };
      }
    }
  });
  if (!found) {
    return -1;
  }
  if (old_name == new_name) {
    return 0;
  }

  int updated = 0;
  sdk_detail::WalkModelAll(*ctx.tree, [&](auto& other) {
    sdk_detail::ScanRefs(other, [&](int, const char*, std::string& rn,
                                    const std::vector<mj::ElementType>& tgts) {
      if (rn == old_name && sdk_detail::Contains(tgts, etype)) {
        rn = new_name;
        ++updated;
      }
    });
  });
  set_name();
  return updated;
}

namespace {

// Every element pointer on `p`'s root-to-ancestor chain includes `root` when `p`
// is in `root`'s subtree (root included).
bool InSubtree(const ps::sdk::ParentMap& pm, const void* root, const void* p) {
  for (const void* q = p; q != nullptr;) {
    if (q == root) {
      return true;
    }
    const ps::sdk::ParentMap::Node* n = pm.Lookup(q);
    if (!n) {
      break;
    }
    q = n->parent;
  }
  return false;
}

struct DeleteOutcome {
  bool found = false;
  bool removed = false;
  std::vector<std::string> dangling;
};

// Runtime-typed subtree delete with referrer reporting/cascade. Mirrors
// sdk::DeleteRecursive but keyed on the runtime ElementType (see the compile-cost
// note above): locate the target, gather its subtree's names via a ParentMap,
// remove it by pointer, then scan/optionally clear the references it orphaned.
DeleteOutcome DeleteCore(mj::Model& model, std::uint64_t serial, bool cascade) {
  DeleteOutcome out;

  const void* target = nullptr;
  sdk_detail::WalkModelAll(model, [&](auto& e) {
    using E = std::decay_t<decltype(e)>;
    if (target) {
      return;
    }
    if constexpr (requires { e.serial; }) {
      if constexpr (!std::is_same_v<E, mj::Model>) {
        if (e.serial == serial) {
          target = &e;
          out.found = true;
        }
      }
    }
  });
  if (!target) {
    return out;
  }

  ps::sdk::ParentMap pm(model);
  std::vector<sdk_detail::NameType> deleted;
  sdk_detail::WalkModelAll(model, [&](auto& e) {
    using E = std::decay_t<decltype(e)>;
    if (InSubtree(pm, target, &e)) {
      if (const std::string* nm = sdk_detail::NameOf(e)) {
        deleted.push_back({*nm, mj::element_type_of<E>::value});
      }
    }
  });

  out.removed = sdk_detail::RemoveByPtr(model, target);
  if (!out.removed) {
    return out;
  }

  // Referrers inside the removed subtree are gone; the scan only sees survivors.
  ps::sdk::ParentMap after(model);
  sdk_detail::WalkModelAll(model, [&](auto& other) {
    const sdk_detail::Handle h = sdk_detail::MakeHandle(other);
    sdk_detail::ScanRefs(other, [&](int, const char* fname, std::string& rn,
                                    const std::vector<mj::ElementType>& tgts) {
      if (sdk_detail::IsDeleted(deleted, rn, tgts)) {
        out.dangling.push_back(after.PathToPtr(h.ptr) + "." + fname + " -> '" +
                               rn + "'");
      }
    });
  });

  if (cascade && !out.dangling.empty()) {
    sdk_detail::WalkModelAll(model, [&](auto& other) {
      sdk_detail::ClearRefs cr{&deleted};
      mj::Visit(other, cr);
    });
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
  DeleteOutcome o = DeleteCore(*ctx.tree, serial, cascade);
  out.found = o.found;
  out.removed = o.removed;
  out.dangling = std::move(o.dangling);
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
  return DeleteCore(*probe, serial, /*cascade=*/false).dangling;
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
  ctx.Log("undo: " + (label.empty() ? std::string("edit") : label));
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
  ctx.Log("redo: " + (label.empty() ? std::string("edit") : label));
  return true;
}

}  // namespace ps::studio
