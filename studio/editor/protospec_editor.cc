// ProtoSpec Studio: the model-authority plugins (ps::studio, ours). R1 re-target
// onto upstream Studio's existing plugin surface -- the editor drives the host
// through the four stock plugin types only:
//
//   SpecEditorPlugin  pre_compile caches the host's live mjvCamera* (a stable
//                     &App::camera_) and always returns false. This is the only
//                     conduit to the camera on the stock surface; it is dispatched
//                     per frame (INVARIANT: app.cc ProcessPendingLoads dispatches
//                     pre_compile unconditionally -- matching merge-base, restored
//                     in R2 -- so caching the stable pointer once suffices).
//
//   ModelPlugin       get_model_to_load serializes the freshly compiled tree to
//                     the compile-XML (CompileToXml -- byte-parity with the
//                     mjs-compiled Binding is the standing 31/31 XmlPath==MjsPath
//                     gate) with an ABSOLUTE meshdir/texturedir injected, handed to
//                     the host as text/xml. MJB is not usable here: the host's
//                     LoadModelFromBuffer null-derefs on a spec-less MJB load
//                     (spec_editor_.Reset(*spec())), and text/xml round-trips
//                     element ids stably (verification (a)/(c)). post_model_loaded
//                     restores the cached camera flicker-free; do_update freezes
//                     the sim in Edit and services the editor pipeline each tick.
//
// The host never touches ps::Model; the editor never registers a fork-only seam.

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>

#include <mujoco/mujoco.h>

#include "bridge.h"  // ps::mjcf::CompileToXml, CompileOptions
#include "editor/editor_ops.h"
#include "editor/plugin_abi.h"
#include "editor/plugins.h"
#include "platform/ux/plugin.h"
#include "types.h"  // ps::mjcf::Model

namespace ps::studio {
namespace {

namespace mj = ps::mjcf;

// Adoption state shared by the camera + model plugins (leaked for app lifetime,
// mirroring the plugin registry's static storage). `adopt_xml` must outlive the
// get_model_to_load return until the host copies the bytes (same call stack).
struct EditorModelHost {
  EditorContext* ctx = nullptr;
  std::string adopt_xml;
  std::uint64_t emitted_generation = 0;
  mjvCamera saved_camera;
  bool saved_camera_valid = false;
  std::uint64_t diagnosed_bad_generation = 0;  // dim-assert dedup (see DoUpdate)
};

// The mjModel dimension/count fields a correct adopt MUST reproduce exactly.
// Written against the CURRENT pin's field set (MuJoCo 3.10.1, see
// docs/SYNC_STATE.md) -- note `nu`/`nactuator`/`nout`, the post-3.10.0 MIMO
// actuator count split. Returns true when `host` matches `ref` on every field;
// on the first divergence returns false and (when `diag`) names the field with
// both values.
bool AdoptDimsMatch(const mjModel* host, const mjModel* ref, std::string* diag) {
  if (!host || !ref) return true;  // nothing to compare yet
  const struct {
    const char* name;
    long long h;
    long long r;
  } dims[] = {
      {"nq", host->nq, ref->nq},
      {"nv", host->nv, ref->nv},
      {"na", host->na, ref->na},
      {"nu", host->nu, ref->nu},
      {"nactuator", host->nactuator, ref->nactuator},
      {"nout", host->nout, ref->nout},
      {"nbody", host->nbody, ref->nbody},
      {"njnt", host->njnt, ref->njnt},
      {"ngeom", host->ngeom, ref->ngeom},
      {"nsite", host->nsite, ref->nsite},
      {"ncam", host->ncam, ref->ncam},
      {"nlight", host->nlight, ref->nlight},
      {"nmesh", host->nmesh, ref->nmesh},
      {"nhfield", host->nhfield, ref->nhfield},
      {"nmat", host->nmat, ref->nmat},
      {"neq", host->neq, ref->neq},
      {"ntendon", host->ntendon, ref->ntendon},
      {"nsensor", host->nsensor, ref->nsensor},
      {"nkey", host->nkey, ref->nkey},
  };
  for (const auto& d : dims) {
    if (d.h != d.r) {
      if (diag) {
        *diag = std::string(d.name) + " host=" + std::to_string(d.h) +
                " compiled=" + std::to_string(d.r);
      }
      return false;
    }
  }
  return true;
}

std::string AbsBaseDir(const std::string& base) {
  std::error_code ec;
  const std::filesystem::path p =
      base.empty() ? std::filesystem::current_path(ec)
                   : std::filesystem::absolute(base, ec);
  return ec ? base : p.string();
}

// The host reparses adoption bytes with an EMPTY VFS and NO model dir
// (ModelHolder::FromBuffer(text/xml) -> mj_parseXMLString), so on-disk
// meshdir/texturedir assets resolve only if the emitted XML carries ABSOLUTE
// asset dirs (verification (c): probe_bytes -- a dir-less reparse fails with
// "Error opening file"). Add them to the emitted <compiler> tag, or insert one.
void InjectAssetDirs(std::string& xml, const std::string& absdir) {
  const std::string attrs =
      " meshdir=\"" + absdir + "\" texturedir=\"" + absdir + "\"";
  const std::size_t cpos = xml.find("<compiler");
  if (cpos != std::string::npos) {
    const std::size_t end = xml.find('>', cpos);
    if (end != std::string::npos) {
      const std::size_t ins = (end > cpos && xml[end - 1] == '/') ? end - 1 : end;
      xml.insert(ins, attrs);
      return;
    }
  }
  const std::size_t mjpos = xml.find("<mujoco");
  if (mjpos != std::string::npos) {
    const std::size_t end = xml.find('>', mjpos);
    if (end != std::string::npos) {
      xml.insert(end + 1, "\n  <compiler" + attrs + "/>");
    }
  }
}

const char* GetModelToLoad(ModelPlugin* self, int* size, char* content_type,
                           int content_type_size, char* model_name,
                           int model_name_size) {
  EditorModelHost* h = ctx_cast<EditorModelHost>(self);
  EditorContext* c = h->ctx;
  if (!c->model_ready || c->compile_generation == h->emitted_generation) {
    return nullptr;  // nothing new to adopt
  }
  // The exact compile input BuildCompileModel used: ctx.tree, or the pruned
  // serial-preserving clone when a layer is disabled (kept in compile_tree).
  const mj::Model* src = c->compile_tree ? c->compile_tree.get() : c->tree.get();
  if (!src) return nullptr;

  mj::CompileOptions opts;
  if (!c->base_dir.empty()) opts.base_dir = c->base_dir;
  opts.vfs_assets = c->vfs_assets;
  std::string xml = mj::CompileToXml(*src, opts);
  if (xml.empty()) return nullptr;
  InjectAssetDirs(xml, AbsBaseDir(c->base_dir));

  // Snapshot the live camera so post_model_loaded restores it flicker-free on a
  // recompile adopt (OnModelLoaded resets the free camera first). A fresh file /
  // New load is allowed to reframe.
  if (!c->fresh_load && c->camera_ready && c->camera) {
    h->saved_camera = *c->camera;
    h->saved_camera_valid = true;
  } else {
    h->saved_camera_valid = false;
  }

  h->adopt_xml = std::move(xml);
  h->emitted_generation = c->compile_generation;
  // The host's ModelHolder::FromBuffer switches on EXACTLY "text/xml".
  std::snprintf(content_type, content_type_size, "%s", "text/xml");
  // Tag the load with our source path so post_model_loaded can tell our own
  // adoption apart from a host stock file load it must ingest.
  std::snprintf(model_name, model_name_size, "%s", c->source_path.c_str());
  *size = static_cast<int>(h->adopt_xml.size());
  return h->adopt_xml.c_str();
}

void PostModelLoaded(ModelPlugin* self, const char* model_path) {
  EditorModelHost* h = ctx_cast<EditorModelHost>(self);
  EditorContext* c = h->ctx;
  const std::string path = model_path ? model_path : "";

  // A path that is NOT our own adoption tag means the host stock-loaded a file
  // (CLI --model_file / drag-drop) the editor does not yet own. Ingest it so the
  // editor becomes the authority; the next get_model_to_load re-emits it as the
  // editor's compiled bytes. On parse failure the host's model simply stands.
  if (!path.empty() && path != c->source_path) {
    if (LoadModel(*c, path)) {
      c->fresh_load = true;  // a genuine file open reframes the camera
    }
    return;
  }

  // Our own adopt: restore the pre-adopt camera on a recompile; a fresh load /
  // New keeps the host's reframing.
  if (h->saved_camera_valid && c->camera) {
    *c->camera = h->saved_camera;
    h->saved_camera_valid = false;
  }
  c->fresh_load = false;
}

bool DoUpdate(ModelPlugin* self, mjModel* host_model, mjData* host_data) {
  EditorModelHost* h = ctx_cast<EditorModelHost>(self);
  EditorContext* c = h->ctx;

  // The editor's per-tick pipeline: pending load, deferred revert, debounced
  // recompile, mesh fixup. Bumps compile_generation for get_model_to_load.
  ServiceEditorModel(*c);

  // Latch what physics is doing for CanEdit() (the editor owns the mode now).
  c->sim_paused = (c->mode == EditorMode::Edit) || c->play_paused;
  c->sim_time = host_data ? host_data->time : 0.0;

  // ADOPTION DIM-ASSERT (silent-corruption self-check). Once the host has adopted
  // the current generation (compile_generation == emitted_generation), its
  // freshly reparsed mjModel must have the same dimensions/counts as
  // ctx.compiled -- both were compiled from the same authored tree. A mismatch
  // means the adopt silently corrupted: an asset that failed to resolve on the
  // host's dir-less reparse, a VFS gap, or an engine count-model change. Rendering
  // or stepping physics on it would show garbage keyed to the wrong dimensions.
  // Refuse it -- freeze on the last good frame (return true skips
  // StepControl::Advance) and surface a loud diagnostic, once per generation.
  if (host_model && c->compiled.model &&
      c->compile_generation == h->emitted_generation) {
    std::string diag;
    if (!AdoptDimsMatch(host_model, c->compiled.model.get(), &diag)) {
      if (h->diagnosed_bad_generation != c->compile_generation) {
        h->diagnosed_bad_generation = c->compile_generation;
        const std::string msg =
            "ADOPT DIM MISMATCH: host mjModel disagrees with the compiled tree "
            "(" + diag + ") -- refusing to render the adopted model (likely an "
            "unresolved asset on the host's dir-less reparse, or an engine "
            "count-model change).";
        c->Log(msg);
        std::fprintf(stderr, "[protospec] %s\n", msg.c_str());
      }
      return true;  // freeze; never forward/step a dimensionally-corrupt model
    }
  }

  if (c->mode != EditorMode::Edit) {
    // Play: the host advances physics under its own StepControl, unless the
    // editor-emulated pause is on (Space mid-flight) -- then freeze in place:
    // no reset, no forward needed (data holds the last stepped kinematics).
    return c->play_paused;
  }

  // Edit mode: freeze at the reset state. INVARIANT: returning true makes the
  // host skip StepControl::Advance (verified at merge-base app.cc UpdatePhysics:
  // `if (do_update(...)) stepped=true;` then `if(!stepped){ ...Advance }`).
  if (host_model && host_data) {
    if (c->gizmo_active || c->rig_preview.active) {
      // A gizmo drag OR a joint scrub is in flight: the viewport plugin
      // (dispatched AFTER this one, so it sees THIS frame's LivePatch / scrub
      // forward) mirrors the preview kinematics onto host_data for every element
      // kind. Do nothing here -- a bare mj_forward would recompute host_data from
      // the host model's un-patched body_pos/geom_pos and un-scrubbed qpos0, and
      // snap the subtree back to the rest pose, fighting the mirror (the old
      // "geom only moves on release" regression). The rig_preview arm is the
      // scrub twin of gizmo_active; both are pinned by test_plugin_windowless
      // (do_update deferral) so this contract cannot silently regress. See
      // MirrorDragKinematics in viewport_plugin.cc.
    } else if (c->compile_generation != h->emitted_generation) {
      // A fresh compile is queued but the host has NOT adopted it yet (the adopt
      // runs next frame in ProcessPendingLoads -> get_model_to_load). host_model
      // is the PREVIOUS model, so forwarding it now would render the stale pose
      // for one frame before the adopt replaces it -- the visible flicker seen
      // on a gizmo commit (drag pose -> pre-drag snap -> committed pose). Leave
      // host_data untouched so the last good frame holds until the adopt lands.
    } else {
      // Hold the reset pose AND keep the render scene coherent. The host never
      // populates a freshly adopted mjData: ModelHolder makes data with
      // mj_makeData only (no forward), and stock Studio relies on
      // StepControl::Advance to run mj_forward even while paused -- but our Edit
      // freeze returns true, so the host skips Advance entirely. Without this
      // the first frame after every adoption (fresh load or recompile) renders
      // with all geom_xpos == 0 (the model collapsed at the origin, off-camera:
      // the "black viewport until Backspace" regression, Backspace == host
      // ResetPhysics == mj_forward).
      //
      // RESET-ON-ENTER (single Edit toggle): entering Edit must show qpos0, NOT
      // the mid-flight pose the host left when it was running. `time != 0` is
      // exactly that transition -- the sim only advances while Edit is OFF (our
      // freeze holds time at 0), so the first Edit tick after handing back to the
      // host resets to qpos0; every later Edit tick sees time == 0 and just holds
      // the frozen pose. Leaving Edit never resets (Play returns false above).
      if (host_data->time != 0.0) {
        mj_resetData(host_model, host_data);  // enter-Edit snap to qpos0
      }
      mj_forward(host_model, host_data);
    }
  }
  return true;  // freeze
}

}  // namespace

void RegisterProtoSpecEditorPlugin(EditorContext& ctx) {
  EditorModelHost* h = new EditorModelHost{&ctx};

  SpecEditorPlugin cam;
  cam.name = "ProtoSpec Camera";
  cam.data = h;
  cam.pre_compile = [](SpecEditorPlugin* self, mjSpec*, const mjModel*,
                       const mjData*, const mjvCamera* camera) -> bool {
    EditorModelHost* hh = ctx_cast<EditorModelHost>(self);
    // Cache the stable &App::camera_ (const_cast: App::camera_ is non-const, and
    // the editor writes lookat for F-focus / restores it across adoptions).
    hh->ctx->camera = const_cast<mjvCamera*>(camera);
    hh->ctx->camera_ready = (camera != nullptr);
    return false;  // never trigger the host recompile
  };
  RegisterPlugin<SpecEditorPlugin>(cam);

  ModelPlugin model;
  model.name = "ProtoSpec Model";
  model.data = h;
  model.get_model_to_load = GetModelToLoad;
  model.post_model_loaded = PostModelLoaded;
  model.do_update = DoUpdate;
  RegisterPlugin<ModelPlugin>(model);
}

}  // namespace ps::studio
