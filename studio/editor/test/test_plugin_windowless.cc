// Windowless tests of the editor's plugin surface (transport state machine,
// adoption gating, edit-gate). Compiles protospec_editor.cc into this TU with a
// stub plugin registry and stub editor-ops, so the ModelPlugin/SpecEditorPlugin
// callbacks are driven directly -- no window, no ImGui, no host.
//
// Covers the host-behavior assumptions documented in protospec_editor.cc:
//   * pre_compile caches the host camera pointer and never triggers recompile
//   * get_model_to_load emits exactly once per compile_generation (double-adopt
//     suppression), tags content_type "text/xml" and model_name = source_path
//   * post_model_loaded distinguishes our own adoption (camera restore) from a
//     host stock file load (ingest via LoadModel)
//   * do_update freeze semantics: Edit returns true (host skips Advance), Play
//     returns play_paused; reset-on-enter snaps time!=0 to qpos0; a pending
//     un-adopted compile leaves host_data untouched; gizmo_active defers
//   * InjectAssetDirs rewrites <compiler> correctly in all three shapes
//
// Driven by tests/test_plugin_windowless.py (g++ against libprotospec_core.a +
// libmujoco.so, same recipe as test_path_diff).

#include <cassert>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>

// --- stub plugin registry (replaces the host's platform library) ---------- //
#include "platform/ux/plugin.h"

namespace mujoco::platform {
namespace {
template <typename T>
std::vector<T>& Registry() {
  static std::vector<T> r;
  return r;
}
}  // namespace
template <typename T>
void RegisterPlugin(T plugin) {
  Registry<T>().push_back(plugin);
}
template <typename T>
void ForEachPlugin(const std::function<void(T*)>& fn) {
  for (T& p : Registry<T>()) fn(&p);
}
}  // namespace mujoco::platform

// --- stub editor ops (protospec_editor.cc calls these) -------------------- //
#include "editor/editor_context.h"
#include "mjcf.h"  // ps::mjcf::io::ParseMjcfString

namespace ps::studio {
namespace teststub {
int service_calls = 0;
int load_calls = 0;
std::string last_load_path;
bool load_result = true;
}  // namespace teststub

void ServiceEditorModel(EditorContext&) { ++teststub::service_calls; }
bool LoadModel(EditorContext&, const std::string& path) {
  ++teststub::load_calls;
  teststub::last_load_path = path;
  return teststub::load_result;
}
}  // namespace ps::studio

// Splice the plugin TU in so its anonymous-namespace callbacks are reachable.
#include "editor/protospec_editor.cc"

namespace ps::studio {
namespace {

#define CHECK(cond)                                                       \
  do {                                                                    \
    if (!(cond)) {                                                        \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      return 1;                                                           \
    }                                                                     \
  } while (0)

constexpr char kTinyMjcf[] = R"(<mujoco model="tiny">
  <worldbody>
    <body name="b" pos="0 0 1">
      <freejoint/>
      <geom name="g" type="sphere" size="0.1"/>
    </body>
  </worldbody>
</mujoco>
)";

struct Plugins {
  ModelPlugin* model = nullptr;
  SpecEditorPlugin* cam = nullptr;
};

Plugins RegisteredPlugins() {
  Plugins out;
  ForEachPlugin<ModelPlugin>([&](ModelPlugin* p) {
    if (std::strcmp(p->name, "ProtoSpec Model") == 0) out.model = p;
  });
  ForEachPlugin<SpecEditorPlugin>([&](SpecEditorPlugin* p) {
    if (std::strcmp(p->name, "ProtoSpec Camera") == 0) out.cam = p;
  });
  return out;
}

const char* CallGet(ModelPlugin* mp, int* size, char* ctype, char* mname) {
  return mp->get_model_to_load(mp, size, ctype, 64, mname, 512);
}

int RunTests() {
  EditorContext ctx;
  RegisterProtoSpecEditorPlugin(ctx);
  Plugins p = RegisteredPlugins();
  CHECK(p.model && p.model->get_model_to_load && p.model->post_model_loaded &&
        p.model->do_update);
  CHECK(p.cam && p.cam->pre_compile);

  // Assumption probe (2026-07 upstream sync): ModelPlugin grew pre_step/post_step
  // hooks. The editor drives physics solely through the do_update freeze and must
  // leave these null, so the host's new PreStep/PostStep dispatch is a no-op for
  // us. If a future change wires them, this fires and DoUpdate's freeze reasoning
  // must be re-derived against the new step path. (That these fields exist at all
  // also exercises the ModelPlugin ABI the plugin_abi.h static_asserts pin.)
  CHECK(p.model->pre_step == nullptr);
  CHECK(p.model->post_step == nullptr);

  // --- pre_compile: camera conduit, never triggers host recompile --------- //
  mjvCamera cam;
  mjv_defaultCamera(&cam);
  CHECK(p.cam->pre_compile(p.cam, nullptr, nullptr, nullptr, &cam) == false);
  CHECK(ctx.camera == &cam);
  CHECK(ctx.camera_ready);

  // --- adoption gating ---------------------------------------------------- //
  int size = 0;
  char ctype[64] = {0};
  char mname[512] = {0};

  // No model ready: nothing to adopt.
  CHECK(CallGet(p.model, &size, ctype, mname) == nullptr);

  auto parsed = ps::mjcf::io::ParseMjcfString(kTinyMjcf);
  CHECK(parsed.ok());
  ctx.tree = std::move(parsed.model);
  ctx.model_ready = true;
  ctx.compile_generation = 1;
  ctx.source_path = "/virtual/tiny.xml";

  const char* xml = CallGet(p.model, &size, ctype, mname);
  CHECK(xml != nullptr);
  CHECK(size > 0 && static_cast<int>(std::strlen(xml)) == size);
  CHECK(std::string(ctype) == "text/xml");         // host switches on EXACTLY this
  CHECK(std::string(mname) == ctx.source_path);    // our adoption tag
  CHECK(std::string(xml).find("<mujoco") != std::string::npos);
  // Absolute asset dirs must have been injected for the host's dir-less reparse.
  CHECK(std::string(xml).find("meshdir=\"") != std::string::npos);

  // Same generation: emitted once, never twice (double-adopt suppression).
  CHECK(CallGet(p.model, &size, ctype, mname) == nullptr);

  // New generation: emitted again.
  ctx.compile_generation = 2;
  CHECK(CallGet(p.model, &size, ctype, mname) != nullptr);
  CHECK(CallGet(p.model, &size, ctype, mname) == nullptr);

  // --- post_model_loaded: own-adopt camera restore vs stock-load ingest --- //
  // Recompile adopt (not a fresh load): the pre-adopt camera was snapshotted at
  // get_model_to_load time; a host reframe in between must be undone.
  ctx.fresh_load = false;
  cam.distance = 7.5;
  ctx.compile_generation = 3;
  CHECK(CallGet(p.model, &size, ctype, mname) != nullptr);  // snapshots camera
  cam.distance = 99.0;  // host's OnModelLoaded reframes the free camera
  p.model->post_model_loaded(p.model, ctx.source_path.c_str());
  CHECK(cam.distance == 7.5);  // flicker-free restore
  CHECK(!ctx.fresh_load);

  // Fresh load (File > Open / New): the host's reframe must stand.
  ctx.fresh_load = true;
  ctx.compile_generation = 4;
  CHECK(CallGet(p.model, &size, ctype, mname) != nullptr);
  cam.distance = 42.0;
  p.model->post_model_loaded(p.model, ctx.source_path.c_str());
  CHECK(cam.distance == 42.0);  // reframe kept
  CHECK(!ctx.fresh_load);       // one-shot consumed

  // A path that is not our tag is a host stock file load: ingest it.
  teststub::load_calls = 0;
  teststub::load_result = true;
  p.model->post_model_loaded(p.model, "/somewhere/else/model.xml");
  CHECK(teststub::load_calls == 1);
  CHECK(teststub::last_load_path == "/somewhere/else/model.xml");
  CHECK(ctx.fresh_load);  // a genuine file open reframes the camera

  // Failed ingest: the host's model stands, no fresh_load flag.
  ctx.fresh_load = false;
  teststub::load_result = false;
  p.model->post_model_loaded(p.model, "/somewhere/else/broken.xml");
  CHECK(!ctx.fresh_load);

  // --- do_update: freeze semantics + reset-on-enter ----------------------- //
  // Real host model/data via MuJoCo, loaded from the emitted bytes.
  mjVFS vfs;
  mj_defaultVFS(&vfs);
  CHECK(mj_addBufferVFS(&vfs, "tiny.xml", kTinyMjcf,
                        static_cast<int>(sizeof(kTinyMjcf) - 1)) == 0);
  char err[256] = {0};
  mjModel* hm = mj_loadXML("tiny.xml", &vfs, err, sizeof(err));
  CHECK(hm != nullptr);
  mjData* hd = mj_makeData(hm);
  CHECK(hd != nullptr);

  // Sync the emitted generation so do_update takes the steady-state branch.
  ctx.compile_generation = 4;

  // Play mode: host advances physics; return value is the emulated pause.
  ctx.mode = EditorMode::Play;
  ctx.play_paused = false;
  teststub::service_calls = 0;
  CHECK(p.model->do_update(p.model, hm, hd) == false);
  CHECK(teststub::service_calls == 1);  // pipeline serviced every tick
  CHECK(!ctx.sim_paused);
  ctx.play_paused = true;
  CHECK(p.model->do_update(p.model, hm, hd) == true);  // Space: freeze in place
  CHECK(ctx.sim_paused);

  // Edit mode with mid-flight time: reset-on-enter snaps to qpos0, time 0.
  ctx.mode = EditorMode::Edit;
  ctx.play_paused = false;
  hd->time = 5.0;
  hd->qpos[2] = 0.123;  // perturbed mid-flight pose
  CHECK(p.model->do_update(p.model, hm, hd) == true);  // freeze
  CHECK(hd->time == 0.0);
  CHECK(hd->qpos[2] == hm->qpos0[2]);
  CHECK(ctx.sim_paused);
  // Steady Edit tick: holds the frozen pose (no reset, still forwarded).
  CHECK(p.model->do_update(p.model, hm, hd) == true);
  CHECK(hd->time == 0.0);

  // Pending un-adopted compile: host_data must be left untouched (anti-flicker).
  ctx.compile_generation = 5;  // > emitted_generation (4)
  hd->time = 3.0;
  CHECK(p.model->do_update(p.model, hm, hd) == true);
  CHECK(hd->time == 3.0);  // no reset, no forward: last good frame holds
  ctx.compile_generation = 4;

  // Gizmo drag in flight: the model plugin defers all host-data writes.
  ctx.gizmo_active = true;
  hd->time = 2.0;
  CHECK(p.model->do_update(p.model, hm, hd) == true);
  CHECK(hd->time == 2.0);
  ctx.gizmo_active = false;

  mj_deleteData(hd);
  mj_deleteModel(hm);
  mj_deleteVFS(&vfs);

  // --- InjectAssetDirs shapes --------------------------------------------- //
  {
    std::string a = "<mujoco>\n  <compiler angle=\"degree\"/>\n</mujoco>";
    InjectAssetDirs(a, "/abs/dir");
    CHECK(a.find("<compiler angle=\"degree\" meshdir=\"/abs/dir\" "
                 "texturedir=\"/abs/dir\"/>") != std::string::npos);
    std::string b = "<mujoco>\n  <compiler angle=\"degree\"></compiler>\n</mujoco>";
    InjectAssetDirs(b, "/abs/dir");
    CHECK(b.find("<compiler angle=\"degree\" meshdir=\"/abs/dir\" "
                 "texturedir=\"/abs/dir\">") != std::string::npos);
    std::string c = "<mujoco>\n  <worldbody/>\n</mujoco>";
    InjectAssetDirs(c, "/abs/dir");
    CHECK(c.find("<compiler meshdir=\"/abs/dir\" texturedir=\"/abs/dir\"/>") !=
          std::string::npos);
  }

  std::printf("plugin_windowless: all checks passed\n");
  return 0;
}

}  // namespace
}  // namespace ps::studio

int main() { return ps::studio::RunTests(); }
