// ProtoSpec Studio host-level certification tests (windowless).
//
// The batteries under test/ exercise the editor's pure ops; these drive the
// HOST wiring (app.cc / shell.cc) that A19 & friends leave uncovered: the
// Play/Stop mode machine and physics-state discard (G1), the Save As +
// VFS-externalization menu path (G4), and the delete-confirm referrer flow
// (G7). The host is a thin adapter over the shared EditorContext: its whole
// contribution is three reactions -- the EditorShellPlugin::set_mode fan-out
// (app.cc:1608), StepControl pause (app.cc:1621), and ResetPhysics /
// mj_resetData (app.cc:279). We reproduce those reactions verbatim over the
// real editor ops, so the path a Play/Stop/Save click travels is exercised end
// to end with no SDL window or renderer.

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

#include <mujoco/mujoco.h>

#include "binding.h"
#include "compile.h"
#include "editor/asset_import.h"
#include "editor/authoring_ops.h"
#include "editor/editor_context.h"
#include "editor/editor_ops.h"
#include "mjcf.h"
#include "protospec/refs.h"
#include "protospec/traversal.h"
#include "types.h"
#include "validate.h"

namespace mj = ps::mjcf;
namespace bridge = ps::mjcf::bridge;
using namespace ps::studio;

static int g_failed = 0;
static int g_checks = 0;

#define CHECK(cond)                                                \
  do {                                                             \
    ++g_checks;                                                    \
    if (!(cond)) {                                                 \
      ++g_failed;                                                  \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);  \
    }                                                              \
  } while (0)

static bool Near(double a, double b, double eps = 1e-6) {
  return (a - b) <= eps && (b - a) <= eps;
}

// A private scratch directory that survives for one test then self-deletes.
struct TempDir {
  std::filesystem::path dir;
  TempDir() {
    const auto stamp =
        std::chrono::steady_clock::now().time_since_epoch().count();
    dir = std::filesystem::temp_directory_path() /
          ("ps_host_" + std::to_string(stamp) + "_" + std::to_string(g_checks));
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
  }
  ~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
  }
  std::string file(const char* name) const { return (dir / name).string(); }
};

static void WriteFile(const std::string& path, const std::string& body) {
  std::ofstream out(path, std::ios::binary);
  out << body;
}

// ------------------------------------------------------------------------- //
// The host's three reactions, lifted verbatim from app.cc so the test drives
// exactly what a toolbar click drives. `HostSim` stands in for App's mjData +
// StepControl pause bit.
// ------------------------------------------------------------------------- //
struct HostSim {
  mjData* data = nullptr;
  bool paused = true;  // StepControl starts paused on a fresh adopt (app.cc:475)

  ~HostSim() {
    if (data) mj_deleteData(data);
  }

  // app.cc ProcessPendingLoads -> AdoptCompiledModel: build data for the model
  // the ModelSource plugin just published, reset to qpos0, forward once.
  void Adopt(const mjModel* m) {
    if (data) mj_deleteData(data);
    data = mj_makeData(m);
    mj_resetData(m, data);
    mj_forward(m, data);
  }

  // app.cc UpdatePhysics: one host frame steps the sim only while unpaused.
  void Frame(const mjModel* m) {
    if (!paused) mj_step(m, data);
  }

  // app.cc:279 ResetPhysics: back to qpos0 of the currently compiled model.
  void ResetPhysics(const mjModel* m) {
    mj_resetData(m, data);
    mj_forward(m, data);
  }
};

// The host Play toolbar (app.cc:1615): set_editor_mode(1) fans out to the shell's
// EditorShellPlugin::set_mode (shell.cc:211), then the host unpauses + the
// ModelSource poll services any forced recompile before the next frame.
static void HostPlay(EditorContext& ctx, HostSim& sim) {
  // --- EditorShellPlugin::set_mode(1) == shell.cc OnSetMode(mode=1) ---
  ctx.mode = EditorMode::Play;
  ctx.play_paused = false;
  if (ctx.dirty) {
    ctx.apply_edits = true;
    ctx.RequestRecompile();
  }
  // --- host ModelSource poll (protospec_editor.cc:38): the apply_edits one-shot
  // forces the dirty compile before physics runs ---
  if (ShouldServiceRecompile(ctx)) {
    RecompileTree(ctx);
    ctx.recompile_requested = false;
    ctx.apply_edits = false;
    sim.Adopt(ctx.compiled.model.get());
  }
  sim.paused = false;  // app.cc:1617 SetPauseState(kUnpaused)
}

// The host Stop toolbar (app.cc:1626): set_editor_mode(0) + pause + ResetPhysics.
static void HostStop(EditorContext& ctx, HostSim& sim) {
  ctx.mode = EditorMode::Edit;             // shell.cc OnSetMode(mode=0)
  sim.paused = true;                       // app.cc:1628 SetPauseState(kPaused)
  sim.ResetPhysics(ctx.compiled.model.get());  // app.cc:1629
}

// ------------------------------------------------------------------------- //
// G1: Play compiles the dirty edit then runs (time advances); Pause freezes;
// Stop returns to Edit at the EDITED definition's qpos0, discarding sim state.
// ------------------------------------------------------------------------- //
static void TestPlayStopStateDiscard() {
  TempDir tmp;
  const std::string path = tmp.file("fall.xml");
  WriteFile(path, R"(
  <mujoco>
    <worldbody>
      <body name="ball" pos="0 0 1">
        <freejoint/>
        <geom name="g" type="sphere" size="0.1"/>
      </body>
    </worldbody>
  </mujoco>)");

  EditorContext ctx;
  CHECK(LoadModel(ctx, path));
  CHECK(ctx.compiled.ok());
  CHECK(!ctx.dirty);
  // Free body: qpos = [x y z, quat]. Compiled qpos0 z starts at the authored 1.
  CHECK(Near(ctx.compiled.model->qpos0[2], 1.0));

  HostSim sim;
  sim.Adopt(ctx.compiled.model.get());

  // A dirty Details edit made in Edit mode: raise the ball to z=2. Left pending
  // (the debounced Edit-mode recompile has not fired yet), so the compiled model
  // still reads z=1 -- Play must compile it.
  mj::Body* ball = ps::sdk::Find<mj::Body>(*ctx.tree, "ball");
  CHECK(ball != nullptr);
  ctx.BeginEdit();
  ball->pos = std::array<double, 3>{0, 0, 2};
  ctx.CommitEdit("raise ball");
  CHECK(ctx.dirty);
  CHECK(Near(ctx.compiled.model->qpos0[2], 1.0));  // not yet applied to compiled

  // --- Play: compiles the dirty edit (qpos0 z -> 2), then runs. ---
  HostPlay(ctx, sim);
  CHECK(ctx.mode == EditorMode::Play);
  CHECK(Near(ctx.compiled.model->qpos0[2], 2.0));  // Play forced the compile
  const mjModel* m = ctx.compiled.model.get();
  CHECK(Near(sim.data->qpos[2], 2.0));  // adopted at the edited qpos0

  // Time advances and the ball falls under gravity while unpaused.
  for (int i = 0; i < 300; ++i) sim.Frame(m);
  CHECK(sim.data->time > 0.0);
  CHECK(sim.data->qpos[2] < 1.9);  // fell well below the start height
  const double fallen_z = sim.data->qpos[2];

  // --- Pause: the running sim freezes (host stops stepping). ---
  sim.paused = true;  // app.cc:1621 SetPauseState(kNormalPaused)
  for (int i = 0; i < 100; ++i) sim.Frame(m);
  CHECK(Near(sim.data->qpos[2], fallen_z));  // no motion while paused

  // --- Stop: back to Edit at the EDITED qpos0 (z=2), sim state discarded. ---
  HostStop(ctx, sim);
  CHECK(ctx.mode == EditorMode::Edit);
  CHECK(Near(sim.data->qpos[2], 2.0));         // edited qpos0, NOT the fallen pose
  CHECK(!Near(sim.data->qpos[2], fallen_z));   // fall discarded
  CHECK(!Near(sim.data->qpos[2], 1.0));        // and NOT the original definition
  // The edit lives on in the authored tree and takes effect on the next Play.
  mj::Body* ball2 = ps::sdk::Find<mj::Body>(*ctx.tree, "ball");
  CHECK(ball2 != nullptr && ball2->pos.has_value());
  CHECK(Near((*ball2->pos)[2], 2.0));
}

// ------------------------------------------------------------------------- //
// G4: Save As + VFS externalization through the host menu path (SaveExternalize
// == SaveModel + ExternalizeVfsAssets, shell.cc:27). A New model with an
// imported (in-memory / VFS) mesh, saved to a *different* directory, must land
// the mesh bytes next to the written .xml, clear the VFS store + dirty flag, and
// reload+compile straight off disk.
// ------------------------------------------------------------------------- //
static void TestSaveAsExternalizeHostPath() {
  TempDir src;   // where the imported mesh originally lives
  TempDir dst;   // the Save As destination (a different directory)

  // A minimal tetrahedron OBJ; the editor never parses it, the compiler does.
  const std::string obj = src.file("tetra.obj");
  WriteFile(obj,
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n"
            "f 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n");

  EditorContext ctx;
  CHECK(NewModelOp(ctx));         // File > New: an unsaved (in-memory) model
  CHECK(ctx.source_path.empty());

  // File > Import Mesh: for an unsaved model the bytes are registered as a VFS
  // asset (not copied), so nothing is on disk yet next to a (nonexistent) .xml.
  MeshImportResult mr = ImportMesh(ctx, obj, nullptr);
  CHECK(mr.ok);
  CHECK(mr.vfs);
  CHECK(!ctx.vfs_assets.empty());
  CHECK(ctx.dirty);
  CHECK(ps::sdk::Find<mj::Mesh>(*ctx.tree, "tetra") != nullptr);
  CHECK(RecompileTree(ctx) && ctx.compiled.ok());
  CHECK(ctx.compiled.model->nmesh >= 1);

  // --- File > Save As <dst>/scene.xml  ==  shell.cc SaveExternalize ---
  const std::string xml = dst.file("scene.xml");
  CHECK(SaveModel(ctx, xml));         // writes the .xml, updates source_path,
  CHECK(!ctx.dirty);                  //   clears dirty (title-bar dot clears)
  CHECK(ctx.source_path == xml);
  ExternalizeVfsAssets(ctx, xml);     // spills the VFS bytes next to the .xml
  CHECK(ctx.vfs_assets.empty());      // store drained

  // The .xml and its sibling mesh both exist in the *destination* directory.
  CHECK(std::filesystem::exists(xml));
  CHECK(std::filesystem::exists(dst.dir / "tetra.obj"));

  // A fresh disk load of the saved model compiles straight off the file.
  EditorContext reload;
  CHECK(LoadModel(reload, xml));
  CHECK(reload.compiled.ok());
  CHECK(reload.compiled.model->nmesh >= 1);
  CHECK(!reload.dirty);
}

// ------------------------------------------------------------------------- //
// G7: the delete-confirm referrer flow (shell.cc:104 delete_request_serial ->
// hierarchy_panel.cc modal -> confirm/cancel). The panel owns the ImGui modal;
// its decision + effects are the pure ops PreviewDeleteReferrers / DeleteBySerial
// this test drives directly. Two arms: a referenced element (dialog, then
// cancel-is-inert and confirm-cascades) and a leaf (no dialog, immediate delete).
// ------------------------------------------------------------------------- //
static void TestDeleteConfirmFlow() {
  TempDir tmp;
  const std::string path = tmp.file("rig.xml");
  WriteFile(path, R"(
  <mujoco>
    <worldbody>
      <body name="b" pos="0 0 1">
        <joint name="j" type="hinge" axis="0 1 0"/>
        <geom name="g" type="box" size="0.1 0.1 0.1"/>
        <geom name="leaf" type="sphere" size="0.05" pos="0.3 0 0"/>
      </body>
    </worldbody>
    <sensor>
      <jointpos name="s" joint="j"/>
    </sensor>
  </mujoco>)");

  EditorContext ctx;
  CHECK(LoadModel(ctx, path));
  CHECK(ctx.compiled.ok());

  const std::uint64_t j = ps::sdk::Find<mj::Joint>(*ctx.tree, "j")->serial;

  // Edit > Delete on the joint: the host records the request (shell.cc:104).
  CHECK(SelectBySerial(ctx, j));
  ctx.delete_request_serial = ctx.selected_serial;
  CHECK(ctx.delete_request_serial == j);

  // The panel services the request: the joint has a referrer (the jointpos
  // sensor), so the confirm dialog opens rather than deleting silently.
  std::vector<std::string> referrers = PreviewDeleteReferrers(ctx, j);
  CHECK(!referrers.empty());
  // Read-only preview: the tree is untouched and still compiles.
  CHECK(ps::sdk::Find<mj::Joint>(*ctx.tree, "j") != nullptr);
  CHECK(ps::sdk::Find<mj::Jointpos>(*ctx.tree, "s") != nullptr);

  // --- Cancel: the modal closes, the request clears, nothing is deleted. ---
  ctx.delete_request_serial = 0;
  CHECK(ps::sdk::Find<mj::Joint>(*ctx.tree, "j") != nullptr);
  CHECK(ps::sdk::Find<mj::Jointpos>(*ctx.tree, "s") != nullptr);
  CHECK(RecompileTree(ctx) && ctx.compiled.ok());

  // --- Confirm: cascade the delete; the joint goes and no referrer is left
  // dangling (A15 referrer safety). ---
  ctx.BeginEdit();
  DeleteResult dr = DeleteBySerial(ctx, j, /*cascade=*/true);
  ctx.CommitEdit("delete joint");
  CHECK(dr.found && dr.removed);
  CHECK(ps::sdk::Find<mj::Joint>(*ctx.tree, "j") == nullptr);
  CHECK(ps::sdk::FindReferrers(*ctx.tree, "j", mj::ElementType::Joint).empty());

  // --- Leaf arm: a geom with no referrers needs no dialog (immediate delete). ---
  const std::uint64_t leaf = ps::sdk::Find<mj::Geom>(*ctx.tree, "leaf")->serial;
  CHECK(PreviewDeleteReferrers(ctx, leaf).empty());  // -> panel deletes at once
  ctx.BeginEdit();
  DeleteResult ld = DeleteBySerial(ctx, leaf, /*cascade=*/false);
  ctx.CommitEdit("delete leaf");
  CHECK(ld.found && ld.removed);
  CHECK(ps::sdk::Find<mj::Geom>(*ctx.tree, "leaf") == nullptr);
  CHECK(RecompileTree(ctx) && ctx.compiled.ok());
}

// ------------------------------------------------------------------------- //
// G8: Diagnostics click-to-select navigation. Three units behind the panel's
// clickable rows: (1) the structured append contract -- severity / serial /
// SourceLoc, the retained legacy plain-string path, and the ring cap; (2)
// serial routing on a row click, at the state-machine level (per G7): a real
// producer stamps the acting serial and the exact call the Selectable makes
// (SelectBySerial) selects that element; (3) the Validate -> diagnostic ->
// serial mapping on a planted tier-2 (referential) error.
// ------------------------------------------------------------------------- //
static void TestDiagnosticStructuredAppend() {
  EditorContext ctx;

  // Legacy plain-string path: an Info entry with no serial and no loc.
  ctx.Log("plain line");
  CHECK(ctx.diagnostics.size() == 1);
  CHECK(ctx.diagnostics.back().severity == DiagEntry::Severity::Info);
  CHECK(ctx.diagnostics.back().message == "plain line");
  CHECK(!ctx.diagnostics.back().serial.has_value());
  CHECK(!ctx.diagnostics.back().loc.has_value());

  // Structured path: severity, serial, and SourceLoc all carried through.
  ctx.Diagnose(DiagEntry{DiagEntry::Severity::Error, "boom", std::uint64_t{99},
                         ps::SourceLoc{"model.xml", 12}});
  const DiagEntry& e = ctx.diagnostics.back();
  CHECK(e.severity == DiagEntry::Severity::Error);
  CHECK(e.serial.has_value() && *e.serial == 99);
  CHECK(e.loc.has_value() && e.loc->file == "model.xml" && e.loc->line == 12);

  // Ring cap: the deque never grows past the bound; the oldest rows are evicted.
  ctx.ClearDiagnostics();
  CHECK(ctx.diagnostics.empty());
  for (std::size_t i = 0; i < EditorContext::kMaxDiagnostics + 50; ++i) {
    ctx.Log("row " + std::to_string(i));
  }
  CHECK(ctx.diagnostics.size() == EditorContext::kMaxDiagnostics);
  CHECK(ctx.diagnostics.front().message == "row 50");  // first 50 evicted
  CHECK(ctx.diagnostics.back().message ==
        "row " + std::to_string(EditorContext::kMaxDiagnostics + 49));

  ctx.ClearDiagnostics();
  CHECK(ctx.diagnostics.empty());
}

static void TestDiagnosticClickSelectsElement() {
  TempDir tmp;
  const std::string path = tmp.file("pickrig.xml");
  WriteFile(path, R"(
  <mujoco>
    <worldbody>
      <body name="b" pos="0 0 1">
        <geom name="g" type="box" size="0.1 0.1 0.1"/>
      </body>
    </worldbody>
  </mujoco>)");

  EditorContext ctx;
  CHECK(LoadModel(ctx, path));
  CHECK(ctx.compiled.ok());
  const std::uint64_t g = ps::sdk::Find<mj::Geom>(*ctx.tree, "g")->serial;

  // A producer (the viewport pick logger) stamps the acting element's serial
  // onto its Diagnostics row. This model has a single geom -> compiled id 0.
  ctx.ClearDiagnostics();
  const PickResolution r = ResolvePick(ctx, /*geom_id=*/0, /*body_id=*/-1);
  CHECK(r.hit && r.serial == g);
  CHECK(!ctx.diagnostics.empty());
  const DiagEntry& row = ctx.diagnostics.back();
  CHECK(row.serial.has_value() && *row.serial == g);

  // Simulate the row click: drop the selection, then route the row's serial
  // through the exact call the panel's Selectable makes (SelectBySerial). The
  // element is reselected -> Hierarchy/viewport highlight follows.
  ctx.selected_serial = 0;
  ctx.selected_desc.clear();
  CHECK(SelectBySerial(ctx, *row.serial));
  CHECK(ctx.selected_serial == g);
  CHECK(!ctx.selected_desc.empty());

  // A row with no serial (a plain Info line) is inert -- nothing to route.
  ctx.Log("just a note");
  CHECK(!ctx.diagnostics.back().serial.has_value());
}

static void TestDiagnosticValidateMapsSerial() {
  // A tier-2 (referential) error: a sensor references a joint that does not
  // exist. Validate flags it at the sensor's element path; the mapping resolves
  // that path back to the sensor's creation serial so a click selects it. (The
  // model also fails compile, so in the live load path the tree rolls back --
  // this exercises the mapping unit on the retained parse, the part a click
  // depends on.)
  const std::string xml = R"(
  <mujoco>
    <worldbody>
      <body name="b" pos="0 0 1">
        <joint name="j" type="hinge" axis="0 1 0"/>
        <geom name="g" type="box" size="0.1 0.1 0.1"/>
      </body>
    </worldbody>
    <sensor>
      <jointpos name="badsensor" joint="ghost"/>
    </sensor>
  </mujoco>)";

  mj::io::ParseResult parsed = mj::io::ParseMjcfString(xml, "planted.xml");
  CHECK(parsed.ok());

  const std::vector<mj::validate::Diagnostic> diags = mj::validate::Validate(
      *parsed.model,
      mj::validate::kTierStructural | mj::validate::kTierReferential);

  const mj::validate::Diagnostic* bad = nullptr;
  for (const mj::validate::Diagnostic& d : diags) {
    if (d.severity == mj::validate::Severity::Error &&
        d.message.find("ghost") != std::string::npos) {
      bad = &d;
    }
  }
  CHECK(bad != nullptr);

  // The mapping resolves the diagnostic's path to the offending sensor.
  const std::uint64_t want =
      ps::sdk::Find<mj::Jointpos>(*parsed.model, "badsensor")->serial;
  const std::optional<std::uint64_t> got =
      SerialForValidatePath(*parsed.model, bad->path);
  CHECK(got.has_value() && *got == want);

  // And routing that serial (the click) selects the sensor on the live tree.
  EditorContext ctx;
  ctx.tree = std::move(parsed.model);
  CHECK(SelectBySerial(ctx, *got));
  CHECK(ctx.selected_serial == want);
}

int main() {
  TestPlayStopStateDiscard();
  TestSaveAsExternalizeHostPath();
  TestDeleteConfirmFlow();
  TestDiagnosticStructuredAppend();
  TestDiagnosticClickSelectsElement();
  TestDiagnosticValidateMapsSerial();

  std::printf("%d checks, %d failed\n", g_checks, g_failed);
  return g_failed == 0 ? 0 : 1;
}
