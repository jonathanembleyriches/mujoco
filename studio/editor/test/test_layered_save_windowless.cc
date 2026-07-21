// Windowless test of the layered in-place save invariant (ps::studio, ours):
//
//   A layered in-place save must ALWAYS produce a set of files that reloads to
//   an equivalent tree -- Save can never make a model unloadable.
//
// The regression it guards: SaveLayeredMjcf used to write EVERY layer -- the
// SOURCE/root document included -- as a <mujocoinclude> fragment, overwriting
// the root file so no <mujoco> root survived. The saved model then loaded in
// NEITHER ProtoSpec ("root element must be <mujoco>") nor stock mj_loadXML
// ("Unrecognized XML model type: 'mujocoinclude'"). A plain Ctrl+S silently
// corrupted the user's files.
//
// Fixture mirrors the franka scene/fr3 pair: a <mujoco> root that <include>s a
// full <mujoco> child document, so the load splits into two provenance layers.
// Checks: (a) the root file stays a <mujoco> root carrying an <include>; (b) the
// child file is a <mujocoinclude> fragment; (c) ProtoSpec reload reproduces the
// tree (WriteMjcf fixpoint -- serial-independent); (d) stock mj_loadXML accepts
// the saved set (both loaders); (e) save->reload->save is idempotent (no drift);
// plus a single-layer regression. Splices layers.cc / undo.cc; links
// libprotospec_core.a + libmujoco.so, same recipe as test_rigger_windowless.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>

#include "editor/editor_context.h"
#include "editor/layers.h"
#include "mjcf.h"                  // ps::mjcf::io::ParseMjcfFile / WriteMjcf
#include "protospec/traversal.h"   // ps::sdk::ForEachElement

// Splice the units under test (not in the linked core lib).
#include "editor/layers.cc"
#include "editor/undo.cc"

namespace ps::studio {

// layers.cc (RemoveLayer) references SelectBySerial, defined in editor_ops.cc --
// not part of the save path this test exercises. Stub it so the spliced TU links
// without dragging the whole editor_ops unit (and MuJoCo compile bridge) in.
bool SelectBySerial(EditorContext&, std::uint64_t) { return false; }

namespace {

namespace mj = ps::mjcf;
namespace io = ps::mjcf::io;

int g_failures = 0;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++g_failures;                                                        \
    }                                                                      \
  } while (0)

// Root document that includes a full-<mujoco> child -- the franka scene/fr3
// shape (include first, then the root's own sections).
constexpr char kRootXml[] = R"(<mujoco model="root scene">
  <include file="child.xml"/>

  <visual>
    <global azimuth="120" elevation="-20"/>
  </visual>

  <asset>
    <material name="grid" rgba="0.2 0.3 0.4 1"/>
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="5 5 0.1" material="grid"/>
  </worldbody>
</mujoco>
)";

constexpr char kChildXml[] = R"(<mujoco model="child">
  <compiler autolimits="true"/>

  <default>
    <default class="arm">
      <geom type="box" size="0.1 0.1 0.1"/>
    </default>
  </default>

  <worldbody>
    <body name="link1" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0" range="-1 1"/>
      <geom class="arm" name="g1"/>
      <body name="link2" pos="0.3 0 0">
        <joint name="j2" type="hinge" axis="0 1 0" range="-1 1"/>
        <geom class="arm" name="g2"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor name="m1" joint="j1"/>
    <motor name="m2" joint="j2"/>
  </actuator>
</mujoco>
)";

// A standalone single-layer model (no include) for the regression check.
constexpr char kSoloXml[] = R"(<mujoco model="solo">
  <compiler autolimits="true"/>
  <worldbody>
    <body name="b" pos="0 0 1">
      <joint name="jj" type="hinge" axis="0 1 0" range="-1 1"/>
      <geom name="gg" type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="mm" joint="jj"/>
  </actuator>
</mujoco>
)";

void WriteFile(const std::filesystem::path& p, const std::string& text) {
  std::ofstream out(p, std::ios::binary);
  out.write(text.data(), static_cast<std::streamsize>(text.size()));
}

std::string ReadFile(const std::filesystem::path& p) {
  std::ifstream in(p, std::ios::binary);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

int CountElements(const mj::Model& m) {
  int n = 0;
  ps::sdk::ForEachElement(const_cast<mj::Model&>(m), [&](auto&) { ++n; });
  return n;
}

bool StartsWithRootTag(const std::string& xml, const char* tag) {
  // First '<' that opens an element (skip the XML declaration/comments/space).
  std::size_t i = 0;
  while (i < xml.size()) {
    if (xml[i] == '<') {
      if (xml.compare(i, 2, "<?") == 0 || xml.compare(i, 4, "<!--") == 0) {
        i = xml.find('>', i);
        if (i == std::string::npos) return false;
        ++i;
        continue;
      }
      return xml.compare(i + 1, std::strlen(tag), tag) == 0;
    }
    ++i;
  }
  return false;
}

// Load `path` with stock MuJoCo; returns true iff it compiled (both the root
// <mujoco> and every <include>d fragment were accepted).
bool StockLoads(const std::filesystem::path& path, std::string* err) {
  char buf[1024] = {0};
  mjModel* m = mj_loadXML(path.string().c_str(), nullptr, buf, sizeof(buf));
  if (!m) {
    if (err) *err = buf;
    return false;
  }
  mj_deleteModel(m);
  return true;
}

// Build a layered EditorContext from a root file on disk (parse -> partition).
bool LoadLayered(EditorContext& ctx, const std::filesystem::path& root,
                 std::string* err) {
  io::ParseResult parsed = io::ParseMjcfFile(root.string());
  if (!parsed.ok()) {
    if (err) *err = parsed.errors.empty() ? "parse failed"
                                          : parsed.errors.front().message;
    return false;
  }
  ctx.tree = std::move(parsed.model);
  SplitLayersFromTree(ctx, root.string(), root.stem().string());
  return true;
}

// --- The invariant ---------------------------------------------------------- //

void TestLayeredSaveInvariant(const std::filesystem::path& dir) {
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  const std::filesystem::path root = dir / "root.xml";
  const std::filesystem::path child = dir / "child.xml";
  WriteFile(root, kRootXml);
  WriteFile(child, kChildXml);

  // Load + partition: two provenance layers (root scene + child).
  EditorContext ctx;
  std::string err;
  CHECK(LoadLayered(ctx, root, &err));
  CHECK(ctx.layers.size() == 2);
  const std::string canonical = io::WriteMjcf(*ctx.tree);  // reference form
  const int n_before = CountElements(*ctx.tree);

  // Save in place.
  CHECK(SaveLayeredMjcf(ctx, root.string(), &err));

  // (a) the root file is still a <mujoco> ROOT carrying an <include> ref.
  const std::string root_txt = ReadFile(root);
  CHECK(StartsWithRootTag(root_txt, "mujoco"));
  CHECK(!StartsWithRootTag(root_txt, "mujocoinclude"));
  CHECK(root_txt.find("<include") != std::string::npos);
  CHECK(root_txt.find("child.xml") != std::string::npos);

  // (b) the child file is a <mujocoinclude> fragment.
  const std::string child_txt = ReadFile(child);
  CHECK(StartsWithRootTag(child_txt, "mujocoinclude"));

  // (c) ProtoSpec reload reproduces the tree (WriteMjcf fixpoint).
  io::ParseResult re = io::ParseMjcfFile(root.string());
  CHECK(re.ok());
  if (re.ok()) {
    CHECK(CountElements(*re.model) == n_before);
    CHECK(io::WriteMjcf(*re.model) == canonical);
  }

  // (d) stock mj_loadXML accepts the saved set (both loaders).
  std::string stock_err;
  CHECK(StockLoads(root, &stock_err));
  if (!stock_err.empty()) std::fprintf(stderr, "  stock: %s\n", stock_err.c_str());

  // (e) save->reload->save is idempotent (no drift).
  const std::string root_txt1 = root_txt;
  const std::string child_txt1 = child_txt;
  EditorContext ctx2;
  CHECK(LoadLayered(ctx2, root, &err));
  CHECK(ctx2.layers.size() == 2);
  CHECK(SaveLayeredMjcf(ctx2, root.string(), &err));
  CHECK(ReadFile(root) == root_txt1);
  CHECK(ReadFile(child) == child_txt1);
}

// --- Single-layer regression ------------------------------------------------ //
//
// A one-layer in-place save (root_path == the sole layer's key) must write the
// whole model back as a <mujoco> root and reload equivalently.
void TestSingleLayerSave(const std::filesystem::path& dir) {
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  const std::filesystem::path solo = dir / "solo.xml";
  WriteFile(solo, kSoloXml);

  EditorContext ctx;
  std::string err;
  CHECK(LoadLayered(ctx, solo, &err));
  CHECK(ctx.layers.size() == 1);
  const std::string canonical = io::WriteMjcf(*ctx.tree);

  CHECK(SaveLayeredMjcf(ctx, solo.string(), &err));
  const std::string txt = ReadFile(solo);
  CHECK(StartsWithRootTag(txt, "mujoco"));
  CHECK(!StartsWithRootTag(txt, "mujocoinclude"));

  io::ParseResult re = io::ParseMjcfFile(solo.string());
  CHECK(re.ok());
  if (re.ok()) CHECK(io::WriteMjcf(*re.model) == canonical);

  std::string stock_err;
  CHECK(StockLoads(solo, &stock_err));
}

}  // namespace
}  // namespace ps::studio

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <scratch_dir>\n", argv[0]);
    return 2;
  }
  const std::filesystem::path dir(argv[1]);
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);

  ps::studio::TestLayeredSaveInvariant(dir / "layered");
  ps::studio::TestSingleLayerSave(dir / "solo");

  if (ps::studio::g_failures == 0) {
    std::printf("all checks passed\n");
    return 0;
  }
  std::fprintf(stderr, "%d check(s) failed\n", ps::studio::g_failures);
  return 1;
}
