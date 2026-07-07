// MJCF IO tests: fixpoint round-trips plus a quirk-by-quirk battery over
// embedded snippets and the matching negative cases. The reader/writer are
// exercised end to end; the differential-against-MuJoCo dimension belongs to the
// harness (cpp/harness/, separate owner).

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <variant>

#include "keywords.h"
#include "mjcf.h"
#include "numeric.h"
#include "types.h"

using namespace ps::mjcf;
using namespace ps::mjcf::io;

static int g_failed = 0;
static int g_checks = 0;

#define CHECK(cond)                                                    \
  do {                                                                 \
    ++g_checks;                                                        \
    if (!(cond)) {                                                     \
      ++g_failed;                                                      \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);      \
    }                                                                  \
  } while (0)

static bool Near(double a, double b) { return std::fabs(a - b) < 1e-9; }

static ParseResult Parse(const std::string& xml) {
  return ParseMjcfString(xml, "<test>");
}

// The world Body of a parsed model (worldbody[0]).
static const Body* World(const Model& m) {
  return m.worldbody.empty() ? nullptr : m.worldbody.front().get();
}

// --- fixpoint: Read(Write(Read(x))) == Read(x); Write is deterministic ----- //
static void TestFixpoint() {
  const std::string xml = R"(<mujoco model="rig">
  <compiler angle="degree" eulerseq="xyz" meshdir="meshes"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast">
    <flag contact="disable" energy="enable"/>
  </option>
  <size memory="1M" nkey="2"/>
  <visual>
    <global fovy="50" offwidth="640"/>
    <map force="0.01"/>
    <rgba haze="1 1 1 1"/>
  </visual>
  <statistic meansize="0.05" extent="2"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="10 10 0.1" rgba="0.8 0.8 0.8 1"/>
    <body name="torso" pos="0 0 1" euler="90 0 0">
      <inertial pos="0 0 0" mass="5" diaginertia="0.1 0.1 0.1"/>
      <joint name="hinge" type="hinge" axis="0 1 0" range="-90 90" ref="10"/>
      <geom name="cap" type="capsule" fromto="0 0 0 0 0 0.5" size="0.05"
            friction="1 0.5"/>
      <site name="s0" pos="0 0 0.5" quat="1 0 0 0"/>
      <camera name="eye" pos="0 -1 0" fovy="45"/>
      <body name="link" pos="0 0 0.5">
        <joint name="slide" type="slide" axis="0 0 1" range="0 0.3"/>
        <geom name="g2" type="sphere" size="0.03"/>
        <frame name="f0" pos="0 0 0.1">
          <geom name="g3" type="box" size="0.02 0.02 0.02"/>
        </frame>
      </body>
    </body>
  </worldbody>
</mujoco>)";

  ParseResult a = Parse(xml);
  CHECK(a.ok());
  if (!a.ok()) {
    for (const auto& e : a.errors) std::printf("  err: %s\n", e.Render().c_str());
    return;
  }
  std::string out1 = WriteMjcf(*a.model);
  ParseResult b = ParseMjcfString(out1, "<rt>");
  CHECK(b.ok());
  if (!b.ok()) {
    for (const auto& e : b.errors) std::printf("  err: %s\n", e.Render().c_str());
    return;
  }
  // Read(Write(Read(x))) == Read(x) under generated deep equality.
  CHECK(*a.model == *b.model);
  // Write is deterministic: Write(Read(Write(Read(x)))) is byte-identical.
  std::string out2 = WriteMjcf(*b.model);
  CHECK(out1 == out2);
}

// --- Q-ANGLE: values and compiler unit are preserved verbatim -------------- //
// MuJoCo converts joint range only for LIMITED hinge/ball and ref/springref
// only for hinge, resolving those conditions per consuming element at compile
// (user_objects.cc:3207-3282); a shared default class can feed elements that
// resolve them differently, so no read-time conversion is correct. ProtoSpec
// therefore stores every angle exactly as authored and round trips the compiler
// unit, deferring the conversion to MuJoCo.
static void TestAngle() {
  auto r = Parse(R"(<mujoco>
    <compiler angle="degree"/>
    <worldbody>
      <body name="b" euler="90 0 0">
        <joint name="h" type="hinge" range="-90 180" ref="90" springref="45"/>
        <joint name="s" type="slide" range="0 2" ref="1"/>
      </body>
    </worldbody>
  </mujoco>)");
  CHECK(r.ok());
  const Body* b = World(*r.model)->bodies.front().get();
  // The body carries the euler orientation exactly as authored (degrees here).
  CHECK(b->orient.has_value());
  const auto& eu = std::get<Euler>(*b->orient);
  CHECK(Near(eu.angles[0], 90.0));
  CHECK(Near(eu.angles[1], 0.0) && Near(eu.angles[2], 0.0));

  const Joint& h = *b->joints[0];
  CHECK(Near((*h.range)[0], -90.0) && Near((*h.range)[1], 180.0));
  CHECK(Near(*h.ref, 90.0));
  CHECK(Near(*h.springref, 45.0));
  const Joint& s = *b->joints[1];
  CHECK(Near((*s.range)[0], 0.0) && Near((*s.range)[1], 2.0));
  CHECK(Near(*s.ref, 1.0));

  // The authored compiler unit round trips verbatim: no pinning to radian.
  std::string out = WriteMjcf(*r.model);
  CHECK(out.find("angle=\"degree\"") != std::string::npos);
  CHECK(out.find("angle=\"radian\"") == std::string::npos);

  // A radian-authored model likewise keeps its unit and values verbatim.
  auto rad = Parse(R"(<mujoco>
    <compiler angle="radian"/>
    <worldbody><body euler="1.5707963 0 0"/></worldbody>
  </mujoco>)");
  CHECK(rad.ok());
  const auto& eu2 = std::get<Euler>(*World(*rad.model)->bodies.front()->orient);
  CHECK(Near(eu2.angles[0], 1.5707963));
  std::string rad_out = WriteMjcf(*rad.model);
  CHECK(rad_out.find("angle=\"radian\"") != std::string::npos);
}

// --- Q-ORIENT: each encoding + multiple-specifier and zero-quat errors ----- //
static void TestOrient() {
  auto ok = Parse(R"(<mujoco><worldbody>
    <geom name="g" type="sphere" size="1" axisangle="0 0 1 90"/>
  </worldbody></mujoco>)");
  CHECK(ok.ok());
  const auto& g = *World(*ok.model)->geoms.front();
  CHECK(g.orient.has_value());
  const auto& aa = std::get<AxisAngle>(*g.orient);
  CHECK(Near(aa.axis[2], 1.0) && Near(aa.angle, 90.0));  // authored, verbatim

  auto multi = Parse(R"(<mujoco><worldbody>
    <geom type="sphere" size="1" quat="1 0 0 0" euler="0 0 0"/>
  </worldbody></mujoco>)");
  CHECK(!multi.ok());
  CHECK(multi.errors[0].message.find("multiple orientation") != std::string::npos);
  CHECK(multi.errors[0].loc.line > 0);

  auto zero = Parse(R"(<mujoco><worldbody>
    <geom type="sphere" size="1" quat="0 0 0 0"/>
  </worldbody></mujoco>)");
  CHECK(!zero.ok());
  CHECK(zero.errors[0].message.find("zero quaternion") != std::string::npos);
}

// --- Q-FROMTO: routes into GeomShape; size stays a plain field ------------- //
static void TestFromto() {
  auto r = Parse(R"(<mujoco><worldbody>
    <geom name="c" type="capsule" fromto="0 0 0 0 0 1" size="0.1"/>
    <geom name="s" type="sphere" size="0.2"/>
  </worldbody></mujoco>)");
  CHECK(r.ok());
  const auto& c = *World(*r.model)->geoms[0];
  CHECK(c.shape.has_value() && std::holds_alternative<FromTo>(*c.shape));
  CHECK(Near(std::get<FromTo>(*c.shape).fromto[5], 1.0));
  CHECK(c.size.has_value() && Near((*c.size)[0], 0.1));
  // No fromto -> shape unset (implicit explicit form), size present.
  const auto& s = *World(*r.model)->geoms[1];
  CHECK(!s.shape.has_value());
  CHECK(s.size.has_value() && Near((*s.size)[0], 0.2));
}

// --- Q-INERTIA: full vs diagonal; fullinertia + orientation exclusion ------ //
static void TestInertia() {
  auto full = Parse(R"(<mujoco><worldbody><body>
    <inertial pos="0 0 0" mass="1" fullinertia="1 2 3 0.1 0.2 0.3"/>
  </body></worldbody></mujoco>)");
  CHECK(full.ok());
  const auto& body = *World(*full.model)->bodies.front();
  CHECK(!body.inertial.empty());
  const auto& in = *body.inertial.front();
  CHECK(in.inertia.has_value() && std::holds_alternative<FullInertia>(*in.inertia));

  auto both = Parse(R"(<mujoco><worldbody><body>
    <inertial pos="0 0 0" mass="1" fullinertia="1 2 3 0 0 0" euler="0 0 0"/>
  </body></worldbody></mujoco>)");
  CHECK(!both.ok());
  CHECK(both.errors[0].message.find("fullinertia and inertial orientation") !=
        std::string::npos);
}

// --- Q-ARITY: fewer-than-max OK, more-than-max errors ---------------------- //
static void TestArity() {
  auto r = Parse(R"(<mujoco><worldbody>
    <geom name="g" type="sphere" size="1" friction="1.5"/>
  </worldbody></mujoco>)");
  CHECK(r.ok());
  const auto& g = *World(*r.model)->geoms.front();
  CHECK(g.friction.has_value() && g.friction->size() == 1);  // filled count = 1
  CHECK(Near((*g.friction)[0], 1.5));

  auto too_many = Parse(R"(<mujoco><worldbody>
    <geom type="sphere" size="1" friction="1 2 3 4"/>
  </worldbody></mujoco>)");
  CHECK(!too_many.ok());
  CHECK(too_many.errors[0].message.find("too much data") != std::string::npos);
}

// --- Q-NUM: inf, memory suffix, overflow, NaN warning ---------------------- //
static void TestNumeric() {
  auto r = Parse(R"(<mujoco>
    <size memory="2M"/>
    <worldbody><body>
      <joint name="j" type="hinge" armature="inf"/>
    </body></worldbody>
  </mujoco>)");
  CHECK(r.ok());
  CHECK(r.model->sizes.front()->memory.has_value());
  CHECK(*r.model->sizes.front()->memory == std::to_string(2ull << 20));
  const auto& j = *World(*r.model)->bodies.front()->joints.front();
  CHECK(j.armature.has_value() && std::isinf(*j.armature));

  // Integer overflow is an error.
  auto over = Parse(R"(<mujoco><size nkey="99999999999999999999"/></mujoco>)");
  CHECK(!over.ok());
  CHECK(over.errors[0].message.find("too large") != std::string::npos);

  // NaN parses (MuJoCo parity) but raises a non-fatal warning.
  auto nan = Parse(R"(<mujoco><worldbody><body>
    <joint type="hinge" armature="nan"/>
  </body></worldbody></mujoco>)");
  CHECK(nan.ok());
  CHECK(!nan.warnings.empty());

  // memory canonical-byte parsing in isolation.
  std::uint64_t bytes = 0;
  CHECK(num::ParseMemory("1K", bytes) == num::MemStatus::Ok && bytes == 1024);
  CHECK(num::ParseMemory("-1", bytes) == num::MemStatus::Unset);
  CHECK(num::ParseMemory("1Q", bytes) == num::MemStatus::Bad);
}

// --- presence (DR-1): only authored attributes set fields ------------------ //
static void TestPresence() {
  auto r = Parse(R"(<mujoco><worldbody>
    <geom name="g" type="box" size="1 1 1"/>
  </worldbody></mujoco>)");
  CHECK(r.ok());
  const auto& g = *World(*r.model)->geoms.front();
  CHECK(g.type.has_value() && *g.type == GeomType::box);
  CHECK(g.size.has_value());
  CHECK(!g.mass.has_value());       // unauthored -> unset (no default written)
  CHECK(!g.density.has_value());
  CHECK(!g.rgba.has_value());
  CHECK(!g.contype.has_value());
  // The writer emits only authored attributes.
  std::string out = WriteMjcf(*r.model);
  CHECK(out.find("mass=") == std::string::npos);
  CHECK(out.find("density=") == std::string::npos);
  CHECK(out.find("type=\"box\"") != std::string::npos);
}

// --- unknown attribute / unknown element: malformed (not unsupported) ------ //
static void TestMalformed() {
  auto attr = Parse(R"(<mujoco><worldbody>
    <geom type="sphere" size="1" bogus="3"/>
  </worldbody></mujoco>)");
  CHECK(!attr.ok());
  CHECK(!attr.unsupported_only());
  CHECK(attr.errors[0].kind == Diagnostic::Kind::MalformedInput);
  CHECK(attr.errors[0].message.find("unknown attribute 'bogus'") !=
        std::string::npos);
  CHECK(attr.errors[0].loc.line > 0);

  auto elem = Parse(R"(<mujoco><worldbody>
    <notathing/>
  </worldbody></mujoco>)");
  CHECK(!elem.ok());
  CHECK(!elem.unsupported_only());
  CHECK(elem.errors[0].message.find("unknown element 'notathing'") !=
        std::string::npos);
}

// --- unsupported element: distinct, machine-detectable skip signal --------- //
static void TestUnsupported() {
  auto r = Parse(R"(<mujoco>
    <worldbody><geom type="sphere" size="1"/></worldbody>
    <sensor/>
  </mujoco>)");
  CHECK(!r.ok());
  CHECK(r.unsupported_only());
  bool found = false;
  for (const auto& e : r.errors) {
    if (e.kind == Diagnostic::Kind::UnsupportedElement && e.element == "sensor")
      found = true;
  }
  CHECK(found);

  // An unsupported child of a supported element also signals skip (geom plugin).
  auto plug = Parse(R"(<mujoco><worldbody>
    <geom type="sphere" size="1"><plugin plugin="x"/></geom>
  </worldbody></mujoco>)");
  CHECK(!plug.ok());
  CHECK(plug.unsupported_only());
}

// --- root must be <mujoco> ------------------------------------------------- //
static void TestRoot() {
  auto r = Parse("<notmujoco/>");
  CHECK(!r.ok());
  CHECK(r.errors[0].message.find("root element must be <mujoco>") !=
        std::string::npos);

  auto bad = Parse("<mujoco><worldbody></mujoco>");  // malformed XML
  CHECK(!bad.ok());
}

// --- Defaults (family c): classes are DATA, read verbatim, resolve nothing - //
static void TestDefaults() {
  const std::string xml = R"(<mujoco>
  <default>
    <geom rgba="1 0 0 1" friction="1 0.1 0.1"/>
    <joint damping="0.5"/>
    <position kp="100" ctrlrange="-1 1"/>
    <pair friction="1 1 0.01 0.01 0.01"/>
    <equality solref="0.02 1"/>
    <mesh scale="2 2 2"/>
    <material specular="0.5"/>
    <default class="sub">
      <geom rgba="0 1 0 1"/>
      <default class="leaf">
        <joint damping="2"/>
      </default>
    </default>
  </default>
  <worldbody>
    <geom type="sphere" size="1" class="sub"/>
  </worldbody>
</mujoco>)";
  ParseResult r = Parse(xml);
  CHECK(r.ok());
  if (!r.ok()) {
    for (const auto& e : r.errors) std::printf("  err: %s\n", e.Render().c_str());
    return;
  }
  CHECK(r.model->defaults.size() == 1);
  const Default& root = *r.model->defaults.front();
  // The root class is unnamed (implicitly "main"); sub-elements are stored as
  // partial specs with only their authored fields present.
  CHECK(!root.dclass.has_value());
  CHECK(root.geom.size() == 1 && root.geom.front()->rgba.has_value());
  CHECK(root.joint.size() == 1 &&
        Near((*root.joint.front()->damping)[0], 0.5));
  CHECK(root.position.size() == 1 && Near(*root.position.front()->kp, 100));
  CHECK(root.pair.size() == 1);
  CHECK(root.equality.size() == 1);
  CHECK(root.mesh.size() == 1);
  CHECK(root.material.size() == 1);
  // Nested class tree preserved verbatim; nothing resolved into the geom.
  CHECK(root.subclasses.size() == 1);
  const Default& sub = *root.subclasses.front();
  CHECK(sub.dclass.has_value() && *sub.dclass == "sub");
  CHECK(sub.subclasses.size() == 1 && *sub.subclasses.front()->dclass == "leaf");
  // The worldbody geom keeps its class ref by name; no default is applied.
  const Geom& g = *World(*r.model)->geoms.front();
  CHECK(g.dclass.has_value() && g.dclass->name == "sub");
  CHECK(!g.rgba.has_value());  // class value is NOT merged into the element

  // Fixpoint: the whole default tree round trips under deep equality.
  std::string out1 = WriteMjcf(*r.model);
  ParseResult b = ParseMjcfString(out1, "<rt>");
  CHECK(b.ok());
  CHECK(*r.model == *b.model);
  CHECK(out1 == WriteMjcf(*b.model));
}

// --- Q-AUTO/defaults: root class name rules (xml_native_reader.cc:3041-3055) //
static void TestDefaultClassNames() {
  // Top level may be unnamed or exactly "main".
  CHECK(Parse(R"(<mujoco><default><geom size="1"/></default></mujoco>)").ok());
  CHECK(Parse(R"(<mujoco><default class="main"/></mujoco>)").ok());

  auto renamed = Parse(R"(<mujoco><default class="root"/></mujoco>)");
  CHECK(!renamed.ok());
  CHECK(renamed.errors[0].message.find(
            "top-level default class 'main' cannot be renamed") !=
        std::string::npos);
  CHECK(renamed.errors[0].loc.line > 0);

  // A nested default must name a non-empty class.
  auto empty = Parse(R"(<mujoco><default>
    <default><geom size="1"/></default>
  </default></mujoco>)");
  CHECK(!empty.ok());
  CHECK(empty.errors[0].message.find("empty class name") != std::string::npos);
}

// --- Unknown class reference: NOT a read-time error (deferred to tier 2) ---- //
// MuJoCo resolves classes during parse and errors immediately on a dangling
// name (xml_native_reader.cc:4705-4719). ProtoSpec instead stores every ref by
// name and resolves none at read (DR-8, plan Section 9 tier 2), exactly as it
// treats mesh/material/target refs -- referential validation, not IO, reports
// dangling names with provenance later. This is harness-neutral: a genuinely
// dangling class makes MuJoCo reject the model at compile on both sides.
static void TestUnknownClassRef() {
  auto r = Parse(R"(<mujoco><worldbody>
    <geom type="sphere" size="1" class="does_not_exist"/>
  </worldbody></mujoco>)");
  CHECK(r.ok());
  const Geom& g = *World(*r.model)->geoms.front();
  CHECK(g.dclass.has_value() && g.dclass->name == "does_not_exist");
}

// --- Assets (family d): mesh/hfield/material/skin read as data -------------- //
static void TestAssets() {
  const std::string xml = R"(<mujoco>
  <asset>
    <mesh name="m" file="parts/arm.obj" scale="0.1 0.1 0.1" refpos="0 0 1"/>
    <mesh name="verts" vertex="0 0 0 1 0 0 0 1 0" face="0 1 2"/>
    <hfield name="h" file="terrain.png" nrow="4" ncol="4" size="1 1 1 0.1"/>
    <material name="mat" specular="0.3" rgba="1 1 1 1">
      <layer texture="t_rough" role="rgb"/>
    </material>
    <skin name="s" file="skin.skn" inflate="0.01"/>
    <model name="nested" file="submodel.xml"/>
  </asset>
  <worldbody>
    <geom type="mesh" mesh="m"/>
  </worldbody>
</mujoco>)";
  ParseResult r = Parse(xml);
  CHECK(r.ok());
  if (!r.ok()) {
    for (const auto& e : r.errors) std::printf("  err: %s\n", e.Render().c_str());
    return;
  }
  CHECK(r.model->assets.size() == 1);
  const Asset& a = *r.model->assets.front();
  CHECK(a.meshs.size() == 2);
  const Mesh& m0 = *a.meshs.front();
  // Asset file paths are DATA: stored verbatim, contents never loaded (DR-7).
  CHECK(m0.file.has_value() && *m0.file == "parts/arm.obj");
  CHECK(m0.scale.has_value() && Near((*m0.scale)[0], 0.1));
  CHECK(a.meshs[1]->vertex.has_value() && a.meshs[1]->vertex->size() == 9);
  CHECK(a.hfields.size() == 1 && *a.hfields.front()->nrow == 4);
  CHECK(a.materials.size() == 1);
  const Material& mat = *a.materials.front();
  CHECK(mat.layers.size() == 1 && mat.layers.front()->role.has_value());
  CHECK(a.skins.size() == 1 && *a.skins.front()->file == "skin.skn");
  CHECK(a.modelAssets.size() == 1 && *a.modelAssets.front()->file == "submodel.xml");

  std::string out1 = WriteMjcf(*r.model);
  ParseResult b = ParseMjcfString(out1, "<rt>");
  CHECK(b.ok());
  CHECK(*r.model == *b.model);
  CHECK(out1 == WriteMjcf(*b.model));
}

// --- Q-TEX: the texture source variant (builtin vs file) ------------------- //
static void TestTextureSource() {
  auto tex = Parse(R"(<mujoco><asset>
    <texture name="grid" type="2d" builtin="checker" width="8" height="8"/>
    <texture name="img" type="2d" file="wood.png"/>
  </asset></mujoco>)");
  CHECK(tex.ok());
  const Asset& a = *tex.model->assets.front();
  CHECK(a.textures.size() == 2);
  const Texture& t0 = *a.textures[0];
  CHECK(t0.source.has_value() &&
        std::holds_alternative<TextureBuiltin>(*t0.source));
  CHECK(std::get<TextureBuiltin>(*t0.source) == TextureBuiltin::checker);
  const Texture& t1 = *a.textures[1];
  CHECK(t1.source.has_value() && std::holds_alternative<TexFile>(*t1.source));
  CHECK(std::get<TexFile>(*t1.source).file == "wood.png");

  std::string out = WriteMjcf(*tex.model);
  CHECK(out.find("builtin=\"checker\"") != std::string::npos);
  CHECK(out.find("file=\"wood.png\"") != std::string::npos);
  // A file wins over a redundant builtin="none" (single-variant modeling).
  auto both = Parse(R"(<mujoco><asset>
    <texture name="x" type="2d" builtin="none" file="a.png"/>
  </asset></mujoco>)");
  CHECK(both.ok());
  CHECK(std::holds_alternative<TexFile>(*both.model->assets.front()
                                             ->textures.front()->source));
}

// --- Include (Q-INC): splice-in-place, provenance, once-per-file globally --- //
namespace {
int g_tmp_counter = 0;
std::filesystem::path TempDir() {
  auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
  std::filesystem::path d =
      std::filesystem::temp_directory_path() /
      ("ps_io_inc_" + std::to_string(stamp) + "_" +
       std::to_string(++g_tmp_counter));
  std::filesystem::remove_all(d);
  std::filesystem::create_directories(d);
  return d;
}
void WriteText(const std::filesystem::path& p, const std::string& s) {
  std::filesystem::create_directories(p.parent_path());
  std::ofstream(p) << s;
}
}  // namespace

static void TestInclude() {
  std::filesystem::path dir = TempDir();
  WriteText(dir / "main.xml", R"(<mujoco model="m">
  <include file="defs.xml"/>
  <worldbody>
    <include file="sub/body.xml"/>
    <geom name="floor" type="plane" size="1 1 1"/>
  </worldbody>
</mujoco>)");
  WriteText(dir / "defs.xml", R"(<mujoco>
  <default><default class="vis"><geom rgba="1 0 0 1"/></default></default>
</mujoco>)");
  WriteText(dir / "sub" / "body.xml", R"(<mujocoinclude>
  <body name="b" pos="0 0 1">
    <geom name="g" type="sphere" size="0.1" class="vis"/>
  </body>
</mujocoinclude>)");

  ParseResult r = ParseMjcfFile((dir / "main.xml").string());
  CHECK(r.ok());
  if (!r.ok()) {
    for (const auto& e : r.errors) std::printf("  err: %s\n", e.Render().c_str());
  } else {
    // The <default> from defs.xml and the <body> from sub/body.xml are spliced
    // in place; the include elements vanish (flattened tree).
    CHECK(r.model->defaults.size() == 1);
    CHECK(r.model->defaults.front()->subclasses.size() == 1);
    const Body* w = World(*r.model);
    CHECK(w->geoms.size() == 1 && *w->geoms.front()->name == "floor");
    CHECK(w->bodies.size() == 1 && *w->bodies.front()->name == "b");
    // Provenance (DR-9): the spliced body's SourceLoc names the INCLUDED file.
    const Body& b = *w->bodies.front();
    CHECK(b.loc.file.find("body.xml") != std::string::npos);
    CHECK(b.loc.line == 2);
  }
  std::filesystem::remove_all(dir);
}

static void TestIncludeErrors() {
  // Self-include is caught by the once-per-file rule (seeded with the top file).
  {
    std::filesystem::path dir = TempDir();
    WriteText(dir / "c.xml", R"(<mujoco><include file="c.xml"/></mujoco>)");
    ParseResult r = ParseMjcfFile((dir / "c.xml").string());
    CHECK(!r.ok());
    CHECK(r.errors[0].message.find("already included") != std::string::npos);
    std::filesystem::remove_all(dir);
  }
  // A duplicate include of the same file anywhere is rejected globally.
  {
    std::filesystem::path dir = TempDir();
    WriteText(dir / "m.xml", R"(<mujoco>
      <include file="a.xml"/><include file="a.xml"/>
    </mujoco>)");
    WriteText(dir / "a.xml", R"(<mujoco><compiler/></mujoco>)");
    ParseResult r = ParseMjcfFile((dir / "m.xml").string());
    CHECK(!r.ok());
    CHECK(r.errors[0].message.find("already included") != std::string::npos);
    std::filesystem::remove_all(dir);
  }
  // Missing file attribute.
  {
    std::filesystem::path dir = TempDir();
    WriteText(dir / "m.xml", R"(<mujoco><include/></mujoco>)");
    ParseResult r = ParseMjcfFile((dir / "m.xml").string());
    CHECK(!r.ok());
    CHECK(r.errors[0].message.find("missing file attribute") !=
          std::string::npos);
    std::filesystem::remove_all(dir);
  }
  // Include element with children.
  {
    std::filesystem::path dir = TempDir();
    WriteText(dir / "m.xml",
              R"(<mujoco><include file="a.xml"><geom/></include></mujoco>)");
    WriteText(dir / "a.xml", R"(<mujoco><compiler/></mujoco>)");
    ParseResult r = ParseMjcfFile((dir / "m.xml").string());
    CHECK(!r.ok());
    CHECK(r.errors[0].message.find("cannot have children") != std::string::npos);
    std::filesystem::remove_all(dir);
  }
  // Unresolvable file.
  {
    std::filesystem::path dir = TempDir();
    WriteText(dir / "m.xml", R"(<mujoco><include file="nope.xml"/></mujoco>)");
    ParseResult r = ParseMjcfFile((dir / "m.xml").string());
    CHECK(!r.ok());
    CHECK(r.errors[0].message.find("cannot open") != std::string::npos);
    std::filesystem::remove_all(dir);
  }
}

// Fetch a union member pointer of type M from an item, or nullptr.
template <class M, class U>
static const M* Member(const U& item) {
  auto* p = std::get_if<std::unique_ptr<M>>(&item.node);
  return p ? p->get() : nullptr;
}

// Round-trip a whole model under deep equality + deterministic output.
static void Fixpoint(const ParseResult& a) {
  CHECK(a.ok());
  if (!a.ok()) {
    for (const auto& e : a.errors) std::printf("  err: %s\n", e.Render().c_str());
    return;
  }
  std::string out1 = WriteMjcf(*a.model);
  ParseResult b = ParseMjcfString(out1, "<rt>");
  CHECK(b.ok());
  if (!b.ok()) {
    for (const auto& e : b.errors) std::printf("  err: %s\n", e.Render().c_str());
    return;
  }
  CHECK(*a.model == *b.model);
  CHECK(out1 == WriteMjcf(*b.model));
}

// --- Contact (family e): pair 5-value friction + solreffriction, exclude ---- //
static void TestContact() {
  auto r = Parse(R"(<mujoco>
    <contact>
      <pair name="p" geom1="a" geom2="b" condim="6"
            friction="1 1 0.01 0.02 0.03" solreffriction="0.01 1" gap="0.001"/>
      <exclude name="e" body1="t" body2="l"/>
    </contact>
  </mujoco>)");
  CHECK(r.ok());
  if (!r.ok()) {
    for (const auto& e : r.errors) std::printf("  err: %s\n", e.Render().c_str());
    return;
  }
  const Contact& c = *r.model->contacts.front();
  CHECK(c.pairs.size() == 1 && c.excludes.size() == 1);
  const Pair& p = *c.pairs.front();
  CHECK(*p.condim == 6);
  CHECK(p.friction.has_value() && p.friction->size() == 5);
  CHECK(Near((*p.friction)[4], 0.03));
  CHECK(p.solreffriction.has_value() && p.solreffriction->size() == 2);
  CHECK(p.geom1->name == "a" && p.geom2->name == "b");
  CHECK(c.excludes.front()->body1->name == "t");
  Fixpoint(r);

  // 3-value pair friction (Q-ARITY, fewer than the 5 max) round trips.
  auto few = Parse(R"(<mujoco><contact>
    <pair geom1="a" geom2="b" friction="1 1 0.01"/>
  </contact></mujoco>)");
  CHECK(few.ok());
  CHECK(few.model->contacts.front()->pairs.front()->friction->size() == 3);
  Fixpoint(few);
}

// --- Equality (family e): every spelling, document order, alt attr sets ----- //
static void TestEquality() {
  auto r = Parse(R"(<mujoco>
    <equality>
      <connect name="cb" body1="b1" body2="b2" anchor="0 0 1"/>
      <weld name="w" body1="b1" relpose="0 0 0 1 0 0 0" torquescale="2"/>
      <joint name="ej" joint1="j1" joint2="j2" polycoef="0 1 0 0 0"/>
      <tendon name="et" tendon1="t1" tendon2="t2"/>
      <connect name="cs" site1="s1" site2="s2"/>
      <flex name="ef" flex="f1"/>
      <flexvert name="fv" flex="f1"/>
      <flexstrain name="fs" flex="f1" cell="0 1 2"/>
    </equality>
  </mujoco>)");
  CHECK(r.ok());
  if (!r.ok()) {
    for (const auto& e : r.errors) std::printf("  err: %s\n", e.Render().c_str());
    return;
  }
  const Equality& eq = *r.model->equalitys.front();
  CHECK(eq.equalities.size() == 8);
  // Union child list keeps document order across spellings (Section 6).
  using K = EqualityAny::Kind;
  CHECK(eq.equalities[0].kind() == K::Connect);
  CHECK(eq.equalities[1].kind() == K::Weld);
  CHECK(eq.equalities[2].kind() == K::EqualityJoint);
  CHECK(eq.equalities[3].kind() == K::EqualityTendon);
  CHECK(eq.equalities[4].kind() == K::Connect);
  CHECK(eq.equalities[5].kind() == K::EqualityFlex);
  CHECK(eq.equalities[6].kind() == K::Flexvert);
  CHECK(eq.equalities[7].kind() == K::Flexstrain);
  // Body-form connect keeps anchor+bodies; site-form connect keeps sites.
  const Connect* cb = Member<Connect>(eq.equalities[0]);
  CHECK(cb && cb->anchor.has_value() && cb->body1->name == "b1");
  CHECK(!cb->site1.has_value());
  const Connect* cs = Member<Connect>(eq.equalities[4]);
  CHECK(cs && cs->site1->name == "s1" && !cs->anchor.has_value());
  const Weld* w = Member<Weld>(eq.equalities[1]);
  CHECK(w && Near(*w->torquescale, 2.0) && w->relpose.has_value());
  const Flexstrain* fs = Member<Flexstrain>(eq.equalities[7]);
  CHECK(fs && fs->cell.has_value() && Near((*fs->cell)[2], 2.0));
  Fixpoint(r);
}

static void TestEqualityNegatives() {
  // connect: body and site semantics cannot be mixed.
  auto mixed = Parse(R"(<mujoco><equality>
    <connect body1="b" anchor="0 0 0" site1="s"/>
  </equality></mujoco>)");
  CHECK(!mixed.ok());
  CHECK(mixed.errors[0].message.find("cannot be mixed") != std::string::npos);

  // connect: neither complete body form nor complete site form.
  auto neither = Parse(R"(<mujoco><equality>
    <connect body1="b"/>
  </equality></mujoco>)");
  CHECK(!neither.ok());
  CHECK(neither.errors[0].message.find(
            "either both body1 and anchor") != std::string::npos);

  // weld: site1 without site2 is not a complete site form (and no body form).
  auto weld = Parse(R"(<mujoco><equality>
    <weld site1="s"/>
  </equality></mujoco>)");
  CHECK(!weld.ok());
  CHECK(weld.errors[0].message.find("body1 must be defined") !=
        std::string::npos);
}

// --- Tendon (family e): spatial path interleave (THE ordering test) + fixed - //
static void TestTendon() {
  auto r = Parse(R"(<mujoco>
    <tendon>
      <spatial name="sp" width="0.005" limited="true" range="0 1"
               stiffness="10" damping="0.1" springlength="0.2">
        <site site="s0"/>
        <geom geom="g0" sidesite="sd"/>
        <pulley divisor="2"/>
        <site site="s1"/>
      </spatial>
      <fixed name="fx">
        <joint joint="j0" coef="1"/>
        <joint joint="j1" coef="-1"/>
      </fixed>
    </tendon>
  </mujoco>)");
  CHECK(r.ok());
  if (!r.ok()) {
    for (const auto& e : r.errors) std::printf("  err: %s\n", e.Render().c_str());
    return;
  }
  const Tendon& t = *r.model->tendons.front();
  CHECK(t.tendons.size() == 2);
  CHECK(t.tendons[0].kind() == TendonAny::Kind::Spatial);
  CHECK(t.tendons[1].kind() == TendonAny::Kind::Fixed);
  const Spatial* sp = Member<Spatial>(t.tendons[0]);
  CHECK(sp && sp->path.size() == 4);
  // The interleaved site/geom/pulley order is preserved exactly (the schema was
  // redesigned around this: a per-type list would lose site-geom-pulley-site).
  using PK = PathItemAny::Kind;
  CHECK(sp->path[0].kind() == PK::SpatialSite);
  CHECK(sp->path[1].kind() == PK::SpatialGeom);
  CHECK(sp->path[2].kind() == PK::Pulley);
  CHECK(sp->path[3].kind() == PK::SpatialSite);
  CHECK(Member<SpatialGeom>(sp->path[1])->sidesite->name == "sd");
  CHECK(Near(Member<Pulley>(sp->path[2])->divisor.value(), 2.0));
  CHECK(sp->stiffness->size() == 1 && Near((*sp->stiffness)[0], 10.0));
  const Fixed* fx = Member<Fixed>(t.tendons[1]);
  CHECK(fx && fx->fixedJoints.size() == 2);
  CHECK(fx->fixedJoints[0]->joint->name == "j0" &&
        Near(*fx->fixedJoints[0]->coef, 1.0));
  Fixpoint(r);
}

// --- Actuators (family f): id order, typed-stays-typed, params, plugin ------ //
static void TestActuators() {
  auto r = Parse(R"(<mujoco>
    <actuator>
      <motor name="m" joint="j0" gear="2 0 0 0 0 0" ctrllimited="true"
             ctrlrange="-1 1"/>
      <position name="p" joint="j0" kp="100" kv="5" inheritrange="1"/>
      <general name="g" joint="j1" dyntype="filter" gaintype="fixed"
               biastype="affine" gainprm="35" biasprm="0 -35 0" actdim="1"/>
      <velocity name="v" joint="j1" kv="7"/>
      <intvelocity name="iv" joint="j0" kp="3" actrange="-2 2"/>
      <damper name="d" joint="j1" kv="4" ctrlrange="0 1"/>
      <cylinder name="cy" joint="j0" timeconst="0.1" area="2" bias="1 0 0"/>
      <muscle name="mu" joint="j1" timeconst="0.01 0.04" force="500" scale="200"/>
      <adhesion name="ad" body="b0" gain="1" ctrlrange="0 1"/>
      <dcmotor name="dc" joint="j0" motorconst="0.5 0" resistance="1.2"
               controller="1 2 3 4 5 6" thermal="1 2 3 4 5 6" input="velocity"/>
      <plugin name="pl" joint="j1" plugin="mujoco.pid" actdim="1">
        <config key="kp" value="10"/>
        <config key="ki" value="1"/>
      </plugin>
    </actuator>
  </mujoco>)");
  CHECK(r.ok());
  if (!r.ok()) {
    for (const auto& e : r.errors) std::printf("  err: %s\n", e.Render().c_str());
    return;
  }
  const Actuator& a = *r.model->actuators.front();
  CHECK(a.actuators.size() == 11);
  // Document order == compile id order; each spelling stays its own type
  // (DR-3/Q-ACT: no lowering to <general> in IO).
  using K = ActuatorAny::Kind;
  const K expect[] = {K::Motor,       K::Position, K::ActuatorGeneral,
                      K::Velocity,    K::IntVelocity, K::Damper,
                      K::Cylinder,    K::Muscle,   K::Adhesion,
                      K::DcMotor,     K::ActuatorPlugin};
  for (int i = 0; i < 11; ++i) CHECK(a.actuators[i].kind() == expect[i]);

  const Motor* m = Member<Motor>(a.actuators[0]);
  CHECK(m && m->joint->name == "j0" && m->gear->size() == 6);
  CHECK(*m->ctrllimited == TriState::true_);  // tri-state stays a tri-state
  const Position* p = Member<Position>(a.actuators[1]);
  CHECK(p && Near(*p->kp, 100) && Near(*p->kv, 5) && Near(*p->inheritrange, 1));
  const ActuatorGeneral* g = Member<ActuatorGeneral>(a.actuators[2]);
  CHECK(g && *g->dyntype == DynType::filter && *g->biastype == BiasType::affine);
  CHECK(g->gainprm->size() == 1 && g->biasprm->size() == 3);
  const Muscle* mu = Member<Muscle>(a.actuators[7]);
  CHECK(mu && mu->timeconst->size() == 2 && Near(*mu->force, 500));
  const DcMotor* dc = Member<DcMotor>(a.actuators[9]);
  CHECK(dc && dc->motorconst->size() == 2 && dc->controller->size() == 6 &&
        dc->thermal->size() == 6 && *dc->input == DcMotorInput::velocity);
  const ActuatorPlugin* pl = Member<ActuatorPlugin>(a.actuators[10]);
  CHECK(pl && *pl->plugin == "mujoco.pid" && pl->config.size() == 2);
  CHECK(*pl->config[0]->key == "kp" && *pl->config[1]->value == "1");
  Fixpoint(r);
}

static void TestActuatorNegatives() {
  // At most one transmission target.
  auto two = Parse(R"(<mujoco><actuator>
    <motor joint="j0" site="s0"/>
  </actuator></mujoco>)");
  CHECK(!two.ok());
  CHECK(two.errors[0].message.find("at most one of transmission") !=
        std::string::npos);

  // refsite requires a site transmission.
  auto refsite = Parse(R"(<mujoco><actuator>
    <position joint="j0" refsite="r"/>
  </actuator></mujoco>)");
  CHECK(!refsite.ok());
  CHECK(refsite.errors[0].message.find("refsite can only be used") !=
        std::string::npos);

  // slidersite requires a slidercrank transmission.
  auto slider = Parse(R"(<mujoco><actuator>
    <motor joint="j0" slidersite="s"/>
  </actuator></mujoco>)");
  CHECK(!slider.ok());
  CHECK(slider.errors[0].message.find("slidercrank transmission") !=
        std::string::npos);
}

// --- Union interleave preserved through Write->Read (both directions) ------- //
static void TestUnionOrderRoundTrip() {
  // A spelling sequence with repeats; the written MJCF must re-read to the same
  // ordered union, and re-writing must be byte-identical.
  auto r = Parse(R"(<mujoco><actuator>
    <position name="a" joint="j"/>
    <motor name="b" joint="j"/>
    <position name="c" joint="j"/>
    <velocity name="d" joint="j"/>
    <motor name="e" joint="j"/>
  </actuator></mujoco>)");
  CHECK(r.ok());
  const auto& list = r.model->actuators.front()->actuators;
  CHECK(list.size() == 5);
  std::string out1 = WriteMjcf(*r.model);
  // The five spellings appear in the written order motor/position after position.
  CHECK(out1.find("name=\"a\"") < out1.find("name=\"b\""));
  CHECK(out1.find("name=\"b\"") < out1.find("name=\"c\""));
  ParseResult b = ParseMjcfString(out1, "<rt>");
  CHECK(b.ok());
  const auto& list2 = b.model->actuators.front()->actuators;
  CHECK(list2.size() == 5);
  for (std::size_t i = 0; i < list.size(); ++i)
    CHECK(list[i].kind() == list2[i].kind());
  CHECK(out1 == WriteMjcf(*b.model));
}

int main() {
  TestFixpoint();
  TestAngle();
  TestOrient();
  TestFromto();
  TestInertia();
  TestArity();
  TestNumeric();
  TestPresence();
  TestMalformed();
  TestUnsupported();
  TestRoot();
  TestDefaults();
  TestDefaultClassNames();
  TestUnknownClassRef();
  TestAssets();
  TestTextureSource();
  TestInclude();
  TestIncludeErrors();
  TestContact();
  TestEquality();
  TestEqualityNegatives();
  TestTendon();
  TestActuators();
  TestActuatorNegatives();
  TestUnionOrderRoundTrip();

  std::printf("%d checks, %d failures\n", g_checks, g_failed);
  return g_failed == 0 ? 0 : 1;
}
