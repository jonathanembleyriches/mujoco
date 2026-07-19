// ProtoSpec Python module: the top-level `protospec` extension.
//
// Ties the generated typed element classes (cpp/python/generated) to the
// hand-written surface the owner drives from Python:
//   * IO           load / loads / write / save
//   * validation   validate(model, tiers) -> list of diagnostic dicts
//   * compile      compile(model, base_dir) -> Compiled (mjModel + Binding +
//                  a minimal sim surface: nq/nv/..., step, forward, qpos, xpos)
//   * recompile    recompile(model, prev, keep_state) -> Compiled with state
//   * builders     Augment(Body/Frame/Replicate/Model): add_body / add_geom /
//                  add_joint / add_freejoint / ... and friendly child views
//
// This file is inside the MuJoCo quarantine (it includes mujoco.h) because the
// sim surface owns an mjData; the pure object model / IO / validation it calls
// stay MuJoCo-free.
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <mujoco/mujoco.h>

#include "ps_bind.h"
#include "generated/py_bind_gen.h"

#include "protospec/sdk.h"
#include "mjcf.h"       // ps::mjcf::io  (ParseMjcf*, WriteMjcf)
#include "compile.h"    // ps::mjcf (Compile, Recompile, CompileOptions)
#include "validate.h"   // ps::mjcf::validate (Validate)

namespace pyb = pybind11;

namespace ps::py {
namespace {

namespace io = ps::mjcf::io;
namespace validate = ps::mjcf::validate;

// Raised (as a ValueError subclass) when a document is well-formed but uses
// elements outside the supported families; lets a corpus sweep skip cleanly.
struct UnsupportedElementError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// --- Builders ------------------------------------------------------------- //

// Wrap a freshly built child as a live handle internal to `self`, apply any
// keyword arguments as attribute assignments (reusing the generated field
// properties), and return it.
template <class Child>
pyb::object FinishChild(pyb::object self, Child& child, const pyb::kwargs& kw) {
  pyb::object obj = pyb::cast(
      &child, pyb::return_value_policy::reference_internal, self);
  for (auto item : kw)
    obj.attr(item.first) = item.second;
  return obj;
}

// A read-only, type-filtered view over a body-context element's `subtree`.
template <class Parent, class T>
pyb::list SubtreeOf(pyb::object self) {
  Parent& p = self.cast<Parent&>();
  pyb::list out;
  for (auto& item : p.subtree)
    if (auto* pp = std::get_if<std::unique_ptr<T>>(&item.node))
      if (*pp)
        out.append(pyb::cast(pp->get(),
                            pyb::return_value_policy::reference_internal, self));
  return out;
}

// The add_* builders + filtered child views shared by Body / Frame / Replicate.
template <class Parent>
void BindBodyContext(pyb::class_<Parent>& c) {
  namespace sdk = ps::sdk;
  c.def(
      "add_body",
      [](pyb::object self, pyb::kwargs kw) {
        return FinishChild(self, sdk::AddBody(self.cast<Parent&>()), kw);
      },
      "Append a child body; pass fields as keywords (pos=[...], name=...).");
  c.def(
      "add_geom",
      [](pyb::object self, pyb::kwargs kw) {
        return FinishChild(self, sdk::AddGeom(self.cast<Parent&>()), kw);
      },
      "Append a geom (defaults to a sphere); type=\"box\", size=[...], ...");
  c.def(
      "add_joint",
      [](pyb::object self, pyb::kwargs kw) {
        return FinishChild(self, sdk::AddJoint(self.cast<Parent&>()), kw);
      },
      "Append a joint (defaults to a hinge); type=\"slide\", axis=[...], ...");
  c.def(
      "add_freejoint",
      [](pyb::object self, pyb::kwargs kw) {
        return FinishChild(self, sdk::AddFreeJoint(self.cast<Parent&>()), kw);
      },
      "Append a free joint.");
  c.def("add_site", [](pyb::object self, pyb::kwargs kw) {
    return FinishChild(self, sdk::AddSite(self.cast<Parent&>()), kw);
  });
  c.def("add_camera", [](pyb::object self, pyb::kwargs kw) {
    return FinishChild(self, sdk::AddCamera(self.cast<Parent&>()), kw);
  });
  c.def("add_light", [](pyb::object self, pyb::kwargs kw) {
    return FinishChild(self, sdk::AddLight(self.cast<Parent&>()), kw);
  });
  c.def("add_frame", [](pyb::object self, pyb::kwargs kw) {
    return FinishChild(self, sdk::AddFrame(self.cast<Parent&>()), kw);
  });
  c.def("add_inertial", [](pyb::object self, pyb::kwargs kw) {
    return FinishChild(self, sdk::AddInertial(self.cast<Parent&>()), kw);
  });

  c.def_property_readonly("bodies", &SubtreeOf<Parent, ps::mjcf::Body>);
  c.def_property_readonly("geoms", &SubtreeOf<Parent, ps::mjcf::Geom>);
  c.def_property_readonly("joints", &SubtreeOf<Parent, ps::mjcf::Joint>);
  c.def_property_readonly("sites", &SubtreeOf<Parent, ps::mjcf::Site>);
  c.def_property_readonly("frames", &SubtreeOf<Parent, ps::mjcf::Frame>);
  c.def_property_readonly("cameras", &SubtreeOf<Parent, ps::mjcf::Camera>);
  c.def_property_readonly("lights", &SubtreeOf<Parent, ps::mjcf::Light>);
}

}  // namespace

// --- Augment overloads (declared in ps_bind.h) ---------------------------- //

void Augment(pyb::class_<ps::mjcf::Body>& c) { BindBodyContext(c); }
void Augment(pyb::class_<ps::mjcf::Frame>& c) { BindBodyContext(c); }
void Augment(pyb::class_<ps::mjcf::Replicate>& c) { BindBodyContext(c); }

namespace {

// Dispatch add_actuator("motor"/"position"/...) to the typed SDK builder.
pyb::object AddActuatorByKind(pyb::object self, const std::string& kind,
                            const pyb::kwargs& kw) {
  namespace sdk = ps::sdk;
  ps::mjcf::Model& m = self.cast<ps::mjcf::Model&>();
#define PS_ACT(tag, T)                                            \
  if (kind == tag) return FinishChild(self, sdk::AddActuator<ps::mjcf::T>(m), kw)
  PS_ACT("motor", Motor);
  PS_ACT("position", Position);
  PS_ACT("velocity", Velocity);
  PS_ACT("intvelocity", IntVelocity);
  PS_ACT("damper", Damper);
  PS_ACT("cylinder", Cylinder);
  PS_ACT("muscle", Muscle);
  PS_ACT("adhesion", Adhesion);
  PS_ACT("general", ActuatorGeneral);
  PS_ACT("dcmotor", DcMotor);
#undef PS_ACT
  throw pyb::value_error("unknown actuator kind '" + kind +
                        "' (motor/position/velocity/intvelocity/damper/"
                        "cylinder/muscle/adhesion/general/dcmotor)");
}

}  // namespace

namespace {

// --- Compiled: mjModel + Binding + minimal sim surface -------------------- //

struct PyCompiled {
  ps::mjcf::Compiled c;
  mjData* d = nullptr;
  pyb::object model_keepalive;  // keeps the Python Model (and its tree) alive

  PyCompiled(ps::mjcf::Compiled&& compiled, mjData* data, pyb::object model)
      : c(std::move(compiled)), d(data), model_keepalive(std::move(model)) {}
  PyCompiled(const PyCompiled&) = delete;
  PyCompiled& operator=(const PyCompiled&) = delete;
  ~PyCompiled() {
    if (d) mj_deleteData(d);
  }

  mjModel* m() const { return c.model.get(); }
};

int ObjTypeFromString(const std::string& s) {
  int t = mju_str2Type(s.c_str());
  return t;  // mjOBJ_UNKNOWN (0) when unrecognized
}

pyb::array_t<double> ViewOf(pyb::object self, double* ptr, int n) {
  // A live, writable view over mjData memory; `self` (the PyCompiled) is the
  // base so the array cannot outlive the data it points into.
  return pyb::array_t<double>({static_cast<pyb::ssize_t>(n)},
                             {static_cast<pyb::ssize_t>(sizeof(double))}, ptr,
                             self);
}

// --- Binding wrapper ------------------------------------------------------ //

struct PyBinding {
  const ps::mjcf::Binding* b = nullptr;
  const mjModel* m = nullptr;

  // serial -> (objtype, id); built once from the binding entries.
  std::unordered_map<std::uint64_t, std::pair<int, int>> by_serial;

  void Build() {
    for (const auto& e : b->entries())
      by_serial[e.serial] = {e.objtype, e.id};
  }

  // (objtype, id) for a Python element handle, via its stable serial.
  const std::pair<int, int>* Lookup(pyb::handle elem) const {
    if (!pyb::hasattr(elem, "serial")) return nullptr;
    auto serial = elem.attr("serial").cast<std::uint64_t>();
    auto it = by_serial.find(serial);
    return it == by_serial.end() ? nullptr : &it->second;
  }

  pyb::object Id(pyb::handle elem) const {
    const auto* hit = Lookup(elem);
    if (!hit || hit->second < 0) return pyb::none();
    return pyb::int_(hit->second);
  }

  pyb::object AddrHelper(pyb::handle elem, int want_objtype, const int* table,
                        const char* what) const {
    const auto* hit = Lookup(elem);
    if (!hit || hit->second < 0 || hit->first != want_objtype) return pyb::none();
    (void)what;
    return pyb::int_(table[hit->second]);
  }

  pyb::object QposAdr(pyb::handle j) const {
    return AddrHelper(j, mjOBJ_JOINT, m->jnt_qposadr, "qpos_adr");
  }
  pyb::object DofAdr(pyb::handle j) const {
    return AddrHelper(j, mjOBJ_JOINT, m->jnt_dofadr, "dof_adr");
  }
  pyb::object ActId(pyb::handle a) const {
    const auto* hit = Lookup(a);
    if (!hit || hit->second < 0 || hit->first != mjOBJ_ACTUATOR)
      return pyb::none();
    return pyb::int_(hit->second);
  }
  pyb::object SensorAdr(pyb::handle s) const {
    return AddrHelper(s, mjOBJ_SENSOR, m->sensor_adr, "sensor_adr");
  }

  std::vector<int> Find(const std::string& type, const std::string& glob) const {
    int objtype = ObjTypeFromString(type);
    return b->Find(objtype, glob);
  }
};

// --- IO ------------------------------------------------------------------- //

[[noreturn]] void RaiseParse(const io::ParseResult& r) {
  std::string msg;
  for (const auto& e : r.errors) {
    if (!msg.empty()) msg += "\n";
    msg += e.Render();
  }
  if (r.unsupported_only())
    throw UnsupportedElementError(msg.empty() ? "unsupported elements" : msg);
  throw pyb::value_error(msg.empty() ? "parse failed" : msg);
}

pyb::object ModelFromResult(io::ParseResult&& r) {
  if (!r.ok()) RaiseParse(r);
  return pyb::cast(std::move(r.model));  // pybind adopts the unique_ptr
}

pyb::object LoadFile(const std::string& path) {
  return ModelFromResult(io::ParseMjcfFile(path));
}
pyb::object LoadString(const std::string& xml) {
  return ModelFromResult(io::ParseMjcfString(xml));
}

// Auto-detect: an XML document (contains a '<') is parsed as a string; anything
// else is treated as a path.
pyb::object Load(const std::string& path_or_xml) {
  if (path_or_xml.find('<') != std::string::npos) return LoadString(path_or_xml);
  return LoadFile(path_or_xml);
}

std::string Write(pyb::handle model) {
  return io::WriteMjcf(model.cast<const ps::mjcf::Model&>());
}
void Save(pyb::handle model, const std::string& path) {
  std::string xml = Write(model);
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) throw pyb::value_error("could not open '" + path + "' for writing");
  std::fwrite(xml.data(), 1, xml.size(), f);
  std::fclose(f);
}

// --- Validation ----------------------------------------------------------- //

const char* TierName(validate::Tier t) {
  switch (t) {
    case validate::Tier::Structural: return "structural";
    case validate::Tier::Referential: return "referential";
    case validate::Tier::Semantic: return "semantic";
    case validate::Tier::None: break;
  }
  return "";
}

pyb::dict DiagToDict(const ps::Diagnostic& d) {
  pyb::dict out;
  out["tier"] = static_cast<int>(d.tier);
  out["tier_name"] = TierName(d.tier);
  out["severity"] =
      d.severity == validate::Severity::Error ? "error" : "warning";
  out["message"] = d.message;
  out["file"] = d.loc.file;
  out["line"] = d.loc.line;
  out["path"] = d.tag;  // validator element path rides in the shared `tag` field
  return out;
}

pyb::list Validate(pyb::handle model, pyb::object tiers) {
  const ps::mjcf::Model& m = model.cast<const ps::mjcf::Model&>();
  validate::TierMask mask = validate::kAllTiers;
  if (!tiers.is_none()) {
    mask = 0;
    // Accept an int bitmask, an int tier (1/2/3), or an iterable of tiers.
    auto add_tier = [&](int t) {
      if (t == 1) mask |= validate::kTierStructural;
      else if (t == 2) mask |= validate::kTierReferential;
      else if (t == 3) mask |= validate::kTierSemantic;
      else mask |= static_cast<validate::TierMask>(t);  // already a bitmask
    };
    if (pyb::isinstance<pyb::int_>(tiers)) {
      add_tier(tiers.cast<int>());
    } else {
      for (auto item : tiers) add_tier(item.cast<int>());
    }
  }
  pyb::list out;
  for (const auto& d : validate::Validate(m, mask)) out.append(DiagToDict(d));
  return out;
}

// --- Compile / recompile -------------------------------------------------- //

ps::mjcf::CompileOptions MakeOptions(pyb::object base_dir) {
  ps::mjcf::CompileOptions opts;
  if (!base_dir.is_none()) opts.base_dir = base_dir.cast<std::string>();
  return opts;
}

pyb::dict ReportToDict(const ps::mjcf::CompileReport& r) {
  pyb::dict out;
  out["ok"] = r.ok();
  out["path_taken"] =
      r.taken == ps::mjcf::CompilePath::XmlPath ? "xml"
      : r.taken == ps::mjcf::CompilePath::NativePath ? "native" : "auto";
  pyb::list errs, warns;
  for (const auto& e : r.errors) errs.append(e.Render());
  for (const auto& w : r.warnings) warns.append(w.Render());
  out["errors"] = errs;
  out["warnings"] = warns;
  return out;
}

std::unique_ptr<PyCompiled> CompileModel(pyb::object model, pyb::object base_dir) {
  ps::mjcf::Model& m = model.cast<ps::mjcf::Model&>();
  ps::mjcf::Compiled c = ps::mjcf::Compile(m, MakeOptions(base_dir));
  if (!c.ok()) {
    std::string msg;
    for (const auto& e : c.report.errors) {
      if (!msg.empty()) msg += "\n";
      msg += e.Render();
    }
    throw pyb::value_error(msg.empty() ? "compile failed" : msg);
  }
  mjData* d = mj_makeData(c.model.get());
  if (!d) throw pyb::value_error("mj_makeData failed on the compiled model");
  mj_forward(c.model.get(), d);
  return std::make_unique<PyCompiled>(std::move(c), d, std::move(model));
}

std::unique_ptr<PyCompiled> RecompileModel(pyb::object model, PyCompiled& prev,
                                     bool keep_state, pyb::object base_dir) {
  ps::mjcf::Model& m = model.cast<ps::mjcf::Model&>();
  ps::mjcf::CompileOptions opts = MakeOptions(base_dir);
  if (keep_state) {
    mjData* nd = nullptr;
    ps::mjcf::Compiled c = ps::mjcf::Recompile(m, prev.c, prev.d, &nd, opts);
    if (!c.ok()) throw pyb::value_error("recompile failed");
    if (!nd) nd = mj_makeData(c.model.get());
    mj_forward(c.model.get(), nd);
    return std::make_unique<PyCompiled>(std::move(c), nd, std::move(model));
  }
  return CompileModel(std::move(model), base_dir);
}

int ResolveBody(const mjModel* m, pyb::handle body) {
  if (pyb::isinstance<pyb::int_>(body)) return body.cast<int>();
  if (pyb::isinstance<pyb::str>(body))
    return mj_name2id(m, mjOBJ_BODY, body.cast<std::string>().c_str());
  // A bound element handle: use its serial through... fall back to name attr.
  if (pyb::hasattr(body, "name")) {
    pyb::object n = body.attr("name");
    if (!n.is_none())
      return mj_name2id(m, mjOBJ_BODY, n.cast<std::string>().c_str());
  }
  return -1;
}

}  // namespace

// Model builders + convenience methods. Defined here (after the IO / validate /
// compile helpers) so the methods can delegate straight to them.
void Augment(pyb::class_<ps::mjcf::Model>& c) {
  c.def_property_readonly(
      "worldbody",
      [](pyb::object self) {
        ps::mjcf::Body& w = ps::sdk::World(self.cast<ps::mjcf::Model&>());
        return pyb::cast(&w, pyb::return_value_policy::reference_internal, self);
      },
      "The single world body (created on first access).");
  c.def(
      "add_body",
      [](pyb::object self, pyb::kwargs kw) {
        ps::mjcf::Body& w = ps::sdk::World(self.cast<ps::mjcf::Model&>());
        pyb::object world =
            pyb::cast(&w, pyb::return_value_policy::reference_internal, self);
        return FinishChild(world, ps::sdk::AddBody(w), kw);
      },
      "Append a top-level body under the world body.");
  c.def(
      "add_actuator",
      [](pyb::object self, const std::string& kind, pyb::kwargs kw) {
        return AddActuatorByKind(self, kind, kw);
      },
      pyb::arg("kind") = "motor",
      "Append an actuator of the given spelling; joint=..., kp=..., ...");

  // Convenience methods mirroring the module-level functions.
  c.def("to_xml", [](pyb::handle self) { return Write(self); },
        "Serialize this model to a canonical MJCF string.");
  c.def("save", [](pyb::handle self, const std::string& path) { Save(self, path); },
        pyb::arg("path"), "Write this model to an MJCF file.");
  c.def(
      "validate",
      [](pyb::handle self, pyb::object tiers) { return Validate(self, tiers); },
      pyb::arg("tiers") = pyb::none(),
      "Validate this model; returns a list of diagnostic dicts.");
  c.def(
      "compile",
      [](pyb::object self, pyb::object base_dir) {
        return CompileModel(self, base_dir);
      },
      pyb::arg("base_dir") = pyb::none(),
      "Compile this model to an mjModel + Binding + sim surface.");
}

}  // namespace ps::py

// --------------------------------------------------------------------------- //
// Module definition                                                            //
// --------------------------------------------------------------------------- //
PYBIND11_MODULE(protospec, m) {
  using namespace ps::py;
  namespace psm = ps::mjcf;

  m.doc() =
      "ProtoSpec: a clean, IDL-generated object model for MuJoCo models.\n"
      "load()/loads() an MJCF file or string, edit typed elements, write()/\n"
      "save() it back, validate() it, or compile() it to a stepping model.";

  pyb::register_exception<UnsupportedElementError>(m, "UnsupportedElement",
                                                  PyExc_ValueError);

  // Generated typed element + struct classes (structs first: variant arms).
  RegisterGenerated(m);

  // --- Compiled (sim surface) --------------------------------------------- //
  pyb::class_<PyCompiled>(m, "Compiled")
      .def_property_readonly("ok", [](PyCompiled& c) { return c.c.ok(); })
      .def_property_readonly("report",
                             [](PyCompiled& c) { return ReportToDict(c.c.report); })
      .def_property_readonly("nq", [](PyCompiled& c) { return c.m()->nq; })
      .def_property_readonly("nv", [](PyCompiled& c) { return c.m()->nv; })
      .def_property_readonly("na", [](PyCompiled& c) { return c.m()->na; })
      .def_property_readonly("nu", [](PyCompiled& c) { return c.m()->nu; })
      .def_property_readonly("nbody", [](PyCompiled& c) { return c.m()->nbody; })
      .def_property_readonly("ngeom", [](PyCompiled& c) { return c.m()->ngeom; })
      .def_property_readonly("njnt", [](PyCompiled& c) { return c.m()->njnt; })
      .def_property_readonly("nsensor",
                             [](PyCompiled& c) { return c.m()->nsensor; })
      .def(
          "step",
          [](PyCompiled& c, int n) {
            for (int i = 0; i < n; ++i) mj_step(c.m(), c.d);
          },
          pyb::arg("n") = 1, "Advance the simulation by n steps.")
      .def("forward", [](PyCompiled& c) { mj_forward(c.m(), c.d); },
           "Recompute derived quantities without integrating time.")
      .def("reset", [](PyCompiled& c) { mj_resetData(c.m(), c.d); },
           "Reset mjData to the model's initial state.")
      .def_property(
          "time", [](PyCompiled& c) { return c.d->time; },
          [](PyCompiled& c, double t) { c.d->time = t; })
      .def_property_readonly(
          "qpos",
          [](pyb::object self) {
            PyCompiled& c = self.cast<PyCompiled&>();
            return ViewOf(self, c.d->qpos, c.m()->nq);
          },
          "Live, writable view of mjData.qpos.")
      .def_property_readonly(
          "qvel",
          [](pyb::object self) {
            PyCompiled& c = self.cast<PyCompiled&>();
            return ViewOf(self, c.d->qvel, c.m()->nv);
          })
      .def_property_readonly(
          "ctrl",
          [](pyb::object self) {
            PyCompiled& c = self.cast<PyCompiled&>();
            return ViewOf(self, c.d->ctrl, c.m()->nu);
          })
      .def(
          "xpos",
          [](PyCompiled& c, pyb::handle body) {
            int id = ResolveBody(c.m(), body);
            if (id < 0 || id >= c.m()->nbody)
              throw pyb::value_error("no such body");
            const double* p = c.d->xpos + 3 * id;
            return pyb::make_tuple(p[0], p[1], p[2]);
          },
          "World position (x, y, z) of a body, by id or name.")
      .def_property_readonly(
          "binding",
          [](pyb::object self) {
            PyCompiled& c = self.cast<PyCompiled&>();
            auto b = std::make_unique<PyBinding>();
            b->b = &c.c.binding;
            b->m = c.m();
            b->Build();
            return pyb::cast(std::move(b),
                            pyb::return_value_policy::take_ownership, self);
          },
          "The name-based Binding (element identity -> compiled id).");

  // --- Binding ------------------------------------------------------------ //
  pyb::class_<PyBinding>(m, "Binding")
      .def("id", &PyBinding::Id, pyb::arg("elem"),
           "Compiled mjModel id of a tree element (None if unbound).")
      .def("qpos_adr", &PyBinding::QposAdr, pyb::arg("joint"))
      .def("dof_adr", &PyBinding::DofAdr, pyb::arg("joint"))
      .def("act_id", &PyBinding::ActId, pyb::arg("actuator"))
      .def("sensor_adr", &PyBinding::SensorAdr, pyb::arg("sensor"))
      .def("find", &PyBinding::Find, pyb::arg("type"), pyb::arg("glob"),
           "mjModel ids whose name matches a glob, for an object type "
           "(\"body\", \"geom\", \"joint\", ...).");

  // --- Module functions --------------------------------------------------- //
  m.def("load", &Load, pyb::arg("path_or_xml"),
        "Parse an MJCF file (a path) or document (a string) into a Model.");
  m.def("loads", &LoadString, pyb::arg("xml"),
        "Parse an MJCF document string into a Model.");
  m.def("load_file", &LoadFile, pyb::arg("path"));
  m.def("write", &Write, pyb::arg("model"),
        "Serialize a Model to a canonical MJCF string.");
  m.def("save", &Save, pyb::arg("model"), pyb::arg("path"),
        "Write a Model to an MJCF file.");
  m.def("validate", &Validate, pyb::arg("model"), pyb::arg("tiers") = pyb::none(),
        "Validate a Model; returns a list of diagnostic dicts "
        "(tier/severity/message/file/line/path).");
  m.def("compile", &ps::py::CompileModel, pyb::arg("model"),
        pyb::arg("base_dir") = pyb::none(),
        "Compile a Model to an mjModel + Binding + sim surface.");
  m.def("recompile", &ps::py::RecompileModel, pyb::arg("model"), pyb::arg("prev"),
        pyb::arg("keep_state") = true, pyb::arg("base_dir") = pyb::none(),
        "Recompile after a structural edit, migrating simulation state.");

  m.attr("__version__") = "0.0.1";
}
