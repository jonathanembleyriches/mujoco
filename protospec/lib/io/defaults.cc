#include "defaults.h"

#include <string>

namespace ps::mjcf::io {
namespace {

void Err(std::vector<ps::Diagnostic>& errors, const Default& d, std::string msg) {
  ps::Diagnostic diag;
  diag.source = "parse";
  diag.kind = ps::Diagnostic::Kind::MalformedInput;
  diag.message = std::move(msg);
  diag.loc = d.loc;
  errors.push_back(std::move(diag));
}

// Recurse into subclasses, where every default must name a non-empty class.
void CheckNested(const Default& d, std::vector<ps::Diagnostic>& errors) {
  for (const auto& sub : d.subclasses) {
    if (!sub->dclass || sub->dclass->empty()) {
      Err(errors, *sub, "empty class name");
    }
    CheckNested(*sub, errors);
  }
}

}  // namespace

void ValidateDefaultClasses(const Model& model,
                            std::vector<ps::Diagnostic>& errors) {
  for (const auto& d : model.defaults) {
    // Top level: unnamed or exactly "main" (xml_native_reader.cc:3052-3055).
    if (d->dclass && !d->dclass->empty() && *d->dclass != "main") {
      Err(errors, *d, "top-level default class 'main' cannot be renamed");
    }
    CheckNested(*d, errors);
  }
}

}  // namespace ps::mjcf::io
