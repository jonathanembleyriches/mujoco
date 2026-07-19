#include "include.h"

#include <cstring>
#include <filesystem>
#include <string>
#include <unordered_set>
#include <vector>

#include "tinyxml2.h"

namespace ps::mjcf::io {
namespace {

namespace fs = std::filesystem;

using tinyxml2::XMLDocument;
using tinyxml2::XMLElement;
using tinyxml2::XMLNode;

bool IsInclude(const XMLElement* e) {
  return e != nullptr && std::strcmp(e->Value(), "include") == 0;
}

// Chained includes recurse once per file, so nesting depth is bounded only by
// this cap; without it a long enough chain exhausts the native stack. 64 is far
// beyond any real model (the once-per-file rule already forces every level to
// be a distinct file).
constexpr int kMaxIncludeDepth = 64;

// The parent directory of a path, as authored (empty stays empty).
std::string ParentDir(const std::string& path) {
  if (path.empty()) return "";
  fs::path p(path);
  return p.parent_path().string();
}

// A stable key for the once-per-file rule: the absolute, lexically normal path
// when it exists on disk, else the path as given.
std::string Canonical(const std::string& path) {
  std::error_code ec;
  fs::path c = fs::weakly_canonical(fs::path(path), ec);
  if (ec) return path;
  return c.string();
}

class Expander {
 public:
  Expander(XMLDocument& doc, std::string model_dir, std::string model_file)
      : doc_(doc),
        model_dir_(std::move(model_dir)),
        model_file_(std::move(model_file)) {
    included_.insert(Canonical(model_file_));
  }

  IncludeResult Run(XMLElement* root) {
    ProcessChildren(root, model_dir_);
    return {std::move(provenance_), std::move(errors_)};
  }

 private:
  void Err(const XMLElement* e, std::string msg) {
    ps::Diagnostic d;
    d.source = "parse";
    d.kind = ps::Diagnostic::Kind::MalformedInput;
    d.message = std::move(msg);
    d.loc = ps::SourceLoc{model_file_loc(e), e->GetLineNum()};
    errors_.push_back(std::move(d));
  }

  // The source file for an include error: an <include> spliced from a nested
  // file carries that file's path via the provenance map; a top-level <include>
  // is absent from the map and falls back to model_file_.
  std::string model_file_loc(const XMLElement* e) const {
    auto it = provenance_.find(e);
    return it == provenance_.end() ? model_file_ : it->second.file;
  }

  void ProcessChildren(XMLElement* parent, const std::string& search_dir) {
    XMLElement* child = parent->FirstChildElement();
    while (child) {
      XMLElement* next = child->NextSiblingElement();
      if (IsInclude(child)) {
        HandleInclude(child, parent, search_dir);
      } else {
        ProcessChildren(child, search_dir);
      }
      child = next;
    }
  }

  // Resolve a relative include file against the model dir first, then the
  // including file's dir (xml.cc:154-171). Absolute paths are used verbatim.
  bool Resolve(const std::string& file, const std::string& search_dir,
               std::string& out) const {
    fs::path f(file);
    if (f.is_absolute()) {
      std::error_code ec;
      if (fs::exists(f, ec)) {
        out = f.string();
        return true;
      }
      return false;
    }
    for (const std::string* base : {&model_dir_, &search_dir}) {
      fs::path cand = base->empty() ? f : (fs::path(*base) / f);
      std::error_code ec;
      if (fs::exists(cand, ec)) {
        out = cand.string();
        return true;
      }
    }
    return false;
  }

  void HandleInclude(XMLElement* inc, XMLElement* parent,
                     const std::string& search_dir) {
    if (inc->FirstChildElement() != nullptr) {
      Err(inc, "include element cannot have children");
      return;
    }
    const char* file = inc->Attribute("file");
    if (!file) {
      Err(inc, "include element missing file attribute");
      return;
    }
    if (depth_ >= kMaxIncludeDepth) {
      Err(inc, "include depth exceeds " + std::to_string(kMaxIncludeDepth) +
                   " at file '" + std::string(file) + "'");
      return;
    }

    std::string resolved;
    if (!Resolve(file, search_dir, resolved)) {
      Err(inc, "cannot open included file '" + std::string(file) + "'");
      return;
    }
    if (!included_.insert(Canonical(resolved)).second) {
      Err(inc, "file '" + std::string(file) + "' already included");
      return;
    }

    XMLDocument idoc;
    if (idoc.LoadFile(resolved.c_str()) != tinyxml2::XML_SUCCESS) {
      const char* what = idoc.ErrorStr();
      Err(inc, "error reading included file '" + std::string(file) +
                   "': " + (what ? what : "parse error"));
      return;
    }
    XMLElement* iroot = idoc.RootElement();
    if (!iroot) {
      Err(inc, "root element missing in included file '" + std::string(file) +
                   "'");
      return;
    }
    XMLElement* first = iroot->FirstChildElement();
    if (!first) {
      Err(inc, "empty include file '" + std::string(file) + "'");
      return;
    }

    // Splice the included root's children where the <include> sat, cloning into
    // the main document and recording per-element provenance from the included
    // file (whose parse line numbers are correct; the clones' are not).
    const std::string idir = ParentDir(resolved);
    std::vector<XMLElement*> spliced;
    XMLNode* after = inc;
    for (XMLElement* src = first; src; src = src->NextSiblingElement()) {
      XMLNode* clone = src->DeepClone(&doc_);
      after = parent->InsertAfterChild(after, clone);
      XMLElement* clone_elem = clone->ToElement();
      RecordProvenance(src, clone_elem, resolved);
      spliced.push_back(clone_elem);
    }
    parent->DeleteChild(inc);

    // Nested includes inside the spliced content resolve against the included
    // file's own directory. Everything reached from here is one include level
    // deeper.
    ++depth_;
    for (XMLElement* s : spliced) {
      if (IsInclude(s)) {
        HandleInclude(s, parent, idir);
      } else {
        ProcessChildren(s, idir);
      }
    }
    --depth_;
  }

  // Walk the source and clone subtrees in lockstep (DeepClone preserves
  // structure) so every spliced element gets the included file's path and its
  // own line number.
  void RecordProvenance(const XMLElement* src, XMLElement* clone,
                        const std::string& file) {
    provenance_[clone] = ps::SourceLoc{file, src->GetLineNum()};
    const XMLElement* sc = src->FirstChildElement();
    XMLElement* cc = clone->FirstChildElement();
    while (sc && cc) {
      RecordProvenance(sc, cc, file);
      sc = sc->NextSiblingElement();
      cc = cc->NextSiblingElement();
    }
  }

  XMLDocument& doc_;
  std::string model_dir_;
  std::string model_file_;
  int depth_ = 0;
  std::unordered_set<std::string> included_;
  ProvenanceMap provenance_;
  std::vector<ps::Diagnostic> errors_;
};

}  // namespace

IncludeResult ExpandIncludes(XMLDocument& doc, const std::string& model_dir,
                             const std::string& model_file) {
  XMLElement* root = doc.RootElement();
  if (!root) return {};
  Expander expander(doc, model_dir, model_file);
  return expander.Run(root);
}

}  // namespace ps::mjcf::io
