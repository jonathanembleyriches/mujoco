// Default-class read-time checks (family c, Q-AUTO context).
//
// ProtoSpec treats defaults as data: classes are read verbatim into the Default
// tree, written back verbatim, and NOTHING is resolved or applied (DR-1). The
// only rules enforced at read time are the two structural ones MuJoCo enforces
// in mjXReader::Default (xml_native_reader.cc:3034-3056), because they gate a
// well-formed <default> tree and MuJoCo rejects violators before compile:
//
//   * a top-level default class must be unnamed or exactly "main"
//     (:3052-3055, "top-level default class 'main' cannot be renamed");
//   * a nested default class name must be non-empty
//     (:3041-3044, "empty class name").
//
// Class-reference resolution (does a referenced class exist?) is deliberately
// NOT done here: it is referential validation (plan Section 9 tier 2, DR-8),
// consistent with how every other ref<T> in the reader is stored by name and
// resolved later. See test_io.cc TestUnknownClassRef for the grounding.
#ifndef PROTOSPEC_IO_DEFAULTS_H
#define PROTOSPEC_IO_DEFAULTS_H

#include <vector>

#include "mjcf.h"
#include "types.h"

namespace ps::mjcf::io {

// Append a MalformedInput diagnostic for each default-tree class-name
// violation, each carrying the offending <default>'s SourceLoc.
void ValidateDefaultClasses(const Model& model,
                            std::vector<ps::Diagnostic>& errors);

}  // namespace ps::mjcf::io

#endif  // PROTOSPEC_IO_DEFAULTS_H
