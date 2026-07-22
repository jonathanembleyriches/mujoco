// ProtoSpec SDK: the ergonomic authoring layer over the plain-data object model.
//
// The generated types are plain owned values (DR-2) with presence-tracked
// fields (DR-1) and typed references stored by name (DR-8). This SDK adds the
// convenience that makes them pleasant to build and edit, entirely on top of the
// generated Visit/reflect hooks -- it never needs regenerating when the schema
// grows. Nothing here touches MuJoCo; it is a pure tree library.
//
//   builders.h   typed Add* helpers into the right child/union list
//   traversal.h  Find, ForEach*, ParentMap, path-to-element
//   refs.h       Resolve, FindReferrers, Rename, DeleteRecursive
//   classes.h    Effective (query) + FlattenDefaults / ExtractClass (mutating)
//   attach.h     namespaced deep-clone splice
#ifndef PROTOSPEC_SDK_H
#define PROTOSPEC_SDK_H

#include "protospec/attach.h"
#include "protospec/builders.h"
#include "protospec/classes.h"
#include "protospec/refs.h"
#include "protospec/traversal.h"

#endif  // PROTOSPEC_SDK_H
