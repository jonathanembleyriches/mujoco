// ProtoSpec Studio plugin registry: instantiations for the Studio-compatible
// core plugin types (ps::studio, ours). See registry.inc.h for the mechanism.

#include "platform/ux/plugin.h"
#include "platform/ux/registry.inc.h"

PS_STUDIO_INSTANTIATE_PLUGIN(ps::studio::GuiPlugin);
PS_STUDIO_INSTANTIATE_PLUGIN(ps::studio::ModelPlugin);
PS_STUDIO_INSTANTIATE_PLUGIN(ps::studio::KeyHandlerPlugin);
PS_STUDIO_INSTANTIATE_PLUGIN(ps::studio::SpecEditorPlugin);
