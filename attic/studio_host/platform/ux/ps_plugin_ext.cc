// ProtoSpec Studio plugin registry: instantiations for the extension plugin
// types (ps::studio, ours). See registry.inc.h for the mechanism.

#include "platform/ux/ps_plugin_ext.h"
#include "platform/ux/registry.inc.h"

PS_STUDIO_INSTANTIATE_PLUGIN(ps::studio::ViewportPlugin);
PS_STUDIO_INSTANTIATE_PLUGIN(ps::studio::ViewportGuiPlugin);
PS_STUDIO_INSTANTIATE_PLUGIN(ps::studio::OverlayPlugin);
PS_STUDIO_INSTANTIATE_PLUGIN(ps::studio::ModelSourcePlugin);
