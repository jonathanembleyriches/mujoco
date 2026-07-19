// ProtoSpec Studio curated default dock layout (ps::studio, ours): plugin-side.
//
// A Unity/Unreal-style default: Hierarchy left, Viewport centre (the host's
// passthrough central node), Details right with Layers beneath it, and
// Assets / Diagnostics / Profiler as tabs along the bottom. The host's own
// Options / Explorer / Inspector / Editor / Properties panels are docked into
// the same nodes (so they land somewhere sane, tabbed behind the curated
// panels, whenever they are shown) rather than floating.
//
// HOW this works with ZERO host patches: the host builds its stock layout in
// ConfigureDockingLayout only when the "Root" dockspace node does not yet
// exist (its first_time check). This service pre-empts that -- its first
// ModelPlugin::do_update of the run fires after ImGui::NewFrame (so dock nodes
// restored from the saved .ini are already materialized) and before the host's
// BuildGui ever runs. If, and only if, no "Root" node exists, it builds the
// curated layout under that same id; the host then sees an existing node and
// skips its stock layout. A saved layout therefore wins exactly as it does
// upstream, and the host's DockSpace() submission and font-rescale
// (RescaleDock) keep operating on the very same node.
//
// ID note: ImGui::GetID("Root") here and at the host's call sites
// (ConfigureDockingLayout, RescaleDock) all execute with no window Begin
// active, i.e. against the fallback Debug##Default window's ID seed, so the
// three ids agree by construction.

#include "editor/plugins.h"
#include "imgui.h"
#include "imgui_internal.h"
#include "platform/ux/plugin.h"

namespace ps::studio {
namespace {

struct DockLayoutService {
  enum class Phase { kCurate, kFocus, kDone };
  Phase phase = Phase::kCurate;
  int focus_attempts = 0;
};

// Makes `name` the selected tab of its dock node. Returns false while the
// window has not yet been absorbed into a node (dock requests queued by the
// builder are applied during the window's Begin, i.e. after this plugin's
// do_update slot), so the caller retries on a later frame.
bool SelectDockedTab(const char* name) {
  ImGuiWindow* window = ImGui::FindWindowByName(name);
  if (!window || !window->DockNode || !window->DockNode->TabBar) {
    return false;
  }
  window->DockNode->TabBar->NextSelectedTabId = window->TabId;
  return true;
}

void BuildCuratedLayout(ImGuiID root) {
  // Match the host's stock sizing convention (viewport work area; the toolbar /
  // status bar are off by default, and DockSpace() re-fits the node each frame
  // anyway).
  const ImGuiViewport* viewport = ImGui::GetMainViewport();

  ImGui::DockBuilderAddNode(root, ImGuiDockNodeFlags_DockSpace);
  ImGui::DockBuilderSetNodeSize(root, viewport->WorkSize);

  ImGuiID main = root;

  ImGuiID hierarchy = 0;
  ImGui::DockBuilderSplitNode(main, ImGuiDir_Left, 0.20f, &hierarchy, &main);

  ImGuiID details = 0;
  ImGui::DockBuilderSplitNode(main, ImGuiDir_Right, 0.24f, &details, &main);

  // Right column splits: Details on top, Layers beneath it.
  ImGuiID layers = 0;
  ImGui::DockBuilderSplitNode(details, ImGuiDir_Down, 0.40f, &layers, &details);

  ImGuiID bottom = 0;
  ImGui::DockBuilderSplitNode(main, ImGuiDir_Down, 0.26f, &bottom, &main);

  // Centre (main) is the passthrough viewport; "Dockspace" is the host's
  // full-workspace dummy window (mirrors the stock layout's assignment).
  ImGui::DockBuilderDockWindow("Dockspace", main);
  // Left: Hierarchy, with the host Options panel tabbed behind it.
  ImGui::DockBuilderDockWindow("Hierarchy", hierarchy);
  ImGui::DockBuilderDockWindow("Options", hierarchy);
  // Right: Details, with the host spec/data panels and the ProtoSpec utility
  // panels (File / + Add) tabbed behind it -- reachable, never floating.
  ImGui::DockBuilderDockWindow("Details", details);
  ImGui::DockBuilderDockWindow("Inspector", details);
  ImGui::DockBuilderDockWindow("Explorer", details);
  ImGui::DockBuilderDockWindow("Editor", details);
  ImGui::DockBuilderDockWindow("Properties", details);
  ImGui::DockBuilderDockWindow("File", details);
  ImGui::DockBuilderDockWindow("+ Add", details);
  ImGui::DockBuilderDockWindow("Layers", layers);
  // Bottom: Assets + Diagnostics tabs (Profiler joins them when shown).
  ImGui::DockBuilderDockWindow("Assets", bottom);
  ImGui::DockBuilderDockWindow("Diagnostics", bottom);
  ImGui::DockBuilderDockWindow("Profiler", bottom);
  ImGui::DockBuilderFinish(root);
}

bool DockLayoutUpdate(ModelPlugin* self, mjModel*, mjData*) {
  auto* svc = static_cast<DockLayoutService*>(self->data);
  switch (svc->phase) {
    case DockLayoutService::Phase::kCurate: {
      const ImGuiID root = ImGui::GetID("Root");
      if (ImGui::DockBuilderGetNode(root) == nullptr) {
        BuildCuratedLayout(root);
        svc->phase = DockLayoutService::Phase::kFocus;
      } else {
        // A saved (or already-curated) layout exists -- leave it alone.
        svc->phase = DockLayoutService::Phase::kDone;
      }
      break;
    }
    case DockLayoutService::Phase::kFocus: {
      // Once the windows exist AND have been absorbed into their nodes, make
      // the curated panels the selected tabs (the host panels default to
      // visible and would otherwise contend for tab focus). Bounded retry: a
      // panel the user disabled would otherwise pin this phase forever.
      const bool details = SelectDockedTab("Details");
      const bool hierarchy = SelectDockedTab("Hierarchy");
      if ((details && hierarchy) || ++svc->focus_attempts > 30) {
        svc->phase = DockLayoutService::Phase::kDone;
      }
      break;
    }
    case DockLayoutService::Phase::kDone:
      break;
  }
  return false;  // never steps the simulation
}

}  // namespace

void RegisterDockLayoutService() {
  // Leaked for the app lifetime (mirrors the plugin registry's static storage).
  auto* svc = new DockLayoutService;
  ModelPlugin plugin;
  plugin.name = "ProtoSpec Dock Layout";
  plugin.do_update = DockLayoutUpdate;
  plugin.data = svc;
  RegisterPlugin<ModelPlugin>(plugin);
}

}  // namespace ps::studio
