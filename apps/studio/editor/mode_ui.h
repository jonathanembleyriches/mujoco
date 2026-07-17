// Mode readout for the Unity-style mode machine (EditorMode + play_paused):
// a chip for the host toolbar and a viewport border. Both hosts draw the chip
// from here so the two toolbars agree on the vocabulary; the border is drawn
// once from the editor's viewport hook, so it follows the editor to any host.
//
// Header-only: the fork lists the editor's .cc files explicitly, so a new one
// would have to be added to two build files to reach both hosts.

#ifndef PS_STUDIO_EDITOR_MODE_UI_H_
#define PS_STUDIO_EDITOR_MODE_UI_H_

#include <imgui.h>

#include "editor/editor_context.h"

namespace ps::studio {

// The three states the mode machine can be observed in. Play splits on
// play_paused: physics is only advancing in the running case, and "paused
// inside Play" is not "back in Edit" -- sim state is still live, and Stop is
// what discards it.
struct ModeBadge {
  const char* label;
  ImU32 color;
};

inline ModeBadge CurrentModeBadge(const EditorContext& c) {
  if (c.mode == EditorMode::Play) {
    if (c.play_paused) return {"PAUSED", IM_COL32(232, 172, 46, 255)};
    return {"PLAYING", IM_COL32(78, 201, 96, 255)};
  }
  return {"EDIT", IM_COL32(150, 152, 158, 255)};
}

inline const char* ModeTooltip(const EditorContext& c) {
  if (c.mode == EditorMode::Play) {
    if (c.play_paused) {
      return "Paused mid-simulation: sim state is live and gizmos are inert.\n"
             "Play resumes; Stop discards sim state and returns to Edit.";
    }
    return "Simulating: gizmos are inert and edits do not apply.\n"
           "Stop discards sim state and returns to Edit.";
  }
  return "Edit mode: physics paused at the spec pose, gizmos live.";
}

// The mode chip, drawn inline at the end of the editor's own toolbar section.
// Not right-aligned to the bar: a host may keep drawing its own controls on the
// same row after this section (the fork's camera/label/frame widgets do), and
// they would land on top of a chip pinned to the far edge.
//
// Call it outside any BeginDisabled -- the mode must stay legible when no model
// is loaded.
inline void DrawModeChip(const EditorContext& c) {
  const ModeBadge b = CurrentModeBadge(c);
  const float dot_r = ImGui::GetFontSize() * 0.26f;
  const float gap = ImGui::GetStyle().ItemInnerSpacing.x;

  ImGui::SameLine();
  ImGui::TextDisabled("|");
  ImGui::SameLine();

  const ImVec2 p = ImGui::GetCursorScreenPos();
  ImGui::GetWindowDrawList()->AddCircleFilled(
      ImVec2(p.x + dot_r, p.y + ImGui::GetTextLineHeight() * 0.5f), dot_r,
      b.color);
  ImGui::Dummy(ImVec2(dot_r * 2.0f + gap, ImGui::GetTextLineHeight()));
  ImGui::SameLine(0.0f, 0.0f);
  ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(b.color), "%s", b.label);
  ImGui::SetItemTooltip("%s", ModeTooltip(c));
}

// Frames the viewport while the sim owns it. Edit draws nothing: the neutral
// state is the one the user is in most of the time, and a border that is always
// present stops carrying a signal.
inline void DrawModeBorder(const EditorContext& c) {
  if (c.mode != EditorMode::Play) return;
  const ImGuiViewport* v = ImGui::GetMainViewport();
  ImGui::GetForegroundDrawList()->AddRect(
      v->Pos, ImVec2(v->Pos.x + v->Size.x, v->Pos.y + v->Size.y),
      CurrentModeBadge(c).color, 0.0f, 0, 2.0f);
}

}  // namespace ps::studio

#endif  // PS_STUDIO_EDITOR_MODE_UI_H_
