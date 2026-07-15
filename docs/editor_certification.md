# ProtoSpec Studio editor certification

How the editor cluster (`src/experimental/studio/protospec/editor`) is verified.
Every mapping/mutation decision is a pure function tested windowless; the live
window is only exercised for render/layout, captured by the host self-screenshot.

## Windowless test batteries

Built and run from `C:/tmp/studio_spike/build_ps`. All eight must be green.

| Battery | Target | Covers |
| --- | --- | --- |
| studio | `protospec_studio_tests` | model source / compile pipeline, diagnostics |
| gizmo | `protospec_studio_gizmo_tests` | transform gizmo math + interaction |
| movability | `protospec_studio_movability_tests` | reparent / keep-world-pose |
| details | `protospec_studio_details_tests` | Details field->widget classifier, presence layers, ref combos, colour classifier, numeric width |
| authoring | `protospec_studio_authoring_tests` | add / duplicate / reparent / new-model ops |
| se5 | `protospec_studio_se5_tests` | multi-import, batch add, material/texture create, geom material assign, **texture-layer editing**, **dialog validation** |
| host | `protospec_studio_host_tests` | host plugin registry / file-dialog state machine |
| hostui | `protospec_studio_hostui_tests` | host UI wiring |

## Manual (live-window) verification

Injected input does not reach the window in the headless capture environment, so
these are driven by hand in a real session. Launch:

```
mujoco_studio.exe <scene.xml> --gfx=classic
```

- **Material texture layers.** Select a Material (Assets panel or Hierarchy). The
  Details panel shows a **Texture Layers** section below Properties. Each row is
  `layer N | texture-ref combo | role combo | x`, aligned to the panel's value
  column. `+ Add layer` appends a row (role `rgb`, first texture pre-filled); `x`
  removes it; the texture combo offers every model texture plus `(none)`; the role
  combo offers the `TexRole` keywords. Each change is one undo step. This is the
  one appearance surface the generic reflection visitor cannot reach (a Material's
  `layers` are an owned child list), so it is authored here explicitly. The pure
  mutators (`AddMaterialLayer` / `RemoveMaterialLayer` / `SetLayerTexture` /
  `SetLayerRole`, in `details_panel.h`) are covered windowless by the se5 battery.
- **rgba colour picker.** A Material's `rgba` row (and a geom's) renders as an
  ImGui `ColorEdit4` swatch/picker, not a bare numeric row (details battery
  `TestColorClassifier`).
- **New Material / New Texture dialogs.** `Create` is disabled until the spec is
  valid: a non-empty name is required; a builtin texture needs width/height >= 1;
  a file texture needs a path. `Cancel` closes without a change. The gate is the
  pure `MaterialSpecValid` / `TextureSpecValid` predicate (se5 battery
  `TestDialogValidation`).

## Self-screenshot (render/layout)

```
mujoco_studio.exe <scene.xml> --gfx=classic \
  --screenshot_seq=<dir> --screenshot_after=90 --screenshot_exit
```

Writes `<dir>/shot_0000.png`. Note absl flag names use underscores. To bring a
docked tab (e.g. Assets) to the front for a capture, front it in the ImGui ini
(`$HOME/.mujoco.ini`) since input injection is unavailable.

Verified renders: viewport materials (metal/rubber/checker-floor), Hierarchy body
+ asset trees, and the Assets panel (New Material / New Texture buttons, material
rgba swatches, texture builtin-preview swatches with `[checker]`/`[gradient]`
kind tags).
