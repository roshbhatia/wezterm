# `ToggleFloatingPane`

Shows or hides the current tab's floating pane. If the tab has no floating
pane, starts the default shell in a new floating pane using the current
pane's domain and directory. Hiding a pane keeps its process running.

```lua
config.keys = {
  { key = 'f', mods = 'SUPER|SHIFT', action = wezterm.action.ToggleFloatingPane },
}
```

Normal panes keep their size while a floating pane is visible. Clicking outside
the floating pane hides it. Left/Right or Previous/Next pane navigation cycles
between the tab's floating panes. Closing a floating pane terminates its process;
hiding it does not.

The command palette also provides `SpawnCommandInNewFloatingPane`,
`MoveFloatingPaneToHorizontalSplit`, and `MoveFloatingPaneToVerticalSplit`.

Floating panes are configured with `floating_pane_padding` and
`floating_pane_border`. The default padding is 20 percent on each side.
The CLI can create one with:

```sh
wezterm cli --no-auto-start spawn --pane-id "$WEZTERM_PANE" --floating-pane
```

`wezterm cli list --format json`, `tab:panes()`, and `tab:panes_with_info()`
include hidden floating panes. Pane information exposes `is_floating`.
Mux clients and servers must both use codec version 48.

This fork integrates the original author's work from
[wezterm/wezterm#5576](https://github.com/wezterm/wezterm/pull/5576).
