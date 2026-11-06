# `pane:spawn_floating_pane{}`

Spawns one floating pane in the tab containing `pane` and returns the new
`Pane`. This operation works without a GUI window, including in an inactive
workspace during session recovery.

The command fields `args`, `cwd`, `set_environment_variables`, and `domain`
have the same meaning as in [pane:split](split.md). Optional `width` and
`height` specify cells. `visible` defaults to `true`.

```lua
local floating = pane:spawn_floating_pane {
  cwd = wezterm.home_dir,
  width = 60,
  height = 18,
  visible = false,
}
```

The existing tab-local floating-pane controls can show or hide the pane later.
