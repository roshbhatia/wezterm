# `AdjustPaneSize`

{{since('20201031-154415-9614e117')}}

`AdjustPaneSize` manipulates the size of the active pane, allowing the
size to be adjusted by an integer amount in a specific direction.

When a floating pane is active, `Left` and `Right` decrease and increase its
width; `Up` and `Down` decrease and increase its height. The pane stays centered,
with a minimum size of one cell and a maximum that leaves a one-cell margin.
The underlying splits are unchanged. Hiding and restoring the floating pane
preserves its size, and resizing the window scales it to fit.

If the pane is on the right hand side of a split and you adjust the size
left by 1 then the split grows larger by 1 cell by expanding its size to
the left.  The pane to its left is reduced in size by 1 cell to accommodate
the growth.   If you were to adjust this same right hand side right by 1 cell,
then the pane will shrink by 1 cell and move the split 1 cell to the right.

Here's a sample configuration that uses `CTRL-A H` to increase the size
of the active pane by 5 cells in the left direction.  The other `vi` style
motion keys are used to adjust the size in their conventional directions,
respectively.

```lua
local wezterm = require 'wezterm'
local act = wezterm.action
local config = {}

config.leader = { key = 'a', mods = 'CTRL' }
config.keys = {
  {
    key = 'H',
    mods = 'LEADER',
    action = act.AdjustPaneSize { 'Left', 5 },
  },
  {
    key = 'J',
    mods = 'LEADER',
    action = act.AdjustPaneSize { 'Down', 5 },
  },
  { key = 'K', mods = 'LEADER', action = act.AdjustPaneSize { 'Up', 5 } },
  {
    key = 'L',
    mods = 'LEADER',
    action = act.AdjustPaneSize { 'Right', 5 },
  },
}
return config
```
