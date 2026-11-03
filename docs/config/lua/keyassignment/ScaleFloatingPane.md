# `ScaleFloatingPane`

This fork provides `ScaleFloatingPane` to scale the active floating pane's width
and height together. The signed integer argument is a percentage change: `-5`
shrinks both dimensions by approximately 5%, and `5` grows them. Cell rounding
may change the exact proportion. Very small panes grow by at least one cell.

Scaling stops both axes together when either reaches a size limit. It keeps the
pane centered, does not resize underlying splits, and sends one size change to
the terminal application. With a normal pane active, the action does nothing.

```lua
config.keys = {
  { key = 'j', mods = 'SUPER|CTRL', action = wezterm.action.ScaleFloatingPane(-5) },
  { key = 'k', mods = 'SUPER|CTRL', action = wezterm.action.ScaleFloatingPane(5) },
}
```
