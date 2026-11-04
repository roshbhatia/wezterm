# `ToggleGlobalFloatingPane`

Creates a singleton shell in the local domain, or moves that same floating pane
to the current tab. Invoking the action while the singleton is focused hides it
without terminating its process. The singleton belongs to the running mux and
survives closing its host tab. Separate mux servers have separate singletons.

The shell starts in the invoking pane's directory when it is local, otherwise
in the local domain's default directory. Moving it preserves its process and
current directory. Its size is retained, clamped to the destination tab.

`ToggleFloatingPane` controls only the tab's own floating panes. Local and global
floats retain separate processes; selecting a local float hides the singleton.

```lua
config.keys = {
  { key = 'f', mods = 'SUPER', action = wezterm.action.ToggleFloatingPane },
  { key = 'f', mods = 'SUPER|SHIFT', action = wezterm.action.ToggleGlobalFloatingPane },
}
```
