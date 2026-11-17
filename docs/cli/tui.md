# Native terminal attachment

`wezterm tui` attaches an ordinary terminal to panes in the configured local
WezTerm mux. It uses Ratatui and the existing mux protocol. An SSH client such as
Termius does not need to implement the WezTerm protocol.

Connect with an interactive SSH terminal, then run:

```console
wezterm-remote tui
```

`wezterm-remote` is the sysinit wrapper that selects the persistent mux's
configuration. Other installations can use `wezterm --config-file FILE tui`.
Use the same fork revision for the client and mux. The client does not start a
missing mux unless `--start-server` is specified.

Type to filter the pane picker by workspace, title, directory, or pane ID. Use
arrow keys and Enter to attach. Escape returns to the SSH shell. `--pane-id ID`
attaches directly, and `--read-only` disables application input initially.
Floating panes carry an `F` prefix, including when their desktop float is hidden.

While attached, Control+B opens the controls menu. Arrow keys and Enter work
without desktop modifiers. The menu also accepts these keys:

| Key | Operation |
| --- | --- |
| `p` | Choose another pane |
| `h` | Browse scrollback |
| `v` | Toggle view-only or shared input |
| `n` | Create a mobile shell |
| `d` | Detach to the SSH shell |

Control+B twice sends a literal Control+B. Escape dismisses the menu. In history,
arrows, PageUp/PageDown, Home/End, and `j`/`k` navigate; Escape or `q` returns to
live output. Other application input, including Control+C, Control+D, Escape,
and bracketed paste, reaches the pane in shared mode.

Existing panes retain their terminal dimensions. The phone shows a cropped view
that follows the cursor; rotating the phone changes only that view. A new mobile
shell uses the current terminal dimensions and the `mobile` workspace. Its size
also remains unchanged on later client resize. There is no automatic size
takeover, mouse forwarding, or split-layout rendering in this initial client.

Detaching or losing SSH leaves mux processes running. If the mux itself exits,
the client restores the terminal and exits with an error. Reopen the picker to
discover the current panes. It does not reconnect to a potentially reused pane
ID or replay input. Existing recovery owns restoration after a mux restart.
