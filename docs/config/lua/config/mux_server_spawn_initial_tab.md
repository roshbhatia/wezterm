# `mux_server_spawn_initial_tab = true`

Controls whether `wezterm-mux-server` starts a default tab when its default
domain has no panes after `mux-startup` runs. The default is `true`.

Set this to `false` in a dedicated mux server configuration when a GUI session
recovery handler owns the initial layout. The server then listens without
starting a shell. A connecting client can restore its saved panes or request
a new tab explicitly.
