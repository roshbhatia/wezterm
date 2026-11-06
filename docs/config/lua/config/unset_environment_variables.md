# `unset_environment_variables = {}`

Removes the named environment variables when spawning local terminal commands.
Removal runs after `set_environment_variables`. The default empty list preserves
the inherited environment.

```lua
config.unset_environment_variables = { "NO_COLOR" }
```

This prevents an automation client's `NO_COLOR` preference from disabling color
in an interactive pane it launches. Commands may explicitly set the variable
again, for example with `env NO_COLOR=1 command`.
