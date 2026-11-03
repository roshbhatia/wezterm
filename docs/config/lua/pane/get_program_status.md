# `pane:get_program_status()`

Returns the pane's current [OSC 7501 program status](https://www.superlogical.com/rex/docs/build/program-status)
records as a Lua array, ordered from least to most recently updated.

Each record has `id`, `state`, and optional `kind`, `progress`, `app`, `title`,
and `msg` fields. The root record has an empty `id`. Text is decoded from
base64; `app` includes inheritance from the nearest ancestor that declares it.

```lua
wezterm.on('program-status-changed', function(window, pane)
  for _, record in ipairs(pane:get_program_status()) do
    wezterm.log_info(record.id, record.state, record.msg)
  end
end)
```

The event receives the GUI window and the pane whose records changed. Event
callbacks may be coalesced; read the current records in the callback. Polling
from `update-status` also works. Records are available for local and mux panes,
including when a client attaches after the program reported its status.

A report replaces the addressed record. `clear` removes a record and its children.
Starting a shell prompt (OSC 133 A) or exiting the attached process drops
`working` and `blocked` records. `done` and `error` remain until replaced,
explicitly cleared, the terminal is reset, or the pane is removed. Switching
screens and soft resets preserve records. A full reset clears all records.

Reports are limited to 4096 bytes and 256 records per pane. Invalid base64,
control characters in decoded text, invalid ids, and oversized reports are
rejected without changing records. Programs can query support with
`ESC ] 7501 ; ? ST`; the reply is `ESC ] 7501 ; ? ST`.

The protocol does not trigger desktop notifications or execute commands.
Display `title` and `msg` as plain text if adding them to a status bar.
Mux clients and servers must both use this fork's codec version 48.
