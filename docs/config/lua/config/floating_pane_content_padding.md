# `floating_pane_content_padding`

This fork uses `floating_pane_content_padding` for the space between floating
terminal content and its border. Each side defaults to 6 pixels. The padding is
included in the opaque background and mouse hit area; terminal coordinates do
not change.

```lua
config.floating_pane_content_padding = {
  left = 6, right = 6, top = 6, bottom = 6,
}
```

The separate `floating_pane_padding` setting controls the initial outer margins
between the floating pane and its parent tab, not this content inset.
