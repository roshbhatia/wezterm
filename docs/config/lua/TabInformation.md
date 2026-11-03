# TabInformation

The `TabInformation` struct describes a tab.  `TabInformation` is purely a
snapshot of some of the key characteristics of the tab, intended for use in
synchronous, fast, event callbacks that format GUI elements such as the window
and tab title bars.

The `TabInformation` struct contains the following fields:

* `tab_id` - the identifier for the tab
* `tab_index` - the logical tab position within its containing window, with 0 indicating the leftmost tab
* `is_active` - is true if this tab is the active tab
* `is_last_active` - is true if this tab is the previously active tab. {{since('nightly', inline=True)}}
* `active_pane` - the [PaneInformation](PaneInformation.md) for the active pane in this tab
* `active_normal_pane` - in this fork, the active non-floating pane, independent of floating focus and zoom. Use this to keep the parent tab label stable while a float is open.
* `floating_pane_count` - in this fork, the number of floating panes, including hidden ones
* `floating_pane_visible` - in this fork, true when this snapshot's active pane is one of the tab's floats, including a zoomed float
* `panes` - the [PaneInformation](PaneInformation.md) for all panes in this tab {{since('20220319-142410-0fcdea07', inline=True)}}
* `window_id` - the ID of the window that contains this tab {{since('20220807-113146-c2fee766', inline=True)}}
* `window_title` - the title of the window that contains this tab {{since('20220807-113146-c2fee766', inline=True)}}
* `tab_title` - the title of the tab {{since('20220807-113146-c2fee766', inline=True)}}


