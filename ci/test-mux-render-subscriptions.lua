local w = require 'wezterm'
local state = assert(os.getenv 'WEZTERM_TEST_STATE')
w.on('gui-attached', function()
  w.time.call_after(0.5, function()
    local pane = w.mux.all_windows()[1]:active_pane()
    local title = pane:get_title()
    assert(
      pane:get_user_vars().SUBSCRIPTION_STARTUP == 'boot',
      'initial OSC state was lost'
    )
    w.time.call_after(0.7, function()
      assert(
        pane:get_title() ~= title,
        'GUI stopped receiving render updates'
      )
      local file = assert(io.open(state .. '/gui-passed', 'w'))
      file:write 'passed'
      file:close()
    end)
  end)
end)
return {
  unix_domains = {
    { name = 'unix', socket_path = assert(os.getenv 'WEZTERM_TEST_SOCKET') },
  },
  default_domain = 'unix',
  check_for_updates = false,
  automatically_reload_config = false,
}
