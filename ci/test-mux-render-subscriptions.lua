local w = require 'wezterm'
local state = assert(os.getenv 'WEZTERM_TEST_STATE')
local function after(delay, callback)
  w.time.call_after(delay, function()
    local ok, err = pcall(callback)
    if not ok then
      local file = assert(io.open(state .. '/gui-failed', 'w'))
      file:write(tostring(err))
      file:close()
    end
  end)
end
local function exercise_float_focus(window, origin, finished)
  local floating = origin:spawn_floating_pane { width = 48, height = 12 }
  local round = 0
  local cycle
  cycle = function()
    round = round + 1
    window:perform_action(
      w.action.Multiple {
        w.action.ActivatePaneById(floating:pane_id()),
        w.action.InputSelector {
          title = 'Focus regression',
          scope = 'Pane',
          choices = { { id = 'fixture', label = 'Fixture row' } },
          action = w.action_callback(function()
            origin:activate()
          end),
        },
      },
      floating
    )
    local samples = 0
    local check
    check = function()
      assert(
        window:mux_window():active_pane():pane_id() == floating:pane_id(),
        string.format(
          'floating picker lost focus in round %d sample %d: expected %d, got %d',
          round,
          samples,
          floating:pane_id(),
          window:mux_window():active_pane():pane_id()
        )
      )
      local displayed = window:active_pane()
      assert(
        displayed:pane_id() ~= floating:pane_id()
          and displayed:get_lines_as_text():find('Fixture row', 1, true),
        string.format(
          'picker was replaced by the floating shell in round %d sample %d',
          round,
          samples
        )
      )
      samples = samples + 1
      if samples < 4 then
        after(0.05, check)
        return
      end
      window:perform_action(
        w.action.SendKey { key = 'Escape', mods = 'NONE' },
        window:active_pane()
      )
      after(0.1, function()
        assert(
          window:mux_window():active_pane():pane_id() == origin:pane_id(),
          'closing the picker did not restore the original pane'
        )
        if round < 5 then
          cycle()
        else
          finished()
        end
      end)
    end
    after(0.2, check)
  end
  local changes = {}
  for _ = 1, 5 do
    changes[#changes + 1] = w.action.ActivatePaneById(origin:pane_id())
    changes[#changes + 1] = w.action.ActivatePaneById(floating:pane_id())
  end
  window:perform_action(w.action.Multiple(changes), floating)
  after(0.2, cycle)
end
w.on('gui-attached', function()
  after(0.5, function()
    local pane = w.mux.all_windows()[1]:active_pane()
    local title = pane:get_title()
    assert(
      pane:get_user_vars().SUBSCRIPTION_STARTUP == 'boot',
      'initial OSC state was lost'
    )
    after(0.7, function()
      assert(
        pane:get_title() ~= title,
        'GUI stopped receiving render updates'
      )
      local window = w.mux.all_windows()[1]:gui_window()
      exercise_float_focus(window, pane, function()
        local file = assert(io.open(state .. '/gui-passed', 'w'))
        file:write 'passed'
        file:close()
      end)
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
