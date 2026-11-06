local w = require 'wezterm'
local root = assert(os.getenv 'WEZTERM_TEST_STATE')
local counts = {}
local quiet, busy
w.on('format-window-title', function(tab)
  local key = tostring(tab.window_id)
  counts[key] = (counts[key] or 0) + 1
  return tab.active_pane.title
end)
w.on('gui-startup', function()
  local qt, qp, qw = w.mux.spawn_window {
    workspace = 'performance-test',
    args = { '/bin/sleep', '60' },
  }
  quiet = qw:window_id()
  local bt, bp, bw = w.mux.spawn_window {
    workspace = 'performance-test',
    args = {
      assert(os.getenv 'WEZTERM_TEST_PYTHON'),
      '-u',
      '-c',
      "import sys,time; time.sleep(2); [(sys.stdout.write(chr(27)+']2;perf-'+str(i)+chr(7)),sys.stdout.flush(),time.sleep(.02)) for i in range(100)]; time.sleep(60)",
    },
  }
  busy = bw:window_id()
  w.time.call_after(1.5, function()
    local initial = counts[tostring(quiet)] or 0
    w.time.call_after(4, function()
      local f = assert(io.open(root .. '/result.json', 'w'))
      f:write(w.json_encode {
        quiet_updates = (counts[tostring(quiet)] or 0) - initial,
        counts = counts,
        quiet = quiet,
        busy = busy,
      })
      f:close()
    end)
  end)
end)
return {
  initial_cols = 90,
  initial_rows = 35,
  check_for_updates = false,
  automatically_reload_config = false,
  default_prog = { '/bin/sleep', '60' },
}
