"""Check title-alert isolation in a temporary GUI; requires a macOS GUI session."""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

with tempfile.TemporaryDirectory(prefix="wezterm-title-routing-") as state:
    env = os.environ.copy()
    for key in ("WEZTERM_PANE", "WEZTERM_UNIX_SOCKET"):
        env.pop(key, None)
    env["WEZTERM_TEST_STATE"] = state
    env["WEZTERM_TEST_PYTHON"] = sys.executable
    config = Path(__file__).with_suffix(".lua").resolve()
    result = Path(state) / "result.json"
    with open(Path(state) / "gui.log", "w+") as log:
        process = subprocess.Popen(
            [
                sys.argv[1],
                "--config-file",
                str(config),
                "start",
                "--always-new-process",
            ],
            env=env,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 12
            while not result.exists() and time.monotonic() < deadline:
                if process.poll() is not None:
                    break
                time.sleep(0.2)
            if not result.exists():
                log.seek(0)
                raise RuntimeError(log.read() or "GUI fixture did not finish")
            observed = json.loads(result.read_text())
            assert observed["quiet_updates"] == 0, observed
            print(json.dumps(observed))
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
