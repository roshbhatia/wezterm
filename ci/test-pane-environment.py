"""Check that CLI-launched interactive panes discard configured environment keys."""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

with tempfile.TemporaryDirectory(prefix="wezterm-environment-", dir="/tmp") as state:
    root = Path(state)
    sock = root / "mux.sock"
    config = root / "mux.lua"
    result = root / "result"
    config.write_text(
        'return {default_prog={"/bin/sleep","60"},'
        'unset_environment_variables={"NO_COLOR"},'
        'set_environment_variables={PROJECT_CONTEXT="retained"},'
        'unix_domains={{name="unix",socket_path=' + json.dumps(str(sock)) + "}}}"
    )
    env = os.environ.copy()
    env.pop("WEZTERM_PANE", None)
    env["NO_COLOR"] = "1"
    env["WEZTERM_UNIX_SOCKET"] = str(sock)
    with open(root / "server.log", "w+") as log:
        server = subprocess.Popen(
            [sys.argv[1], "--config-file", str(config)],
            env=env,
            stdout=log,
            stderr=log,
        )
        try:
            deadline = time.monotonic() + 8
            while (
                not sock.exists()
                and server.poll() is None
                and time.monotonic() < deadline
            ):
                time.sleep(0.05)
            if not sock.exists():
                log.seek(0)
                raise RuntimeError(log.read() or "mux socket did not appear")
            subprocess.run(
                [
                    str(Path(sys.argv[1]).with_name("wezterm")),
                    "--config-file",
                    str(config),
                    "cli",
                    "--no-auto-start",
                    "--prefer-mux",
                    "spawn",
                    "--new-window",
                    "--",
                    "/bin/sh",
                    "-c",
                    'printf "%s" "${NO_COLOR-unset}:${PROJECT_CONTEXT-unset}" > "$1"; exec /bin/sleep 60',
                    "sh",
                    str(result),
                ],
                env=env,
                capture_output=True,
                text=True,
                check=True,
                timeout=8,
            )
            while not result.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            assert result.read_text() == "unset:retained", (
                "interactive pane inherited NO_COLOR or lost context"
            )
            print("CLI pane environment: NO_COLOR removed; project context retained")
        finally:
            server.terminate()
            server.wait(timeout=5)
