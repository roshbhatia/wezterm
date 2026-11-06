"""Check that idle mux connections do not receive unsolicited render frames."""

import json
import os
import select
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

with tempfile.TemporaryDirectory(prefix="wezterm-render-", dir="/tmp") as state:
    root = Path(state)
    sock = root / "mux.sock"
    config = root / "mux.lua"
    producer = [
        sys.executable,
        "-u",
        "-c",
        "import time; print(chr(27)+']1337;SetUserVar=SUBSCRIPTION_STARTUP=Ym9vdA=='+chr(7),flush=True); [(print(((chr(27)+']2;render-'+str(i)+chr(7)) if i>40 else '')+'render-subscription-test '+str(i),flush=True),time.sleep(.05)) for i in range(200)]",
    ]
    config.write_text(
        'local w=require("wezterm"); return {unix_domains={{name="unix",socket_path='
        + json.dumps(str(sock))
        + "}},default_prog=w.json_parse("
        + json.dumps(json.dumps(producer))
        + ")}"
    )
    env = os.environ.copy()
    for key in ("WEZTERM_PANE", "WEZTERM_UNIX_SOCKET"):
        env.pop(key, None)
    clients = []
    gui = None
    with open(root / "server.log", "w+") as log:
        process = subprocess.Popen(
            [sys.argv[1], "--config-file", str(config)], env=env, stdout=log, stderr=log
        )
        try:
            deadline = time.monotonic() + 8
            while not sock.exists() and time.monotonic() < deadline:
                if process.poll() is not None:
                    break
                time.sleep(0.05)
            if not sock.exists():
                log.seek(0)
                raise RuntimeError(log.read() or "mux socket did not appear")
            cli_env = {**env, "WEZTERM_UNIX_SOCKET": str(sock)}
            cli = [
                str(Path(sys.argv[1]).with_name("wezterm")),
                "--config-file",
                str(config),
                "cli",
                "--no-auto-start",
                "--prefer-mux",
            ]
            panes = []
            while not panes and time.monotonic() < deadline:
                listed = subprocess.run(
                    cli + ["list", "--format", "json"],
                    env=cli_env,
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=True,
                )
                panes = json.loads(listed.stdout)
                if not panes:
                    time.sleep(0.05)
            assert panes, "mux did not spawn the producer"
            for _ in range(4):
                client = socket.socket(socket.AF_UNIX)
                client.connect(str(sock))
                clients.append(client)
            time.sleep(0.5)
            received = sum(
                len(client.recv(65536))
                for client in select.select(clients, [], [], 0.3)[0]
            )
            text = subprocess.run(
                cli + ["get-text", "--pane-id", str(panes[0]["pane_id"])],
                env=cli_env,
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            assert "render-subscription-test" in text.stdout, (
                "CLI could not read the producer"
            )
            print(
                json.dumps(
                    {"idle_clients": len(clients), "unsolicited_bytes": received}
                ),
                flush=True,
            )
            assert received == 0, "idle client received unrequested render data"
            if len(sys.argv) > 2:
                env["WEZTERM_TEST_SOCKET"] = str(sock)
                env["WEZTERM_TEST_STATE"] = state
                with open(root / "gui.log", "w+") as gui_log:
                    gui = subprocess.Popen(
                        [
                            sys.argv[2],
                            "--config-file",
                            str(Path(__file__).with_suffix(".lua").resolve()),
                            "connect",
                            "unix",
                        ],
                        env=env,
                        stdout=gui_log,
                        stderr=gui_log,
                    )
                    marker = root / "gui-passed"
                    deadline = time.monotonic() + 7
                    while not marker.exists() and time.monotonic() < deadline:
                        if gui.poll() is not None:
                            break
                        time.sleep(0.1)
                    if not marker.exists():
                        gui_log.seek(0)
                        raise RuntimeError(
                            gui_log.read() or "GUI subscriber did not receive updates"
                        )
                    print("GUI receives changing titles and initial OSC user variables")
        finally:
            if gui is not None:
                gui.terminate()
                gui.wait(timeout=5)
            for client in clients:
                client.close()
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
