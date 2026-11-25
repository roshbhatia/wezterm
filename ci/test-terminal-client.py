"""Exercise the native terminal client against an isolated mux and real PTYs."""

import errno
import fcntl
import json
import os
import pty
import select
import signal
import struct
import subprocess
import sys
import tempfile
import termios
import time
from pathlib import Path


def wait_for(predicate, message, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for client in TerminalClient.active:
            client.read()
        if predicate():
            return
        time.sleep(0.03)
    raise AssertionError(message)


class TerminalClient:
    active = []

    def __init__(self, command, env, cols=39, rows=14):
        self.pid, self.fd = pty.fork()
        if self.pid == 0:
            os.execvpe(command[0], command, env)
        self.output = bytearray()
        self.cursor_reply_offset = 0
        self.status = None
        self.resize(cols, rows)
        self.active.append(self)

    def resize(self, cols, rows):
        fcntl.ioctl(self.fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))

    def read(self):
        while select.select([self.fd], [], [], 0)[0]:
            try:
                chunk = os.read(self.fd, 65536)
            except OSError as error:
                if error.errno == errno.EIO:
                    break
                raise
            if not chunk:
                break
            self.output.extend(chunk)
        if b"\x1b[6n" in self.output[self.cursor_reply_offset:]:
            self.send(b"\x1b[1;1R")
            self.cursor_reply_offset = len(self.output)
        return bytes(self.output)

    def send(self, data):
        os.write(self.fd, data)

    def exited(self):
        if self.status is None:
            pid, status = os.waitpid(self.pid, os.WNOHANG)
            if pid:
                self.status = os.waitstatus_to_exitcode(status)
        return self.status is not None

    def close(self):
        if not self.exited():
            os.kill(self.pid, signal.SIGTERM)
            try:
                wait_for(self.exited, "fixture client did not terminate", timeout=2)
            except AssertionError:
                os.kill(self.pid, signal.SIGKILL)
                os.waitpid(self.pid, 0)
        self.active.remove(self)
        os.close(self.fd)


def main():
    binary = str(Path(sys.argv[1]).resolve())
    server_binary = str(Path(binary).with_name("wezterm-mux-server"))
    clients = []
    env = os.environ.copy()
    for key in ("WEZTERM_PANE", "WEZTERM_UNIX_SOCKET"):
        env.pop(key, None)
    env["TERM"] = "xterm-256color"
    with tempfile.TemporaryDirectory(prefix="wezterm-terminal-", dir="/tmp") as state:
        root = Path(state)
        sock = root / "mux.sock"
        received = root / "input.bin"
        received.touch()
        producer = root / "producer.py"
        producer.write_text(
            "import os, tty\n"
            "tty.setraw(0)\n"
            "print('@' * 20 + '\\r', flush=True)\n"
            "for i in range(100): print('history-line-%03d\\r' % i, flush=True)\n"
            "print('native-client-ready 界\\r', flush=True)\n"
            f"with open({str(received)!r}, 'ab', buffering=0) as log:\n"
            " while True:\n"
            "  data=os.read(0,4096)\n"
            "  if not data: break\n"
            "  log.write(data)\n"
        )
        config = root / "mux.lua"
        config.write_text(
            'local w=require("wezterm"); return {unix_domains={{name="unix",socket_path='
            + json.dumps(str(sock))
            + "}},default_prog=w.json_parse("
            + json.dumps(json.dumps([sys.executable, "-u", str(producer)]))
            + "),scrollback_lines=1000}"
        )
        command = [binary, "--config-file", str(config)]
        cli = command + ["cli", "--no-auto-start", "--prefer-mux"]
        noninteractive = subprocess.run(command + ["tui"], env=env, capture_output=True, timeout=5)
        assert noninteractive.returncode != 0 and b"interactive terminal" in noninteractive.stderr
        missing = TerminalClient(command + ["tui"], env)
        clients.append(missing)
        wait_for(missing.exited, "missing mux did not fail without auto-start")
        assert missing.status != 0 and not sock.exists(), "client automatically started a missing mux"
        with open(root / "server.log", "w+") as log:
            server = subprocess.Popen(
                [server_binary, "--config-file", str(config)],
                env=env, stdout=log, stderr=log,
            )
            try:
                wait_for(sock.exists, "isolated mux socket did not appear")
                env["WEZTERM_UNIX_SOCKET"] = str(sock)

                def panes():
                    result = subprocess.run(
                        cli + ["list", "--format", "json"], env=env,
                        capture_output=True, text=True, check=True, timeout=5,
                    )
                    return json.loads(result.stdout)

                wait_for(panes, "mux did not create a pane")
                pane = panes()[0]
                pane_id = pane["pane_id"]
                original_size = pane["size"]
                attach = command + ["tui", "--pane-id", str(pane_id)]
                phone = TerminalClient(attach, env)
                clients.append(phone)
                wait_for(lambda: b"native-client-ready" in phone.read(), "phone did not render pane")
                desktop = TerminalClient(attach, env, 100, 30)
                clients.append(desktop)
                wait_for(lambda: b"native-client-ready" in desktop.read(), "second client did not render pane")
                phone.resize(24, 10)
                time.sleep(0.2)
                assert panes()[0]["size"] == original_size, "attachment resized the shared pane"

                phone.send(b"phone-input\x03\x04")
                wait_for(lambda: b"phone-input\x03\x04" in received.read_bytes(), "application control input was consumed")
                phone.send(b"\x1b")
                wait_for(lambda: received.read_bytes().endswith(b"\x1b"), "Escape did not reach application")
                phone.send(b"\x02")
                wait_for(lambda: b"Controls" in phone.read(), "prefix did not open controls")
                phone.send(b"\x02")
                wait_for(lambda: received.read_bytes().endswith(b"\x02"), "double prefix did not reach application")

                phone.send(b"\x02v")
                wait_for(lambda: b"VIEW ONLY" in phone.read(), "view-only mode did not activate")
                before = received.read_bytes()
                phone.send(b"must-not-send")
                time.sleep(0.2)
                assert received.read_bytes() == before, "view-only mode sent input"
                phone.send(b"\x02v")
                phone.send(b"\x1b[200~paste-value\x1b[201~")
                wait_for(lambda: received.read_bytes().endswith(b"paste-value"), "bracketed paste did not reach application")

                phone.send(b"\x02h\x1b[H")
                wait_for(lambda: b"@@@@@@" in phone.read(), "scrollback did not render")
                phone.send(b"q\x02d")
                wait_for(phone.exited, "detach did not return to the SSH shell")
                assert phone.status == 0
                assert any(p["pane_id"] == pane_id for p in panes()), "detach destroyed the pane"
                assert not desktop.exited(), "detach disconnected the other client"

                picker = TerminalClient(command + ["tui"], env)
                clients.append(picker)
                wait_for(lambda: b"Persistent" in picker.read() and b"panes" in picker.read(), "picker did not appear")
                wait_for(lambda: b"Ctrl+N" in picker.read() and b"Esc" in picker.read(), "phone picker hid its new-shell or exit shortcut")
                picker.send(b"\x1b")
                wait_for(picker.exited, "picker Escape did not exit")
                assert picker.status == 0

                mobile = TerminalClient(command + ["tui"], env)
                clients.append(mobile)
                wait_for(lambda: b"Persistent" in mobile.read(), "new-shell picker did not appear")
                mobile.send(b"\x0e")
                wait_for(lambda: len(panes()) == 2, "new mobile shell was not created")
                new_pane = next(p for p in panes() if p["pane_id"] != pane_id)
                assert new_pane["workspace"] == "default"
                assert new_pane["size"]["cols"] == 39 and new_pane["size"]["rows"] == 13
                wait_for(lambda: b"native-client-ready" in mobile.read(), "new mobile shell did not attach")
                mobile.send(b"\x02d")
                wait_for(mobile.exited, "new mobile shell did not detach")
                assert mobile.status == 0 and len(panes()) == 2

                server.terminate()
                server.wait(timeout=5)
                wait_for(desktop.exited, "client did not exit after mux loss")
                assert desktop.status != 0, "mux loss was reported as a clean detach"
                print(json.dumps({
                    "clients": 2, "shared_size_preserved": True,
                    "control_input": True, "paste": True, "view_only": True,
                    "history": True, "detach_preserves_pane": True,
                    "picker": True, "mux_loss_exits": True,
                    "no_auto_start": True, "new_mobile_shell": True,
                }))
            except BaseException:
                print(repr(received.read_bytes()), file=sys.stderr)
                for client in clients:
                    output = client.read()
                    print(repr(output[:1000] + output[-500:]), file=sys.stderr)
                log.seek(0)
                print(log.read()[-4000:], file=sys.stderr)
                raise
            finally:
                for client in clients:
                    client.close()
                if server.poll() is None:
                    server.terminate()
                    server.wait(timeout=5)


if __name__ == "__main__":
    main()
