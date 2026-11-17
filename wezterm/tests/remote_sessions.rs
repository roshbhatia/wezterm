#![cfg(unix)]

use portable_pty::{native_pty_system, Child, CommandBuilder, MasterPty, PtySize};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};

struct Server(std::process::Child);
impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

struct Daemon(PathBuf);
impl Drop for Daemon {
    fn drop(&mut self) {
        if let Ok(pid) = std::fs::read_to_string(&self.0) {
            if let Ok(pid) = pid.trim().parse::<i32>() {
                unsafe {
                    libc::kill(pid, libc::SIGTERM);
                }
            }
        }
    }
}

struct Terminal {
    master: Box<dyn MasterPty + Send>,
    child: Box<dyn Child + Send + Sync>,
    writer: Box<dyn Write + Send>,
    output: Receiver<Vec<u8>>,
    bytes: Vec<u8>,
    cursor_reply_offset: usize,
}

impl Terminal {
    fn spawn(binary: &Path, config: &Path, socket: &Path, arguments: &[&str]) -> Self {
        let pair = native_pty_system()
            .openpty(PtySize {
                rows: 12,
                cols: 40,
                pixel_width: 0,
                pixel_height: 0,
            })
            .unwrap();
        let mut command = CommandBuilder::new(binary);
        command.args(["--config-file", config.to_str().unwrap()]);
        command.args(arguments);
        command.env("WEZTERM_UNIX_SOCKET", socket);
        command.env("TERM", "xterm-256color");
        command.env_remove("NO_COLOR");
        command.env("COLORTERM", "truecolor");
        let child = pair.slave.spawn_command(command).unwrap();
        drop(pair.slave);
        let mut reader = pair.master.try_clone_reader().unwrap();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let mut bytes = [0; 8192];
            while let Ok(count) = reader.read(&mut bytes) {
                if count == 0 || tx.send(bytes[..count].to_vec()).is_err() {
                    break;
                }
            }
        });
        let writer = pair.master.take_writer().unwrap();
        Self {
            master: pair.master,
            child,
            writer,
            output: rx,
            bytes: Vec::new(),
            cursor_reply_offset: 0,
        }
    }

    fn sees(&mut self, text: &str) -> bool {
        while let Ok(bytes) = self.output.try_recv() {
            self.bytes.extend(bytes);
        }
        // Crossterm may query cursor position on an attached PTY.
        if self.bytes[self.cursor_reply_offset..]
            .windows(4)
            .any(|bytes| bytes == b"\x1b[6n")
        {
            self.writer.write_all(b"\x1b[1;1R").unwrap();
            self.cursor_reply_offset = self.bytes.len();
        }
        String::from_utf8_lossy(&self.bytes).contains(text)
    }

    fn detach(&mut self) {
        self.writer.write_all(b"\x02d").unwrap();
        until(|| {
            self.child
                .try_wait()
                .unwrap()
                .is_some_and(|status| status.success())
        });
    }
}

impl Drop for Terminal {
    fn drop(&mut self) {
        if std::thread::panicking() {
            eprintln!("terminal output: {}", String::from_utf8_lossy(&self.bytes));
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[track_caller]
fn until(mut ready: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(15);
    while !ready() {
        assert!(
            Instant::now() < deadline,
            "terminal session did not reach the expected state"
        );
        std::thread::sleep(Duration::from_millis(30));
    }
}

fn run(binary: &Path, config: &Path, socket: &Path, arguments: &[&str]) -> std::process::Output {
    Command::new(binary)
        .args(["--config-file", config.to_str().unwrap()])
        .args(arguments)
        .env("WEZTERM_UNIX_SOCKET", socket)
        .output()
        .unwrap()
}

fn panes(binary: &Path, config: &Path, socket: &Path) -> Vec<serde_json::Value> {
    let response = run(
        binary,
        config,
        socket,
        &[
            "cli",
            "--no-auto-start",
            "--prefer-mux",
            "list",
            "--format",
            "json",
        ],
    );
    assert!(
        response.status.success(),
        "{}",
        String::from_utf8_lossy(&response.stderr)
    );
    serde_json::from_slice(&response.stdout).unwrap()
}

#[test]
fn named_sessions_survive_detach_and_use_the_configured_palette() {
    let binary = PathBuf::from(env!("CARGO_BIN_EXE_wezterm"));
    let server_binary = binary.with_file_name("wezterm-mux-server");
    assert!(
        server_binary.exists(),
        "build wezterm-mux-server before this integration test"
    );
    let temporary = tempfile::tempdir().unwrap();
    let socket = temporary.path().join("mux.sock");
    let config = temporary.path().join("wezterm.lua");
    let pid_file = temporary.path().join("mux.pid");
    let daemon_log = temporary.path().join("daemon.log");
    std::fs::write(&config, format!(
        "return {{unix_domains={{{{name='unix',socket_path={:?}}}}},daemon_options={{pid_file={:?},stdout={:?},stderr={:?}}},mux_server_spawn_initial_tab=false,colors={{foreground='#d0d0d0',background='#112233',selection_fg='#ffffff',selection_bg='#445566'}},remote_client={{interface='cli'}}}}",
        socket.to_str().unwrap(), pid_file.to_str().unwrap(), daemon_log.to_str().unwrap(), daemon_log.to_str().unwrap()
    )).unwrap();

    let missing = run(&binary, &config, &socket, &["remote", "list"]);
    assert!(missing.status.success());
    assert!(!socket.exists(), "listing started an unwanted mux");
    let invalid = run(
        &binary,
        &config,
        &socket,
        &["remote", "attach", "two names"],
    );
    assert!(!invalid.status.success());
    assert!(!socket.exists(), "invalid input started a mux");

    let _server = Server(
        Command::new(server_binary)
            .args(["--config-file", config.to_str().unwrap()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap(),
    );
    until(|| socket.exists());
    let mut attached = Terminal::spawn(
        &binary,
        &config,
        &socket,
        &[
            "remote",
            "attach",
            "integration",
            "--",
            "/bin/sh",
            "-c",
            "stty raw -echo; printf 'named-session-ready\\r\\n'; cat",
        ],
    );
    until(|| attached.sees("named-session-ready"));
    assert!(
        !attached.sees("SHARED |"),
        "direct attach rendered a persistent status bar"
    );
    let original = panes(&binary, &config, &socket);
    assert_eq!(original.len(), 1);
    assert_eq!(original[0]["workspace"], "integration");
    let remote_alias = temporary.path().join("wezterm-remote");
    std::os::unix::fs::symlink(&binary, &remote_alias).unwrap();
    let native_cli = Command::new(&remote_alias)
        .args([
            "--config-file",
            config.to_str().unwrap(),
            "cli",
            "--no-auto-start",
            "--prefer-mux",
            "list",
            "--format",
            "json",
        ])
        .env_remove("WEZTERM_UNIX_SOCKET")
        .output()
        .unwrap();
    assert!(
        native_cli.status.success(),
        "{}",
        String::from_utf8_lossy(&native_cli.stderr)
    );
    assert_eq!(
        serde_json::from_slice::<Vec<serde_json::Value>>(&native_cli.stdout)
            .unwrap()
            .len(),
        1
    );
    let pane_id = original[0]["pane_id"].clone();
    attached
        .master
        .resize(PtySize {
            rows: 18,
            cols: 64,
            pixel_width: 0,
            pixel_height: 0,
        })
        .unwrap();
    until(|| panes(&binary, &config, &socket)[0]["size"]["cols"] == 64);
    attached.detach();
    assert_eq!(panes(&binary, &config, &socket)[0]["pane_id"], pane_id);

    let listed = run(&binary, &config, &socket, &["remote", "list"]);
    assert!(listed.status.success());
    assert!(String::from_utf8_lossy(&listed.stdout).contains("integration"));
    let mut reattached = Terminal::spawn(
        &binary,
        &config,
        &socket,
        &["remote", "attach", "integration", "--", "/does/not/exist"],
    );
    until(|| reattached.sees("\x1b[?2004h"));
    reattached.writer.write_all(b"reattached-input\n").unwrap();
    until(|| {
        reattached.sees("reattached");
        let response = run(
            &binary,
            &config,
            &socket,
            &[
                "cli",
                "--no-auto-start",
                "--prefer-mux",
                "get-text",
                "--pane-id",
                &pane_id.to_string(),
            ],
        );
        response.status.success()
            && String::from_utf8_lossy(&response.stdout).contains("reattached-input")
    });
    assert_eq!(
        panes(&binary, &config, &socket).len(),
        1,
        "reattachment created a duplicate session"
    );
    reattached.detach();

    let mut picker = Terminal::spawn(&binary, &config, &socket, &["remote", "ui"]);
    until(|| picker.sees("Persistent"));
    assert!(
        picker.sees("48;2;17;34;51"),
        "picker did not inherit the configured background"
    );
    picker.writer.write_all(b"\x1b").unwrap();
    until(|| picker.child.try_wait().unwrap().is_some());

    let original_config = std::fs::read_to_string(&config).unwrap();
    std::fs::write(
        &config,
        original_config.replace("interface='cli'", "interface='ui'"),
    )
    .unwrap();
    let mut default_ui = Terminal::spawn(&binary, &config, &socket, &["remote"]);
    until(|| default_ui.sees("Persistent"));
    default_ui.writer.write_all(b"\x1b").unwrap();
    until(|| default_ui.child.try_wait().unwrap().is_some());
    std::fs::write(
        &config,
        original_config.replace("interface='cli'", "interface='invalid'"),
    )
    .unwrap();
    assert!(!run(&binary, &config, &socket, &["remote", "list"])
        .status
        .success());
    std::fs::write(&config, original_config).unwrap();

    drop(_server);
    let _daemon = Daemon(pid_file);
    let mut autostarted = Terminal::spawn(
        &binary,
        &config,
        &socket,
        &[
            "remote",
            "attach",
            "autostart",
            "--",
            "/bin/sh",
            "-c",
            "stty raw -echo; printf 'autostart-ready\\r\\n'; cat",
        ],
    );
    until(|| autostarted.sees("autostart-ready"));
    autostarted.detach();
}
