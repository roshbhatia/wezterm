use anyhow::{bail, Context};
use clap::{Parser, Subcommand};
use mux::connui::ConnectionUI;
use portable_pty::CommandBuilder;
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::io::{self, IsTerminal};
use wezterm_client::client::Client;

#[derive(Debug, Clone, Parser)]
pub struct RemoteCommand {
    #[command(subcommand)]
    command: Option<RemoteAction>,
}

#[derive(Debug, Clone, Subcommand)]
enum RemoteAction {
    /// List named sessions without starting a missing mux.
    List,
    /// Attach a named session, creating it when it does not exist.
    Attach {
        name: String,
        /// Run this command only when creating the session; defaults to the configured shell.
        #[arg(last = true)]
        command: Vec<OsString>,
    },
    /// Open the themed picker and controls.
    Ui,
}

impl RemoteCommand {
    pub fn run(self, opts: &crate::Opt) -> anyhow::Result<()> {
        crate::init_config(opts)?;
        let config =
            config::configuration_result().context("loading remote client configuration")?;
        let action = self.command.unwrap_or_else(|| {
            if config.remote_client.interface == "ui" {
                RemoteAction::Ui
            } else {
                RemoteAction::List
            }
        });
        if matches!(action, RemoteAction::Ui) {
            return crate::tui::TuiCommand::picker().run(opts);
        }
        if let RemoteAction::Attach { name, .. } = &action {
            validate_name(name)?;
            if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
                bail!("wezterm remote attach requires an interactive terminal (use ssh -t)");
            }
        }
        let executor = promise::spawn::ScopedExecutor::new();
        let pane = promise::spawn::block_on(executor.run(async {
            let mut ui = ConnectionUI::new_headless();
            let domain =
                Client::default_unix_domain(true, wezterm_gui_subcommands::DEFAULT_WINDOW_CLASS)?;
            let create = matches!(action, RemoteAction::Attach { .. });
            let client = match Client::new_unix_domain(None, &domain, true, &mut ui, !create) {
                Ok(client) => client,
                Err(_) if !create && !domain.socket_path().exists() => {
                    println!("No persistent sessions. Create one with: wezterm-remote attach NAME");
                    return Ok(None);
                }
                Err(error) => return Err(error).context("connecting to the persistent mux"),
            };
            client.verify_version_compat(&ui).await?;
            let mut panes = Vec::new();
            for tab in client.list_panes().await?.tabs {
                crate::tui::collect_panes(tab.panes, &mut panes);
            }
            panes.sort_by_key(|pane| pane.pane_id);
            match action {
                RemoteAction::List => {
                    let mut sessions = BTreeMap::new();
                    for pane in &panes {
                        let entry = sessions.entry(&pane.workspace).or_insert((0, &pane.title));
                        entry.0 += 1;
                    }
                    if sessions.is_empty() {
                        println!(
                            "No persistent sessions. Create one with: wezterm-remote attach NAME"
                        );
                    } else {
                        println!("{:<24} {:>5}  TITLE", "NAME", "PANES");
                        for (name, (count, title)) in sessions {
                            println!("{name:<24} {count:>5}  {title}");
                        }
                    }
                    Ok(None)
                }
                RemoteAction::Attach { name, command } => {
                    if let Some(pane) = panes.iter().find(|pane| pane.workspace == name) {
                        return Ok(Some(pane.pane_id));
                    }
                    let (cols, rows) = ratatui::crossterm::terminal::size()?;
                    let result = client
                        .spawn_v2(codec::SpawnV2 {
                            domain: config::keyassignment::SpawnTabDomain::DefaultDomain,
                            window_id: None,
                            command: if command.is_empty() {
                                None
                            } else {
                                Some(CommandBuilder::from_argv(command))
                            },
                            command_dir: None,
                            size: wezterm_term::TerminalSize {
                                rows: usize::from(rows.max(1)),
                                cols: usize::from(cols.max(1)),
                                pixel_width: 0,
                                pixel_height: 0,
                                dpi: 96,
                            },
                            workspace: name,
                        })
                        .await?;
                    Ok(Some(result.pane_id))
                }
                RemoteAction::Ui => unreachable!(),
            }
        }))?;
        if let Some(pane_id) = pane {
            crate::tui::TuiCommand::direct(pane_id).run(opts)?;
        }
        Ok(())
    }
}

fn validate_name(name: &str) -> anyhow::Result<()> {
    if name.is_empty() || name.chars().any(|c| c.is_control() || c.is_whitespace()) {
        bail!("session names must be nonempty and contain no whitespace or control characters");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_name_is_validated_before_connecting() {
        for name in ["k9s", "project-api", "build.42"] {
            assert!(validate_name(name).is_ok());
        }
        for name in ["", "two names", "bad\nname", "bad\x1bname"] {
            assert!(validate_name(name).is_err());
        }
    }

    #[test]
    fn command_arguments_are_preserved() {
        let parsed = RemoteCommand::try_parse_from([
            "remote",
            "attach",
            "work",
            "--",
            "nu",
            "-c",
            "echo quoted value",
        ])
        .unwrap();
        let Some(RemoteAction::Attach { name, command }) = parsed.command else {
            panic!("expected attach")
        };
        assert_eq!(name, "work");
        assert_eq!(
            command,
            ["nu", "-c", "echo quoted value"].map(OsString::from)
        );
    }
}
