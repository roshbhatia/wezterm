use crate::activity::Activity;
use crate::domain::{alloc_domain_id, Domain, DomainId, DomainState, SplitSource};
use crate::pane::{Pane, PaneId};
use crate::tab::{SplitRequest, Tab, TabId};
use crate::tmux_commands::{
    CommandReply, DetachClient, ListAllPanes, ListAllWindows, ListCommands, NewWindow, SplitPane,
    TmuxCommand,
};
use crate::window::WindowId;
use crate::{Mux, MuxWindowBuilder};
use anyhow::Context;
use async_trait::async_trait;
use filedescriptor::FileDescriptor;
use parking_lot::{Condvar, Mutex};
use portable_pty::CommandBuilder;
use std::collections::{HashMap, HashSet, VecDeque};
use std::io::Write;
use std::sync::Arc;
use termwiz::tmux_cc::*;
use wezterm_term::TerminalSize;

#[derive(PartialEq, Eq, Debug, Copy, Clone)]
pub enum AttachState {
    Init,
    Done,
}

#[derive(PartialEq, Eq, Debug, Copy, Clone)]
enum State {
    WaitForInitialGuard,
    Idle,
    WaitingForResponse,
    ProcessingResponse,
    Exit,
}

#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct TmuxRemotePane {
    // members for local
    pub local_pane_id: PaneId,
    pub output_write: FileDescriptor,
    pub active_lock: Arc<(Mutex<bool>, Condvar)>,
    // members sync with remote
    pub session_id: TmuxSessionId,
    pub window_id: TmuxWindowId,
    pub pane_id: TmuxPaneId,
    pub cursor_x: u64,
    pub cursor_y: u64,
    pub pane_width: u64,
    pub pane_height: u64,
    pub pane_left: u64,
    pub pane_top: u64,
}

pub(crate) type RefTmuxRemotePane = Arc<Mutex<TmuxRemotePane>>;

impl TmuxRemotePane {
    pub fn release(&self) {
        let (released, wake) = &*self.active_lock;
        *released.lock() = true;
        wake.notify_all();
    }
}

/// As a remote TmuxTab, keeping the TmuxPanes ID
/// within the remote tab.
#[allow(dead_code)]
pub(crate) struct TmuxTab {
    pub tab_id: TabId, // local tab ID
    pub tmux_window_id: TmuxWindowId,
    pub layout_csum: String,
    pub panes: HashSet<TmuxPaneId>, // tmux panes within tmux window
}

pub(crate) type TmuxCmdQueue = VecDeque<Box<dyn TmuxCommand>>;
pub(crate) struct TmuxDomainState {
    pub pane_id: PaneId,     // ID of the original pane
    pub domain_id: DomainId, // ID of TmuxDomain
    state: Mutex<State>,
    pub cmd_queue: Arc<Mutex<TmuxCmdQueue>>,
    pub gui_window: Mutex<Option<MuxWindowBuilder>>,
    pub gui_tabs: Mutex<HashMap<TmuxWindowId, TmuxTab>>,
    pub remote_panes: Mutex<HashMap<TmuxPaneId, RefTmuxRemotePane>>,
    pub tmux_session: Mutex<Option<TmuxSessionId>>,
    pub support_commands: Mutex<HashMap<String, String>>,
    pub attach_state: Mutex<AttachState>,
    pub backlog: Mutex<HashMap<TmuxPaneId, Vec<u8>>>,
}

pub struct TmuxDomain {
    pub(crate) inner: Arc<TmuxDomainState>,
}

impl TmuxDomainState {
    pub fn is_connected(&self) -> bool {
        *self.state.lock() != State::Exit
    }

    fn stop(&self) {
        *self.state.lock() = State::Exit;
        for command in std::mem::take(&mut *self.cmd_queue.lock()) {
            command.cancel();
        }
        for pane in self.remote_panes.lock().values() {
            pane.lock().release();
        }
    }

    pub fn disconnected(&self) {
        self.stop();
        let domain_id = self.domain_id;
        let pane_id = self.pane_id;
        promise::spawn::spawn_into_main_thread(async move {
            if let Some(mux) = Mux::try_get() {
                if let Some(pane) = mux.get_pane(pane_id) {
                    if let Some(pane) = pane.downcast_ref::<crate::localpane::LocalPane>() {
                        pane.clear_tmux_domain(domain_id);
                    }
                }
                if let Some(domain) = mux.get_domain(domain_id) {
                    if let Some(domain) = domain.downcast_ref::<TmuxDomain>() {
                        let window = domain.inner.gui_window.lock().take();
                        drop(window);
                    }
                }
                mux.domain_was_detached(domain_id);
                mux.remove_tmux_domain(domain_id);
            }
        })
        .detach();
    }

    pub fn advance(&self, events: Box<Vec<Event>>) {
        for event in events.iter() {
            let state = *self.state.lock();
            log::debug!("tmux: {:?} in state {:?}", event, state);
            match event {
                // Tmux generic events
                Event::Guarded(response) => match state {
                    State::WaitForInitialGuard => {
                        if response.error {
                            self.disconnected();
                            return;
                        }
                        *self.state.lock() = State::Idle;
                    }
                    State::WaitingForResponse => {
                        let mut cmd_queue = self.cmd_queue.as_ref().lock();
                        if let Some(cmd) = cmd_queue.pop_front() {
                            let domain_id = self.domain_id;
                            *self.state.lock() = State::ProcessingResponse;
                            let resp = response.clone();
                            promise::spawn::spawn_into_main_thread(async move {
                                if let Err(err) = cmd.process_result(domain_id, &resp) {
                                    log::error!("Tmux processing command result error: {}", err);
                                }
                                if let Some(domain) = Mux::get().get_domain(domain_id) {
                                    if let Some(tmux) = domain.downcast_ref::<TmuxDomain>() {
                                        let mut state = tmux.inner.state.lock();
                                        if *state == State::ProcessingResponse {
                                            *state = State::Idle;
                                        }
                                        drop(state);
                                        tmux.send_next_command();
                                    }
                                }
                            })
                            .detach();
                        }
                    }
                    State::Idle | State::ProcessingResponse | State::Exit => {}
                },

                // Tmux specific events
                Event::ConfigError { error } => {
                    // tmux config file error, not our fault, just log it and go
                    log::warn!("tmux configuration error: {error}");
                }
                Event::Exit { reason: _ } => {
                    self.stop();

                    // Force to quit the tmux mode
                    let (pane_id, domain_id) = (self.pane_id, self.domain_id);
                    promise::spawn::spawn_into_main_thread_with_low_priority(async move {
                        let mux = Mux::get();
                        if mux.get_domain(domain_id).is_some() {
                            if let Some(x) = mux.get_pane(pane_id) {
                                let _ = writeln!(x.writer());
                            }
                        }
                    })
                    .detach();

                    return;
                }
                Event::LayoutChange {
                    window,
                    layout,
                    visible_layout: _,
                    raw_flags: _,
                } => {
                    let mut cmd_queue = self.cmd_queue.as_ref().lock();
                    cmd_queue.push_back(Box::new(ListAllPanes {
                        window_id: *window,
                        prune: true,
                        layout_csum: if let Some(l) = layout.get(0..4) {
                            l.to_string()
                        } else {
                            "".to_string()
                        },
                    }));
                }
                Event::Output { pane, text } | Event::ExtendedOutput { pane, text } => {
                    let pane_map = self.remote_panes.lock();
                    if let Some(ref_pane) = pane_map.get(pane).cloned() {
                        drop(pane_map);
                        let mut tmux_pane = ref_pane.lock();
                        if let Err(err) = tmux_pane.output_write.write_all(text) {
                            log::error!("Failed to write tmux data to output: {:#}", err);
                        }
                    } else {
                        // the output may come early then pane is ready, in this case we
                        // backlog it
                        self.backlog.lock().entry(*pane).or_default().extend(text);
                        log::debug!("Tmux pane {} haven't been attached", pane);
                    }
                }
                Event::SessionChanged { session, name: _ } => {
                    let previous = self.tmux_session.lock().replace(*session);
                    let mut cmd_queue = self.cmd_queue.as_ref().lock();
                    cmd_queue.push_back(Box::new(ListCommands));

                    if previous.is_none() {
                        self.subscribe_notification();
                    }
                    log::info!("tmux session changed:{}", session);
                }
                Event::WindowAdd { window } => {
                    // Only handle the new tab, the first empty window handled by sync_window_state
                    if !self.gui_window.lock().is_none() {
                        if let Some(session) = *self.tmux_session.lock() {
                            let mut cmd_queue = self.cmd_queue.as_ref().lock();
                            cmd_queue.push_back(Box::new(ListAllWindows {
                                session_id: session,
                                window_id: Some(*window),
                                reply: None,
                            }));
                            log::info!("tmux window add: {}:{}", session, window);
                        }
                    }
                }
                Event::WindowPaneChanged { window, pane } => {
                    log::info!("tmux window pane changed: {}:{}", window, pane);
                }
                Event::WindowRenamed { window, name } => {
                    let gui_tabs = self.gui_tabs.lock();
                    if let Some(x) = gui_tabs.get(&window) {
                        let mux = Mux::get();
                        if let Some(tab) = mux.get_tab(x.tab_id) {
                            tab.set_title(&format!("{}", name));
                        }
                    }
                }
                Event::WindowClose { window } | Event::UnlinkedWindowClose { window } => {
                    let (domain_id, window_id) = (self.domain_id, *window);
                    promise::spawn::spawn_into_main_thread(async move {
                        if let Some(domain) = Mux::get().get_domain(domain_id) {
                            if let Some(domain) = domain.downcast_ref::<TmuxDomain>() {
                                let _ = domain.inner.remove_detached_window(window_id);
                            }
                        }
                    })
                    .detach();
                }
                _ => {}
            }
        }

        // send pending commands to tmux
        let cmd_queue = self.cmd_queue.as_ref().lock();
        if *self.state.lock() == State::Idle && !cmd_queue.is_empty() {
            TmuxDomainState::schedule_send_next_command(self.domain_id);
        }
    }

    /// send next command at the front of cmd_queue.
    /// must be called inside main thread
    fn send_next_command(&self) {
        if *self.state.lock() != State::Idle {
            return;
        }
        let mut cmd_queue = self.cmd_queue.as_ref().lock();
        while let Some(first) = cmd_queue.front() {
            let cmd = first.get_command(self.domain_id);
            if cmd.is_empty() {
                cmd_queue.pop_front();
                continue;
            }
            log::debug!("sending cmd {:?}", cmd);
            *self.state.lock() = State::WaitingForResponse;
            drop(cmd_queue);
            let write_result = Mux::get()
                .get_pane(self.pane_id)
                .context("tmux control pane closed")
                .and_then(|pane| {
                    let mut writer = pane.writer();
                    writer.write_all(cmd.as_bytes())?;
                    writer.flush()?;
                    Ok(())
                });
            if let Err(err) = write_result {
                log::debug!("tmux control write failed: {err:#}");
                self.disconnected();
            }
            break;
        }
    }

    /// schedule a `send_next_command` into main thread
    pub fn schedule_send_next_command(domain_id: usize) {
        promise::spawn::spawn_into_main_thread(async move {
            let mux = Mux::get();
            if let Some(domain) = mux.get_domain(domain_id) {
                if let Some(tmux_domain) = domain.downcast_ref::<TmuxDomain>() {
                    tmux_domain.send_next_command();
                }
            }
        })
        .detach();
    }

    /// create a standalone window for tmux tabs
    pub fn create_gui_window(&self) {
        if self.gui_window.lock().is_none() {
            let mux = Mux::get();
            let window_builder =
                if let Some((_domain, window_id, _tab)) = mux.resolve_pane_id(self.pane_id) {
                    MuxWindowBuilder {
                        window_id,
                        activity: Some(Activity::new()),
                        notified: false,
                    }
                } else {
                    mux.new_empty_window(
                        None, /* TODO: pass session here */
                        None, /* position */
                    )
                };

            log::info!("Tmux create window id {}", window_builder.window_id);
            {
                let mut window_id = self.gui_window.lock();
                *window_id = Some(window_builder); // keep the builder so it won't be purged
            }
        };
    }

    /// create a tmux window
    fn create_tmux_window(&self, args: String) -> anyhow::Result<promise::Future<Arc<Tab>>> {
        let session = (*self.tmux_session.lock()).context("tmux session is not attached")?;
        let (reply, future) = CommandReply::new();
        let mut cmd_queue = self.cmd_queue.as_ref().lock();
        anyhow::ensure!(self.is_connected(), "tmux client detached");
        cmd_queue.push_back(Box::new(NewWindow {
            session,
            args,
            reply: Arc::new(reply),
        }));
        TmuxDomainState::schedule_send_next_command(self.domain_id);
        Ok(future)
    }

    /// split the tmux pane
    fn split_tmux_pane(
        &self,
        _tab: TabId,
        pane_id: PaneId,
        split_request: SplitRequest,
        args: String,
    ) -> anyhow::Result<promise::Future<TmuxPaneId>> {
        let tmux_pane_id = self
            .remote_panes
            .lock()
            .iter()
            .find(|(_, ref_pane)| ref_pane.lock().local_pane_id == pane_id)
            .map(|p| p.1.lock().pane_id);

        if let Some(id) = tmux_pane_id {
            let mut reply = promise::Promise::new();
            let future = reply.get_future().unwrap();
            let mut cmd_queue = self.cmd_queue.as_ref().lock();
            anyhow::ensure!(*self.state.lock() != State::Exit, "tmux client detached");
            cmd_queue.push_back(Box::new(SplitPane {
                pane_id: id,
                request: split_request,
                args,
                reply: Mutex::new(reply),
            }));
            TmuxDomainState::schedule_send_next_command(self.domain_id);
            return Ok(future);
        } else {
            anyhow::bail!("Could not find the tmux pane peer for local pane: {pane_id}");
        }
    }
}

fn spawn_arguments(
    command: Option<&CommandBuilder>,
    directory: Option<&str>,
) -> anyhow::Result<String> {
    fn quote(value: &str) -> anyhow::Result<String> {
        anyhow::ensure!(
            !value.contains(['\n', '\r', '\0']),
            "tmux arguments cannot contain control-mode line delimiters"
        );
        let value = value
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('$', "\\$");
        Ok(format!("\"{value}\""))
    }

    let mut args = String::new();
    let cwd = directory.or_else(|| command.and_then(|cmd| cmd.get_cwd()?.to_str()));
    if let Some(cwd) = cwd {
        args.push_str(" -c ");
        args.push_str(&quote(cwd)?);
    }
    if let Some(command) = command {
        for (key, value) in command.iter_extra_env_as_str() {
            args.push_str(" -e ");
            args.push_str(&quote(&format!("{key}={value}"))?);
        }
        if !command.is_default_prog() {
            args.push_str(" --");
            for arg in command.get_argv() {
                args.push(' ');
                args.push_str(&quote(
                    arg.to_str().context("tmux arguments must be UTF-8")?,
                )?);
            }
        }
    }
    Ok(args)
}

impl TmuxDomain {
    pub fn new(pane_id: PaneId) -> Self {
        let domain_id = alloc_domain_id();
        let cmd_queue = VecDeque::new();
        let inner = Arc::new(TmuxDomainState {
            domain_id,
            pane_id,
            // parser,
            state: Mutex::new(State::WaitForInitialGuard),
            cmd_queue: Arc::new(Mutex::new(cmd_queue)),
            gui_window: Mutex::new(None),
            gui_tabs: Mutex::new(HashMap::default()),
            remote_panes: Mutex::new(HashMap::default()),
            tmux_session: Mutex::new(None),
            support_commands: Mutex::new(HashMap::default()),
            attach_state: Mutex::new(AttachState::Init),
            backlog: Mutex::new(HashMap::default()),
        });

        Self { inner }
    }

    fn send_next_command(&self) {
        self.inner.send_next_command();
    }
}

#[async_trait(?Send)]
impl Domain for TmuxDomain {
    async fn spawn(
        &self,
        _size: TerminalSize,
        command: Option<CommandBuilder>,
        command_dir: Option<String>,
        _window: WindowId,
    ) -> anyhow::Result<Arc<Tab>> {
        let args = spawn_arguments(command.as_ref(), command_dir.as_deref())?;
        self.inner.create_tmux_window(args)?.await
    }

    async fn split_pane(
        &self,
        source: SplitSource,
        tab: TabId,
        pane_id: PaneId,
        split_request: SplitRequest,
    ) -> anyhow::Result<Arc<dyn Pane>> {
        let SplitSource::Spawn {
            command,
            command_dir,
        } = source
        else {
            anyhow::bail!("tmux split cannot move an existing WezTerm pane");
        };
        let args = spawn_arguments(command.as_ref(), command_dir.as_deref())?;
        let id = self
            .inner
            .split_tmux_pane(tab, pane_id, split_request, args)?
            .await?;
        self.inner.split_pane(tab, pane_id, id, split_request)
    }

    async fn spawn_pane(
        &self,
        _size: TerminalSize,
        _command: Option<CommandBuilder>,
        _command_dir: Option<String>,
    ) -> anyhow::Result<Arc<dyn Pane>> {
        anyhow::bail!("Spawn_pane not yet implemented for TmuxDomain");
    }

    fn domain_id(&self) -> DomainId {
        self.inner.domain_id
    }

    fn domain_name(&self) -> &str {
        "tmux"
    }

    async fn attach(&self, _window_id: Option<crate::WindowId>) -> anyhow::Result<()> {
        Ok(())
    }

    fn detachable(&self) -> bool {
        self.state() == DomainState::Attached
    }

    fn detach(&self) -> anyhow::Result<()> {
        let mut queue = self.inner.cmd_queue.lock();
        anyhow::ensure!(
            *self.inner.state.lock() != State::Exit,
            "tmux client detached"
        );
        queue.push_back(Box::new(DetachClient));
        TmuxDomainState::schedule_send_next_command(self.domain_id());
        Ok(())
    }

    fn state(&self) -> DomainState {
        if *self.inner.state.lock() != State::Exit
            && *self.inner.attach_state.lock() == AttachState::Done
        {
            DomainState::Attached
        } else {
            DomainState::Detached
        }
    }
}
