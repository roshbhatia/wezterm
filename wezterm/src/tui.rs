use anyhow::{bail, Context};
use clap::Parser;
use codec::{GetLines, GetPaneRenderChanges, Pdu, SendKeyDown, SendPaste};
use mux::connui::ConnectionUI;
use mux::renderable::{RenderableDimensions, StableCursorPosition};
use mux::tab::{PaneEntry, PaneNode};
use ratatui::buffer::{Buffer, CellDiffOption};
use ratatui::crossterm::event::{
    self, DisableBracketedPaste, EnableBracketedPaste, Event, KeyCode, KeyEvent, KeyEventKind,
    KeyModifiers,
};
use ratatui::crossterm::execute;
use ratatui::crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::layout::Rect;
use ratatui::prelude::{Color, CrosstermBackend, Modifier, Style, Widget};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph};
use ratatui::{Frame, Terminal};
use smol::channel::Receiver;
use std::collections::BTreeMap;
use std::convert::TryFrom;
use std::io::{self, IsTerminal};
use std::num::NonZeroU16;
use std::time::{Duration, Instant};
use termwiz::cell::{Intensity, Underline};
use termwiz::color::{ColorAttribute, SrgbaTuple};
use termwiz::input::{KeyCode as MuxKey, KeyEvent as MuxKeyEvent, Modifiers};
use wezterm_client::client::Client;
use wezterm_term::color::ColorPalette;
use wezterm_term::{Line, StableRowIndex};

#[derive(Debug, Clone, Parser)]
pub struct TuiCommand {
    /// Attach directly instead of opening the pane picker.
    #[arg(long)]
    pane_id: Option<usize>,
    /// Observe the pane without sending keyboard or paste input.
    #[arg(long)]
    read_only: bool,
    /// Start the configured mux server if its socket is unavailable.
    #[arg(long)]
    start_server: bool,
    #[arg(skip)]
    direct: bool,
}

impl TuiCommand {
    pub(crate) fn picker() -> Self {
        Self {
            pane_id: None,
            read_only: false,
            start_server: false,
            direct: false,
        }
    }

    pub(crate) fn direct(pane_id: usize) -> Self {
        Self {
            pane_id: Some(pane_id),
            direct: true,
            ..Self::picker()
        }
    }

    pub fn run(self, opts: &crate::Opt) -> anyhow::Result<()> {
        if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
            bail!("wezterm tui requires an interactive terminal (use ssh -t)");
        }
        crate::init_config(opts)?;
        let executor = promise::spawn::ScopedExecutor::new();
        promise::spawn::block_on(executor.run(self.run_async()))
    }

    async fn run_async(self) -> anyhow::Result<()> {
        let mut ui = ConnectionUI::new_headless();
        let domain =
            Client::default_unix_domain(true, wezterm_gui_subcommands::DEFAULT_WINDOW_CLASS)?;
        let (client, events) =
            Client::new_unix_domain_with_events(&domain, &mut ui, !self.start_server).context(
                "connecting to the persistent mux; use --start-server to start it explicitly",
            )?;
        client.verify_version_compat(&ui).await?;
        let mut app = App::new(client, events, self.read_only);
        app.show_ui = !self.direct;
        app.resize_session = self.direct;
        app.refresh().await?;
        if let Some(id) = self.pane_id {
            let entry = app
                .panes
                .iter()
                .find(|p| p.pane_id == id)
                .cloned()
                .with_context(|| format!("pane {id} does not exist in this mux"))?;
            app.attach(entry).await?;
        }
        if self.direct {
            let (cols, rows) = ratatui::crossterm::terminal::size()?;
            app.resize(cols, rows).await?;
        }

        let _restore = RestoreTerminal;
        enable_raw_mode()?;
        execute!(io::stdout(), EnterAlternateScreen, EnableBracketedPaste)?;
        let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;
        terminal.clear()?;
        let mut last_refresh = Instant::now();
        let mut needs_draw = true;
        loop {
            while let Ok(pdu) = app.events.try_recv() {
                app.apply_event(pdu);
                needs_draw = true;
            }
            if app.events.is_closed() {
                bail!("mux disconnected; the pane may have ended or the mux restarted. Reconnect through the picker; input was not replayed");
            }
            if self.direct && app.screen.is_none() {
                return Ok(());
            }
            if app.screen.is_none() && last_refresh.elapsed() >= Duration::from_secs(2) {
                app.refresh().await?;
                last_refresh = Instant::now();
                needs_draw = true;
            }
            if needs_draw {
                let size = terminal.size()?;
                app.fetch_view(
                    size.width,
                    size.height.saturating_sub(u16::from(app.show_ui)),
                )
                .await?;
                terminal.draw(|frame| app.draw(frame))?;
                needs_draw = false;
            }
            if event::poll(Duration::from_millis(33))? {
                if app.handle(event::read()?).await? {
                    return Ok(());
                }
                needs_draw = true;
            }
        }
    }
}

struct RestoreTerminal;

impl Drop for RestoreTerminal {
    fn drop(&mut self) {
        let _ = execute!(io::stdout(), DisableBracketedPaste, LeaveAlternateScreen);
        let _ = disable_raw_mode();
    }
}

struct Screen {
    entry: PaneEntry,
    dimensions: RenderableDimensions,
    cursor: StableCursorPosition,
    lines: BTreeMap<StableRowIndex, Line>,
    palette: ColorPalette,
    history_top: Option<StableRowIndex>,
}

impl Screen {
    fn view_origin(&self, width: u16, height: u16) -> (usize, StableRowIndex) {
        let x = self
            .cursor
            .x
            .saturating_sub(usize::from(width).saturating_sub(1));
        let top = self.history_top.unwrap_or_else(|| {
            self.dimensions.physical_top.max(
                self.cursor
                    .y
                    .saturating_sub(isize::try_from(height).unwrap_or(0).saturating_sub(1)),
            )
        });
        (x, top)
    }

    fn insert_lines(&mut self, lines: codec::SerializedLines) {
        for (row, line) in lines.extract_data().0 {
            if self
                .lines
                .get(&row)
                .is_none_or(|old| old.current_seqno() <= line.current_seqno())
            {
                self.lines.insert(row, line);
            }
        }
    }
}

struct App {
    client: Client,
    events: Receiver<Pdu>,
    panes: Vec<PaneEntry>,
    selected: usize,
    query: String,
    screen: Option<Screen>,
    read_only: bool,
    menu: bool,
    menu_selected: usize,
    message: String,
    show_ui: bool,
    resize_session: bool,
    palette: ColorPalette,
}

impl App {
    fn new(client: Client, events: Receiver<Pdu>, read_only: bool) -> Self {
        Self {
            client,
            events,
            panes: Vec::new(),
            selected: 0,
            query: String::new(),
            screen: None,
            read_only,
            menu: false,
            menu_selected: 0,
            message: String::new(),
            show_ui: true,
            resize_session: false,
            palette: config::configuration().resolved_palette.clone().into(),
        }
    }

    async fn refresh(&mut self) -> anyhow::Result<()> {
        let response = self.client.list_panes().await?;
        self.panes.clear();
        for tab in response.tabs {
            collect_panes(tab.panes, &mut self.panes);
            for node in tab.floating_panes {
                collect_panes(node, &mut self.panes);
            }
        }
        self.panes.sort_by(|a, b| {
            (&a.workspace, a.tab_id, a.pane_id).cmp(&(&b.workspace, b.tab_id, b.pane_id))
        });
        self.selected = self.selected.min(self.filtered().len().saturating_sub(1));
        Ok(())
    }

    fn filtered(&self) -> Vec<&PaneEntry> {
        let query = self.query.to_lowercase();
        self.panes
            .iter()
            .filter(|p| {
                format!(
                    "{} {} {} {}",
                    p.workspace,
                    p.title,
                    p.pane_id,
                    p.working_dir.as_ref().map(|u| u.url.path()).unwrap_or("")
                )
                .to_lowercase()
                .contains(&query)
            })
            .collect()
    }

    async fn attach(&mut self, entry: PaneEntry) -> anyhow::Result<()> {
        let info = self
            .client
            .get_dimensions(codec::GetPaneRenderableDimensions {
                pane_id: entry.pane_id,
            })
            .await?;
        self.screen = Some(Screen {
            entry: entry.clone(),
            dimensions: info.dimensions,
            cursor: info.cursor_position,
            lines: BTreeMap::new(),
            palette: self.palette.clone(),
            history_top: None,
        });
        self.client
            .get_pane_render_changes(GetPaneRenderChanges {
                pane_id: entry.pane_id,
            })
            .await?;
        self.menu = false;
        self.message.clear();
        Ok(())
    }

    fn apply_event(&mut self, event: Pdu) {
        let Some(screen) = &mut self.screen else {
            return;
        };
        let id = screen.entry.pane_id;
        match event {
            Pdu::GetPaneRenderChangesResponse(delta) if delta.pane_id == id => {
                screen.dimensions = delta.dimensions;
                screen.cursor = delta.cursor_position;
                screen.entry.title = delta.title;
                screen.insert_lines(delta.bonus_lines);
                for range in delta.dirty_lines {
                    screen.lines.retain(|row, _| !range.contains(row));
                }
                let oldest = screen.dimensions.scrollback_top;
                screen.lines.retain(|row, _| *row >= oldest);
                if let Some(top) = &mut screen.history_top {
                    *top = (*top).clamp(oldest, screen.dimensions.physical_top);
                }
            }
            Pdu::SetPalette(update) if update.pane_id == id => {
                screen.palette = update.palette;
            }
            Pdu::PaneRemoved(update) if update.pane_id == id => {
                self.screen = None;
                self.menu = false;
                self.message =
                    format!("Pane {id} ended. Select another pane or press Esc for the shell.");
            }
            _ => {}
        }
    }

    async fn fetch_view(&mut self, width: u16, height: u16) -> anyhow::Result<()> {
        let Some(screen) = &mut self.screen else {
            return Ok(());
        };
        let (_, top) = screen.view_origin(width, height);
        let end = (top + isize::try_from(height)?)
            .min(screen.dimensions.physical_top + screen.dimensions.viewport_rows as isize);
        if (top..end).any(|row| !screen.lines.contains_key(&row)) {
            let response = self
                .client
                .get_lines(GetLines {
                    pane_id: screen.entry.pane_id,
                    lines: vec![top..end],
                })
                .await?;
            screen.insert_lines(response.lines);
        }
        Ok(())
    }

    fn draw(&self, frame: &mut Frame) {
        let area = frame.area();
        let palette = self
            .screen
            .as_ref()
            .map(|screen| &screen.palette)
            .unwrap_or(&self.palette);
        let normal = Style::default()
            .fg(rgb(palette.foreground))
            .bg(rgb(palette.background));
        let selected = Style::default()
            .fg(rgb(palette.selection_fg))
            .bg(rgb(palette.selection_bg));
        frame.render_widget(Block::default().style(normal), area);
        let body = Rect::new(
            area.x,
            area.y,
            area.width,
            area.height.saturating_sub(u16::from(self.show_ui)),
        );
        if let Some(screen) = &self.screen {
            let (left, top) = screen.view_origin(body.width, body.height);
            frame.render_widget(
                PaneViewport {
                    lines: &screen.lines,
                    palette: &screen.palette,
                    reverse: screen.dimensions.reverse_video,
                    left,
                    top,
                },
                body,
            );
            if screen.history_top.is_none()
                && !self.menu
                && screen.cursor.visibility == termwiz::surface::CursorVisibility::Visible
            {
                let x = screen.cursor.x.saturating_sub(left);
                let y = screen.cursor.y - top;
                if x < usize::from(body.width)
                    && y >= 0
                    && y < isize::try_from(body.height).unwrap_or(0)
                {
                    frame.set_cursor_position((body.x + x as u16, body.y + y as u16));
                }
            }
            let mode = if screen.history_top.is_some() {
                "HISTORY"
            } else if self.read_only {
                "VIEW ONLY"
            } else {
                "SHARED"
            };
            let status = if self.menu {
                "Controls | arrows/Enter | Esc back | Ctrl+B literal".to_owned()
            } else if screen.history_top.is_some() {
                "History | arrows/PageUp/PageDown | Esc back".to_owned()
            } else {
                format!(
                    "{mode} | Ctrl+B menu | {} | {}x{} | {}",
                    screen.entry.workspace,
                    screen.dimensions.cols,
                    screen.dimensions.viewport_rows,
                    self.client.client_id.hostname
                )
            };
            if self.show_ui || screen.history_top.is_some() {
                frame.render_widget(
                    Paragraph::new(status).style(selected),
                    Rect::new(
                        area.x,
                        area.y + area.height.saturating_sub(1),
                        area.width,
                        u16::from(area.height > 0),
                    ),
                );
            }
            if self.menu {
                let menu = Rect::new(body.x, body.y, body.width.min(48), body.height.min(7));
                frame.render_widget(Clear, menu);
                let items = [
                    "p  Choose pane",
                    "h  Scrollback history",
                    if self.read_only {
                        "v  Enable shared input"
                    } else {
                        "v  View only"
                    },
                    "n  New mobile shell",
                    "d  Detach to SSH shell",
                ];
                let mut state = ListState::default().with_selected(Some(self.menu_selected));
                frame.render_stateful_widget(
                    List::new(items)
                        .style(normal)
                        .block(Block::default().borders(Borders::ALL).title("Controls"))
                        .highlight_style(selected),
                    menu,
                    &mut state,
                );
            }
        } else {
            let panes = self.filtered();
            let items: Vec<ListItem> = panes
                .iter()
                .map(|p| {
                    ListItem::new(format!(
                        "{}{} / {}\n  #{}  {}x{}  {}",
                        if p.is_floating_pane { "F " } else { "" },
                        p.workspace,
                        p.title,
                        p.pane_id,
                        p.size.cols,
                        p.size.rows,
                        p.working_dir.as_ref().map(|u| u.url.path()).unwrap_or("")
                    ))
                })
                .collect();
            let mut state = ListState::default().with_selected(if panes.is_empty() {
                None
            } else {
                Some(self.selected)
            });
            frame.render_stateful_widget(
                List::new(items)
                    .style(normal)
                    .block(
                        Block::default()
                            .borders(Borders::ALL)
                            .title(format!("Persistent panes | {}", self.query)),
                    )
                    .highlight_style(selected),
                body,
                &mut state,
            );
            let status = if self.message.is_empty() {
                "Enter open | Ctrl+N new | Esc shell | type to search | arrows select"
            } else {
                &self.message
            };
            frame.render_widget(
                Paragraph::new(status).style(normal),
                Rect::new(
                    area.x,
                    area.y + area.height.saturating_sub(1),
                    area.width,
                    u16::from(area.height > 0),
                ),
            );
        }
    }

    async fn handle(&mut self, event: Event) -> anyhow::Result<bool> {
        match event {
            Event::Resize(cols, rows) => self.resize(cols, rows).await?,
            Event::Paste(text) => {
                if self.screen.is_none() {
                    self.query.push_str(&text);
                    self.selected = 0;
                } else if !self.menu
                    && !self.read_only
                    && self
                        .screen
                        .as_ref()
                        .is_some_and(|s| s.history_top.is_none())
                {
                    self.client
                        .send_paste(SendPaste {
                            pane_id: self.screen.as_ref().unwrap().entry.pane_id,
                            data: text,
                        })
                        .await?;
                }
            }
            Event::Key(key) if key.kind != KeyEventKind::Release => {
                return self.handle_key(key).await
            }
            _ => {}
        }
        Ok(false)
    }

    async fn handle_key(&mut self, key: KeyEvent) -> anyhow::Result<bool> {
        let prefix =
            key.code == KeyCode::Char('b') && key.modifiers.contains(KeyModifiers::CONTROL);
        if self.screen.is_none() {
            match key.code {
                KeyCode::Esc => return Ok(true),
                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    return Ok(true)
                }
                KeyCode::Char('n') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    self.new_shell().await?
                }
                KeyCode::Up => self.selected = self.selected.saturating_sub(1),
                KeyCode::Down => {
                    self.selected = (self.selected + 1).min(self.filtered().len().saturating_sub(1))
                }
                KeyCode::Enter => {
                    if let Some(entry) = self.filtered().get(self.selected).map(|p| (*p).clone()) {
                        if let Err(err) = self.attach(entry).await {
                            self.screen = None;
                            self.message = format!("{err:#}");
                            self.refresh().await?;
                        }
                    }
                }
                KeyCode::Backspace => {
                    self.query.pop();
                    self.selected = 0;
                }
                KeyCode::Char(c)
                    if !key
                        .modifiers
                        .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
                {
                    self.query.push(c);
                    self.selected = 0;
                }
                _ => {}
            }
        } else if self.menu {
            if key.code == KeyCode::Up {
                self.menu_selected = self.menu_selected.saturating_sub(1);
                return Ok(false);
            }
            if key.code == KeyCode::Down {
                self.menu_selected = (self.menu_selected + 1).min(4);
                return Ok(false);
            }
            self.menu = false;
            if prefix {
                self.forward_key(key).await?;
            } else {
                match if key.code == KeyCode::Enter {
                    KeyCode::Char(['p', 'h', 'v', 'n', 'd'][self.menu_selected])
                } else {
                    key.code
                } {
                    KeyCode::Char('d') => return Ok(true),
                    KeyCode::Char('p') => {
                        self.screen = None;
                        self.query.clear();
                        self.refresh().await?;
                    }
                    KeyCode::Char('v') => self.read_only = !self.read_only,
                    KeyCode::Char('n') => self.new_shell().await?,
                    KeyCode::Char('h') => {
                        let s = self.screen.as_mut().unwrap();
                        s.history_top = Some(s.dimensions.physical_top);
                    }
                    _ => {}
                }
            }
        } else if prefix {
            self.menu = true;
            self.menu_selected = 0;
        } else if self
            .screen
            .as_ref()
            .is_some_and(|s| s.history_top.is_some())
        {
            let s = self.screen.as_mut().unwrap();
            let top = s.history_top.unwrap();
            s.history_top = match key.code {
                KeyCode::Esc | KeyCode::Char('q') => None,
                KeyCode::Up | KeyCode::Char('k') => {
                    Some((top - 1).max(s.dimensions.scrollback_top))
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    Some((top + 1).min(s.dimensions.physical_top))
                }
                KeyCode::PageUp => Some((top - 20).max(s.dimensions.scrollback_top)),
                KeyCode::PageDown => Some((top + 20).min(s.dimensions.physical_top)),
                KeyCode::Home => Some(s.dimensions.scrollback_top),
                KeyCode::End => Some(s.dimensions.physical_top),
                _ => Some(top),
            };
        } else {
            self.forward_key(key).await?;
        }
        Ok(false)
    }

    async fn forward_key(&self, key: KeyEvent) -> anyhow::Result<()> {
        if self.read_only {
            return Ok(());
        }
        if let Some(event) = mux_key(key) {
            self.client
                .key_down(SendKeyDown {
                    pane_id: self.screen.as_ref().unwrap().entry.pane_id,
                    event,
                    input_serial: codec::InputSerial::now(),
                })
                .await?;
        }
        Ok(())
    }

    async fn resize(&mut self, cols: u16, rows: u16) -> anyhow::Result<()> {
        if self.resize_session {
            if let Some(screen) = &mut self.screen {
                self.client
                    .resize(codec::Resize {
                        containing_tab_id: screen.entry.tab_id,
                        pane_id: screen.entry.pane_id,
                        size: wezterm_term::TerminalSize {
                            cols: usize::from(cols.max(1)),
                            rows: usize::from(rows.max(1)),
                            pixel_width: 0,
                            pixel_height: 0,
                            dpi: 96,
                        },
                    })
                    .await?;
                screen.lines.clear();
            }
        }
        Ok(())
    }

    async fn new_shell(&mut self) -> anyhow::Result<()> {
        let size = ratatui::crossterm::terminal::size()?;
        let response = self
            .client
            .spawn_v2(codec::SpawnV2 {
                domain: config::keyassignment::SpawnTabDomain::DefaultDomain,
                window_id: None,
                command: None,
                command_dir: None,
                size: wezterm_term::TerminalSize {
                    rows: usize::from(size.1.saturating_sub(1).max(1)),
                    cols: usize::from(size.0.max(1)),
                    pixel_width: 0,
                    pixel_height: 0,
                    dpi: 96,
                },
                workspace: "mobile".to_owned(),
            })
            .await?;
        self.refresh().await?;
        let entry = self
            .panes
            .iter()
            .find(|p| p.pane_id == response.pane_id)
            .cloned()
            .context("new pane was absent from the mux")?;
        self.read_only = false;
        self.attach(entry).await
    }
}

pub(crate) fn collect_panes(node: PaneNode, result: &mut Vec<PaneEntry>) {
    match node {
        PaneNode::Empty => {}
        PaneNode::Leaf(entry) => result.push(entry),
        PaneNode::Split { left, right, .. } => {
            collect_panes(*left, result);
            collect_panes(*right, result);
        }
    }
}

fn mux_key(key: KeyEvent) -> Option<MuxKeyEvent> {
    let code = match key.code {
        KeyCode::Char(c) => MuxKey::Char(c),
        KeyCode::Enter => MuxKey::Enter,
        KeyCode::Backspace => MuxKey::Backspace,
        KeyCode::Tab => MuxKey::Tab,
        KeyCode::BackTab => MuxKey::Tab,
        KeyCode::Esc => MuxKey::Escape,
        KeyCode::Left => MuxKey::LeftArrow,
        KeyCode::Right => MuxKey::RightArrow,
        KeyCode::Up => MuxKey::UpArrow,
        KeyCode::Down => MuxKey::DownArrow,
        KeyCode::Home => MuxKey::Home,
        KeyCode::End => MuxKey::End,
        KeyCode::PageUp => MuxKey::PageUp,
        KeyCode::PageDown => MuxKey::PageDown,
        KeyCode::Delete => MuxKey::Delete,
        KeyCode::Insert => MuxKey::Insert,
        KeyCode::F(n) => MuxKey::Function(n),
        _ => return None,
    };
    let mut modifiers = Modifiers::empty();
    for (source, target) in [
        (KeyModifiers::CONTROL, Modifiers::CTRL),
        (KeyModifiers::ALT, Modifiers::ALT),
        (KeyModifiers::SHIFT, Modifiers::SHIFT),
        (KeyModifiers::SUPER, Modifiers::SUPER),
    ] {
        if key.modifiers.contains(source) {
            modifiers |= target;
        }
    }
    if key.code == KeyCode::BackTab {
        modifiers |= Modifiers::SHIFT;
    }
    Some(MuxKeyEvent {
        key: code,
        modifiers,
    })
}

struct PaneViewport<'a> {
    lines: &'a BTreeMap<StableRowIndex, Line>,
    palette: &'a ColorPalette,
    reverse: bool,
    left: usize,
    top: StableRowIndex,
}

impl Widget for PaneViewport<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let palette = self.palette;
        let mut base = Style::default()
            .fg(rgb(palette.foreground))
            .bg(rgb(palette.background));
        if self.reverse {
            base = base.add_modifier(Modifier::REVERSED);
        }
        buf.set_style(area, base);
        for y in 0..area.height {
            let Some(line) = self
                .lines
                .get(&(self.top + isize::try_from(y).unwrap_or(0)))
            else {
                continue;
            };
            for cell in line.visible_cells() {
                let x = cell.cell_index();
                let width = cell.width();
                if x < self.left || x + width > self.left + usize::from(area.width) {
                    continue;
                }
                let target = (area.x + (x - self.left) as u16, area.y + y);
                let attr = cell.attrs();
                let mut style = Style::default()
                    .fg(cell_color(attr.foreground(), palette, palette.foreground))
                    .bg(cell_color(attr.background(), palette, palette.background));
                if attr.intensity() == Intensity::Bold {
                    style = style.add_modifier(Modifier::BOLD);
                }
                if attr.intensity() == Intensity::Half {
                    style = style.add_modifier(Modifier::DIM);
                }
                if attr.italic() {
                    style = style.add_modifier(Modifier::ITALIC);
                }
                if attr.reverse() != self.reverse {
                    style = style.add_modifier(Modifier::REVERSED);
                }
                if attr.underline() != Underline::None {
                    style = style.add_modifier(Modifier::UNDERLINED);
                }
                if attr.strikethrough() {
                    style = style.add_modifier(Modifier::CROSSED_OUT);
                }
                if attr.invisible() {
                    style = style.add_modifier(Modifier::HIDDEN);
                }
                let out = &mut buf[target];
                out.set_symbol(cell.str()).set_style(style);
                if let Some(width) = NonZeroU16::new(width as u16) {
                    out.set_diff_option(CellDiffOption::ForcedWidth(width));
                }
                for offset in 1..width {
                    buf[(target.0 + offset as u16, target.1)].set_diff_option(CellDiffOption::Skip);
                }
            }
        }
    }
}

fn rgb(color: SrgbaTuple) -> Color {
    let (r, g, b, _) = color.to_srgb_u8();
    Color::Rgb(r, g, b)
}

fn cell_color(color: ColorAttribute, palette: &ColorPalette, default: SrgbaTuple) -> Color {
    match color {
        ColorAttribute::Default => rgb(default),
        ColorAttribute::PaletteIndex(index) => rgb(palette.colors.0[usize::from(index)]),
        ColorAttribute::TrueColorWithDefaultFallback(color)
        | ColorAttribute::TrueColorWithPaletteFallback(color, _) => rgb(color),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use termwiz::cell::CellAttributes;

    #[test]
    fn control_and_application_keys_keep_their_meaning() {
        for key in ['c', 'd', 'w', 'v', 't'] {
            let actual = mux_key(KeyEvent::new(KeyCode::Char(key), KeyModifiers::CONTROL)).unwrap();
            assert_eq!(actual.key, MuxKey::Char(key));
            assert!(actual.modifiers.contains(Modifiers::CTRL));
        }
        assert_eq!(
            mux_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
                .unwrap()
                .key,
            MuxKey::Escape
        );
        assert!(mux_key(KeyEvent::new(KeyCode::BackTab, KeyModifiers::NONE))
            .unwrap()
            .modifiers
            .contains(Modifiers::SHIFT));
    }

    #[test]
    fn viewport_preserves_graphemes_width_and_clips_partial_cells() {
        let mut attrs = CellAttributes::default();
        attrs
            .set_intensity(Intensity::Bold)
            .set_foreground(ColorAttribute::PaletteIndex(1));
        let line = Line::from_text("A界e\u{301}", &attrs, 1, None);
        let lines = BTreeMap::from([(100, line)]);
        let palette = ColorPalette::default();
        let area = Rect::new(0, 0, 5, 1);
        let mut buf = Buffer::empty(area);
        PaneViewport {
            lines: &lines,
            palette: &palette,
            reverse: false,
            left: 0,
            top: 100,
        }
        .render(area, &mut buf);
        assert_eq!(buf[(0, 0)].symbol(), "A");
        assert_eq!(buf[(1, 0)].symbol(), "界");
        assert_eq!(buf[(2, 0)].diff_option, CellDiffOption::Skip);
        assert_eq!(buf[(3, 0)].symbol(), "e\u{301}");
        assert!(buf[(0, 0)].modifier.contains(Modifier::BOLD));
        assert_eq!(buf[(0, 0)].fg, rgb(palette.colors.0[1]));
        let mut clipped = Buffer::empty(Rect::new(0, 0, 2, 1));
        PaneViewport {
            lines: &lines,
            palette: &palette,
            reverse: false,
            left: 2,
            top: 100,
        }
        .render(clipped.area, &mut clipped);
        assert_eq!(clipped[(0, 0)].symbol(), " ");
        assert_eq!(clipped[(1, 0)].symbol(), "e\u{301}");
    }
}
