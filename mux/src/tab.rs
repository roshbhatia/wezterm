use crate::domain::DomainId;
use crate::pane::*;
use crate::renderable::StableCursorPosition;
use crate::{Mux, MuxNotification, WindowId};
use bintree::PathBranch;
use config::configuration;
use config::keyassignment::PaneDirection;
use parking_lot::Mutex;
use rangeset::intersects_range;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::convert::TryInto;
use std::sync::Arc;
use url::Url;
use wezterm_term::{StableRowIndex, TerminalSize};

pub type Tree = bintree::Tree<Arc<dyn Pane>, SplitDirectionAndSize>;
pub type Cursor = bintree::Cursor<Arc<dyn Pane>, SplitDirectionAndSize>;

static TAB_ID: ::std::sync::atomic::AtomicUsize = ::std::sync::atomic::AtomicUsize::new(0);
pub type TabId = usize;

#[derive(Default)]
struct Recency {
    count: usize,
    by_idx: HashMap<usize, usize>,
}

impl Recency {
    fn tag(&mut self, idx: usize) {
        self.by_idx.insert(idx, self.count);
        self.count += 1;
    }

    fn score(&self, idx: usize) -> usize {
        self.by_idx.get(&idx).copied().unwrap_or(0)
    }
}

struct TabInner {
    id: TabId,
    pane: Option<Tree>,
    size: TerminalSize,
    size_before_zoom: TerminalSize,
    active: usize,
    zoomed: Option<Arc<dyn Pane>>,
    title: String,
    floating_panes: Vec<Arc<dyn Pane>>,
    active_floating_pane: usize,
    floating_pane_visible: bool,
    recency: Recency,
}

/// A Tab is a container of Panes
pub struct Tab {
    inner: Mutex<TabInner>,
    tab_id: TabId,
}

#[derive(Clone)]
pub struct PositionedPane {
    /// The topological pane index that can be used to reference this pane
    pub index: usize,
    /// true if this is the active pane at the time the position was computed
    pub is_active: bool,
    /// true if this pane is zoomed
    pub is_zoomed: bool,
    /// true if this pane is floating
    pub is_floating: bool,
    /// The offset from the top left corner of the containing tab to the top
    /// left corner of this pane, in cells.
    pub left: usize,
    /// The offset from the top left corner of the containing tab to the top
    /// left corner of this pane, in cells.
    pub top: usize,
    /// The width of this pane in cells
    pub width: usize,
    pub pixel_width: usize,
    /// The height of this pane in cells
    pub height: usize,
    pub pixel_height: usize,
    /// The pane instance
    pub pane: Arc<dyn Pane>,
}

impl std::fmt::Debug for PositionedPane {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::result::Result<(), std::fmt::Error> {
        fmt.debug_struct("PositionedPane")
            .field("index", &self.index)
            .field("is_active", &self.is_active)
            .field("left", &self.left)
            .field("top", &self.top)
            .field("width", &self.width)
            .field("height", &self.height)
            .field("pane_id", &self.pane.pane_id())
            .finish()
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub enum SplitDirection {
    Horizontal,
    Vertical,
}

/// The size is of the (first, second) child of the split
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct SplitDirectionAndSize {
    pub direction: SplitDirection,
    pub first: TerminalSize,
    pub second: TerminalSize,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub enum SplitSize {
    Cells(usize),
    Percent(u8),
}

impl Default for SplitSize {
    fn default() -> Self {
        Self::Percent(50)
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct SplitRequest {
    pub direction: SplitDirection,
    /// Whether the newly created item will be in the second part
    /// of the split (right/bottom)
    pub target_is_second: bool,
    /// Split across the top of the tab rather than the active pane
    pub top_level: bool,
    /// The size of the new item
    pub size: SplitSize,
}

impl Default for SplitRequest {
    fn default() -> Self {
        Self {
            direction: SplitDirection::Horizontal,
            target_is_second: true,
            top_level: false,
            size: SplitSize::default(),
        }
    }
}

impl SplitDirectionAndSize {
    fn top_of_second(&self) -> usize {
        match self.direction {
            SplitDirection::Horizontal => 0,
            SplitDirection::Vertical => self.first.rows as usize + 1,
        }
    }

    fn left_of_second(&self) -> usize {
        match self.direction {
            SplitDirection::Horizontal => self.first.cols as usize + 1,
            SplitDirection::Vertical => 0,
        }
    }

    pub fn width(&self) -> usize {
        if self.direction == SplitDirection::Horizontal {
            self.first.cols + self.second.cols + 1
        } else {
            self.first.cols
        }
    }

    pub fn height(&self) -> usize {
        if self.direction == SplitDirection::Vertical {
            self.first.rows + self.second.rows + 1
        } else {
            self.first.rows
        }
    }

    pub fn size(&self) -> TerminalSize {
        let cell_width = self.first.pixel_width / self.first.cols;
        let cell_height = self.first.pixel_height / self.first.rows;

        let rows = self.height();
        let cols = self.width();

        TerminalSize {
            rows,
            cols,
            pixel_height: cell_height * rows,
            pixel_width: cell_width * cols,
            dpi: self.first.dpi,
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct PositionedSplit {
    /// The topological node index that can be used to reference this split
    pub index: usize,
    pub direction: SplitDirection,
    /// The offset from the top left corner of the containing tab to the top
    /// left corner of this split, in cells.
    pub left: usize,
    /// The offset from the top left corner of the containing tab to the top
    /// left corner of this split, in cells.
    pub top: usize,
    /// For Horizontal splits, how tall the split should be, for Vertical
    /// splits how wide it should be
    pub size: usize,
}

fn is_pane(pane: &Arc<dyn Pane>, other: &Option<&Arc<dyn Pane>>) -> bool {
    if let Some(other) = other {
        other.pane_id() == pane.pane_id()
    } else {
        false
    }
}

fn pane_tree(
    tree: &Tree,
    tab_id: TabId,
    window_id: WindowId,
    active: Option<&Arc<dyn Pane>>,
    zoomed: Option<&Arc<dyn Pane>>,
    workspace: &str,
    left_col: usize,
    top_row: usize,
) -> PaneNode {
    match tree {
        Tree::Empty => PaneNode::Empty,
        Tree::Node { left, right, data } => {
            let data = data.unwrap();
            PaneNode::Split {
                left: Box::new(pane_tree(
                    &*left, tab_id, window_id, active, zoomed, workspace, left_col, top_row,
                )),
                right: Box::new(pane_tree(
                    &*right,
                    tab_id,
                    window_id,
                    active,
                    zoomed,
                    workspace,
                    if data.direction == SplitDirection::Vertical {
                        left_col
                    } else {
                        left_col + data.left_of_second()
                    },
                    if data.direction == SplitDirection::Horizontal {
                        top_row
                    } else {
                        top_row + data.top_of_second()
                    },
                )),
                node: data,
            }
        }
        Tree::Leaf(pane) => {
            let dims = pane.get_dimensions();
            let working_dir = pane.get_current_working_dir(CachePolicy::AllowStale);
            let cursor_pos = pane.get_cursor_position();

            PaneNode::Leaf(PaneEntry {
                window_id,
                tab_id,
                pane_id: pane.pane_id(),
                title: pane.get_title(),
                is_active_pane: is_pane(pane, &active),
                is_zoomed_pane: is_pane(pane, &zoomed),
                is_floating_pane: false,
                size: TerminalSize {
                    cols: dims.cols,
                    rows: dims.viewport_rows,
                    pixel_height: dims.pixel_height,
                    pixel_width: dims.pixel_width,
                    dpi: dims.dpi,
                },
                working_dir: working_dir.map(Into::into),
                workspace: workspace.to_string(),
                cursor_pos,
                physical_top: dims.physical_top,
                left_col,
                top_row,
                tty_name: pane.tty_name(),
            })
        }
    }
}

fn build_from_pane_tree<F>(
    tree: bintree::Tree<PaneEntry, SplitDirectionAndSize>,
    active: &mut Option<Arc<dyn Pane>>,
    zoomed: &mut Option<Arc<dyn Pane>>,
    make_pane: &mut F,
) -> Tree
where
    F: FnMut(PaneEntry) -> Arc<dyn Pane>,
{
    match tree {
        bintree::Tree::Empty => Tree::Empty,
        bintree::Tree::Node { left, right, data } => Tree::Node {
            left: Box::new(build_from_pane_tree(*left, active, zoomed, make_pane)),
            right: Box::new(build_from_pane_tree(*right, active, zoomed, make_pane)),
            data,
        },
        bintree::Tree::Leaf(entry) => {
            let is_zoomed_pane = entry.is_zoomed_pane;
            let is_active_pane = entry.is_active_pane;
            let pane = make_pane(entry);
            if is_zoomed_pane {
                zoomed.replace(Arc::clone(&pane));
            }
            if is_active_pane {
                active.replace(Arc::clone(&pane));
            }
            Tree::Leaf(pane)
        }
    }
}

/// Computes the minimum (x, y) size based on the panes in this portion
/// of the tree.
fn compute_min_size(tree: &mut Tree) -> (usize, usize) {
    match tree {
        Tree::Node { data: None, .. } | Tree::Empty => (1, 1),
        Tree::Node {
            left,
            right,
            data: Some(data),
        } => {
            let (left_x, left_y) = compute_min_size(&mut *left);
            let (right_x, right_y) = compute_min_size(&mut *right);
            match data.direction {
                SplitDirection::Vertical => (left_x.max(right_x), left_y + right_y + 1),
                SplitDirection::Horizontal => (left_x + right_x + 1, left_y.max(right_y)),
            }
        }
        Tree::Leaf(_) => (1, 1),
    }
}

fn adjust_x_size(tree: &mut Tree, mut x_adjust: isize, cell_dimensions: &TerminalSize) {
    let (min_x, _) = compute_min_size(tree);
    while x_adjust != 0 {
        match tree {
            Tree::Empty | Tree::Leaf(_) => return,
            Tree::Node { data: None, .. } => return,
            Tree::Node {
                left,
                right,
                data: Some(data),
            } => {
                data.first.dpi = cell_dimensions.dpi;
                data.second.dpi = cell_dimensions.dpi;
                match data.direction {
                    SplitDirection::Vertical => {
                        let new_cols = (data.first.cols as isize)
                            .saturating_add(x_adjust)
                            .max(min_x as isize);
                        x_adjust = new_cols.saturating_sub(data.first.cols as isize);

                        if x_adjust != 0 {
                            adjust_x_size(&mut *left, x_adjust, cell_dimensions);
                            data.first.cols = new_cols.try_into().unwrap();
                            data.first.pixel_width =
                                data.first.cols.saturating_mul(cell_dimensions.pixel_width);

                            adjust_x_size(&mut *right, x_adjust, cell_dimensions);
                            data.second.cols = data.first.cols;
                            data.second.pixel_width = data.first.pixel_width;
                        }
                        return;
                    }
                    SplitDirection::Horizontal if x_adjust > 0 => {
                        adjust_x_size(&mut *left, 1, cell_dimensions);
                        data.first.cols += 1;
                        data.first.pixel_width =
                            data.first.cols.saturating_mul(cell_dimensions.pixel_width);
                        x_adjust -= 1;

                        if x_adjust > 0 {
                            adjust_x_size(&mut *right, 1, cell_dimensions);
                            data.second.cols += 1;
                            data.second.pixel_width =
                                data.second.cols.saturating_mul(cell_dimensions.pixel_width);
                            x_adjust -= 1;
                        }
                    }
                    SplitDirection::Horizontal => {
                        // x_adjust is negative.
                        // Each child can only shrink down to the minimum
                        // size of the panes it contains. The sizes recorded
                        // in `data` may disagree with the children (they are
                        // synced independently from a mux server), so we
                        // must stop as soon as neither side can give up a
                        // column rather than loop until `x_adjust` is zero.
                        let (first_min, _) = compute_min_size(&mut *left);
                        let (second_min, _) = compute_min_size(&mut *right);
                        let mut shrunk = false;
                        if data.first.cols > first_min {
                            adjust_x_size(&mut *left, -1, cell_dimensions);
                            data.first.cols -= 1;
                            data.first.pixel_width =
                                data.first.cols.saturating_mul(cell_dimensions.pixel_width);
                            x_adjust += 1;
                            shrunk = true;
                        }
                        if x_adjust < 0 && data.second.cols > second_min {
                            adjust_x_size(&mut *right, -1, cell_dimensions);
                            data.second.cols -= 1;
                            data.second.pixel_width =
                                data.second.cols.saturating_mul(cell_dimensions.pixel_width);
                            x_adjust += 1;
                            shrunk = true;
                        }
                        if !shrunk {
                            return;
                        }
                    }
                }
            }
        }
    }
}

fn adjust_y_size(tree: &mut Tree, mut y_adjust: isize, cell_dimensions: &TerminalSize) {
    let (_, min_y) = compute_min_size(tree);
    while y_adjust != 0 {
        match tree {
            Tree::Empty | Tree::Leaf(_) => return,
            Tree::Node { data: None, .. } => return,
            Tree::Node {
                left,
                right,
                data: Some(data),
            } => {
                data.first.dpi = cell_dimensions.dpi;
                data.second.dpi = cell_dimensions.dpi;
                match data.direction {
                    SplitDirection::Horizontal => {
                        let new_rows = (data.first.rows as isize)
                            .saturating_add(y_adjust)
                            .max(min_y as isize);
                        y_adjust = new_rows.saturating_sub(data.first.rows as isize);

                        if y_adjust != 0 {
                            adjust_y_size(&mut *left, y_adjust, cell_dimensions);
                            data.first.rows = new_rows.try_into().unwrap();
                            data.first.pixel_height =
                                data.first.rows.saturating_mul(cell_dimensions.pixel_height);

                            adjust_y_size(&mut *right, y_adjust, cell_dimensions);
                            data.second.rows = data.first.rows;
                            data.second.pixel_height = data.first.pixel_height;
                        }
                        return;
                    }
                    SplitDirection::Vertical if y_adjust > 0 => {
                        adjust_y_size(&mut *left, 1, cell_dimensions);
                        data.first.rows += 1;
                        data.first.pixel_height =
                            data.first.rows.saturating_mul(cell_dimensions.pixel_height);
                        y_adjust -= 1;
                        if y_adjust > 0 {
                            adjust_y_size(&mut *right, 1, cell_dimensions);
                            data.second.rows += 1;
                            data.second.pixel_height = data
                                .second
                                .rows
                                .saturating_mul(cell_dimensions.pixel_height);
                            y_adjust -= 1;
                        }
                    }
                    SplitDirection::Vertical => {
                        // y_adjust is negative; see adjust_x_size for why
                        // we bail out when neither child can shrink.
                        let (_, first_min) = compute_min_size(&mut *left);
                        let (_, second_min) = compute_min_size(&mut *right);
                        let mut shrunk = false;
                        if data.first.rows > first_min {
                            adjust_y_size(&mut *left, -1, cell_dimensions);
                            data.first.rows -= 1;
                            data.first.pixel_height =
                                data.first.rows.saturating_mul(cell_dimensions.pixel_height);
                            y_adjust += 1;
                            shrunk = true;
                        }
                        if y_adjust < 0 && data.second.rows > second_min {
                            adjust_y_size(&mut *right, -1, cell_dimensions);
                            data.second.rows -= 1;
                            data.second.pixel_height = data
                                .second
                                .rows
                                .saturating_mul(cell_dimensions.pixel_height);
                            y_adjust += 1;
                            shrunk = true;
                        }
                        if !shrunk {
                            return;
                        }
                    }
                }
            }
        }
    }
}

fn apply_sizes_from_splits(tree: &Tree, size: &TerminalSize) {
    match tree {
        Tree::Empty => return,
        Tree::Node { data: None, .. } => return,
        Tree::Node {
            left,
            right,
            data: Some(data),
        } => {
            apply_sizes_from_splits(&*left, &data.first);
            apply_sizes_from_splits(&*right, &data.second);
        }
        Tree::Leaf(pane) => {
            pane.resize(*size).ok();
        }
    }
}

fn cell_dimensions(size: &TerminalSize) -> TerminalSize {
    TerminalSize {
        rows: 1,
        cols: 1,
        pixel_width: size.pixel_width / size.cols,
        pixel_height: size.pixel_height / size.rows,
        dpi: size.dpi,
    }
}

impl Tab {
    pub fn new(size: &TerminalSize) -> Self {
        let inner = TabInner::new(size);
        let tab_id = inner.id;
        Self {
            inner: Mutex::new(inner),
            tab_id,
        }
    }

    pub fn get_title(&self) -> String {
        self.inner.lock().title.clone()
    }

    pub fn set_title(&self, title: &str) {
        let mut inner = self.inner.lock();
        if inner.title != title {
            inner.title = title.to_string();
            Mux::try_get().map(|mux| {
                mux.notify(MuxNotification::TabTitleChanged {
                    tab_id: inner.id,
                    title: title.to_string(),
                })
            });
        }
    }

    /// Called by the multiplexer client when building a local tab to
    /// mirror a remote tab.  The supplied `root` is the information
    /// about our counterpart in the remote server.
    /// This method builds a local tree based on the remote tree which
    /// then replaces the local tree structure.
    ///
    /// The `make_pane` function is provided by the caller, and its purpose
    /// is to lookup an existing Pane that corresponds to the provided
    /// PaneEntry, or to create a new Pane from that entry.
    /// make_pane is expected to add the pane to the mux if it creates
    /// a new pane, otherwise the pane won't poll/update in the GUI.
    pub fn sync_with_pane_tree<F>(&self, size: TerminalSize, root: TabEntry, make_pane: F)
    where
        F: FnMut(PaneEntry) -> Arc<dyn Pane>,
    {
        self.inner.lock().sync_with_pane_tree(size, root, make_pane)
    }

    pub fn codec_pane_tree(&self) -> TabEntry {
        self.inner.lock().codec_pane_tree()
    }

    /// Returns a count of how many panes are in this tab
    pub fn count_panes(&self) -> Option<usize> {
        self.inner.try_lock().map(|mut inner| inner.count_panes())
    }

    /// Sets the zoom state, returns the prior state
    pub fn set_zoomed(&self, zoomed: bool) -> bool {
        self.inner.lock().set_zoomed(zoomed)
    }

    pub fn toggle_zoom(&self) {
        self.inner.lock().toggle_zoom()
    }

    pub fn contains_pane(&self, pane: PaneId) -> bool {
        self.inner.lock().contains_pane(pane)
    }

    pub fn iter_panes(&self) -> Vec<PositionedPane> {
        self.inner.lock().iter_panes()
    }

    /// List every pane for discovery, including hidden floating panes.
    pub fn iter_panes_including_floating(&self) -> Vec<PositionedPane> {
        let mut inner = self.inner.lock();
        let mut panes = inner.iter_panes_ignoring_zoom();
        if inner.floating_pane_is_visible() {
            for pane in &mut panes {
                pane.is_active = false;
            }
        }
        let cell_width = (inner.size.pixel_width / inner.size.cols).max(1);
        let cell_height = (inner.size.pixel_height / inner.size.rows).max(1);
        let count = panes.len();
        for (index, pane) in inner.floating_panes.iter().enumerate() {
            let size = inner.floating_pane_size(pane);
            panes.push(PositionedPane {
                index: count + index,
                is_active: inner.floating_pane_is_visible() && index == inner.active_floating_pane,
                is_zoomed: inner.zoomed.as_ref().map(|p| p.pane_id()) == Some(pane.pane_id()),
                is_floating: true,
                left: inner.size.pixel_width.saturating_sub(size.pixel_width) / 2 / cell_width,
                top: inner.size.pixel_height.saturating_sub(size.pixel_height) / 2 / cell_height,
                width: size.cols,
                height: size.rows,
                pixel_width: size.pixel_width,
                pixel_height: size.pixel_height,
                pane: Arc::clone(pane),
            });
        }
        panes
    }

    pub fn floating_pane_is_visible(&self) -> bool {
        self.inner.lock().floating_pane_is_visible()
    }

    pub fn get_floating_pane_by_pane_id(&self, pane_id: PaneId) -> Option<(usize, Arc<dyn Pane>)> {
        self.inner.lock().get_floating_pane_by_pane_id(pane_id)
    }

    pub fn get_active_floating_pane(&self) -> Option<PositionedPane> {
        self.inner.lock().get_active_floating_pane()
    }

    pub fn iter_panes_ignoring_zoom(&self) -> Vec<PositionedPane> {
        self.inner.lock().iter_panes_ignoring_zoom()
    }

    pub fn rotate_counter_clockwise(&self) {
        self.inner.lock().rotate_counter_clockwise()
    }

    pub fn rotate_clockwise(&self) {
        self.inner.lock().rotate_clockwise()
    }

    pub fn iter_splits(&self) -> Vec<PositionedSplit> {
        self.inner.lock().iter_splits()
    }

    pub fn tab_id(&self) -> TabId {
        self.tab_id
    }

    pub fn get_size(&self) -> TerminalSize {
        self.inner.lock().get_size()
    }

    /// Apply the new size of the tab to the panes contained within.
    /// The delta between the current and the new size is computed,
    /// and is distributed between the splits.  For small resizes
    /// this algorithm biases towards adjusting the left/top nodes
    /// first.  For large resizes this tends to proportionally adjust
    /// the relative sizes of the elements in a split.
    pub fn resize(&self, size: TerminalSize) {
        self.inner.lock().resize(size)
    }

    /// Called when running in the mux server after an individual pane
    /// has been resized.
    /// Because the split manipulation happened on the GUI we "lost"
    /// the information that would have allowed us to call resize_split_by()
    /// and instead need to back-infer the split size information.
    /// We rely on the client to have resized (or be in the process
    /// of resizing) affected panes consistently with its own Tab
    /// tree model.
    /// This method does a simple tree walk to the leaves to back-propagate
    /// the size of the panes up to their containing node split data.
    /// Without this step, disconnecting and reconnecting would cause
    /// the GUI to use stale size information for the window it spawns
    /// to attach this tab.
    pub fn rebuild_splits_sizes_from_contained_panes(&self) {
        self.inner
            .lock()
            .rebuild_splits_sizes_from_contained_panes()
    }

    /// Given split_index, the topological index of a split returned by
    /// iter_splits() as PositionedSplit::index, revised the split position
    /// by the provided delta; positive values move the split to the right/bottom,
    /// and negative values to the left/top.
    /// The adjusted size is propagated downwards to contained children and
    /// their panes are resized accordingly.
    pub fn resize_split_by(&self, split_index: usize, delta: isize) {
        self.inner.lock().resize_split_by(split_index, delta)
    }

    /// Adjusts the size of the active pane in the specified direction
    /// by the specified amount.
    pub fn adjust_pane_size(&self, direction: PaneDirection, amount: usize) {
        self.inner.lock().adjust_pane_size(direction, amount)
    }

    pub fn scale_floating_pane(&self, percent: i8) {
        self.inner.lock().scale_floating_pane(percent)
    }

    pub fn active_normal_pane(&self) -> Option<PositionedPane> {
        self.inner
            .lock()
            .iter_panes_ignoring_zoom()
            .into_iter()
            .find(|p| p.is_active)
    }

    pub fn floating_pane_count(&self) -> usize {
        self.inner.lock().floating_panes.len()
    }

    /// Activate an adjacent pane in the specified direction.
    /// In cases where there are multiple adjacent panes in the
    /// intended direction, we take the pane that has the largest
    /// edge intersection.
    pub fn activate_pane_direction(&self, direction: PaneDirection) {
        self.inner.lock().activate_pane_direction(direction)
    }

    /// Returns an adjacent pane in the specified direction.
    /// In cases where there are multiple adjacent panes in the
    /// intended direction, we take the pane that has the largest
    /// edge intersection.
    pub fn get_pane_direction(&self, direction: PaneDirection, ignore_zoom: bool) -> Option<usize> {
        let mut inner = self.inner.lock();
        if inner.is_floating_pane_active() {
            let index = match direction {
                PaneDirection::Left | PaneDirection::Prev => {
                    (inner.active_floating_pane + inner.floating_panes.len() - 1)
                        % inner.floating_panes.len()
                }
                PaneDirection::Right | PaneDirection::Next => {
                    (inner.active_floating_pane + 1) % inner.floating_panes.len()
                }
                _ => return None,
            };
            return Some(inner.iter_panes_ignoring_zoom().len() + index);
        }
        inner.get_pane_direction(direction, ignore_zoom)
    }

    pub fn prune_dead_panes(&self) -> bool {
        self.inner.lock().prune_dead_panes()
    }

    pub fn kill_pane(&self, pane_id: PaneId) -> bool {
        self.inner.lock().kill_pane(pane_id)
    }

    pub fn kill_panes_in_domain(&self, domain: DomainId) -> bool {
        self.inner.lock().kill_panes_in_domain(domain)
    }

    /// Remove pane from tab.
    /// The pane is still live in the mux; the intent is for the pane to
    /// be added to a different tab.
    pub fn remove_pane(&self, pane_id: PaneId) -> Option<Arc<dyn Pane>> {
        self.inner.lock().remove_pane(pane_id)
    }

    pub fn can_close_without_prompting(&self, reason: CloseReason) -> bool {
        self.inner.lock().can_close_without_prompting(reason)
    }

    pub fn is_dead(&self) -> bool {
        self.inner.lock().is_dead()
    }

    pub fn is_floating_pane_active(&self) -> bool {
        self.inner.lock().is_floating_pane_active()
    }

    pub fn get_active_pane(&self) -> Option<Arc<dyn Pane>> {
        self.inner.lock().get_active_pane()
    }

    #[allow(unused)]
    pub fn get_active_idx(&self) -> usize {
        self.inner.lock().get_active_idx()
    }

    pub fn set_active_pane(&self, pane: &Arc<dyn Pane>) {
        self.inner.lock().set_active_pane(pane)
    }

    pub fn set_floating_pane_visibility(&self, visible: bool) {
        self.inner
            .lock()
            .set_floating_pane_visibility(visible, true);
    }

    pub fn set_active_floating_pane(&self, index: usize) {
        self.inner.lock().set_active_floating_pane(index, true);
    }

    pub fn append_floating_pane(&self, pane: Arc<dyn Pane>) {
        self.inner.lock().append_floating_pane(pane);
    }

    pub fn set_active_idx(&self, pane_index: usize) {
        self.inner.lock().set_active_idx(pane_index)
    }

    /// Assigns the root pane.
    /// This is suitable when creating a new tab and then assigning
    /// the initial pane
    pub fn assign_pane(&self, pane: &Arc<dyn Pane>) {
        self.inner.lock().assign_pane(pane)
    }

    /// Swap the active pane with the specified pane_index
    pub fn swap_active_with_index(&self, pane_index: usize, keep_focus: bool) -> Option<()> {
        self.inner
            .lock()
            .swap_active_with_index(pane_index, keep_focus)
    }

    /// Computes the size of the pane that would result if the specified
    /// pane was split in a particular direction.
    /// The intent is to call this prior to spawning the new pane so that
    /// you can create it with the correct size.
    /// May return None if the specified pane_index is invalid.
    pub fn compute_split_size(
        &self,
        pane_index: usize,
        request: SplitRequest,
    ) -> Option<SplitDirectionAndSize> {
        self.inner.lock().compute_split_size(pane_index, request)
    }

    pub fn compute_floating_pane_size(&self) -> TerminalSize {
        self.inner.lock().compute_floating_pane_size2()
    }

    pub fn get_active_floating_pane_index(&self) -> usize {
        self.inner.lock().active_floating_pane
    }

    pub fn remove_floating_pane(&self, index: usize) -> anyhow::Result<Arc<dyn Pane>> {
        self.inner.lock().remove_floating_pane(index)
    }

    /// Split the pane that has pane_index in the given direction and assign
    /// the right/bottom pane of the newly created split to the provided Pane
    /// instance.  Returns the resultant index of the newly inserted pane.
    /// Both the split and the inserted pane will be resized.
    pub fn split_and_insert(
        &self,
        pane_index: usize,
        request: SplitRequest,
        pane: Arc<dyn Pane>,
    ) -> anyhow::Result<usize> {
        self.inner
            .lock()
            .split_and_insert(pane_index, request, pane)
    }

    pub fn has_floating_pane(&self) -> bool {
        self.inner.lock().has_floating_pane()
    }

    pub fn toggle_float(&self) {
        self.inner.lock().toggle_floating_pane();
    }

    pub fn add_floating_pane(&self, size: TerminalSize, pane: Arc<dyn Pane>) -> anyhow::Result<()> {
        self.inner.lock().add_floating_pane(size, pane)
    }

    pub fn get_zoomed_pane(&self) -> Option<Arc<dyn Pane>> {
        self.inner.lock().get_zoomed_pane()
    }
}

impl TabInner {
    fn new(size: &TerminalSize) -> Self {
        Self {
            id: TAB_ID.fetch_add(1, ::std::sync::atomic::Ordering::Relaxed),
            pane: Some(Tree::new()),
            size: *size,
            size_before_zoom: *size,
            active: 0,
            zoomed: None,
            title: String::new(),
            recency: Recency::default(),
            floating_panes: vec![],
            active_floating_pane: 0,
            floating_pane_visible: false,
        }
    }

    fn sync_with_pane_tree<F>(&mut self, size: TerminalSize, root: TabEntry, mut make_pane: F)
    where
        F: FnMut(PaneEntry) -> Arc<dyn Pane>,
    {
        let mut active = None;
        let mut zoomed = None;

        log::debug!("sync_with_pane_tree with size {:?}", size);

        let t = build_from_pane_tree(
            root.panes.into_tree(),
            &mut active,
            &mut zoomed,
            &mut make_pane,
        );
        let mut cursor = t.cursor();

        self.active = 0;
        if let Some(active) = active {
            // Resolve the active pane to its index
            let mut index = 0;
            loop {
                if let Some(pane) = cursor.leaf_mut() {
                    if active.pane_id() == pane.pane_id() {
                        // Found it
                        self.active = index;
                        self.recency.tag(index);
                        break;
                    }
                    index += 1;
                }
                match cursor.preorder_next() {
                    Ok(c) => cursor = c,
                    Err(c) => {
                        // Didn't find it
                        cursor = c;
                        break;
                    }
                }
            }
        }
        self.pane.replace(cursor.tree());
        self.size = size;

        self.active_floating_pane = root.active_floating_pane_index;
        self.floating_pane_visible = root.floating_pane_visible;
        self.floating_panes.clear();
        for pane_node in root.floating_panes {
            if let PaneNode::Leaf(entry) = pane_node {
                let is_zoomed = entry.is_zoomed_pane;
                let size = entry.size;
                let floating_pane = make_pane(entry);
                floating_pane.resize(size).ok();
                if is_zoomed {
                    zoomed = Some(Arc::clone(&floating_pane))
                }
                self.floating_panes.push(floating_pane);
            }
        }

        self.zoomed = zoomed;
        self.resize(size);

        log::debug!(
            "sync tab: {:#?} zoomed: {} {:#?}",
            size,
            self.zoomed.is_some(),
            self.iter_panes()
        );
        assert!(self.pane.is_some());
    }

    fn codec_pane_tree(&mut self) -> TabEntry {
        let mux = Mux::get();
        let tab_id = self.id;
        let window_id = match mux.window_containing_tab(tab_id) {
            Some(w) => w,
            None => {
                log::error!("no window contains tab {}", tab_id);
                return TabEntry {
                    panes: PaneNode::Empty,
                    floating_panes: vec![],
                    active_floating_pane_index: self.active_floating_pane,
                    floating_pane_visible: self.floating_pane_visible,
                };
            }
        };

        let workspace = match mux
            .get_window(window_id)
            .map(|w| w.get_workspace().to_string())
        {
            Some(ws) => ws,
            None => {
                log::error!("window id {} doesn't have a window!?", window_id);
                return TabEntry {
                    panes: PaneNode::Empty,
                    floating_panes: vec![],
                    active_floating_pane_index: self.active_floating_pane,
                    floating_pane_visible: self.floating_pane_visible,
                };
            }
        };

        let active = self.get_active_pane();
        let zoomed = self.zoomed.as_ref();

        let mut codec_floating_panes = vec![];
        for (i, floating_pane) in self.floating_panes.iter().enumerate() {
            let dims = floating_pane.get_dimensions();
            let working_dir = floating_pane.get_current_working_dir(CachePolicy::AllowStale);
            let cursor_pos = floating_pane.get_cursor_position();

            let to_add = PaneNode::Leaf(PaneEntry {
                window_id,
                tab_id,
                pane_id: floating_pane.pane_id(),
                title: floating_pane.get_title(),
                size: TerminalSize {
                    cols: dims.cols,
                    rows: dims.viewport_rows,
                    pixel_height: dims.pixel_height,
                    pixel_width: dims.pixel_width,
                    dpi: dims.dpi,
                },
                working_dir: working_dir.map(Into::into),
                is_active_pane: i == self.active_floating_pane && self.floating_pane_visible,
                is_zoomed_pane: is_pane(floating_pane, &zoomed),
                is_floating_pane: true,
                workspace: workspace.to_string(),
                cursor_pos,
                physical_top: 0,
                top_row: 0,
                left_col: 0,
                tty_name: floating_pane.tty_name(),
            });
            codec_floating_panes.push(to_add);
        }

        if let Some(root) = self.pane.as_ref() {
            TabEntry {
                panes: pane_tree(
                    root,
                    tab_id,
                    window_id,
                    active.as_ref(),
                    zoomed,
                    &workspace,
                    0,
                    0,
                ),
                floating_panes: codec_floating_panes,
                active_floating_pane_index: self.active_floating_pane,
                floating_pane_visible: self.floating_pane_visible,
            }
        } else {
            TabEntry {
                panes: PaneNode::Empty,
                floating_panes: codec_floating_panes,
                active_floating_pane_index: self.active_floating_pane,
                floating_pane_visible: self.floating_pane_visible,
            }
        }
    }

    /// Returns a count of how many panes are in this tab
    fn count_panes(&mut self) -> usize {
        let mut count = 0;
        let mut cursor = self.pane.take().unwrap().cursor();

        for _ in &self.floating_panes {
            count += 1;
        }

        loop {
            if cursor.is_leaf() {
                count += 1;
            }
            match cursor.preorder_next() {
                Ok(c) => cursor = c,
                Err(c) => {
                    self.pane.replace(c.tree());
                    return count;
                }
            }
        }
    }

    /// Sets the zoom state, returns the prior state
    fn set_zoomed(&mut self, zoomed: bool) -> bool {
        if self.zoomed.is_some() == zoomed {
            // Current zoom state matches intended zoom state,
            // so we have nothing to do.
            return zoomed;
        }
        self.toggle_zoom();
        !zoomed
    }

    fn toggle_zoom(&mut self) {
        let size = self.size;
        if self.zoomed.take().is_some() {
            // We were zoomed, but now we are not.
            // Re-apply the size to the panes
            if let Some(pane) = self.get_active_pane() {
                pane.set_zoomed(false);
            }
            self.size = self.size_before_zoom;
            self.resize(size);
        } else {
            // We weren't zoomed, but now we want to zoom.
            // Locate the active pane
            self.size_before_zoom = size;
            if let Some(pane) = self.get_active_pane() {
                pane.set_zoomed(true);
                pane.resize(size).ok();
                self.zoomed.replace(pane);
            }
        }
        Mux::try_get().map(|mux| mux.notify(MuxNotification::TabResized(self.id)));
    }

    fn contains_pane(&self, pane: PaneId) -> bool {
        fn contains(tree: &Tree, pane: PaneId) -> bool {
            match tree {
                Tree::Empty => false,
                Tree::Node { left, right, .. } => contains(left, pane) || contains(right, pane),
                Tree::Leaf(p) => p.pane_id() == pane,
            }
        }
        self.floating_panes.iter().any(|p| p.pane_id() == pane)
            || match &self.pane {
                Some(root) => contains(root, pane),
                None => false,
            }
    }

    /// Walks the pane tree to produce the topologically ordered flattened
    /// list of PositionedPane instances along with their positioning information.
    fn iter_panes(&mut self) -> Vec<PositionedPane> {
        self.iter_panes_impl(true)
    }

    fn get_floating_pane_by_pane_id(&self, pane_id: PaneId) -> Option<(usize, Arc<dyn Pane>)> {
        for (i, pane) in self.floating_panes.iter().enumerate() {
            if pane.pane_id() == pane_id {
                return Some((i, Arc::clone(&pane)));
            }
        }
        return None;
    }

    fn floating_pane_is_visible(&self) -> bool {
        self.floating_pane_visible && !self.zoomed.is_some()
    }

    fn get_active_floating_pane(&self) -> Option<PositionedPane> {
        //Even if the zoomed pane is a floating pane. Not returning it since it is currently
        //not visible as a floating pane, instead as a zoomed pane
        if self.zoomed.is_some() {
            return None;
        }

        if self.floating_pane_visible && self.active_floating_pane < self.floating_panes.len() {
            let floating_pane = &self.floating_panes[self.active_floating_pane];
            let root_size = self.size;
            let size = self.floating_pane_size(floating_pane);

            let cell_height = (root_size.pixel_height / root_size.rows).max(1);
            let cell_width = (root_size.pixel_width / root_size.cols).max(1);
            let left = (root_size.pixel_width.saturating_sub(size.pixel_width) / 2) / cell_width;
            let top = root_size.pixel_height.saturating_sub(size.pixel_height) / 2 / cell_height;

            Some(PositionedPane {
                index: 0,
                is_active: true,
                is_zoomed: false,
                is_floating: true,
                left,
                top,
                width: size.cols,
                height: size.rows,
                pixel_width: size.pixel_width,
                pixel_height: size.pixel_height,
                pane: Arc::clone(floating_pane),
            })
        } else {
            None
        }
    }

    /// Like iter_panes, except that it will include all panes, regardless of
    /// whether one of them is currently zoomed.
    fn iter_panes_ignoring_zoom(&mut self) -> Vec<PositionedPane> {
        self.iter_panes_impl(false)
    }

    fn rotate_counter_clockwise(&mut self) {
        let panes = self.iter_panes_ignoring_zoom();
        if panes.is_empty() {
            // Shouldn't happen, but we check for this here so that the
            // expect below cannot trigger a panic
            return;
        }
        let mut pane_to_swap = panes
            .first()
            .map(|p| p.pane.clone())
            .expect("at least one pane");

        let mut cursor = self.pane.take().unwrap().cursor();

        loop {
            if cursor.is_leaf() {
                std::mem::swap(&mut pane_to_swap, cursor.leaf_mut().unwrap());
            }

            match cursor.postorder_next() {
                Ok(c) => cursor = c,
                Err(c) => {
                    self.pane.replace(c.tree());
                    let size = self.size;
                    apply_sizes_from_splits(self.pane.as_mut().unwrap(), &size);
                    break;
                }
            }
        }
    }

    fn rotate_clockwise(&mut self) {
        let panes = self.iter_panes_ignoring_zoom();
        if panes.is_empty() {
            // Shouldn't happen, but we check for this here so that the
            // expect below cannot trigger a panic
            return;
        }
        let mut pane_to_swap = panes
            .last()
            .map(|p| p.pane.clone())
            .expect("at least one pane");

        let mut cursor = self.pane.take().unwrap().cursor();

        loop {
            if cursor.is_leaf() {
                std::mem::swap(&mut pane_to_swap, cursor.leaf_mut().unwrap());
            }

            match cursor.preorder_next() {
                Ok(c) => cursor = c,
                Err(c) => {
                    self.pane.replace(c.tree());
                    let size = self.size;
                    apply_sizes_from_splits(self.pane.as_mut().unwrap(), &size);
                    break;
                }
            }
        }
        Mux::try_get().map(|mux| mux.notify(MuxNotification::TabResized(self.id)));
    }

    fn iter_panes_impl(&mut self, respect_zoom_state: bool) -> Vec<PositionedPane> {
        let mut panes = vec![];

        if respect_zoom_state {
            if let Some(zoomed) = self.zoomed.as_ref() {
                let size = self.size;
                panes.push(PositionedPane {
                    index: 0,
                    is_active: true,
                    is_zoomed: true,
                    is_floating: false,
                    left: 0,
                    top: 0,
                    width: size.cols.into(),
                    pixel_width: size.pixel_width.into(),
                    height: size.rows.into(),
                    pixel_height: size.pixel_height.into(),
                    pane: Arc::clone(zoomed),
                });
                return panes;
            }
        }

        let active_idx = self.active;
        let zoomed_id = self.zoomed.as_ref().map(|p| p.pane_id());
        let root_size = self.size;
        let mut cursor = self.pane.take().unwrap().cursor();

        loop {
            if cursor.is_leaf() {
                let index = panes.len();
                let mut left = 0usize;
                let mut top = 0usize;
                let mut parent_size = None;
                for (branch, node) in cursor.path_to_root() {
                    if let Some(node) = node {
                        if parent_size.is_none() {
                            parent_size.replace(if branch == PathBranch::IsRight {
                                node.second
                            } else {
                                node.first
                            });
                        }
                        if branch == PathBranch::IsRight {
                            top += node.top_of_second();
                            left += node.left_of_second();
                        }
                    }
                }

                let pane = Arc::clone(cursor.leaf_mut().unwrap());
                let dims = parent_size.unwrap_or_else(|| root_size);

                panes.push(PositionedPane {
                    index,
                    is_active: index == active_idx,
                    is_zoomed: zoomed_id == Some(pane.pane_id()),
                    is_floating: false,
                    left,
                    top,
                    width: dims.cols as _,
                    height: dims.rows as _,
                    pixel_width: dims.pixel_width as _,
                    pixel_height: dims.pixel_height as _,
                    pane,
                });
            }

            match cursor.preorder_next() {
                Ok(c) => cursor = c,
                Err(c) => {
                    self.pane.replace(c.tree());
                    break;
                }
            }
        }

        panes
    }

    fn iter_splits(&mut self) -> Vec<PositionedSplit> {
        let mut dividers = vec![];
        if self.zoomed.is_some() {
            return dividers;
        }

        let mut cursor = self.pane.take().unwrap().cursor();
        let mut index = 0;

        loop {
            if !cursor.is_leaf() {
                let mut left = 0usize;
                let mut top = 0usize;
                for (branch, p) in cursor.path_to_root() {
                    if let Some(p) = p {
                        if branch == PathBranch::IsRight {
                            left += p.left_of_second();
                            top += p.top_of_second();
                        }
                    }
                }
                if let Ok(Some(node)) = cursor.node_mut() {
                    match node.direction {
                        SplitDirection::Horizontal => left += node.first.cols as usize,
                        SplitDirection::Vertical => top += node.first.rows as usize,
                    }

                    dividers.push(PositionedSplit {
                        index,
                        direction: node.direction,
                        left,
                        top,
                        size: if node.direction == SplitDirection::Horizontal {
                            node.height() as usize
                        } else {
                            node.width() as usize
                        },
                    })
                }
                index += 1;
            }

            match cursor.preorder_next() {
                Ok(c) => cursor = c,
                Err(c) => {
                    self.pane.replace(c.tree());
                    break;
                }
            }
        }

        dividers
    }

    fn get_size(&self) -> TerminalSize {
        self.size
    }

    fn resize(&mut self, size: TerminalSize) {
        if size.rows == 0 || size.cols == 0 {
            // Ignore "impossible" resize requests
            return;
        }

        let prior_default = self.compute_floating_pane_size(self.size);
        let next_default = self.compute_floating_pane_size(size);
        for floating_pane in &self.floating_panes {
            let current = self.floating_pane_size(floating_pane);
            let next = if current.cols == prior_default.cols && current.rows == prior_default.rows {
                next_default
            } else {
                let cols = (current.cols * size.cols / self.size.cols.max(1))
                    .clamp(1, size.cols.saturating_sub(2).max(1));
                let rows = (current.rows * size.rows / self.size.rows.max(1))
                    .clamp(1, size.rows.saturating_sub(2).max(1));
                TerminalSize {
                    cols,
                    rows,
                    pixel_width: cols * (size.pixel_width / size.cols).max(1),
                    pixel_height: rows * (size.pixel_height / size.rows).max(1),
                    dpi: size.dpi,
                }
            };
            floating_pane.resize(next).ok();
        }

        if let Some(zoomed) = &self.zoomed {
            self.size = size;
            zoomed.resize(size).ok();
        } else {
            let dims = cell_dimensions(&size);
            let (min_x, min_y) = compute_min_size(self.pane.as_mut().unwrap());
            let current_size = self.size;

            // Constrain the new size to the minimum possible dimensions
            let cols = size.cols.max(min_x);
            let rows = size.rows.max(min_y);
            let size = TerminalSize {
                rows,
                cols,
                pixel_width: cols * dims.pixel_width,
                pixel_height: rows * dims.pixel_height,
                dpi: dims.dpi,
            };

            // Update the split nodes with adjusted sizes
            adjust_x_size(
                self.pane.as_mut().unwrap(),
                cols as isize - current_size.cols as isize,
                &dims,
            );
            adjust_y_size(
                self.pane.as_mut().unwrap(),
                rows as isize - current_size.rows as isize,
                &dims,
            );

            self.size = size;

            // And then resize the individual panes to match
            apply_sizes_from_splits(self.pane.as_mut().unwrap(), &size);
        }

        Mux::try_get().map(|mux| mux.notify(MuxNotification::TabResized(self.id)));
    }

    fn apply_pane_size(&mut self, pane_size: TerminalSize, cursor: &mut Cursor) {
        let cell_width = pane_size
            .pixel_width
            .checked_div(pane_size.cols)
            .unwrap_or(1);
        let cell_height = pane_size
            .pixel_height
            .checked_div(pane_size.rows)
            .unwrap_or(1);
        if let Ok(Some(node)) = cursor.node_mut() {
            // Adjust the size of the node; we preserve the size of the first
            // child and adjust the second, so if we are split down the middle
            // and the window is made wider, the right column will grow in
            // size, leaving the left at its current width.
            if node.direction == SplitDirection::Horizontal {
                node.first.rows = pane_size.rows;
                node.second.rows = pane_size.rows;

                node.second.cols = pane_size.cols.saturating_sub(1 + node.first.cols);
            } else {
                node.first.cols = pane_size.cols;
                node.second.cols = pane_size.cols;

                node.second.rows = pane_size.rows.saturating_sub(1 + node.first.rows);
            }
            node.first.pixel_width = node.first.cols * cell_width;
            node.first.pixel_height = node.first.rows * cell_height;

            node.second.pixel_width = node.second.cols * cell_width;
            node.second.pixel_height = node.second.rows * cell_height;
        }
    }

    fn rebuild_splits_sizes_from_contained_panes(&mut self) {
        if self.zoomed.is_some() {
            return;
        }

        fn compute_size(node: &mut Tree) -> Option<TerminalSize> {
            match node {
                Tree::Empty => None,
                Tree::Leaf(pane) => {
                    let dims = pane.get_dimensions();
                    let size = TerminalSize {
                        cols: dims.cols,
                        rows: dims.viewport_rows,
                        pixel_height: dims.pixel_height,
                        pixel_width: dims.pixel_width,
                        dpi: dims.dpi,
                    };
                    Some(size)
                }
                Tree::Node { left, right, data } => {
                    if let Some(data) = data {
                        if let Some(first) = compute_size(left) {
                            data.first = first;
                        }
                        if let Some(second) = compute_size(right) {
                            data.second = second;
                        }
                        Some(data.size())
                    } else {
                        None
                    }
                }
            }
        }

        if let Some(root) = self.pane.as_mut() {
            if let Some(size) = compute_size(root) {
                self.size = size;
            }
        }
        Mux::try_get().map(|mux| mux.notify(MuxNotification::TabResized(self.id)));
    }

    fn resize_split_by(&mut self, split_index: usize, delta: isize) {
        if self.zoomed.is_some() {
            return;
        }

        let mut cursor = self.pane.take().unwrap().cursor();
        let mut index = 0;

        // Position cursor on the specified split
        loop {
            if !cursor.is_leaf() {
                if index == split_index {
                    // Found it
                    break;
                }
                index += 1;
            }
            match cursor.preorder_next() {
                Ok(c) => cursor = c,
                Err(c) => {
                    // Didn't find it
                    self.pane.replace(c.tree());
                    return;
                }
            }
        }

        // Now cursor is looking at the split
        self.adjust_node_at_cursor(&mut cursor, delta);
        self.cascade_size_from_cursor(cursor);
        Mux::try_get().map(|mux| mux.notify(MuxNotification::TabResized(self.id)));
    }

    fn adjust_node_at_cursor(&mut self, cursor: &mut Cursor, delta: isize) {
        let cell_dimensions = self.cell_dimensions();
        if let Ok(Some(node)) = cursor.node_mut() {
            match node.direction {
                SplitDirection::Horizontal => {
                    let width = node.width();

                    let mut cols = node.first.cols as isize;
                    cols = cols
                        .saturating_add(delta)
                        .max(1)
                        .min((width as isize).saturating_sub(2));
                    node.first.cols = cols as usize;
                    node.first.pixel_width =
                        node.first.cols.saturating_mul(cell_dimensions.pixel_width);

                    node.second.cols = width.saturating_sub(node.first.cols.saturating_add(1));
                    node.second.pixel_width =
                        node.second.cols.saturating_mul(cell_dimensions.pixel_width);
                }
                SplitDirection::Vertical => {
                    let height = node.height();

                    let mut rows = node.first.rows as isize;
                    rows = rows
                        .saturating_add(delta)
                        .max(1)
                        .min((height as isize).saturating_sub(2));
                    node.first.rows = rows as usize;
                    node.first.pixel_height =
                        node.first.rows.saturating_mul(cell_dimensions.pixel_height);

                    node.second.rows = height.saturating_sub(node.first.rows.saturating_add(1));
                    node.second.pixel_height = node
                        .second
                        .rows
                        .saturating_mul(cell_dimensions.pixel_height);
                }
            }
        }
    }

    fn cascade_size_from_cursor(&mut self, mut cursor: Cursor) {
        // Now we need to cascade this down to children
        match cursor.preorder_next() {
            Ok(c) => cursor = c,
            Err(c) => {
                self.pane.replace(c.tree());
                return;
            }
        }
        let root_size = self.size;

        loop {
            // Figure out the available size by looking at our immediate parent node.
            // If we are the root, look at the provided new size
            let pane_size = if let Some((branch, Some(parent))) = cursor.path_to_root().next() {
                if branch == PathBranch::IsRight {
                    parent.second
                } else {
                    parent.first
                }
            } else {
                root_size
            };

            if cursor.is_leaf() {
                // Apply our size to the tty
                cursor.leaf_mut().map(|pane| pane.resize(pane_size));
            } else {
                self.apply_pane_size(pane_size, &mut cursor);
            }
            match cursor.preorder_next() {
                Ok(c) => cursor = c,
                Err(c) => {
                    self.pane.replace(c.tree());
                    break;
                }
            }
        }
        Mux::try_get().map(|mux| mux.notify(MuxNotification::TabResized(self.id)));
    }

    fn resize_floating_pane(&self, pane: &Arc<dyn Pane>, mut size: TerminalSize) {
        size.cols = size.cols.min(self.size.cols.saturating_sub(2).max(1));
        size.rows = size.rows.min(self.size.rows.saturating_sub(2).max(1));
        size.pixel_width = size.cols * (self.size.pixel_width / self.size.cols.max(1)).max(1);
        size.pixel_height = size.rows * (self.size.pixel_height / self.size.rows.max(1)).max(1);
        if let Err(err) = pane.resize(size) {
            log::error!("resizing floating pane: {err:#}");
            return;
        }
        Mux::try_get().map(|mux| mux.notify(MuxNotification::TabResized(self.id)));
    }

    fn scale_floating_pane(&mut self, percent: i8) {
        if percent == 0 || !self.is_floating_pane_active() {
            return;
        }
        if let Some(pane) = self.floating_panes.get(self.active_floating_pane) {
            let mut size = self.floating_pane_size(pane);
            let min_axis = size.cols.min(size.rows).max(1) as f64;
            let step = (f64::from(percent).abs() / 100.0).max(1.0 / min_axis);
            let requested = 1.0 + step * f64::from(percent).signum();
            let minimum = (1.0 / size.cols as f64).max(1.0 / size.rows as f64);
            let maximum = (self.size.cols.saturating_sub(2).max(1) as f64 / size.cols as f64)
                .min(self.size.rows.saturating_sub(2).max(1) as f64 / size.rows as f64);
            let factor = requested.clamp(minimum, maximum.max(minimum));
            let cols = (size.cols as f64 * factor).round().max(1.0) as usize;
            let rows = (size.rows as f64 * factor).round().max(1.0) as usize;
            if (cols, rows) != (size.cols, size.rows) {
                size.cols = cols;
                size.rows = rows;
                self.resize_floating_pane(pane, size);
            }
        }
    }

    fn adjust_pane_size(&mut self, direction: PaneDirection, amount: usize) {
        if self.is_floating_pane_active() {
            if let Some(pane) = self.floating_panes.get(self.active_floating_pane) {
                let mut size = self.floating_pane_size(pane);
                match direction {
                    PaneDirection::Left => size.cols = size.cols.saturating_sub(amount).max(1),
                    PaneDirection::Right => size.cols = size.cols.saturating_add(amount),
                    PaneDirection::Up => size.rows = size.rows.saturating_sub(amount).max(1),
                    PaneDirection::Down => size.rows = size.rows.saturating_add(amount),
                    _ => return,
                }
                self.resize_floating_pane(pane, size);
            }
            return;
        }
        if self.zoomed.is_some() {
            return;
        }
        let active_index = self.active;
        let mut cursor = self.pane.take().unwrap().cursor();
        let mut index = 0;

        // Position cursor on the active leaf
        loop {
            if cursor.is_leaf() {
                if index == active_index {
                    // Found it
                    break;
                }
                index += 1;
            }
            match cursor.preorder_next() {
                Ok(c) => cursor = c,
                Err(c) => {
                    // Didn't find it
                    self.pane.replace(c.tree());
                    return;
                }
            }
        }

        // We are on the active leaf.
        // Now we go up until we find the parent node that is
        // aligned with the desired direction.
        let split_direction = match direction {
            PaneDirection::Left | PaneDirection::Right => SplitDirection::Horizontal,
            PaneDirection::Up | PaneDirection::Down => SplitDirection::Vertical,
            PaneDirection::Next | PaneDirection::Prev => unreachable!(),
        };
        let delta = match direction {
            PaneDirection::Down | PaneDirection::Right => amount as isize,
            PaneDirection::Up | PaneDirection::Left => -(amount as isize),
            PaneDirection::Next | PaneDirection::Prev => unreachable!(),
        };
        loop {
            match cursor.go_up() {
                Ok(mut c) => {
                    if let Ok(Some(node)) = c.node_mut() {
                        if node.direction == split_direction {
                            self.adjust_node_at_cursor(&mut c, delta);
                            self.cascade_size_from_cursor(c);
                            return;
                        }
                    }

                    cursor = c;
                }

                Err(c) => {
                    self.pane.replace(c.tree());
                    return;
                }
            }
        }
    }

    fn activate_pane_direction(&mut self, direction: PaneDirection) {
        if self.zoomed.is_some() {
            if !configuration().unzoom_on_switch_pane {
                return;
            }
            self.toggle_zoom();
        }

        if self.floating_pane_visible && self.floating_panes.len() > 0 {
            let len = self.floating_panes.len();

            let new_index = match direction {
                PaneDirection::Right | PaneDirection::Next => (self.active_floating_pane + 1) % len,
                PaneDirection::Left | PaneDirection::Prev => {
                    (self.active_floating_pane + len - 1) % len
                }
                //Not sure what to do with other directions, leave as is for now
                _ => self.active_floating_pane,
            };

            self.set_active_floating_pane(new_index, true);
            return;
        }

        if let Some(panel_idx) = self.get_pane_direction(direction, false) {
            self.set_active_idx(panel_idx);
        }
        let mux = Mux::get();
        if let Some(window_id) = mux.window_containing_tab(self.id) {
            mux.notify(MuxNotification::WindowInvalidated(window_id));
        }
    }

    fn get_pane_direction(&mut self, direction: PaneDirection, ignore_zoom: bool) -> Option<usize> {
        let panes = if ignore_zoom {
            self.iter_panes_ignoring_zoom()
        } else {
            self.iter_panes()
        };

        let active = match panes.iter().find(|pane| pane.is_active) {
            Some(p) => p,
            None => {
                // No active pane somehow...
                return Some(0);
            }
        };

        if matches!(direction, PaneDirection::Next | PaneDirection::Prev) {
            let max_pane_id = panes.iter().map(|p| p.index).max().unwrap_or(active.index);

            return Some(if direction == PaneDirection::Next {
                if active.index == max_pane_id {
                    0
                } else {
                    active.index + 1
                }
            } else {
                if active.index == 0 {
                    max_pane_id
                } else {
                    active.index - 1
                }
            });
        }

        let mut best = None;

        let recency = &self.recency;

        fn edge_intersects(
            active_start: usize,
            active_size: usize,
            current_start: usize,
            current_size: usize,
        ) -> bool {
            intersects_range(
                &(active_start..active_start + active_size),
                &(current_start..current_start + current_size),
            )
        }

        for pane in &panes {
            let score = match direction {
                PaneDirection::Right => {
                    if pane.left == active.left + active.width + 1
                        && edge_intersects(active.top, active.height, pane.top, pane.height)
                    {
                        1 + recency.score(pane.index)
                    } else {
                        0
                    }
                }
                PaneDirection::Left => {
                    if pane.left + pane.width + 1 == active.left
                        && edge_intersects(active.top, active.height, pane.top, pane.height)
                    {
                        1 + recency.score(pane.index)
                    } else {
                        0
                    }
                }
                PaneDirection::Up => {
                    if pane.top + pane.height + 1 == active.top
                        && edge_intersects(active.left, active.width, pane.left, pane.width)
                    {
                        1 + recency.score(pane.index)
                    } else {
                        0
                    }
                }
                PaneDirection::Down => {
                    if active.top + active.height + 1 == pane.top
                        && edge_intersects(active.left, active.width, pane.left, pane.width)
                    {
                        1 + recency.score(pane.index)
                    } else {
                        0
                    }
                }
                PaneDirection::Next | PaneDirection::Prev => unreachable!(),
            };

            if score > 0 {
                let target = match best.take() {
                    Some((best_score, best_pane)) if best_score > score => (best_score, best_pane),
                    _ => (score, pane),
                };
                best.replace(target);
            }
        }

        if let Some((_, target)) = best.take() {
            return Some(target.index);
        }
        None
    }

    fn prune_dead_panes(&mut self) -> bool {
        let mux = Mux::get();
        let previous_floating_pane_visibility = self.floating_pane_visible;
        let previous_active_floating_pane = self.active_floating_pane;
        let mut invalidate = !self
            .remove_pane_if(
                |_, pane| {
                    // If the pane is no longer known to the mux, then its liveness
                    // state isn't guaranteed to be monitored or updated, so let's
                    // consider the pane effectively dead if it isn't in the mux.
                    // <https://github.com/wezterm/wezterm/issues/4030>
                    let in_mux = mux.get_pane(pane.pane_id()).is_some();
                    let dead = pane.is_dead();
                    log::trace!(
                        "prune_dead_panes: pane_id={} dead={} in_mux={}",
                        pane.pane_id(),
                        dead,
                        in_mux
                    );
                    dead || !in_mux
                },
                true,
            )
            .is_empty();

        if self.iter_panes().iter().next().is_none() {
            if let Some(global_id) = Mux::get().global_floating_pane_id() {
                if let Some(index) = self
                    .floating_panes
                    .iter()
                    .position(|pane| pane.pane_id() == global_id)
                {
                    self.remove_floating_pane(index).ok();
                }
            }
            let pane_ids: Vec<_> = self
                .floating_panes
                .iter()
                .map(|floating_pane| floating_pane.pane_id())
                .collect();

            for pane_id in pane_ids {
                self.kill_pane(pane_id);
            }
        }

        if previous_floating_pane_visibility != self.floating_pane_visible
            || previous_active_floating_pane != self.active_floating_pane
        {
            invalidate = true;
        }

        invalidate
    }

    fn kill_pane(&mut self, pane_id: PaneId) -> bool {
        !self
            .remove_pane_if(|_, pane| pane.pane_id() == pane_id, true)
            .is_empty()
    }

    fn kill_panes_in_domain(&mut self, domain: DomainId) -> bool {
        !self
            .remove_pane_if(|_, pane| pane.domain_id() == domain, true)
            .is_empty()
    }

    fn remove_pane(&mut self, pane_id: PaneId) -> Option<Arc<dyn Pane>> {
        let panes = self.remove_pane_if(|_, pane| pane.pane_id() == pane_id, false);
        for pane in panes {
            return Some(pane);
        }
        None
    }

    fn remove_pane_if<F>(&mut self, f: F, kill: bool) -> Vec<Arc<dyn Pane>>
    where
        F: Fn(usize, &Arc<dyn Pane>) -> bool,
    {
        let mut dead_panes = vec![];
        let zoomed_pane = self.zoomed.as_ref().map(|p| p.pane_id());

        {
            let root_size = self.size;
            let mut cursor = self.pane.take().unwrap().cursor();
            let mut pane_index = 0;
            let mut removed_indices = vec![];
            let cell_dims = self.cell_dimensions();

            loop {
                // Figure out the available size by looking at our immediate parent node.
                // If we are the root, look at the tab size
                let pane_size = if let Some((branch, Some(parent))) = cursor.path_to_root().next() {
                    if branch == PathBranch::IsRight {
                        parent.second
                    } else {
                        parent.first
                    }
                } else {
                    root_size
                };

                if cursor.is_leaf() {
                    let pane = Arc::clone(cursor.leaf_mut().unwrap());
                    if f(pane_index, &pane) {
                        removed_indices.push(pane_index);
                        if Some(pane.pane_id()) == zoomed_pane {
                            // If we removed the zoomed pane, un-zoom our state!
                            self.zoomed.take();
                        }
                        let parent;
                        match cursor.unsplit_leaf() {
                            Ok((c, dead, p)) => {
                                dead_panes.push(dead);
                                parent = p.unwrap();
                                cursor = c;
                            }
                            Err(c) => {
                                // We might be the root, for example
                                if c.is_top() && c.is_leaf() {
                                    self.pane.replace(Tree::Empty);
                                    dead_panes.push(pane);
                                } else {
                                    self.pane.replace(c.tree());
                                }
                                break;
                            }
                        };

                        // Now we need to increase the size of the current node
                        // and propagate the revised size to its children.
                        let size = TerminalSize {
                            rows: parent.height(),
                            cols: parent.width(),
                            pixel_width: cell_dims.pixel_width * parent.width(),
                            pixel_height: cell_dims.pixel_height * parent.height(),
                            dpi: cell_dims.dpi,
                        };

                        if let Some(unsplit) = cursor.leaf_mut() {
                            unsplit.resize(size).ok();
                        } else {
                            self.apply_pane_size(size, &mut cursor);
                        }
                    } else if !dead_panes.is_empty() {
                        // Apply our revised size to the tty
                        pane.resize(pane_size).ok();
                    }

                    pane_index += 1;
                } else if !dead_panes.is_empty() {
                    self.apply_pane_size(pane_size, &mut cursor);
                }
                match cursor.preorder_next() {
                    Ok(c) => cursor = c,
                    Err(c) => {
                        self.pane.replace(c.tree());
                        break;
                    }
                }
            }

            // Figure out which pane should now be active.
            // If panes earlier than the active pane were closed, then we
            // need to shift the active pane down
            let active_idx = self.active;
            removed_indices.retain(|&idx| idx <= active_idx);
            self.active = active_idx.saturating_sub(removed_indices.len());
        }

        let mut floating_pane_indices_to_remove = Vec::new();
        let mut active_floating_pane_dead = false;

        for (i, floating_pane) in self.floating_panes.iter().enumerate() {
            if floating_pane.is_dead() || f(0, floating_pane) {
                dead_panes.push(Arc::clone(floating_pane));
                floating_pane_indices_to_remove.push(i);
                if i == self.active_floating_pane {
                    active_floating_pane_dead = true;
                }
            }
        }

        if active_floating_pane_dead {
            self.set_floating_pane_visibility(false, false);
        }

        let removed_before_active = floating_pane_indices_to_remove
            .iter()
            .filter(|&&idx| idx < self.active_floating_pane)
            .count();
        for i in floating_pane_indices_to_remove.into_iter().rev() {
            self.floating_panes.remove(i);
        }
        self.active_floating_pane = self
            .active_floating_pane
            .saturating_sub(removed_before_active)
            .min(self.floating_panes.len().saturating_sub(1));

        if !dead_panes.is_empty() && kill {
            let to_kill: Vec<_> = dead_panes.iter().map(|p| p.pane_id()).collect();
            promise::spawn::spawn_into_main_thread(async move {
                let mux = Mux::get();
                for pane_id in to_kill.into_iter() {
                    mux.remove_pane(pane_id);
                }
            })
            .detach();
        }
        dead_panes
    }

    fn can_close_without_prompting(&mut self, reason: CloseReason) -> bool {
        let panes = self.iter_panes_ignoring_zoom();
        panes
            .iter()
            .map(|pos| &pos.pane)
            .chain(self.floating_panes.iter())
            .all(|pane| pane.can_close_without_prompting(reason))
    }

    fn is_dead(&mut self) -> bool {
        // Make sure we account for all panes, so that we don't
        // kill the whole tab if the zoomed pane is dead!
        let panes = self.iter_panes_ignoring_zoom();
        let mut dead_count = 0;
        for pos in &panes {
            if pos.pane.is_dead() {
                dead_count += 1;
            }
        }
        dead_count == panes.len()
    }

    fn is_floating_pane_active(&self) -> bool {
        self.floating_panes.len() > 0 && self.floating_pane_visible
    }

    fn get_active_pane(&mut self) -> Option<Arc<dyn Pane>> {
        if let Some(zoomed) = self.zoomed.as_ref() {
            return Some(Arc::clone(zoomed));
        }

        if self.is_floating_pane_active() {
            return self
                .floating_panes
                .get(self.active_floating_pane)
                .map(Arc::clone);
        }

        self.iter_panes_ignoring_zoom()
            .iter()
            .nth(self.active)
            .map(|p| Arc::clone(&p.pane))
    }

    fn get_active_idx(&self) -> usize {
        self.active
    }

    fn set_active_pane(&mut self, pane: &Arc<dyn Pane>) {
        for (i, floating_pane) in self.floating_panes.iter().enumerate() {
            if floating_pane.pane_id() == pane.pane_id() {
                self.set_active_floating_pane(i, true);
                self.set_floating_pane_visibility(true, true);
                return;
            }
        }

        if self.floating_pane_visible {
            self.set_floating_pane_visibility(false, true);
        }

        let prior = self.get_active_pane();

        if is_pane(pane, &prior.as_ref()) {
            return;
        }

        if self.zoomed.is_some() {
            if !configuration().unzoom_on_switch_pane {
                return;
            }
            self.toggle_zoom();
        }

        if let Some(item) = self
            .iter_panes_ignoring_zoom()
            .iter()
            .find(|p| p.pane.pane_id() == pane.pane_id())
        {
            self.active = item.index;
            self.recency.tag(item.index);
            self.advise_focus_change(prior);
        }
    }

    fn append_floating_pane(&mut self, pane: Arc<dyn Pane>) {
        self.floating_panes.push(pane);
        self.set_active_floating_pane(self.floating_panes.len() - 1, true);
        self.set_floating_pane_visibility(true, false);
    }

    fn set_floating_pane_visibility(&mut self, visible: bool, invalidate: bool) {
        if visible != self.floating_pane_visible {
            let prior = self.get_active_pane();
            self.floating_pane_visible = visible;
            self.advise_focus_change(prior);
            let mux = Mux::get();
            mux.notify(MuxNotification::FloatingPaneVisibilityChanged {
                tab_id: self.id,
                visible,
            });
            if invalidate {
                if let Some(window_id) = mux.window_containing_tab(self.id) {
                    mux.notify(MuxNotification::WindowInvalidated(window_id));
                }
            }
        }
    }

    fn set_active_floating_pane(&mut self, index: usize, invalidate: bool) {
        if index != self.active_floating_pane && index < self.floating_panes.len() {
            let prior = self.get_active_pane();
            self.active_floating_pane = index;
            self.advise_focus_change(prior);
            let mux = Mux::get();
            mux.notify(MuxNotification::ActiveFloatingPaneChanged {
                tab_id: self.id,
                index: index,
            });
            if invalidate {
                if let Some(window_id) = mux.window_containing_tab(self.id) {
                    mux.notify(MuxNotification::WindowInvalidated(window_id));
                }
            }
        }
    }

    fn advise_focus_change(&mut self, prior: Option<Arc<dyn Pane>>) {
        let mux = Mux::get();
        let current = self.get_active_pane();
        match (prior, current) {
            (Some(prior), Some(current)) if prior.pane_id() != current.pane_id() => {
                prior.focus_changed(false);
                current.focus_changed(true);
                mux.notify(MuxNotification::PaneFocused(current.pane_id()));
            }
            (None, Some(current)) => {
                current.focus_changed(true);
                mux.notify(MuxNotification::PaneFocused(current.pane_id()));
            }
            (Some(prior), None) => {
                prior.focus_changed(false);
            }
            (Some(_), Some(_)) | (None, None) => {
                // no change
            }
        }
    }

    fn set_active_idx(&mut self, pane_index: usize) {
        let prior = self.get_active_pane();
        self.active = pane_index;
        self.recency.tag(pane_index);
        self.advise_focus_change(prior);
    }

    fn assign_pane(&mut self, pane: &Arc<dyn Pane>) {
        match Tree::new().cursor().assign_top(Arc::clone(pane)) {
            Ok(c) => self.pane = Some(c.tree()),
            Err(_) => panic!("tried to assign root pane to non-empty tree"),
        }
    }

    fn cell_dimensions(&self) -> TerminalSize {
        cell_dimensions(&self.size)
    }

    fn swap_active_with_index(&mut self, pane_index: usize, keep_focus: bool) -> Option<()> {
        let active_idx = self.get_active_idx();
        let mut pane = self.get_active_pane()?;
        log::trace!(
            "swap_active_with_index: pane_index {} active {}",
            pane_index,
            active_idx
        );

        {
            let mut cursor = self.pane.take().unwrap().cursor();

            // locate the requested index
            match cursor.go_to_nth_leaf(pane_index) {
                Ok(c) => cursor = c,
                Err(c) => {
                    log::trace!("didn't find pane {pane_index}");
                    self.pane.replace(c.tree());
                    return None;
                }
            };

            std::mem::swap(&mut pane, cursor.leaf_mut().unwrap());

            // re-position to the root
            cursor = cursor.tree().cursor();

            // and now go and update the active idx
            match cursor.go_to_nth_leaf(active_idx) {
                Ok(c) => cursor = c,
                Err(c) => {
                    self.pane.replace(c.tree());
                    log::trace!("didn't find active {active_idx}");
                    return None;
                }
            };

            std::mem::swap(&mut pane, cursor.leaf_mut().unwrap());
            self.pane.replace(cursor.tree());

            // Advise the panes of their new sizes
            let size = self.size;
            apply_sizes_from_splits(self.pane.as_mut().unwrap(), &size);
        }

        // And update focus
        if keep_focus {
            self.set_active_idx(pane_index);
        } else {
            self.advise_focus_change(Some(pane));
        }
        None
    }

    fn remove_floating_pane(&mut self, index: usize) -> anyhow::Result<Arc<dyn Pane>> {
        if index >= self.floating_panes.len() {
            anyhow::bail!("floating_panes index {index} out or range")
        }

        let result = self.floating_panes.remove(index);
        if self.floating_panes.is_empty() {
            self.set_floating_pane_visibility(false, true);
            self.active_floating_pane = 0;
        } else {
            if self.active_floating_pane > index {
                self.active_floating_pane -= 1;
            } else if self.active_floating_pane == index {
                if index >= self.floating_panes.len() {
                    self.active_floating_pane = self.floating_panes.len() - 1;
                } else {
                    self.active_floating_pane = index;
                }
            }
        }
        Ok(Arc::clone(&result))
    }

    fn floating_pane_size(&self, pane: &Arc<dyn Pane>) -> TerminalSize {
        let dims = pane.get_dimensions();
        TerminalSize {
            cols: dims.cols,
            rows: dims.viewport_rows,
            pixel_width: dims.pixel_width,
            pixel_height: dims.pixel_height,
            dpi: dims.dpi,
        }
    }

    fn compute_floating_pane_size2(&self) -> TerminalSize {
        self.compute_floating_pane_size(self.size)
    }

    fn compute_floating_pane_size(&self, root_size: TerminalSize) -> TerminalSize {
        let cell_width = (root_size.pixel_width as f32 / root_size.cols as f32).max(1.0);
        let cell_height = (root_size.pixel_height as f32 / root_size.rows as f32).max(1.0);
        let h_context = config::DimensionContext {
            dpi: root_size.dpi as f32,
            pixel_max: root_size.pixel_width as f32,
            pixel_cell: cell_width,
        };

        let v_context = config::DimensionContext {
            dpi: root_size.dpi as f32,
            pixel_max: root_size.pixel_height as f32,
            pixel_cell: cell_height,
        };

        let padding = configuration().floating_pane_padding;
        let top_padding_pixels = padding.top.evaluate_as_pixels(v_context) as usize;
        let bottom_padding_pixels = padding.bottom.evaluate_as_pixels(v_context) as usize;
        let left_padding_pixels = padding.left.evaluate_as_pixels(h_context) as usize;
        let right_padding_pixels = padding.right.evaluate_as_pixels(h_context) as usize;
        let pixel_width = root_size
            .pixel_width
            .saturating_sub(right_padding_pixels.saturating_add(left_padding_pixels))
            .max(cell_width.ceil() as usize);
        let pixel_height = root_size
            .pixel_height
            .saturating_sub(bottom_padding_pixels.saturating_add(top_padding_pixels))
            .max(cell_height.ceil() as usize);
        let cols = ((pixel_width as f32 / cell_width) as usize).max(1);
        let rows = ((pixel_height as f32 / cell_height) as usize).max(1);

        TerminalSize {
            rows,
            cols,
            pixel_width,
            pixel_height,
            dpi: root_size.dpi,
        }
    }

    fn compute_split_size(
        &mut self,
        pane_index: usize,
        request: SplitRequest,
    ) -> Option<SplitDirectionAndSize> {
        let cell_dims = self.cell_dimensions();

        fn split_dimension(dim: usize, request: SplitRequest) -> (usize, usize) {
            let target_size = match request.size {
                SplitSize::Cells(n) => n,
                SplitSize::Percent(n) => (dim * (n as usize)) / 100,
            }
            .max(1);

            let remain = dim.saturating_sub(target_size + 1);

            if request.target_is_second {
                (remain, target_size)
            } else {
                (target_size, remain)
            }
        }

        if request.top_level {
            let size = self.size;

            let ((width1, width2), (height1, height2)) = match request.direction {
                SplitDirection::Horizontal => (
                    split_dimension(size.cols as usize, request),
                    (size.rows as usize, size.rows as usize),
                ),
                SplitDirection::Vertical => (
                    (size.cols as usize, size.cols as usize),
                    split_dimension(size.rows as usize, request),
                ),
            };

            return Some(SplitDirectionAndSize {
                direction: request.direction,
                first: TerminalSize {
                    rows: height1 as _,
                    cols: width1 as _,
                    pixel_height: cell_dims.pixel_height * height1,
                    pixel_width: cell_dims.pixel_width * width1,
                    dpi: cell_dims.dpi,
                },
                second: TerminalSize {
                    rows: height2 as _,
                    cols: width2 as _,
                    pixel_height: cell_dims.pixel_height * height2,
                    pixel_width: cell_dims.pixel_width * width2,
                    dpi: cell_dims.dpi,
                },
            });
        }

        // Ensure that we're not zoomed, otherwise we'll end up in
        // a bogus split state (https://github.com/wezterm/wezterm/issues/723)
        self.set_zoomed(false);

        self.iter_panes().iter().nth(pane_index).map(|pos| {
            let ((width1, width2), (height1, height2)) = match request.direction {
                SplitDirection::Horizontal => (
                    split_dimension(pos.width, request),
                    (pos.height, pos.height),
                ),
                SplitDirection::Vertical => {
                    ((pos.width, pos.width), split_dimension(pos.height, request))
                }
            };

            SplitDirectionAndSize {
                direction: request.direction,
                first: TerminalSize {
                    rows: height1 as _,
                    cols: width1 as _,
                    pixel_height: cell_dims.pixel_height * height1,
                    pixel_width: cell_dims.pixel_width * width1,
                    dpi: cell_dims.dpi,
                },
                second: TerminalSize {
                    rows: height2 as _,
                    cols: width2 as _,
                    pixel_height: cell_dims.pixel_height * height2,
                    pixel_width: cell_dims.pixel_width * width2,
                    dpi: cell_dims.dpi,
                },
            }
        })
    }

    fn has_floating_pane(&self) -> bool {
        self.floating_panes.len() > 0
    }

    fn toggle_floating_pane(&mut self) {
        self.set_floating_pane_visibility(!self.is_floating_pane_active(), true);
    }

    fn add_floating_pane(&mut self, size: TerminalSize, pane: Arc<dyn Pane>) -> anyhow::Result<()> {
        if self.zoomed.is_some() {
            anyhow::bail!("cannot add floating pane while zoomed");
        }

        self.append_floating_pane(Arc::clone(&pane));

        pane.resize(size)?;
        Ok(())
    }

    fn split_and_insert(
        &mut self,
        pane_index: usize,
        request: SplitRequest,
        pane: Arc<dyn Pane>,
    ) -> anyhow::Result<usize> {
        if self.zoomed.is_some() {
            anyhow::bail!("cannot split while zoomed");
        }

        {
            let split_info = self
                .compute_split_size(pane_index, request)
                .ok_or_else(|| {
                    anyhow::anyhow!("invalid pane_index {}; cannot split!", pane_index)
                })?;

            let tab_size = self.size;
            if split_info.first.rows == 0
                || split_info.first.cols == 0
                || split_info.second.rows == 0
                || split_info.second.cols == 0
                || split_info.top_of_second() + split_info.second.rows > tab_size.rows
                || split_info.left_of_second() + split_info.second.cols > tab_size.cols
            {
                log::error!(
                    "No space for split!!! {:#?} height={} width={} top_of_second={} left_of_second={} tab_size={:?}",
                    split_info,
                    split_info.height(),
                    split_info.width(),
                    split_info.top_of_second(),
                    split_info.left_of_second(),
                    tab_size
                );
                anyhow::bail!("No space for split!");
            }

            let needs_resize = if request.top_level {
                self.pane.as_ref().unwrap().num_leaves() > 1
            } else {
                false
            };

            if needs_resize {
                // Pre-emptively resize the tab contents down to
                // match the target size; it's easier to reuse
                // existing resize logic that way
                let (pane_size, rest_size) = if request.target_is_second {
                    (split_info.second, split_info.first)
                } else {
                    (split_info.first, split_info.second)
                };
                pane.resize(pane_size)?;
                self.resize(rest_size);
                // Restore the overall tab size
                self.size = tab_size;
            }

            let mut cursor = self.pane.take().unwrap().cursor();

            if request.top_level && !cursor.is_leaf() {
                let result = if request.target_is_second {
                    cursor.split_node_and_insert_right(Arc::clone(&pane))
                } else {
                    cursor.split_node_and_insert_left(Arc::clone(&pane))
                };
                cursor = match result {
                    Ok(c) => {
                        cursor = match c.assign_node(Some(split_info)) {
                            Err(c) | Ok(c) => c,
                        };

                        self.pane.replace(cursor.tree());

                        let pane_index = if request.target_is_second {
                            self.pane.as_ref().unwrap().num_leaves().saturating_sub(1)
                        } else {
                            0
                        };

                        self.active = pane_index;
                        self.recency.tag(pane_index);
                        return Ok(pane_index);
                    }
                    Err(cursor) => cursor,
                };
            }

            match cursor.go_to_nth_leaf(pane_index) {
                Ok(c) => cursor = c,
                Err(c) => {
                    self.pane.replace(c.tree());
                    anyhow::bail!("invalid pane_index {}; cannot split!", pane_index);
                }
            };

            let existing_pane = Arc::clone(cursor.leaf_mut().unwrap());

            let (pane1, pane2) = if request.target_is_second {
                (existing_pane, pane)
            } else {
                (pane, existing_pane)
            };

            pane1.resize(split_info.first)?;
            pane2.resize(split_info.second.clone())?;

            *cursor.leaf_mut().unwrap() = pane1;

            match cursor.split_leaf_and_insert_right(pane2) {
                Ok(c) => cursor = c,
                Err(c) => {
                    self.pane.replace(c.tree());
                    anyhow::bail!("invalid pane_index {}; cannot split!", pane_index);
                }
            };

            // cursor now points to the newly created split node;
            // we need to populate its split information
            match cursor.assign_node(Some(split_info)) {
                Err(c) | Ok(c) => self.pane.replace(c.tree()),
            };

            if request.target_is_second {
                self.active = pane_index + 1;
                self.recency.tag(pane_index + 1);
            }
        }

        log::debug!("split info after split: {:#?}", self.iter_splits());
        log::debug!("pane info after split: {:#?}", self.iter_panes());

        Ok(if request.target_is_second {
            pane_index + 1
        } else {
            pane_index
        })
    }

    fn get_zoomed_pane(&self) -> Option<Arc<dyn Pane>> {
        self.zoomed.clone()
    }
}

/// This type is used directly by the codec, take care to bump
/// the codec version if you change this
#[derive(Deserialize, Serialize, PartialEq, Debug)]
pub enum PaneNode {
    Empty,
    Split {
        left: Box<PaneNode>,
        right: Box<PaneNode>,
        node: SplitDirectionAndSize,
    },
    Leaf(PaneEntry),
}

impl PaneNode {
    pub fn into_tree(self) -> bintree::Tree<PaneEntry, SplitDirectionAndSize> {
        match self {
            PaneNode::Empty => bintree::Tree::Empty,
            PaneNode::Split { left, right, node } => bintree::Tree::Node {
                left: Box::new((*left).into_tree()),
                right: Box::new((*right).into_tree()),
                data: Some(node),
            },
            PaneNode::Leaf(e) => bintree::Tree::Leaf(e),
        }
    }

    pub fn root_size(&self) -> Option<TerminalSize> {
        match self {
            PaneNode::Empty => None,
            PaneNode::Split { node, .. } => Some(node.size()),
            PaneNode::Leaf(entry) => Some(entry.size),
        }
    }

    pub fn window_and_tab_ids(&self) -> Option<(WindowId, TabId)> {
        match self {
            PaneNode::Empty => None,
            PaneNode::Split { left, right, .. } => match left.window_and_tab_ids() {
                Some(res) => Some(res),
                None => right.window_and_tab_ids(),
            },
            PaneNode::Leaf(entry) => Some((entry.window_id, entry.tab_id)),
        }
    }
}

/// This type is used directly by the codec, take care to bump
/// the codec version if you change this
#[derive(Deserialize, Serialize, PartialEq, Debug, Clone)]
pub struct PaneEntry {
    pub window_id: WindowId,
    pub tab_id: TabId,
    pub pane_id: PaneId,
    pub title: String,
    pub size: TerminalSize,
    pub working_dir: Option<SerdeUrl>,
    pub is_active_pane: bool,
    pub is_zoomed_pane: bool,
    pub is_floating_pane: bool,
    pub workspace: String,
    pub cursor_pos: StableCursorPosition,
    pub physical_top: StableRowIndex,
    pub top_row: usize,
    pub left_col: usize,
    pub tty_name: Option<String>,
}

#[derive(Deserialize, Serialize, PartialEq, Debug)]
pub struct TabEntry {
    pub panes: PaneNode,
    pub floating_panes: Vec<PaneNode>,
    pub active_floating_pane_index: usize,
    pub floating_pane_visible: bool,
}

#[derive(Deserialize, Clone, Serialize, PartialEq, Debug)]
#[serde(try_from = "String", into = "String")]
pub struct SerdeUrl {
    pub url: Url,
}

impl std::convert::TryFrom<String> for SerdeUrl {
    type Error = url::ParseError;
    fn try_from(s: String) -> Result<SerdeUrl, url::ParseError> {
        let url = Url::parse(&s)?;
        Ok(SerdeUrl { url })
    }
}

impl From<Url> for SerdeUrl {
    fn from(url: Url) -> SerdeUrl {
        SerdeUrl { url }
    }
}

impl Into<Url> for SerdeUrl {
    fn into(self) -> Url {
        self.url
    }
}

impl Into<String> for SerdeUrl {
    fn into(self) -> String {
        self.url.as_str().into()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::renderable::*;
    use parking_lot::{MappedMutexGuard, Mutex};
    use rangeset::RangeSet;
    use std::ops::Range;
    use termwiz::surface::SequenceNo;
    use url::Url;
    use wezterm_term::color::ColorPalette;
    use wezterm_term::{KeyCode, KeyModifiers, Line, MouseEvent, StableRowIndex};

    struct FakePane {
        id: PaneId,
        size: Mutex<TerminalSize>,
        can_close: bool,
        focus: Mutex<Vec<bool>>,
        resize_count: Mutex<usize>,
    }

    impl FakePane {
        fn new(id: PaneId, size: TerminalSize) -> Arc<dyn Pane> {
            Arc::new(Self {
                id,
                size: Mutex::new(size),
                can_close: true,
                focus: Mutex::new(vec![]),
                resize_count: Mutex::new(0),
            })
        }
    }

    impl Pane for FakePane {
        fn can_close_without_prompting(&self, _: CloseReason) -> bool {
            self.can_close
        }

        fn focus_changed(&self, focused: bool) {
            self.focus.lock().push(focused);
        }

        fn pane_id(&self) -> PaneId {
            self.id
        }

        fn get_cursor_position(&self) -> StableCursorPosition {
            unimplemented!();
        }

        fn get_current_seqno(&self) -> SequenceNo {
            unimplemented!();
        }

        fn get_changed_since(
            &self,
            _lines: Range<StableRowIndex>,
            _: SequenceNo,
        ) -> RangeSet<StableRowIndex> {
            unimplemented!();
        }

        fn with_lines_mut(
            &self,
            _stable_range: Range<StableRowIndex>,
            _with_lines: &mut dyn WithPaneLines,
        ) {
            unimplemented!();
        }

        fn for_each_logical_line_in_stable_range_mut(
            &self,
            _lines: Range<StableRowIndex>,
            _for_line: &mut dyn ForEachPaneLogicalLine,
        ) {
            unimplemented!();
        }

        fn get_lines(&self, _lines: Range<StableRowIndex>) -> (StableRowIndex, Vec<Line>) {
            unimplemented!();
        }

        fn get_logical_lines(&self, _lines: Range<StableRowIndex>) -> Vec<LogicalLine> {
            unimplemented!();
        }

        fn get_dimensions(&self) -> RenderableDimensions {
            let size = self.size.lock();
            RenderableDimensions {
                cols: size.cols,
                viewport_rows: size.rows,
                scrollback_rows: size.rows,
                physical_top: 0,
                scrollback_top: 0,
                dpi: size.dpi,
                pixel_width: size.pixel_width,
                pixel_height: size.pixel_height,
                reverse_video: false,
            }
        }

        fn get_title(&self) -> String {
            unimplemented!()
        }
        fn send_paste(&self, _text: &str) -> anyhow::Result<()> {
            unimplemented!()
        }
        fn reader(&self) -> anyhow::Result<Option<Box<dyn std::io::Read + Send>>> {
            Ok(None)
        }
        fn writer(&self) -> MappedMutexGuard<'_, dyn std::io::Write> {
            unimplemented!()
        }
        fn resize(&self, size: TerminalSize) -> anyhow::Result<()> {
            *self.resize_count.lock() += 1;
            *self.size.lock() = size;
            Ok(())
        }

        fn key_down(&self, _key: KeyCode, _mods: KeyModifiers) -> anyhow::Result<()> {
            unimplemented!()
        }
        fn key_up(&self, _: KeyCode, _: KeyModifiers) -> anyhow::Result<()> {
            unimplemented!()
        }
        fn mouse_event(&self, _event: MouseEvent) -> anyhow::Result<()> {
            unimplemented!()
        }
        fn is_dead(&self) -> bool {
            false
        }
        fn palette(&self) -> ColorPalette {
            unimplemented!()
        }
        fn domain_id(&self) -> DomainId {
            1
        }
        fn is_mouse_grabbed(&self) -> bool {
            false
        }
        fn is_alt_screen_active(&self) -> bool {
            false
        }
        fn get_current_working_dir(&self, _policy: CachePolicy) -> Option<Url> {
            None
        }
    }

    #[test]
    fn tab_splitting() {
        let size = TerminalSize {
            rows: 24,
            cols: 80,
            pixel_width: 800,
            pixel_height: 600,
            dpi: 96,
        };

        let tab = Tab::new(&size);
        tab.assign_pane(&FakePane::new(1, size));

        let panes = tab.iter_panes();
        assert_eq!(1, panes.len());
        assert_eq!(0, panes[0].index);
        assert_eq!(true, panes[0].is_active);
        assert_eq!(0, panes[0].left);
        assert_eq!(0, panes[0].top);
        assert_eq!(80, panes[0].width);
        assert_eq!(24, panes[0].height);

        assert!(tab
            .compute_split_size(
                1,
                SplitRequest {
                    direction: SplitDirection::Horizontal,
                    ..Default::default()
                }
            )
            .is_none());

        let horz_size = tab
            .compute_split_size(
                0,
                SplitRequest {
                    direction: SplitDirection::Horizontal,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(
            horz_size,
            SplitDirectionAndSize {
                direction: SplitDirection::Horizontal,
                second: TerminalSize {
                    rows: 24,
                    cols: 40,
                    pixel_width: 400,
                    pixel_height: 600,
                    dpi: 96,
                },
                first: TerminalSize {
                    rows: 24,
                    cols: 39,
                    pixel_width: 390,
                    pixel_height: 600,
                    dpi: 96,
                },
            }
        );

        let vert_size = tab
            .compute_split_size(
                0,
                SplitRequest {
                    direction: SplitDirection::Vertical,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(
            vert_size,
            SplitDirectionAndSize {
                direction: SplitDirection::Vertical,
                second: TerminalSize {
                    rows: 12,
                    cols: 80,
                    pixel_width: 800,
                    pixel_height: 300,
                    dpi: 96,
                },
                first: TerminalSize {
                    rows: 11,
                    cols: 80,
                    pixel_width: 800,
                    pixel_height: 275,
                    dpi: 96,
                }
            }
        );

        let new_index = tab
            .split_and_insert(
                0,
                SplitRequest {
                    direction: SplitDirection::Horizontal,
                    ..Default::default()
                },
                FakePane::new(2, horz_size.second),
            )
            .unwrap();
        assert_eq!(new_index, 1);

        let panes = tab.iter_panes();
        assert_eq!(2, panes.len());

        assert_eq!(0, panes[0].index);
        assert_eq!(false, panes[0].is_active);
        assert_eq!(0, panes[0].left);
        assert_eq!(0, panes[0].top);
        assert_eq!(39, panes[0].width);
        assert_eq!(24, panes[0].height);
        assert_eq!(390, panes[0].pixel_width);
        assert_eq!(600, panes[0].pixel_height);
        assert_eq!(1, panes[0].pane.pane_id());

        assert_eq!(1, panes[1].index);
        assert_eq!(true, panes[1].is_active);
        assert_eq!(40, panes[1].left);
        assert_eq!(0, panes[1].top);
        assert_eq!(40, panes[1].width);
        assert_eq!(24, panes[1].height);
        assert_eq!(400, panes[1].pixel_width);
        assert_eq!(600, panes[1].pixel_height);
        assert_eq!(2, panes[1].pane.pane_id());

        let vert_size = tab
            .compute_split_size(
                0,
                SplitRequest {
                    direction: SplitDirection::Vertical,
                    ..Default::default()
                },
            )
            .unwrap();
        let new_index = tab
            .split_and_insert(
                0,
                SplitRequest {
                    direction: SplitDirection::Vertical,
                    top_level: false,
                    target_is_second: true,
                    size: Default::default(),
                },
                FakePane::new(3, vert_size.second),
            )
            .unwrap();
        assert_eq!(new_index, 1);

        let panes = tab.iter_panes();
        assert_eq!(3, panes.len());

        assert_eq!(0, panes[0].index);
        assert_eq!(false, panes[0].is_active);
        assert_eq!(0, panes[0].left);
        assert_eq!(0, panes[0].top);
        assert_eq!(39, panes[0].width);
        assert_eq!(11, panes[0].height);
        assert_eq!(390, panes[0].pixel_width);
        assert_eq!(275, panes[0].pixel_height);
        assert_eq!(1, panes[0].pane.pane_id());

        assert_eq!(1, panes[1].index);
        assert_eq!(true, panes[1].is_active);
        assert_eq!(0, panes[1].left);
        assert_eq!(12, panes[1].top);
        assert_eq!(39, panes[1].width);
        assert_eq!(12, panes[1].height);
        assert_eq!(390, panes[1].pixel_width);
        assert_eq!(300, panes[1].pixel_height);
        assert_eq!(3, panes[1].pane.pane_id());

        assert_eq!(2, panes[2].index);
        assert_eq!(false, panes[2].is_active);
        assert_eq!(40, panes[2].left);
        assert_eq!(0, panes[2].top);
        assert_eq!(40, panes[2].width);
        assert_eq!(24, panes[2].height);
        assert_eq!(400, panes[2].pixel_width);
        assert_eq!(600, panes[2].pixel_height);
        assert_eq!(2, panes[2].pane.pane_id());

        tab.resize_split_by(1, 1);
        let panes = tab.iter_panes();
        assert_eq!(39, panes[0].width);
        assert_eq!(12, panes[0].height);
        assert_eq!(390, panes[0].pixel_width);
        assert_eq!(300, panes[0].pixel_height);

        assert_eq!(39, panes[1].width);
        assert_eq!(11, panes[1].height);
        assert_eq!(390, panes[1].pixel_width);
        assert_eq!(275, panes[1].pixel_height);

        assert_eq!(40, panes[2].width);
        assert_eq!(24, panes[2].height);
        assert_eq!(400, panes[2].pixel_width);
        assert_eq!(600, panes[2].pixel_height);
    }

    #[test]
    fn floating_pane_removal_keeps_indices_and_removes_all_matches() {
        Mux::set_mux(&Arc::new(Mux::new(None)));
        let size = TerminalSize {
            rows: 24,
            cols: 80,
            pixel_width: 800,
            pixel_height: 600,
            dpi: 96,
        };
        let tab = Tab::new(&size);
        tab.assign_pane(&FakePane::new(1, size));
        let mut inner = tab.inner.lock();
        inner.floating_panes = (2..=4).map(|id| FakePane::new(id, size)).collect();
        inner.active_floating_pane = 2;
        assert!(inner.contains_pane(4));
        assert_eq!(
            inner.remove_pane_if(|_, p| p.pane_id() == 3, false).len(),
            1
        );
        assert_eq!(inner.active_floating_pane, 1);
        assert_eq!(inner.floating_panes[1].pane_id(), 4);
        // Removal must include matches after the active index as well.
        inner.active_floating_pane = 0;
        assert_eq!(
            inner.remove_pane_if(|_, p| p.pane_id() >= 2, false).len(),
            2
        );
        assert!(inner.floating_panes.is_empty());
        assert_eq!(inner.active_floating_pane, 0);
    }

    #[test]
    fn floating_pane_focus_and_close_confirmation() {
        Mux::set_mux(&Arc::new(Mux::new(None)));
        let size = TerminalSize {
            rows: 24,
            cols: 80,
            pixel_width: 800,
            pixel_height: 600,
            dpi: 96,
        };
        let tab = Tab::new(&size);
        tab.assign_pane(&FakePane::new(1, size));
        let floating = Arc::new(FakePane {
            id: 2,
            size: Mutex::new(size),
            can_close: false,
            focus: Mutex::new(vec![]),
            resize_count: Mutex::new(0),
        });
        let pane: Arc<dyn Pane> = floating.clone();
        tab.add_floating_pane(size, pane).unwrap();
        assert_eq!(tab.get_active_pane().unwrap().pane_id(), 2);
        let panes = tab.iter_panes_including_floating();
        assert_eq!(panes.len(), 2);
        assert!(panes[1].is_floating && panes[1].is_active);
        assert!(!panes[0].is_active);
        tab.set_floating_pane_visibility(false);
        let panes = tab.iter_panes_including_floating();
        assert!(panes[0].is_active && !panes[1].is_active);
        assert_eq!(tab.get_active_pane().unwrap().pane_id(), 1);
        assert_eq!(*floating.focus.lock(), vec![true, false]);
        assert!(!tab.can_close_without_prompting(CloseReason::Tab));
        tab.set_floating_pane_visibility(true);
        assert_eq!(*floating.focus.lock(), vec![true, false, true]);
        tab.add_floating_pane(size, FakePane::new(3, size)).unwrap();
        assert_eq!(tab.get_pane_direction(PaneDirection::Prev, true), Some(1));
        tab.activate_pane_direction(PaneDirection::Prev);
        assert_eq!(tab.get_active_pane().unwrap().pane_id(), 2);
        tab.activate_pane_direction(PaneDirection::Next);
        assert_eq!(tab.get_active_pane().unwrap().pane_id(), 3);
        assert_eq!(tab.iter_panes()[0].width, size.cols);
        assert_eq!(tab.iter_panes()[0].height, size.rows);
    }

    #[test]
    fn remote_float_sync_uses_advertised_size_for_existing_pane() {
        Mux::set_mux(&Arc::new(Mux::new(None)));
        let size = TerminalSize {
            rows: 40,
            cols: 100,
            pixel_width: 1000,
            pixel_height: 800,
            dpi: 96,
        };
        let float_size = TerminalSize {
            rows: 12,
            cols: 35,
            pixel_width: 350,
            pixel_height: 240,
            dpi: 96,
        };
        let tab = Tab::new(&size);
        let normal = FakePane::new(1, size);
        let floating = FakePane::new(
            2,
            TerminalSize {
                rows: 1,
                cols: 1,
                ..size
            },
        );
        let entry = |pane_id, size, is_floating_pane| {
            PaneNode::Leaf(PaneEntry {
                window_id: 0,
                tab_id: tab.tab_id(),
                pane_id,
                title: String::new(),
                size,
                working_dir: None,
                is_active_pane: true,
                is_zoomed_pane: false,
                is_floating_pane,
                workspace: "test".to_string(),
                cursor_pos: StableCursorPosition::default(),
                physical_top: 0,
                top_row: 0,
                left_col: 0,
                tty_name: None,
            })
        };
        tab.sync_with_pane_tree(
            size,
            TabEntry {
                panes: entry(1, size, false),
                floating_panes: vec![entry(2, float_size, true)],
                active_floating_pane_index: 0,
                floating_pane_visible: true,
            },
            |entry| {
                if entry.pane_id == 1 {
                    normal.clone()
                } else {
                    floating.clone()
                }
            },
        );
        let restored = tab.get_active_floating_pane().unwrap();
        assert_eq!((restored.width, restored.height), (35, 12));
        assert_eq!((restored.pixel_width, restored.pixel_height), (350, 240));
    }

    #[test]
    fn floating_pane_resize_preserves_underlay_and_hidden_size() {
        Mux::set_mux(&Arc::new(Mux::new(None)));
        let size = TerminalSize {
            rows: 40,
            cols: 100,
            pixel_width: 1000,
            pixel_height: 800,
            dpi: 96,
        };
        let tab = Tab::new(&size);
        tab.assign_pane(&FakePane::new(1, size));
        let initial = tab.compute_floating_pane_size();
        tab.add_floating_pane(initial, FakePane::new(2, initial))
            .unwrap();
        tab.adjust_pane_size(PaneDirection::Right, 3);
        tab.adjust_pane_size(PaneDirection::Down, 3);
        let resized = tab.get_active_floating_pane().unwrap();
        assert_eq!(
            (resized.width, resized.height),
            (initial.cols + 3, initial.rows + 3)
        );
        assert_eq!(
            (resized.pixel_width, resized.pixel_height),
            ((initial.cols + 3) * 10, (initial.rows + 3) * 20)
        );
        assert_eq!(tab.iter_panes()[0].width, size.cols);
        tab.set_floating_pane_visibility(false);
        let hidden = tab.iter_panes_including_floating();
        assert_eq!(
            (hidden[1].width, hidden[1].height),
            (resized.width, resized.height)
        );
        tab.set_floating_pane_visibility(true);
        tab.resize(size);
        let restored = tab.get_active_floating_pane().unwrap();
        assert_eq!(
            (restored.width, restored.height),
            (resized.width, resized.height)
        );
        tab.adjust_pane_size(PaneDirection::Left, usize::MAX);
        tab.adjust_pane_size(PaneDirection::Up, usize::MAX);
        let minimum = tab.get_active_floating_pane().unwrap();
        assert_eq!((minimum.width, minimum.height), (1, 1));
        tab.adjust_pane_size(PaneDirection::Right, usize::MAX);
        tab.adjust_pane_size(PaneDirection::Down, usize::MAX);
        let maximum = tab.get_active_floating_pane().unwrap();
        assert_eq!((maximum.width, maximum.height), (98, 38));
    }

    #[test]
    fn global_float_moves_without_replacing_local_panes_or_ending_its_process() {
        let mux = Arc::new(Mux::new(None));
        Mux::set_mux(&mux);
        let size = TerminalSize {
            cols: 100,
            rows: 40,
            pixel_width: 1000,
            pixel_height: 800,
            dpi: 96,
        };
        let float_size = TerminalSize {
            cols: 60,
            rows: 24,
            pixel_width: 600,
            pixel_height: 480,
            dpi: 96,
        };
        let first = Arc::new(Tab::new(&size));
        let second = Arc::new(Tab::new(&size));
        let normal_first = FakePane::new(101, size);
        let normal_second = FakePane::new(102, size);
        let local_first = FakePane::new(103, float_size);
        let local_second = FakePane::new(104, float_size);
        let global = FakePane::new(105, float_size);
        first.assign_pane(&normal_first);
        second.assign_pane(&normal_second);
        first.add_floating_pane(float_size, local_first).unwrap();
        second.add_floating_pane(float_size, local_second).unwrap();
        mux.add_tab_no_panes(&first);
        mux.add_tab_no_panes(&second);
        mux.add_pane(&normal_first).unwrap();
        mux.add_pane(&normal_second).unwrap();
        mux.add_pane(&global).unwrap();
        *mux.global_floating_pane.write() = Some(global.pane_id());

        assert!(mux
            .toggle_existing_global_floating_pane(first.tab_id())
            .unwrap());
        assert_eq!(first.get_active_pane().unwrap().pane_id(), 105);
        assert_eq!(first.floating_pane_count(), 2);
        assert!(mux
            .toggle_existing_global_floating_pane(second.tab_id())
            .unwrap());
        assert!(first.get_floating_pane_by_pane_id(105).is_none());
        assert_eq!(first.floating_pane_count(), 1);
        assert_eq!(first.get_active_pane().unwrap().pane_id(), 101);
        assert_eq!(second.get_active_pane().unwrap().pane_id(), 105);

        second.scale_floating_pane(5);
        assert!(mux
            .toggle_existing_global_floating_pane(second.tab_id())
            .unwrap());
        assert_eq!(second.floating_pane_count(), 1);
        assert_eq!(second.get_active_pane().unwrap().pane_id(), 102);
        assert!(mux.get_pane(105).is_some());
        assert!(mux
            .toggle_existing_global_floating_pane(second.tab_id())
            .unwrap());
        assert_eq!(global.get_dimensions().cols, 63);
        assert_eq!(global.get_dimensions().viewport_rows, 25);
        mux.remove_tab(second.tab_id());
        assert!(mux.get_pane(105).is_some());
        assert!(mux
            .toggle_existing_global_floating_pane(first.tab_id())
            .unwrap());
        mux.detach_global_floating_pane_from_tab(first.tab_id());
        first.toggle_float();
        assert_eq!(first.get_active_pane().unwrap().pane_id(), 103);
        assert_eq!(normal_first.get_dimensions().cols, 100);
        assert_eq!(normal_second.get_dimensions().viewport_rows, 40);
        mux.remove_pane(105);
        assert!(mux.global_floating_pane_id().is_none());
        assert!(!mux
            .toggle_existing_global_floating_pane(first.tab_id())
            .unwrap());
    }

    #[test]
    fn floating_pane_scales_both_axes_once_and_keeps_owner_identity() {
        Mux::set_mux(&Arc::new(Mux::new(None)));
        let size = TerminalSize {
            rows: 40,
            cols: 100,
            pixel_width: 1000,
            pixel_height: 800,
            dpi: 96,
        };
        let tab = Tab::new(&size);
        tab.assign_pane(&FakePane::new(1, size));
        let initial = TerminalSize {
            rows: 24,
            cols: 60,
            pixel_width: 600,
            pixel_height: 480,
            dpi: 96,
        };
        let floating = Arc::new(FakePane {
            id: 2,
            size: Mutex::new(initial),
            can_close: true,
            focus: Mutex::new(vec![]),
            resize_count: Mutex::new(0),
        });
        tab.add_floating_pane(initial, floating.clone()).unwrap();
        let before = *floating.resize_count.lock();
        tab.scale_floating_pane(5);
        assert_eq!(*floating.resize_count.lock(), before + 1);
        let grown = tab.get_active_floating_pane().unwrap();
        assert_eq!((grown.width, grown.height), (63, 25));
        assert_eq!((grown.pixel_width, grown.pixel_height), (630, 500));
        assert_eq!(tab.active_normal_pane().unwrap().pane.pane_id(), 1);
        assert_eq!(tab.get_active_pane().unwrap().pane_id(), 2);
        assert_eq!(tab.floating_pane_count(), 1);
        tab.scale_floating_pane(-5);
        let shrunk = tab.get_active_floating_pane().unwrap();
        assert_eq!((shrunk.width, shrunk.height), (60, 24));
        tab.scale_floating_pane(127);
        let maximum = tab.get_active_floating_pane().unwrap();
        assert_eq!((maximum.width, maximum.height), (95, 38));
        let before = *floating.resize_count.lock();
        tab.scale_floating_pane(5);
        assert_eq!(*floating.resize_count.lock(), before);
        tab.scale_floating_pane(-100);
        let minimum = tab.get_active_floating_pane().unwrap();
        assert_eq!((minimum.width, minimum.height), (3, 1));
        tab.scale_floating_pane(-5);
        assert_eq!(tab.get_active_floating_pane().unwrap().width, 3);
        tab.scale_floating_pane(5);
        let recovered = tab.get_active_floating_pane().unwrap();
        assert_eq!((recovered.width, recovered.height), (6, 2));
        tab.set_floating_pane_visibility(false);
        let before = *floating.resize_count.lock();
        tab.scale_floating_pane(5);
        assert_eq!(*floating.resize_count.lock(), before);
        assert_eq!(tab.floating_pane_count(), 1);
        assert_eq!(tab.get_active_pane().unwrap().pane_id(), 1);
        assert_eq!(
            (tab.iter_panes()[0].width, tab.iter_panes()[0].height),
            (100, 40)
        );
    }

    #[test]
    fn floating_pane_has_a_cell_in_a_tiny_window() {
        let size = TerminalSize {
            rows: 1,
            cols: 1,
            pixel_width: 1,
            pixel_height: 1,
            dpi: 96,
        };
        let tab = Tab::new(&size);
        let float_size = tab.compute_floating_pane_size();
        assert_eq!((float_size.cols, float_size.rows), (1, 1));
        assert_eq!((float_size.pixel_width, float_size.pixel_height), (1, 1));
    }

    fn cols(cols: usize, rows: usize) -> TerminalSize {
        TerminalSize {
            rows,
            cols,
            pixel_width: cols * 10,
            pixel_height: rows * 25,
            dpi: 96,
        }
    }

    fn leaf(pane_id: PaneId, size: TerminalSize) -> PaneNode {
        PaneNode::Leaf(PaneEntry {
            window_id: 0,
            tab_id: 0,
            pane_id,
            title: String::new(),
            size,
            working_dir: None,
            is_active_pane: pane_id == 1,
            is_zoomed_pane: false,
            is_floating_pane: false,
            workspace: "default".to_string(),
            cursor_pos: StableCursorPosition::default(),
            physical_top: 0,
            top_row: 0,
            left_col: 0,
            tty_name: None,
        })
    }

    fn split(
        direction: SplitDirection,
        first: TerminalSize,
        second: TerminalSize,
        left: PaneNode,
        right: PaneNode,
    ) -> PaneNode {
        PaneNode::Split {
            left: Box::new(left),
            right: Box::new(right),
            node: SplitDirectionAndSize {
                direction,
                first,
                second,
            },
        }
    }

    fn tab_entry(panes: PaneNode) -> TabEntry {
        TabEntry {
            panes,
            floating_panes: vec![],
            active_floating_pane_index: 0,
            floating_pane_visible: false,
        }
    }

    /// Run `f` on a helper thread and fail the test if it hasn't returned
    /// within a few seconds. Used for the resize tests below because the
    /// failure mode we guard against is an infinite loop, which would
    /// otherwise hang the whole test run instead of failing it.
    fn must_terminate<F: FnOnce() + Send + 'static>(what: &str, f: F) {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            f();
            tx.send(()).ok();
        });
        assert!(
            rx.recv_timeout(std::time::Duration::from_secs(5)).is_ok(),
            "{} did not terminate",
            what
        );
    }

    /// The split sizes recorded in a tab's tree can disagree with the
    /// size that the tab believes it has (the tree is synced from a mux
    /// server and the two are maintained separately). Shrinking a tab in
    /// that state used to spin forever in adjust_x_size once every leaf
    /// reached 1 column but the tab still had columns left to remove.
    #[test]
    fn resize_terminates_when_tab_size_exceeds_split_sizes() {
        must_terminate("shrinking an overstated horizontal split", || {
            let tab = Tab::new(&cols(80, 24));
            // Tree says the tab is 3 columns wide, tab says 80.
            let root = split(
                SplitDirection::Horizontal,
                cols(1, 24),
                cols(1, 24),
                leaf(1, cols(1, 24)),
                leaf(2, cols(1, 24)),
            );
            tab.sync_with_pane_tree(cols(80, 24), tab_entry(root), |entry| {
                FakePane::new(entry.pane_id, entry.size)
            });
            tab.resize(cols(5, 24));
            assert_eq!(2, tab.iter_panes().len());
        });

        must_terminate("shrinking an overstated vertical split", || {
            let tab = Tab::new(&cols(80, 24));
            // Tree says the tab is 3 rows tall, tab says 24.
            let root = split(
                SplitDirection::Vertical,
                cols(80, 1),
                cols(80, 1),
                leaf(1, cols(80, 1)),
                leaf(2, cols(80, 1)),
            );
            tab.sync_with_pane_tree(cols(80, 24), tab_entry(root), |entry| {
                FakePane::new(entry.pane_id, entry.size)
            });
            tab.resize(cols(80, 5));
            assert_eq!(2, tab.iter_panes().len());
        });
    }

    /// Same class of bug one level down: a parent split records its child
    /// subtree as wider than the child's own split sizes add up to, so the
    /// parent keeps asking the child to give up a column it doesn't have.
    /// This is the shape of the tree (three panes side by side) that hung
    /// wezterm-gui at 100% CPU while attaching to a remote mux.
    #[test]
    fn resize_terminates_when_parent_overstates_nested_split() {
        must_terminate("shrinking a nested horizontal split", || {
            let tab = Tab::new(&cols(65, 24));
            // Inner split is really 3 columns wide but the parent thinks
            // it is 10 (parent width: 54 + 10 + 1 = 65).
            let inner = split(
                SplitDirection::Horizontal,
                cols(1, 24),
                cols(1, 24),
                leaf(2, cols(1, 24)),
                leaf(3, cols(1, 24)),
            );
            let root = split(
                SplitDirection::Horizontal,
                cols(54, 24),
                cols(10, 24),
                leaf(1, cols(54, 24)),
                inner,
            );
            tab.sync_with_pane_tree(cols(65, 24), tab_entry(root), |entry| {
                FakePane::new(entry.pane_id, entry.size)
            });
            tab.resize(cols(5, 24));
            assert_eq!(3, tab.iter_panes().len());
        });

        must_terminate("shrinking a nested vertical split", || {
            let tab = Tab::new(&cols(80, 65));
            let inner = split(
                SplitDirection::Vertical,
                cols(80, 1),
                cols(80, 1),
                leaf(2, cols(80, 1)),
                leaf(3, cols(80, 1)),
            );
            let root = split(
                SplitDirection::Vertical,
                cols(80, 54),
                cols(80, 10),
                leaf(1, cols(80, 54)),
                inner,
            );
            tab.sync_with_pane_tree(cols(80, 65), tab_entry(root), |entry| {
                FakePane::new(entry.pane_id, entry.size)
            });
            tab.resize(cols(80, 5));
            assert_eq!(3, tab.iter_panes().len());
        });
    }

    fn is_send_and_sync<T: Send + Sync>() -> bool {
        true
    }

    #[test]
    fn tab_is_send_and_sync() {
        assert!(is_send_and_sync::<Tab>());
    }
}
