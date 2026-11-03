use mux::renderable::StableCursorPosition;
use std::time::Instant;

#[derive(Clone)]
pub struct PrevCursorPos {
    pos: StableCursorPosition,
    when: Instant,
}

impl PrevCursorPos {
    pub fn new() -> Self {
        PrevCursorPos {
            pos: StableCursorPosition::default(),
            when: Instant::now(),
        }
    }

    /// Make the cursor look like it moved
    pub fn bump(&mut self) {
        self.when = Instant::now();
    }

    /// Update the cursor position if its different
    pub fn update(&mut self, newpos: &StableCursorPosition) {
        if &self.pos != newpos {
            self.pos = *newpos;
            self.when = Instant::now();
        }
    }

    /// When did the cursor last move?
    pub fn last_cursor_movement(&self) -> Instant {
        self.when
    }
}

/// Active window cursor history for the post-process shader uniforms.
#[derive(Clone, Debug, Default)]
pub struct CursorRenderState {
    /// Cursor rect `{x, y, w, h}` in full-window pixel coords, or `None` if
    /// the cursor is hidden.
    pub current_cursor: Option<[f32; 4]>,
    /// Cursor color as RGBA normalized to `[0, 1]`.
    pub current_cursor_color: [f32; 4],
    pub previous_cursor: Option<[f32; 4]>,
    pub previous_cursor_color: [f32; 4],
    /// The `iTime` value at the last cursor change (position or color).
    pub cursor_change_time: f32,
}

impl CursorRenderState {
    /// Update if the rect or color changed, shifting current to previous.
    pub fn update(&mut self, rect: Option<[f32; 4]>, color: [f32; 4], time: f32) {
        if rect != self.current_cursor || color != self.current_cursor_color {
            self.previous_cursor = self.current_cursor;
            self.previous_cursor_color = self.current_cursor_color;
            self.current_cursor = rect;
            self.current_cursor_color = color;
            self.cursor_change_time = time;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CursorRenderState;

    #[test]
    fn active_cursor_moves_between_panes() {
        let mut state = CursorRenderState::default();
        let left = Some([10.0, 20.0, 8.0, 16.0]);
        let right = Some([500.0, 200.0, 8.0, 16.0]);
        let color = [1.0; 4];
        state.update(left, color, 1.0);
        state.update(right, color, 2.0);
        assert_eq!(state.previous_cursor, left);
        assert_eq!(state.current_cursor, right);
        assert_eq!(state.cursor_change_time, 2.0);
        state.update(right, color, 3.0);
        assert_eq!(state.cursor_change_time, 2.0);
        state.update(left, color, 4.0);
        assert_eq!(state.previous_cursor, right);
        assert_eq!(state.current_cursor, left);
    }
}
