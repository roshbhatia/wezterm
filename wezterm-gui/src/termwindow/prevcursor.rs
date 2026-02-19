use mux::renderable::StableCursorPosition;
use std::time::Instant;

#[derive(Clone)]
pub struct PrevCursorPos {
    pos: StableCursorPosition,
    when: Instant,
    // Physical screen coordinates for cursor trail
    pixel_rect: (f32, f32, f32, f32), // x, y, width, height
    color: (f32, f32, f32, f32),      // RGBA
}

impl PrevCursorPos {
    pub fn new() -> Self {
        PrevCursorPos {
            pos: StableCursorPosition::default(),
            when: Instant::now(),
            pixel_rect: (0.0, 0.0, 0.0, 0.0),
            color: (1.0, 1.0, 1.0, 1.0),
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

    /// Update cursor position with pixel coordinates and color
    pub fn update_with_pixels(
        &mut self,
        newpos: &StableCursorPosition,
        pixel_rect: (f32, f32, f32, f32),
        color: (f32, f32, f32, f32),
    ) {
        if &self.pos != newpos {
            self.pos = *newpos;
            self.pixel_rect = pixel_rect;
            self.color = color;
            self.when = Instant::now();
        }
    }

    /// When did the cursor last move?
    pub fn last_cursor_movement(&self) -> Instant {
        self.when
    }

    /// Get pixel rectangle (x, y, width, height)
    pub fn get_pixel_rect(&self) -> (f32, f32, f32, f32) {
        self.pixel_rect
    }

    /// Get color (RGBA)
    pub fn get_color(&self) -> (f32, f32, f32, f32) {
        self.color
    }
}
