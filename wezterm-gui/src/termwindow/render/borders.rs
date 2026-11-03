use crate::quad::TripleLayerQuadAllocator;
use crate::utilsprites::RenderMetrics;
use ::window::ULength;
use config::{ConfigHandle, Dimension, DimensionContext, FloatingPaneBorderConfig};
use mux::tab::PositionedPane;
use window::parameters::Border;

fn floating_border_rects(
    background: euclid::Rect<f32, window::PixelUnit>,
    border: &Border,
) -> [euclid::Rect<f32, window::PixelUnit>; 4] {
    let left = border.left.get() as f32;
    let right = border.right.get() as f32;
    let top = border.top.get() as f32;
    let bottom = border.bottom.get() as f32;
    let x = background.origin.x;
    let y = background.origin.y;
    let width = background.size.width;
    let height = background.size.height;
    [
        euclid::rect(x - left, y - top, left, height + top + bottom),
        euclid::rect(x, y - top, width, top),
        euclid::rect(x, y + height, width, bottom),
        euclid::rect(x + width, y - top, right, height + top + bottom),
    ]
}

impl crate::TermWindow {
    pub fn paint_window_borders(
        &mut self,
        layers: &mut TripleLayerQuadAllocator,
    ) -> anyhow::Result<()> {
        let border_dimensions = self.get_os_border();

        if border_dimensions.top.get() > 0
            || border_dimensions.bottom.get() > 0
            || border_dimensions.left.get() > 0
            || border_dimensions.right.get() > 0
        {
            let height = self.dimensions.pixel_height as f32;
            let width = self.dimensions.pixel_width as f32;

            let border_top = border_dimensions.top.get() as f32;
            if border_top > 0.0 {
                self.filled_rectangle(
                    layers,
                    1,
                    euclid::rect(0.0, 0.0, width, border_top),
                    self.config
                        .window_frame
                        .border_top_color
                        .map(|c| c.to_linear())
                        .unwrap_or(border_dimensions.color),
                )?;
            }

            let border_left = border_dimensions.left.get() as f32;
            if border_left > 0.0 {
                self.filled_rectangle(
                    layers,
                    1,
                    euclid::rect(0.0, 0.0, border_left, height),
                    self.config
                        .window_frame
                        .border_left_color
                        .map(|c| c.to_linear())
                        .unwrap_or(border_dimensions.color),
                )?;
            }

            let border_bottom = border_dimensions.bottom.get() as f32;
            if border_bottom > 0.0 {
                self.filled_rectangle(
                    layers,
                    1,
                    euclid::rect(0.0, height - border_bottom, width, height),
                    self.config
                        .window_frame
                        .border_bottom_color
                        .map(|c| c.to_linear())
                        .unwrap_or(border_dimensions.color),
                )?;
            }

            let border_right = border_dimensions.right.get() as f32;
            if border_right > 0.0 {
                self.filled_rectangle(
                    layers,
                    1,
                    euclid::rect(width - border_right, 0.0, border_right, height),
                    self.config
                        .window_frame
                        .border_right_color
                        .map(|c| c.to_linear())
                        .unwrap_or(border_dimensions.color),
                )?;
            }
        }

        Ok(())
    }

    pub fn compute_background_rect_with_scrollbar(
        &self,
        pos: &PositionedPane,
        padding_left: f32,
        padding_top: f32,
        border: &Border,
        top_pixel_y: f32,
    ) -> euclid::Rect<f32, window::PixelUnit> {
        let mut base_rect =
            self.compute_background_rect(pos, padding_left, padding_top, border, top_pixel_y);
        if pos.is_floating {
            base_rect.size.width =
                base_rect.size.width + self.get_floating_pane_scroll_thumb_width() as f32;
        };
        base_rect
    }

    pub fn compute_background_rect(
        &self,
        pos: &PositionedPane,
        padding_left: f32,
        padding_top: f32,
        border: &Border,
        top_pixel_y: f32,
    ) -> euclid::Rect<f32, window::PixelUnit> {
        let cell_width = self.render_metrics.cell_size.width as f32;
        let cell_height = self.render_metrics.cell_size.height as f32;

        if pos.is_floating {
            let horizontal = DimensionContext {
                dpi: self.dimensions.dpi as f32,
                pixel_max: pos.width as f32 * cell_width,
                pixel_cell: cell_width,
            };
            let vertical = DimensionContext {
                dpi: self.dimensions.dpi as f32,
                pixel_max: pos.height as f32 * cell_height,
                pixel_cell: cell_height,
            };
            let padding = &self.config.floating_pane_content_padding;
            let left = padding.left.evaluate_as_pixels(horizontal) as f32;
            let right = padding.right.evaluate_as_pixels(horizontal) as f32;
            let top = padding.top.evaluate_as_pixels(vertical) as f32;
            let bottom = padding.bottom.evaluate_as_pixels(vertical) as f32;
            return euclid::rect(
                padding_left + border.left.get() as f32 + pos.left as f32 * cell_width - left,
                top_pixel_y + pos.top as f32 * cell_height - top,
                pos.width as f32 * cell_width + left + right,
                pos.height as f32 * cell_height + top + bottom,
            );
        }

        let (x, width_delta) = if pos.left == 0 {
            (
                0.,
                padding_left + border.left.get() as f32 + (cell_width / 2.0),
            )
        } else {
            (
                padding_left + border.left.get() as f32 - (cell_width / 2.0)
                    + (pos.left as f32 * cell_width),
                cell_width,
            )
        };

        let (y, height_delta) = if pos.top == 0 {
            (
                (top_pixel_y - padding_top),
                padding_top + (cell_height / 2.0),
            )
        } else {
            (
                top_pixel_y + (pos.top as f32 * cell_height) - (cell_height / 2.0),
                cell_height,
            )
        };

        euclid::rect(
            x,
            y,
            // Width calculation
            if pos.left + pos.width >= self.terminal_size.cols as usize {
                self.dimensions.pixel_width as f32 - x
            } else {
                (pos.width as f32 * cell_width) + width_delta
            },
            // Height calculation
            if pos.top + pos.height >= self.terminal_size.rows as usize {
                self.dimensions.pixel_height as f32 - y
            } else {
                (pos.height as f32 * cell_height) + height_delta
            },
        )
    }

    pub fn floating_pane_background_rect(
        &self,
        pos: &PositionedPane,
    ) -> anyhow::Result<euclid::Rect<f32, window::PixelUnit>> {
        let (padding_left, padding_top) = self.padding_left_top();
        let os_border = self.get_os_border();
        let top_bar_height = if self.show_tab_bar && !self.config.tab_bar_at_bottom {
            self.tab_bar_pixel_height()?
        } else {
            0.0
        };
        let top_pixel_y = top_bar_height + padding_top + os_border.top.get() as f32;
        Ok(self.compute_background_rect_with_scrollbar(
            pos,
            padding_left,
            padding_top,
            &os_border,
            top_pixel_y,
        ))
    }

    pub fn paint_floating_pane_border(
        &mut self,
        pos: PositionedPane,
        layers: &mut TripleLayerQuadAllocator,
    ) -> anyhow::Result<()> {
        let config = self.config.floating_pane_border.clone();
        let border = self.get_floating_pane_border();
        let background = self.floating_pane_background_rect(&pos)?;
        let colors = [
            config.left_color,
            config.top_color,
            config.bottom_color,
            config.right_color,
        ];
        for (rect, color) in
            IntoIterator::into_iter(floating_border_rects(background, &border)).zip(colors)
        {
            self.filled_rectangle(
                layers,
                2,
                rect,
                color
                    .map(|c| c.to_linear())
                    .unwrap_or(self.get_os_border().color),
            )?;
        }
        Ok(())
    }

    fn apply_config_to_border(
        border: &mut Border,
        left_width: Dimension,
        right_width: Dimension,
        top_height: Dimension,
        bottom_height: Dimension,
        dimensions: &crate::Dimensions,
        render_metrics: &RenderMetrics,
    ) {
        border.left += ULength::new(
            left_width
                .evaluate_as_pixels(DimensionContext {
                    dpi: dimensions.dpi as f32,
                    pixel_max: dimensions.pixel_width as f32,
                    pixel_cell: render_metrics.cell_size.width as f32,
                })
                .ceil() as usize,
        );
        border.right += ULength::new(
            right_width
                .evaluate_as_pixels(DimensionContext {
                    dpi: dimensions.dpi as f32,
                    pixel_max: dimensions.pixel_width as f32,
                    pixel_cell: render_metrics.cell_size.width as f32,
                })
                .ceil() as usize,
        );
        border.top += ULength::new(
            top_height
                .evaluate_as_pixels(DimensionContext {
                    dpi: dimensions.dpi as f32,
                    pixel_max: dimensions.pixel_height as f32,
                    pixel_cell: render_metrics.cell_size.height as f32,
                })
                .ceil() as usize,
        );
        border.bottom += ULength::new(
            bottom_height
                .evaluate_as_pixels(DimensionContext {
                    dpi: dimensions.dpi as f32,
                    pixel_max: dimensions.pixel_height as f32,
                    pixel_cell: render_metrics.cell_size.height as f32,
                })
                .ceil() as usize,
        );
    }

    pub fn get_os_border_impl(
        os_parameters: &Option<window::parameters::Parameters>,
        config: &ConfigHandle,
        dimensions: &crate::Dimensions,
        render_metrics: &RenderMetrics,
    ) -> window::parameters::Border {
        let mut border = os_parameters
            .as_ref()
            .and_then(|p| p.border_dimensions.clone())
            .unwrap_or_default();

        Self::apply_config_to_border(
            &mut border,
            config.window_frame.border_left_width,
            config.window_frame.border_right_width,
            config.window_frame.border_top_height,
            config.window_frame.border_bottom_height,
            dimensions,
            render_metrics,
        );

        border
    }

    fn get_floating_pane_border_impl(
        dimensions: &crate::Dimensions,
        render_metrics: &RenderMetrics,
        border_config: &FloatingPaneBorderConfig,
    ) -> Border {
        let mut border = Border::default();
        Self::apply_config_to_border(
            &mut border,
            border_config.left_width,
            border_config.right_width,
            border_config.top_height,
            border_config.bottom_height,
            dimensions,
            render_metrics,
        );
        border
    }

    pub fn get_floating_pane_border(&self) -> Border {
        Self::get_floating_pane_border_impl(
            &self.dimensions,
            &self.render_metrics,
            &self.config.floating_pane_border,
        )
    }

    pub fn get_os_border(&self) -> window::parameters::Border {
        Self::get_os_border_impl(
            &self.os_parameters,
            &self.config,
            &self.dimensions,
            &self.render_metrics,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asymmetric_floating_borders_stay_outside_the_background() {
        let border = Border {
            left: ULength::new(4),
            right: ULength::new(8),
            top: ULength::new(6),
            bottom: ULength::new(10),
            ..Default::default()
        };
        let sides = floating_border_rects(euclid::rect(100.0, 80.0, 400.0, 240.0), &border);
        assert_eq!(
            sides,
            [
                euclid::rect(96.0, 74.0, 4.0, 256.0),
                euclid::rect(100.0, 74.0, 400.0, 6.0),
                euclid::rect(100.0, 320.0, 400.0, 10.0),
                euclid::rect(500.0, 74.0, 8.0, 256.0),
            ]
        );
    }
}
