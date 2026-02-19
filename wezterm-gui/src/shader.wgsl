// Vertex shader

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) tex: vec2<f32>,
    @location(2) fg_color: vec4<f32>,
    @location(3) alt_color: vec4<f32>,
    @location(4) hsv: vec3<f32>,
    @location(5) has_color: f32,
    @location(6) mix_value: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex: vec2<f32>,
    @location(1) fg_color: vec4<f32>,
    @location(2) hsv: vec3<f32>,
    @location(3) has_color: f32,
};

// a regular monochrome text glyph
const IS_GLYPH: f32 = 0.0;

// a color emoji glyph
const IS_COLOR_EMOJI: f32 = 1.0;

// a full color texture attached as the
// background image of the window
const IS_BG_IMAGE: f32 = 2.0;

// like 2.0, except that instead of an
// image, we use the solid bg color
const IS_SOLID_COLOR: f32 = 3.0;

// Grayscale poly quad for non-aa text render layers
const IS_GRAY_SCALE: f32 = 4.0;

struct ShaderUniform {
  foreground_text_hsb: vec3<f32>,
  milliseconds: u32,
  projection: mat4x4<f32>,
  current_cursor_rect: vec4<f32>,
  previous_cursor_rect: vec4<f32>,
  current_cursor_color: vec4<f32>,
  cursor_change_time_ms: u32,
  viewport_size: vec2<f32>,
  cursor_trail_params: vec4<f32>,
  _padding: vec2<f32>,
};
@group(0) @binding(0) var<uniform> uniforms: ShaderUniform;

@group(1) @binding(0) var atlas_linear_tex: texture_2d<f32>;
@group(1) @binding(1) var atlas_linear_sampler: sampler;

@group(2) @binding(0) var atlas_nearest_tex: texture_2d<f32>;
@group(2) @binding(1) var atlas_nearest_sampler: sampler;

fn rgb2hsv(c: vec3<f32>) -> vec3<f32>
{
    let K = vec4<f32>(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    let p = mix(vec4<f32>(c.bg, K.wz), vec4<f32>(c.gb, K.xy), step(c.b, c.g));
    let q = mix(vec4<f32>(p.xyw, c.r), vec4<f32>(c.r, p.yzx), step(p.x, c.r));

    let d = q.x - min(q.w, q.y);
    let e = 1.0e-10;
    return vec3<f32>(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

fn hsv2rgb(c: vec3<f32>) -> vec3<f32>
{
    let K = vec4<f32>(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    let p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, vec3(0.0), vec3(1.0)), c.y);
}

fn apply_hsv(c: vec4<f32>, transform: vec3<f32>) -> vec4<f32>
{
  let hsv = rgb2hsv(c.rgb) * transform;
  return vec4<f32>(hsv2rgb(hsv).rgb, c.a);
}

// ============================================================================
// CURSOR TRAIL SHADER FUNCTIONS
// ============================================================================

const PI: f32 = 3.14159265359;
const THRESHOLD_MIN_DISTANCE: f32 = 1.5; // min distance to show trail (units of cursor height)
const BLUR: f32 = 1.0; // blur size in pixels (for antialiasing)

// EaseOutCirc easing function
fn ease(x: f32) -> f32 {
    return sqrt(1.0 - pow(x - 1.0, 2.0));
}

// Normalize pixel coordinates to NDC (-1, 1)
fn normalizeCoord(value: vec2<f32>, isPosition: bool) -> vec2<f32> {
    let factor = select(1.0, 2.0, isPosition);
    return (value * factor - uniforms.viewport_size * select(0.0, 1.0, isPosition)) / uniforms.viewport_size.y;
}

// SDF for rectangle
fn getSdfRectangle(p: vec2<f32>, xy: vec2<f32>, b: vec2<f32>) -> f32 {
    let d = abs(p - xy) - b;
    return length(max(d, vec2<f32>(0.0))) + min(max(d.x, d.y), 0.0);
}

// SDF segment helper for convex quad
fn seg(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>, s: ptr<function, f32>) -> f32 {
    let e = b - a;
    let w = p - a;
    let proj = a + e * clamp(dot(w, e) / dot(e, e), 0.0, 1.0);
    let segd = dot(p - proj, p - proj);
    
    let c0 = step(0.0, p.y - a.y);
    let c1 = 1.0 - step(0.0, p.y - b.y);
    let c2 = 1.0 - step(0.0, e.x * w.y - e.y * w.x);
    let allCond = c0 * c1 * c2;
    let noneCond = (1.0 - c0) * (1.0 - c1) * (1.0 - c2);
    let flip = mix(1.0, -1.0, step(0.5, allCond + noneCond));
    *s *= flip;
    
    return segd;
}

// SDF for convex quad
fn getSdfConvexQuad(p: vec2<f32>, v1: vec2<f32>, v2: vec2<f32>, v3: vec2<f32>, v4: vec2<f32>) -> f32 {
    var s = 1.0;
    var d = dot(p - v1, p - v1);
    
    d = min(d, seg(p, v1, v2, &s));
    d = min(d, seg(p, v2, v3, &s));
    d = min(d, seg(p, v3, v4, &s));
    d = min(d, seg(p, v4, v1, &s));
    
    return s * sqrt(d);
}

// Antialiasing function
fn antialias(distance: f32, blurAmount: f32) -> f32 {
    let normalizedBlur = (vec2<f32>(blurAmount) * 2.0) / uniforms.viewport_size.y;
    return 1.0 - smoothstep(0.0, normalizedBlur.x, distance);
}

// Get duration based on corner alignment with movement direction
fn getDurationFromDot(dot_val: f32, DURATION_LEAD: f32, DURATION_SIDE: f32, DURATION_TRAIL: f32) -> f32 {
    let isLead = step(0.5, dot_val);
    let isSide = step(-0.5, dot_val) * (1.0 - isLead);
    
    var duration = mix(DURATION_TRAIL, DURATION_SIDE, isSide);
    duration = mix(duration, DURATION_LEAD, isLead);
    return duration;
}

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.tex = model.tex;
    out.hsv = model.hsv;
    out.has_color = model.has_color;
    out.fg_color = mix(model.fg_color, model.alt_color, model.mix_value);
    out.clip_position = uniforms.projection * vec4<f32>(model.position, 0.0, 1.0);
    return out;
}

// Fragment shader

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  var color: vec4<f32>;
  var linear_tex: vec4<f32> = textureSample(atlas_linear_tex, atlas_linear_sampler, in.tex);
  var nearest_tex: vec4<f32> = textureSample(atlas_nearest_tex, atlas_nearest_sampler, in.tex);

  var hsv = in.hsv;

  if in.has_color == IS_SOLID_COLOR {
    // Solid color block
    color = in.fg_color;
  } else if in.has_color == IS_BG_IMAGE {
    // Window background attachment
    // Apply window_background_image_opacity to the background image
    color = linear_tex;
    color.a *= in.fg_color.a;
  } else if in.has_color == IS_COLOR_EMOJI {
    // the texture is full color info (eg: color emoji glyph)
    color = nearest_tex;
  } else if in.has_color == IS_GRAY_SCALE {
    // Grayscale poly quad for non-aa text render layers
    color = in.fg_color;
    color.a *= nearest_tex.a;
  } else if in.has_color == IS_GLYPH {
    // the texture is the alpha channel/color mask
    // and we need to tint with the fg_color
    color = in.fg_color;
    color.a = nearest_tex.a;
    hsv *= uniforms.foreground_text_hsb;
  }

  color = apply_hsv(color, hsv);

  // ============================================================================
  // CURSOR TRAIL RENDERING
  // ============================================================================
  
  // Get fragment position in clip space (convert from clip position)
  let fragCoord = in.clip_position.xy;
  
  // Normalize to NDC coordinates (-1, 1)
  let vu = normalizeCoord(fragCoord, true);
  
  // Normalize cursor rects
  let offsetFactor = vec2<f32>(-0.5, 0.5);
  let currentCursor = vec4<f32>(
    normalizeCoord(uniforms.current_cursor_rect.xy, true),
    normalizeCoord(uniforms.current_cursor_rect.zw, false)
  );
  let previousCursor = vec4<f32>(
    normalizeCoord(uniforms.previous_cursor_rect.xy, true),
    normalizeCoord(uniforms.previous_cursor_rect.zw, false)
  );
  
  let centerCC = currentCursor.xy - (currentCursor.zw * offsetFactor);
  let halfSizeCC = currentCursor.zw * 0.5;
  let centerCP = previousCursor.xy - (previousCursor.zw * offsetFactor);
  let halfSizeCP = previousCursor.zw * 0.5;
  
  let sdfCurrentCursor = getSdfRectangle(vu, centerCC, halfSizeCC);
  let lineLength = distance(centerCC, centerCP);
  let minDist = currentCursor.w * THRESHOLD_MIN_DISTANCE;
  
  // Extract trail parameters: [duration, trail_size, blur, thickness]
  let DURATION = uniforms.cursor_trail_params.x;
  let TRAIL_SIZE = uniforms.cursor_trail_params.y;
  let TRAIL_THICKNESS = uniforms.cursor_trail_params.w;
  let TRAIL_THICKNESS_X = 0.9;
  
  let baseProgress = (f32(uniforms.milliseconds) - f32(uniforms.cursor_change_time_ms)) / 1000.0;
  
  // Only render trail if cursor moved far enough and animation is active
  if lineLength > minDist && baseProgress < DURATION - 0.001 {
    // Calculate corner positions for current cursor with thickness
    let cc_half_height = currentCursor.w * 0.5;
    let cc_center_y = currentCursor.y - cc_half_height;
    let cc_new_half_height = cc_half_height * TRAIL_THICKNESS;
    let cc_new_top_y = cc_center_y + cc_new_half_height;
    let cc_new_bottom_y = cc_center_y - cc_new_half_height;
    
    let cc_half_width = currentCursor.z * 0.5;
    let cc_center_x = currentCursor.x + cc_half_width;
    let cc_new_half_width = cc_half_width * TRAIL_THICKNESS_X;
    let cc_new_left_x = cc_center_x - cc_new_half_width;
    let cc_new_right_x = cc_center_x + cc_new_half_width;
    
    let cc_tl = vec2<f32>(cc_new_left_x, cc_new_top_y);
    let cc_tr = vec2<f32>(cc_new_right_x, cc_new_top_y);
    let cc_bl = vec2<f32>(cc_new_left_x, cc_new_bottom_y);
    let cc_br = vec2<f32>(cc_new_right_x, cc_new_bottom_y);
    
    // Calculate corner positions for previous cursor with thickness
    let cp_half_height = previousCursor.w * 0.5;
    let cp_center_y = previousCursor.y - cp_half_height;
    let cp_new_half_height = cp_half_height * TRAIL_THICKNESS;
    let cp_new_top_y = cp_center_y + cp_new_half_height;
    let cp_new_bottom_y = cp_center_y - cp_new_half_height;
    
    let cp_half_width = previousCursor.z * 0.5;
    let cp_center_x = previousCursor.x + cp_half_width;
    let cp_new_half_width = cp_half_width * TRAIL_THICKNESS_X;
    let cp_new_left_x = cp_center_x - cp_new_half_width;
    let cp_new_right_x = cp_center_x + cp_new_half_width;
    
    let cp_tl = vec2<f32>(cp_new_left_x, cp_new_top_y);
    let cp_tr = vec2<f32>(cp_new_right_x, cp_new_top_y);
    let cp_bl = vec2<f32>(cp_new_left_x, cp_new_bottom_y);
    let cp_br = vec2<f32>(cp_new_right_x, cp_new_bottom_y);
    
    // Calculate per-corner durations based on movement direction
    let DURATION_TRAIL = DURATION;
    let DURATION_LEAD = DURATION * (1.0 - TRAIL_SIZE);
    let DURATION_SIDE = (DURATION_LEAD + DURATION_TRAIL) / 2.0;
    
    let moveVec = centerCC - centerCP;
    let s = sign(moveVec);
    
    // Dot products for each corner
    let dot_tl = dot(vec2<f32>(-1.0, 1.0), s);
    let dot_tr = dot(vec2<f32>(1.0, 1.0), s);
    let dot_bl = dot(vec2<f32>(-1.0, -1.0), s);
    let dot_br = dot(vec2<f32>(1.0, -1.0), s);
    
    // Base durations
    var dur_tl = getDurationFromDot(dot_tl, DURATION_LEAD, DURATION_SIDE, DURATION_TRAIL);
    var dur_tr = getDurationFromDot(dot_tr, DURATION_LEAD, DURATION_SIDE, DURATION_TRAIL);
    var dur_bl = getDurationFromDot(dot_bl, DURATION_LEAD, DURATION_SIDE, DURATION_TRAIL);
    var dur_br = getDurationFromDot(dot_br, DURATION_LEAD, DURATION_SIDE, DURATION_TRAIL);
    
    // Adjust for horizontal movement (vertical rail logic)
    let isMovingRight = step(0.5, s.x);
    let isMovingLeft = step(0.5, -s.x);
    
    let dot_right_edge = (dot_tr + dot_br) * 0.5;
    let dur_right_rail = getDurationFromDot(dot_right_edge, DURATION_LEAD, DURATION_SIDE, DURATION_TRAIL);
    
    let dot_left_edge = (dot_tl + dot_bl) * 0.5;
    let dur_left_rail = getDurationFromDot(dot_left_edge, DURATION_LEAD, DURATION_SIDE, DURATION_TRAIL);
    
    dur_tl = mix(dur_tl, dur_left_rail, isMovingLeft);
    dur_bl = mix(dur_bl, dur_left_rail, isMovingLeft);
    dur_tr = mix(dur_tr, dur_right_rail, isMovingRight);
    dur_br = mix(dur_br, dur_right_rail, isMovingRight);
    
    // Calculate progress for each corner
    let prog_tl = ease(clamp(baseProgress / dur_tl, 0.0, 1.0));
    let prog_tr = ease(clamp(baseProgress / dur_tr, 0.0, 1.0));
    let prog_bl = ease(clamp(baseProgress / dur_bl, 0.0, 1.0));
    let prog_br = ease(clamp(baseProgress / dur_br, 0.0, 1.0));
    
    // Interpolate corner positions
    let v_tl = mix(cp_tl, cc_tl, prog_tl);
    let v_tr = mix(cp_tr, cc_tr, prog_tr);
    let v_br = mix(cp_br, cc_br, prog_br);
    let v_bl = mix(cp_bl, cc_bl, prog_bl);
    
    // Calculate SDF for trail quad
    let sdfTrail = getSdfConvexQuad(vu, v_tl, v_tr, v_br, v_bl);
    
    // Apply blur/antialiasing
    var effectiveBlur = BLUR;
    if BLUR < 2.5 {
      // Reduce blur for non-diagonal movement
      let isDiagonal = abs(s.x) * abs(s.y);
      effectiveBlur = mix(0.0, BLUR, isDiagonal);
    }
    
    let shapeAlpha = antialias(sdfTrail, effectiveBlur);
    
    // Use current cursor color for trail
    var trail = uniforms.current_cursor_color;
    let finalAlpha = trail.a * shapeAlpha;
    
    // Blend trail with background, preserving alpha
    color = mix(color, vec4<f32>(trail.rgb, color.a), finalAlpha);
    
    // Punch hole for current cursor (so it renders on top)
    color = mix(color, color, step(0.0, sdfCurrentCursor));
  }

  return color;
}
