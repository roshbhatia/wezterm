@fragment
fn fs_postprocess(input: VertexOutput) -> @location(0) vec4<f32> {
    let base = textureSample(screen_texture, screen_sampler, input.uv);
    let age = max(pp.time - pp.cursor_change_time, 0.0);
    if age >= 0.12 || pp.current_cursor.z <= 0.0 || pp.previous_cursor.z <= 0.0 {
        return base;
    }

    let current = pp.current_cursor.xy + pp.current_cursor.zw * 0.5;
    let previous = pp.previous_cursor.xy + pp.previous_cursor.zw * 0.5;
    let delta = previous - current;
    let distance = length(delta);
    if distance < max(pp.current_cursor.z * 1.5, 12.0) {
        return base;
    }

    let progress = clamp(age / 0.12, 0.0, 1.0);
    let remaining = pow(1.0 - progress, 3.0);
    let tail = current + delta / distance * min(distance * remaining, 120.0);
    let segment = current - tail;
    let length_squared = max(dot(segment, segment), 0.001);
    let point = input.position.xy;
    let along = clamp(dot(point - tail, segment) / length_squared, 0.0, 1.0);
    let nearest = tail + along * segment;
    let radius = max(min(pp.current_cursor.z, pp.current_cursor.w) * 0.12, 1.0);
    let coverage = 1.0 - smoothstep(radius - 0.5, radius + 0.5, length(point - nearest));
    let opacity = coverage * along * (1.0 - progress) * 0.3;

    if all(point >= pp.current_cursor.xy) && all(point < pp.current_cursor.xy + pp.current_cursor.zw) {
        return base;
    }
    return vec4<f32>(mix(base.rgb, pp.current_cursor_color.rgb, opacity), base.a);
}
