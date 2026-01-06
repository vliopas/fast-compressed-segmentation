struct Transform {
    angle: f32,
    scale: f32,
};

@group(0) @binding(0)
var<uniform> transform: Transform;

@vertex
fn vs_main(@location(0) position: vec2<f32>) -> @builtin(position) vec4<f32> {
    let c = cos(transform.angle);
    let s = sin(transform.angle);

    let rotated = vec2<f32>(
        position.x * c - position.y * s,
        position.x * s + position.y * c
    );

    let scaled = rotated * transform.scale;
    return vec4<f32>(scaled, 0.0, 1.0);
}