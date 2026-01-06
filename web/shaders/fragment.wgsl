@group(0) @binding(0) var<uniform> uniforms: vec2<f32>;       // angle, scale
@group(0) @binding(1) var bgTexture: texture_2d<f32>;         // compute-painted texture

@fragment
fn fs_main(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
    let coords = vec2<i32>(fragCoord.xy);
    let bgColor = textureLoad(bgTexture, coords, 0);
    return bgColor; // could blend with triangle later
}