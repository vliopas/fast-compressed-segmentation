/**
 * @file display.wgsl
 * @brief Display compute shader output to screen
 * 
 * Vertex and fragment shaders for rendering the ray-traced output
 * to the screen using a fullscreen triangle.
 */

@group(0) @binding(0)
var<storage, read> rayTraceOutput: array<u32>;

struct Screen {
    width: f32,
    height: f32,
    aspect: f32,
    padding: f32,
}

@group(0) @binding(1)
var<uniform> screen: Screen;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
    // Fullscreen triangle
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0)
    );
    
    var uv = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(2.0, 1.0),
        vec2<f32>(0.0, -1.0)
    );
    
    var output: VertexOutput;
    output.position = vec4<f32>(pos[vertexIndex], 0.0, 1.0);
    output.uv = uv[vertexIndex];
    return output;
}

@fragment
fn fragmentMain(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    // Use current canvas resolution supplied via uniform
    let width = u32(screen.width);
    let height = u32(screen.height);
    
    let x = u32(uv.x * f32(width));
    let y = u32(uv.y * f32(height));
    
    if (x >= width || y >= height) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    
    let index = y * width + x;
    let rgba = rayTraceOutput[index];
    
    // Unpack RGBA8
    let r = f32((rgba >> 0u) & 0xFFu) / 255.0;
    let g = f32((rgba >> 8u) & 0xFFu) / 255.0;
    let b = f32((rgba >> 16u) & 0xFFu) / 255.0;
    let a = f32((rgba >> 24u) & 0xFFu) / 255.0;
    
    // Composite over black opaque background
    let backgroundColor = vec3<f32>(0.0, 0.0, 0.0);
    let volumeColor = vec3<f32>(r, g, b);
    let composited = volumeColor * a + backgroundColor * (1.0 - a);
    
    return vec4<f32>(composited, 1.0);  // Always output opaque
}
