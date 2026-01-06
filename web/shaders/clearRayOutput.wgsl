// Clear ray trace output buffer to black
// This ensures no stale pixel data remains from previous frames

@group(0) @binding(0)
var<storage, read_write> rayTraceOutput: array<u32>;

struct Screen {
    width: f32,
    height: f32,
    aspect: f32,
    padding: f32,
}

@group(0) @binding(1)
var<uniform> screen: Screen;

@compute @workgroup_size(8, 8, 1)
fn clearRayOutput(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel_x = global_id.x;
    let pixel_y = global_id.y;

    let screen_width = u32(screen.width);
    let screen_height = u32(screen.height);

    if (pixel_x >= screen_width || pixel_y >= screen_height) {
        return;
    }

    let pixel_index = pixel_y * screen_width + pixel_x;
    // Clear to transparent black - RGBA(0,0,0,0)
    rayTraceOutput[pixel_index] = 0u;
}
