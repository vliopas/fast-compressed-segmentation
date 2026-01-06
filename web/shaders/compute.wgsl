@group(0) @binding(0) var outputTex : texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8,8)
fn cs_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    // Paint the background blue
    textureStore(outputTex, vec2<i32>(gid.xy), vec4<f32>(0.5, 0.6, 0.9, 1.0));
}