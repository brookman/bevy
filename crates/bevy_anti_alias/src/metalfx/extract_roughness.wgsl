@group(0) @binding(0) var normal_roughness: texture_2d<f32>;
@group(0) @binding(1) var roughness_out: texture_storage_2d<r32float, write>;

// Copies the roughness stored in the alpha channel of the packed normal/roughness
// guide buffer into a standalone texture, as required by MetalFX.
@compute @workgroup_size(8, 8, 1)
fn extract_roughness(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(roughness_out);
    if any(id.xy >= size) {
        return;
    }
    let roughness = textureLoad(normal_roughness, id.xy, 0).a;
    textureStore(roughness_out, id.xy, vec4(roughness, 0.0, 0.0, 0.0));
}
