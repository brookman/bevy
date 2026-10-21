// Adapts the shared denoiser guide buffers to MetalFX's input conventions.

@group(0) @binding(0) var normal_roughness: texture_2d<f32>;
@group(0) @binding(1) var prepass_depth: texture_depth_2d;
@group(0) @binding(2) var guide_depth: texture_2d<f32>;
@group(0) @binding(3) var surface_motion_vectors: texture_2d<f32>;
@group(0) @binding(4) var specular_motion_vectors: texture_2d<f32>;
@group(0) @binding(5) var roughness_out: texture_storage_2d<r32float, write>;
@group(0) @binding(6) var motion_out: texture_storage_2d<rg16float, write>;

@compute @workgroup_size(8, 8, 1)
fn prepare_inputs(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(roughness_out);
    if any(id.xy >= size) {
        return;
    }

    // MetalFX takes roughness as a standalone texture; the guide buffer packs it into alpha.
    let roughness = textureLoad(normal_roughness, id.xy, 0).a;
    textureStore(roughness_out, id.xy, vec4(roughness, 0.0, 0.0, 0.0));

    // Where Solari performed primary surface replacement (mirrors), the guide depth/normal/albedo
    // describe the virtual surface seen through the mirror, and the guide depth is pushed further
    // away than the rasterized depth. MetalFX has no separate specular motion input, so its motion
    // vectors must describe that same virtual surface. Solari only overwrites the main motion
    // vectors itself when the format supports read-write storage, which Metal does not offer for
    // Rg16Float, so do it here.
    let replaced = textureLoad(guide_depth, id.xy, 0).r != textureLoad(prepass_depth, id.xy, 0);
    var motion = textureLoad(surface_motion_vectors, id.xy, 0).xy;
    if replaced {
        motion = textureLoad(specular_motion_vectors, id.xy, 0).xy;
    }
    textureStore(motion_out, id.xy, vec4(motion, 0.0, 0.0));
}
