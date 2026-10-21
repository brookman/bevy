//! Shared types for ray-tracing denoisers (DLSS Ray Reconstruction, MetalFX denoised scaler, ...)
//! that need auxiliary "guide buffers" from the renderer producing the noisy image.

use bevy_ecs::component::Component;
use bevy_render::texture::CachedTexture;

/// Render-world marker placed on a view by a denoiser's extract system to request that the
/// renderer (e.g. Solari) produces [`ViewDenoiserGuideTextures`] for this view.
#[derive(Component, Clone, Copy, Default)]
pub struct DenoiserGuideBuffers;

/// Additional textures needed as inputs for ray-tracing denoisers.
///
/// All textures are sized at the render (pre-upscale) resolution.
#[derive(Component)]
pub struct ViewDenoiserGuideTextures {
    /// `Rgba8Unorm`, linear diffuse albedo.
    pub diffuse_albedo: CachedTexture,
    /// `Rgba8Unorm`, specular albedo (environment BRDF approximation).
    pub specular_albedo: CachedTexture,
    /// `Rgba16Float`, xyz = world-space normal, w = perceptual roughness.
    pub normal_roughness: CachedTexture,
    /// `R32Float`, reverse-Z hardware depth (possibly replaced by virtual depth through mirrors).
    pub depth: CachedTexture,
    /// `Rg16Float`, motion vectors for the specular (reflected) signal.
    pub specular_motion_vectors: CachedTexture,
}
