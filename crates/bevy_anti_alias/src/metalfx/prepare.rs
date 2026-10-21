use super::{ffi, MetalFxDenoiser};
use bevy_camera::{Camera3d, CameraMainTextureUsages, MainPassResolutionOverride};
use bevy_core_pipeline::prepass::{DepthPrepass, MotionVectorPrepass};
use bevy_diagnostic::FrameCount;
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::With,
    system::{Commands, Query, Res},
};
use bevy_image::ToExtents;
use bevy_math::{UVec2, Vec2, Vec4Swizzles};
use bevy_render::{
    camera::{MipBias, TemporalJitter},
    render_resource::{
        TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureViewDescriptor,
    },
    renderer::RenderDevice,
    texture::CachedTexture,
    view::ExtractedView,
};
use std::sync::Mutex;
use tracing::{error, info};

/// Per-view MetalFX state, living in the render world.
#[derive(Component)]
pub struct MetalFxDenoiserContext {
    pub(super) scaler: Mutex<ffi::Scaler>,
    /// Single-channel roughness, extracted from the packed normal/roughness guide buffer.
    pub(super) roughness: CachedTexture,
    pub(super) upscale_ratio: f32,
}

impl MetalFxDenoiserContext {
    pub fn render_resolution(&self) -> UVec2 {
        self.scaler.lock().unwrap().input_size
    }
}

pub fn prepare_metalfx(
    mut query: Query<
        (
            Entity,
            &ExtractedView,
            &MetalFxDenoiser,
            &mut Camera3d,
            &mut CameraMainTextureUsages,
            &mut TemporalJitter,
            &mut MipBias,
            Option<&MetalFxDenoiserContext>,
        ),
        (
            With<Camera3d>,
            With<TemporalJitter>,
            With<DepthPrepass>,
            With<MotionVectorPrepass>,
        ),
    >,
    render_device: Res<RenderDevice>,
    frame_count: Res<FrameCount>,
    mut commands: Commands,
) {
    for (
        entity,
        view,
        denoiser,
        mut camera_3d,
        mut main_texture_usages,
        mut temporal_jitter,
        mut mip_bias,
        context,
    ) in &mut query
    {
        // MetalFX writes the output as a render target / shader write, and reads all inputs.
        main_texture_usages.0 |= TextureUsages::STORAGE_BINDING
            | TextureUsages::TEXTURE_BINDING
            | TextureUsages::RENDER_ATTACHMENT;
        let mut depth_texture_usages = TextureUsages::from(camera_3d.depth_texture_usages);
        depth_texture_usages |= TextureUsages::TEXTURE_BINDING;
        camera_3d.depth_texture_usages = depth_texture_usages.into();

        let output_size = view.viewport.zw();
        if output_size.x == 0 || output_size.y == 0 {
            continue;
        }

        let reusable = context.filter(|c| {
            let scaler = c.scaler.lock().unwrap();
            scaler.output_size == output_size && c.upscale_ratio == denoiser.upscale_ratio
        });

        let render_resolution = match reusable {
            Some(context) => context.render_resolution(),
            None => {
                let (min_scale, max_scale) = ffi::supported_scale_range(&render_device);
                let ratio = denoiser.upscale_ratio.clamp(min_scale, max_scale);
                let input_size = (output_size.as_vec2() / ratio)
                    .ceil()
                    .as_uvec2()
                    .max(UVec2::ONE);

                let Some(scaler) = ffi::Scaler::new(&render_device, input_size, output_size) else {
                    error!("Failed to create MetalFX denoiser ({input_size} -> {output_size})");
                    continue;
                };
                info!("Created MetalFX denoiser ({input_size} -> {output_size})");

                let roughness = create_roughness_texture(&render_device, input_size);
                commands.entity(entity).insert((
                    MetalFxDenoiserContext {
                        scaler: Mutex::new(scaler),
                        roughness,
                        upscale_ratio: denoiser.upscale_ratio,
                    },
                    MainPassResolutionOverride(input_size),
                ));
                mip_bias.0 = (input_size.x as f32 / output_size.x as f32).log2() - 1.0;
                input_size
            }
        };

        let ratio = output_size.as_vec2() / render_resolution.as_vec2();
        temporal_jitter.offset = halton_jitter(frame_count.0, ratio.x * ratio.y);
    }
}

fn create_roughness_texture(render_device: &RenderDevice, size: UVec2) -> CachedTexture {
    let texture = render_device.create_texture(&TextureDescriptor {
        label: Some("metalfx_roughness"),
        size: size.to_extents(),
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::R32Float,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });
    let default_view = texture.create_view(&TextureViewDescriptor::default());
    CachedTexture {
        texture,
        default_view,
    }
}

/// Halton(2, 3) jitter in pixels, in the range [-0.5, 0.5].
///
/// The sequence length grows with the upscale area ratio, so that every output pixel
/// receives several samples.
fn halton_jitter(frame: u32, area_ratio: f32) -> Vec2 {
    let phase_count = (8.0 * area_ratio).ceil().max(8.0) as u32;
    let index = (frame % phase_count) + 1;
    Vec2::new(halton(index, 2), halton(index, 3)) - 0.5
}

fn halton(mut index: u32, base: u32) -> f32 {
    let mut f = 1.0;
    let mut result = 0.0;
    while index > 0 {
        f /= base as f32;
        result += f * (index % base) as f32;
        index /= base;
    }
    result
}
