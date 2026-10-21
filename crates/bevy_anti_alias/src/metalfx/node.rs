use super::{ffi, prepare::MetalFxDenoiserContext, MetalFxDenoiser};
use crate::denoiser_guides::ViewDenoiserGuideTextures;
use bevy_asset::{load_embedded_asset, AssetServer};
use bevy_core_pipeline::prepass::ViewPrepassTextures;
use bevy_ecs::{
    resource::Resource,
    system::{Commands, Res},
};
use bevy_render::{
    camera::TemporalJitter,
    diagnostic::RecordDiagnostics,
    render_resource::{
        binding_types::{texture_2d, texture_storage_2d},
        BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
        CachedComputePipelineId, CommandEncoderDescriptor, ComputePassDescriptor,
        ComputePipelineDescriptor, PipelineCache, ShaderStages, StorageTextureAccess,
        TextureFormat, TextureSampleType,
    },
    renderer::{RenderContext, RenderDevice, ViewQuery},
    view::{ExtractedView, ViewTarget},
};
use bevy_utils::default;

#[derive(Resource)]
pub struct MetalFxPipelines {
    extract_roughness_layout: BindGroupLayoutDescriptor,
    extract_roughness_pipeline: CachedComputePipelineId,
}

pub fn init_metalfx_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    asset_server: Res<AssetServer>,
) {
    let extract_roughness_layout = BindGroupLayoutDescriptor::new(
        "metalfx_extract_roughness_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
            ),
        ),
    );
    let extract_roughness_pipeline =
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("metalfx_extract_roughness_pipeline".into()),
            layout: vec![extract_roughness_layout.clone()],
            shader: load_embedded_asset!(asset_server.as_ref(), "extract_roughness.wgsl"),
            entry_point: Some("extract_roughness".into()),
            ..default()
        });
    commands.insert_resource(MetalFxPipelines {
        extract_roughness_layout,
        extract_roughness_pipeline,
    });
}

pub fn metalfx_denoiser(
    view: ViewQuery<(
        &MetalFxDenoiser,
        &MetalFxDenoiserContext,
        &TemporalJitter,
        &ViewTarget,
        &ViewPrepassTextures,
        &ViewDenoiserGuideTextures,
        &ExtractedView,
    )>,
    pipelines: Option<Res<MetalFxPipelines>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    mut ctx: RenderContext,
) {
    let (denoiser, context, temporal_jitter, view_target, prepass_textures, guides, view) =
        view.into_inner();

    let Some(pipelines) = pipelines else {
        return;
    };
    let Some(extract_roughness_pipeline) =
        pipeline_cache.get_compute_pipeline(pipelines.extract_roughness_pipeline)
    else {
        return;
    };
    let Some(motion_vectors) = &prepass_textures.motion_vectors else {
        return;
    };

    let scaler = context.scaler.lock().unwrap();
    let render_resolution = scaler.input_size;

    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let time_span = diagnostics.time_span(ctx.command_encoder(), "metalfx_denoiser");

    // 1. Unpack roughness into its own texture.
    let bind_group = render_device.create_bind_group(
        "metalfx_extract_roughness_bind_group",
        &pipeline_cache.get_bind_group_layout(&pipelines.extract_roughness_layout),
        &BindGroupEntries::sequential((
            &guides.normal_roughness.default_view,
            &context.roughness.default_view,
        )),
    );
    {
        let mut pass = ctx
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("metalfx_extract_roughness"),
                timestamp_writes: None,
            });
        pass.set_pipeline(extract_roughness_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(
            render_resolution.x.div_ceil(8),
            render_resolution.y.div_ceil(8),
            1,
        );
    }

    // 2. Encode MetalFX into a dedicated encoder (wgpu forbids mixing raw and wgpu encoding).
    let post_process = view_target.post_process_write();
    let params = ffi::EncodeParams {
        color: post_process.source_texture,
        output: post_process.destination_texture,
        depth: &guides.depth.texture,
        motion: &motion_vectors.texture.texture,
        diffuse_albedo: &guides.diffuse_albedo.texture,
        specular_albedo: &guides.specular_albedo.texture,
        normal: &guides.normal_roughness.texture,
        roughness: &context.roughness.texture,
        jitter_offset: -temporal_jitter.offset,
        motion_vector_scale: -render_resolution.as_vec2(),
        world_to_view: view.world_from_view.to_matrix().inverse(),
        view_to_clip: view.clip_from_view,
        reset: denoiser.reset,
    };

    let mut raw_encoder =
        render_device
            .wgpu_device()
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("metalfx_denoiser"),
            });
    scaler.encode(&mut raw_encoder, &params);
    ctx.add_command_buffer(raw_encoder.finish());

    time_span.end(ctx.command_encoder());
}
