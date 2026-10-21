use super::{prepare::MetalFxDenoiserContext, MetalFxDenoiser};
use crate::denoiser_guides::DenoiserGuideBuffers;
use bevy_camera::{Camera, Hdr, MainPassResolutionOverride, Projection};
use bevy_ecs::{
    query::{Has, With},
    system::{Commands, Query, ResMut},
};
use bevy_render::{sync_world::RenderEntity, MainWorld};

pub fn extract_metalfx(
    mut commands: Commands,
    mut main_world: ResMut<MainWorld>,
    cleanup_query: Query<Has<MetalFxDenoiser>>,
) {
    let mut cameras_3d = main_world.query_filtered::<(
        RenderEntity,
        &Camera,
        &Projection,
        Option<&mut MetalFxDenoiser>,
    ), With<Hdr>>();

    for (entity, camera, projection, mut denoiser) in cameras_3d.iter_mut(&mut main_world) {
        let mut entity_commands = commands
            .get_entity(entity)
            .expect("Camera entity wasn't synced.");
        if let Some(denoiser) = denoiser.as_mut()
            && camera.is_active
            && projection.is_perspective()
        {
            entity_commands.insert(((**denoiser).clone(), DenoiserGuideBuffers));
            denoiser.reset = false;
        } else if cleanup_query.get(entity) == Ok(true) {
            entity_commands.remove::<(
                MetalFxDenoiser,
                MetalFxDenoiserContext,
                MainPassResolutionOverride,
                DenoiserGuideBuffers,
            )>();
        }
    }
}
