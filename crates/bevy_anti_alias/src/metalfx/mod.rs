//! Apple MetalFX temporal denoised scaler.
//!
//! A realtime denoiser (and optional upscaler) for raytraced/pathtraced images, the Apple
//! equivalent of DLSS Ray Reconstruction. Intended to be used together with Bevy Solari.
//!
//! Requires macOS 26+, Apple silicon, and the Metal rendering backend.
//!
//! # Usage
//! 1. Enable Bevy's `metalfx` feature
//! 2. Check for the presence of `Option<Res<MetalFxDenoiserSupported>>` at runtime
//! 3. Add the [`MetalFxDenoiser`] component to your camera entity (alongside `SolariLighting`)

mod extract;
mod ffi;
mod node;
mod prepare;

use bevy_app::{App, Plugin};
use bevy_asset::embedded_asset;
use bevy_camera::Hdr;
use bevy_core_pipeline::{
    prepass::{DepthPrepass, MotionVectorPrepass},
    schedule::{Core3d, Core3dSystems},
};
use bevy_ecs::prelude::*;
use bevy_reflect::Reflect;
use bevy_render::{
    camera::{MipBias, TemporalJitter},
    renderer::RenderDevice,
    view::prepare_view_targets,
    ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems,
};
use tracing::info;

pub use prepare::MetalFxDenoiserContext;

/// Enables the MetalFX denoiser on supported systems.
#[derive(Default)]
pub struct MetalFxPlugin;

impl Plugin for MetalFxPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<MetalFxDenoiser>();
        embedded_asset!(app, "prepare_inputs.wgsl");
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app(RenderApp) else {
            return;
        };
        let supported = ffi::is_supported(render_app.world().resource::<RenderDevice>());
        if !supported {
            info!("MetalFX temporal denoised scaler is not supported on this system");
            return;
        }
        info!("MetalFX temporal denoised scaler is supported");
        app.insert_resource(MetalFxDenoiserSupported);

        app.sub_app_mut(RenderApp)
            .add_systems(RenderStartup, node::init_metalfx_pipelines)
            .add_systems(ExtractSchedule, extract::extract_metalfx)
            .add_systems(
                Render,
                prepare::prepare_metalfx
                    .in_set(RenderSystems::PrepareViews)
                    .before(prepare_view_targets),
            )
            .add_systems(
                Core3d,
                node::metalfx_denoiser.in_set(Core3dSystems::EarlyPostProcess),
            );
    }
}

/// Camera component to enable the MetalFX temporal denoised scaler.
///
/// The camera should render with Solari (or another renderer producing
/// [`crate::denoiser_guides::ViewDenoiserGuideTextures`]).
#[derive(Component, Reflect, Clone)]
#[reflect(Component)]
#[require(TemporalJitter, MipBias, DepthPrepass, MotionVectorPrepass, Hdr)]
pub struct MetalFxDenoiser {
    /// Ratio of output resolution to render resolution. `1.0` disables upscaling.
    ///
    /// Clamped to the range supported by the device (typically `1.0..=3.0`).
    ///
    /// Defaults to `2.0`. The denoiser's cost scales with its input resolution, so running it at
    /// native resolution on high-DPI displays can be expensive; upscaling keeps the cost of both
    /// the denoiser and the path tracer down.
    pub upscale_ratio: f32,
    /// Set to true to delete the saved temporal history (past frames).
    ///
    /// Useful for preventing ghosting when the history is no longer
    /// representative of the current frame, such as in sudden camera cuts.
    ///
    /// After setting this to true, it will automatically be toggled
    /// back to false at the end of the frame.
    pub reset: bool,
}

impl Default for MetalFxDenoiser {
    fn default() -> Self {
        Self {
            upscale_ratio: 2.0,
            reset: false,
        }
    }
}

/// When the MetalFX denoiser is supported by the current system, this resource will exist in the
/// main world. Otherwise this resource will be absent.
#[derive(Resource, Clone, Copy)]
pub struct MetalFxDenoiserSupported;
