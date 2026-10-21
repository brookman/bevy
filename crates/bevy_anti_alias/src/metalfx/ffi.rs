//! Thin unsafe wrapper around `MTLFXTemporalDenoisedScaler`, bridging wgpu's Metal backend.

#![expect(
    unsafe_code,
    reason = "MetalFX interop requires raw Metal handles from wgpu-hal and Objective-C calls."
)]

use bevy_math::{Mat4, UVec2, Vec2};
use bevy_render::renderer::RenderDevice;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLDevice, MTLPixelFormat, MTLTexture};
use objc2_metal_fx::{
    MTLFXTemporalDenoisedScaler, MTLFXTemporalDenoisedScalerBase,
    MTLFXTemporalDenoisedScalerDescriptor,
};
use wgpu::hal::api::Metal;

unsafe extern "C" {
    fn bevy_mfx_set_world_to_view_matrix(scaler: *mut core::ffi::c_void, m: *const f32);
    fn bevy_mfx_set_view_to_clip_matrix(scaler: *mut core::ffi::c_void, m: *const f32);
}

/// Formats of the inputs the scaler is created for. Must match the textures passed to
/// [`Scaler::encode`].
pub const COLOR_FORMAT: MTLPixelFormat = MTLPixelFormat::RGBA16Float;
pub const OUTPUT_FORMAT: MTLPixelFormat = MTLPixelFormat::RGBA16Float;
pub const DEPTH_FORMAT: MTLPixelFormat = MTLPixelFormat::R32Float;
pub const MOTION_FORMAT: MTLPixelFormat = MTLPixelFormat::RG16Float;
pub const ALBEDO_FORMAT: MTLPixelFormat = MTLPixelFormat::RGBA8Unorm;
pub const NORMAL_FORMAT: MTLPixelFormat = MTLPixelFormat::RGBA16Float;
pub const ROUGHNESS_FORMAT: MTLPixelFormat = MTLPixelFormat::R32Float;

fn with_raw_device<R>(
    device: &RenderDevice,
    f: impl FnOnce(&ProtocolObject<dyn MTLDevice>) -> R,
) -> Option<R> {
    // SAFETY: the raw device is only borrowed for the duration of `f` and never destroyed.
    let hal_device = unsafe { device.wgpu_device().as_hal::<Metal>() }?;
    Some(f(hal_device.raw_device()))
}

/// Whether the current device and OS support the MetalFX temporal denoised scaler.
pub fn is_supported(device: &RenderDevice) -> bool {
    with_raw_device(device, |d| unsafe {
        MTLFXTemporalDenoisedScalerDescriptor::supportsDevice(d)
    })
    .unwrap_or(false)
}

/// Supported (min, max) ratios of output to input resolution.
pub fn supported_scale_range(device: &RenderDevice) -> (f32, f32) {
    with_raw_device(device, |d| unsafe {
        (
            MTLFXTemporalDenoisedScalerDescriptor::supportedInputContentMinScaleForDevice(d),
            MTLFXTemporalDenoisedScalerDescriptor::supportedInputContentMaxScaleForDevice(d),
        )
    })
    .unwrap_or((1.0, 1.0))
}

/// Owned MetalFX denoiser instance.
pub struct Scaler {
    raw: Retained<ProtocolObject<dyn MTLFXTemporalDenoisedScaler>>,
    pub input_size: UVec2,
    pub output_size: UVec2,
}

// SAFETY: Metal objects are reference counted with atomic refcounts and MetalFX scalers can be
// used from any thread, as long as encoding is externally synchronized (we wrap it in a Mutex).
unsafe impl Send for Scaler {}
unsafe impl Sync for Scaler {}

/// Textures and parameters for a single denoising pass.
pub struct EncodeParams<'a> {
    pub color: &'a wgpu::Texture,
    pub depth: &'a wgpu::Texture,
    pub motion: &'a wgpu::Texture,
    pub diffuse_albedo: &'a wgpu::Texture,
    pub specular_albedo: &'a wgpu::Texture,
    pub normal: &'a wgpu::Texture,
    pub roughness: &'a wgpu::Texture,
    pub output: &'a wgpu::Texture,
    /// Jitter offset in pixels.
    pub jitter_offset: Vec2,
    /// Scale converting the motion vector texture values to pixels.
    pub motion_vector_scale: Vec2,
    pub world_to_view: Mat4,
    pub view_to_clip: Mat4,
    pub reset: bool,
}

impl Scaler {
    pub fn new(device: &RenderDevice, input_size: UVec2, output_size: UVec2) -> Option<Self> {
        with_raw_device(device, |d| unsafe {
            let desc = MTLFXTemporalDenoisedScalerDescriptor::new();
            desc.setColorTextureFormat(COLOR_FORMAT);
            desc.setOutputTextureFormat(OUTPUT_FORMAT);
            desc.setDepthTextureFormat(DEPTH_FORMAT);
            desc.setMotionTextureFormat(MOTION_FORMAT);
            desc.setDiffuseAlbedoTextureFormat(ALBEDO_FORMAT);
            desc.setSpecularAlbedoTextureFormat(ALBEDO_FORMAT);
            desc.setNormalTextureFormat(NORMAL_FORMAT);
            desc.setRoughnessTextureFormat(ROUGHNESS_FORMAT);
            desc.setInputWidth(input_size.x as usize);
            desc.setInputHeight(input_size.y as usize);
            desc.setOutputWidth(output_size.x as usize);
            desc.setOutputHeight(output_size.y as usize);
            desc.setAutoExposureEnabled(false);
            desc.newTemporalDenoisedScalerWithDevice(d)
        })
        .flatten()
        .map(|raw| Self {
            raw,
            input_size,
            output_size,
        })
    }

    /// Encodes the denoiser into `encoder`.
    ///
    /// `encoder` must be a fresh encoder that is not used with the regular wgpu API,
    /// since wgpu forbids mixing raw and wgpu encoding on the same encoder.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, p: &EncodeParams) {
        // SAFETY: hal handles are only borrowed during this call; all textures are kept alive
        // by the caller until the command buffer finishes executing (Bevy keeps view textures
        // alive across frames, and Metal retains resources referenced by command buffers).
        unsafe {
            let tex = |t: &wgpu::Texture| t.as_hal::<Metal>().expect("non-Metal texture");
            let color = tex(p.color);
            let depth = tex(p.depth);
            let motion = tex(p.motion);
            let diffuse = tex(p.diffuse_albedo);
            let specular = tex(p.specular_albedo);
            let normal = tex(p.normal);
            let roughness = tex(p.roughness);
            let output = tex(p.output);
            let raw_tex = |t: &wgpu::hal::metal::Texture| -> *const ProtocolObject<dyn MTLTexture> {
                t.raw_handle()
            };

            let s = &*self.raw;
            s.setColorTexture(Some(&*raw_tex(&color)));
            s.setDepthTexture(Some(&*raw_tex(&depth)));
            s.setMotionTexture(Some(&*raw_tex(&motion)));
            s.setDiffuseAlbedoTexture(Some(&*raw_tex(&diffuse)));
            s.setSpecularAlbedoTexture(Some(&*raw_tex(&specular)));
            s.setNormalTexture(Some(&*raw_tex(&normal)));
            s.setRoughnessTexture(Some(&*raw_tex(&roughness)));
            s.setOutputTexture(Some(&*raw_tex(&output)));

            s.setJitterOffsetX(p.jitter_offset.x);
            s.setJitterOffsetY(p.jitter_offset.y);
            s.setMotionVectorScaleX(p.motion_vector_scale.x);
            s.setMotionVectorScaleY(p.motion_vector_scale.y);
            s.setDepthReversed(true);
            s.setShouldResetHistory(p.reset);

            let ptr = Retained::as_ptr(&self.raw) as *mut core::ffi::c_void;
            bevy_mfx_set_world_to_view_matrix(ptr, p.world_to_view.to_cols_array().as_ptr());
            bevy_mfx_set_view_to_clip_matrix(ptr, p.view_to_clip.to_cols_array().as_ptr());

            encoder.as_hal_mut::<Metal, _, _>(|hal_encoder| {
                let command_buffer = hal_encoder
                    .and_then(|e| e.raw_command_buffer())
                    .expect("MetalFX: no raw Metal command buffer available");
                s.encodeToCommandBuffer(command_buffer);
            });
        }
    }
}
