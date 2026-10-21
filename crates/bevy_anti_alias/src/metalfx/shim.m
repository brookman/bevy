// Small Objective-C shim for MetalFX setters that take `simd_float4x4` by value.
// These cannot be called reliably via `objc2::msg_send!` because `simd_float4x4`
// is passed as a homogeneous vector aggregate on arm64.
#import <MetalFX/MetalFX.h>
#include <simd/simd.h>

static simd_float4x4 bevy_mfx_load(const float *m) {
    simd_float4x4 r;
    for (int c = 0; c < 4; c++) {
        r.columns[c] = simd_make_float4(m[c * 4 + 0], m[c * 4 + 1], m[c * 4 + 2], m[c * 4 + 3]);
    }
    return r;
}

// `m` is a column-major 4x4 matrix (16 floats).
void bevy_mfx_set_world_to_view_matrix(void *scaler, const float *m) {
    [(__bridge id<MTLFXTemporalDenoisedScaler>)scaler setWorldToViewMatrix:bevy_mfx_load(m)];
}

// `m` is a column-major 4x4 matrix (16 floats).
void bevy_mfx_set_view_to_clip_matrix(void *scaler, const float *m) {
    [(__bridge id<MTLFXTemporalDenoisedScaler>)scaler setViewToClipMatrix:bevy_mfx_load(m)];
}
