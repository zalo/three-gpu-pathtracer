#ifndef OPENPBR_BRIDGE_SLANG
#define OPENPBR_BRIDGE_SLANG

// OpenPBR BSDF integration bridge with texture LUT mode
//
// Uses OPENPBR_USE_TEXTURE_LUTS = 1 to avoid large constant arrays
// that don't compile in Slang. LUT data is provided via GPU textures.

#include "common.slang"

// ---- OpenPBR LUT Texture Bindings ----
// These must be declared before the sampling macros below.
// 2D LUTs packed into a Texture2DArray (6 layers, 32x32 each)
// 3D LUTs as separate Texture3D (32x32x32 each)
//
// Binding group 1: OpenPBR LUT data stored in storage buffers
// (avoids rgba32float unfilterable texture issues on WebGPU)
// 2D LUTs: 6 tables x 32x32 = 6144 float4 entries
// 3D LUTs: 2 tables x 32x32x32 = 65536 float4 entries each
[[vk::binding(0, 1)]] StructuredBuffer<float4> g_openpbr_lut_2d;  // 6*32*32 = 6144 entries
[[vk::binding(1, 1)]] StructuredBuffer<float4> g_openpbr_lut_3d_0; // 32^3 = 32768 entries
[[vk::binding(2, 1)]] StructuredBuffer<float4> g_openpbr_lut_3d_1; // 32^3 = 32768 entries

static const int LUT_SIZE = 32;

// Map 2D LUT IDs to buffer offsets:
//   LutId 1 → layer 0, LutId 2 → layer 1, LutId 4 → layer 2,
//   LutId 5 → layer 3, LutId 6 → layer 4, LutId 7 → layer 5
float4 _openpbr_sample_2d_impl(int lut_id, float2 uv)
{
    int layer;
    if      (lut_id == 1) layer = 0;
    else if (lut_id == 2) layer = 1;
    else if (lut_id == 4) layer = 2;
    else if (lut_id == 5) layer = 3;
    else if (lut_id == 6) layer = 4;
    else                  layer = 5;

    // Bilinear interpolation
    float2 coord = uv * float(LUT_SIZE) - 0.5;
    int2 c0 = clamp(int2(floor(coord)), int2(0), int2(LUT_SIZE - 1));
    int2 c1 = clamp(c0 + int2(1), int2(0), int2(LUT_SIZE - 1));
    float2 f = coord - floor(coord);

    int base = layer * LUT_SIZE * LUT_SIZE;
    float4 v00 = g_openpbr_lut_2d[base + c0.y * LUT_SIZE + c0.x];
    float4 v10 = g_openpbr_lut_2d[base + c0.y * LUT_SIZE + c1.x];
    float4 v01 = g_openpbr_lut_2d[base + c1.y * LUT_SIZE + c0.x];
    float4 v11 = g_openpbr_lut_2d[base + c1.y * LUT_SIZE + c1.x];

    return lerp(lerp(v00, v10, f.x), lerp(v01, v11, f.x), f.y);
}

float4 _openpbr_fetch_3d(int lut_id, int idx)
{
    if (lut_id == 0) return g_openpbr_lut_3d_0[idx];
    else             return g_openpbr_lut_3d_1[idx];
}

float4 _openpbr_sample_3d_impl(int lut_id, float3 uvw)
{
    // Trilinear interpolation
    float3 coord = uvw * float(LUT_SIZE) - 0.5;
    int3 c0 = clamp(int3(floor(coord)), int3(0), int3(LUT_SIZE - 1));
    int3 c1 = clamp(c0 + int3(1), int3(0), int3(LUT_SIZE - 1));
    float3 f = coord - floor(coord);

    float4 v000 = _openpbr_fetch_3d(lut_id, c0.z * LUT_SIZE * LUT_SIZE + c0.y * LUT_SIZE + c0.x);
    float4 v100 = _openpbr_fetch_3d(lut_id, c0.z * LUT_SIZE * LUT_SIZE + c0.y * LUT_SIZE + c1.x);
    float4 v010 = _openpbr_fetch_3d(lut_id, c0.z * LUT_SIZE * LUT_SIZE + c1.y * LUT_SIZE + c0.x);
    float4 v110 = _openpbr_fetch_3d(lut_id, c0.z * LUT_SIZE * LUT_SIZE + c1.y * LUT_SIZE + c1.x);
    float4 v001 = _openpbr_fetch_3d(lut_id, c1.z * LUT_SIZE * LUT_SIZE + c0.y * LUT_SIZE + c0.x);
    float4 v101 = _openpbr_fetch_3d(lut_id, c1.z * LUT_SIZE * LUT_SIZE + c0.y * LUT_SIZE + c1.x);
    float4 v011 = _openpbr_fetch_3d(lut_id, c1.z * LUT_SIZE * LUT_SIZE + c1.y * LUT_SIZE + c0.x);
    float4 v111 = _openpbr_fetch_3d(lut_id, c1.z * LUT_SIZE * LUT_SIZE + c1.y * LUT_SIZE + c1.x);

    float4 r00 = lerp(v000, v100, f.x);
    float4 r10 = lerp(v010, v110, f.x);
    float4 r01 = lerp(v001, v101, f.x);
    float4 r11 = lerp(v011, v111, f.x);

    return lerp(lerp(r00, r10, f.y), lerp(r01, r11, f.y), f.z);
}

// ---- Configure OpenPBR for Slang with texture LUTs ----
#define OPENPBR_LANGUAGE_TARGET_SLANG 1
#define OPENPBR_USE_TEXTURE_LUTS 1
#define OPENPBR_SAMPLE_2D_TEXTURE(lut_id, uv) _openpbr_sample_2d_impl(lut_id, uv)
#define OPENPBR_SAMPLE_3D_TEXTURE(lut_id, uvw) _openpbr_sample_3d_impl(lut_id, uvw)

#include "../../third_party/openpbr-bsdf/openpbr.h"

#endif // OPENPBR_BRIDGE_SLANG
