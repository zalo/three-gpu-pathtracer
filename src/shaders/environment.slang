#ifndef ENVIRONMENT_SLANG
#define ENVIRONMENT_SLANG

#include "common.slang"

// ---- Environment Map Bindings (group 3) ----
[[vk::binding(0, 3)]] Texture2D<float4>  g_env_map;
[[vk::binding(1, 3)]] SamplerState       g_env_sampler;
[[vk::binding(2, 3)]] Texture2D<float>   g_marginal_cdf;   // 1 x height
[[vk::binding(3, 3)]] Texture2D<float>   g_conditional_cdf; // width x height
[[vk::binding(4, 3)]] Texture2D<float>   g_luminance_pdf;   // width x height

struct EnvMapParams
{
    uint  width;
    uint  height;
    float intensity;
    uint  has_env_map; // 0 = use procedural sky, 1 = use env map
};

[[vk::binding(9, 0)]] ConstantBuffer<EnvMapParams> g_env_params;

// ---- Binary Search in CDF ----

int binary_search_cdf_row(float rand, int row, int width)
{
    int lo = 0;
    int hi = width - 1;
    while (lo < hi)
    {
        int mid = (lo + hi) / 2;
        float cdf_val = g_conditional_cdf.Load(int3(mid, row, 0)).r;
        if (cdf_val < rand)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

int binary_search_marginal_cdf(float rand, int height)
{
    int lo = 0;
    int hi = height - 1;
    while (lo < hi)
    {
        int mid = (lo + hi) / 2;
        float cdf_val = g_marginal_cdf.Load(int3(0, mid, 0)).r;
        if (cdf_val < rand)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

// ---- Equirectangular Mapping ----

float3 env_uv_to_direction(float2 uv)
{
    float theta = PI * uv.y;
    float phi = TWO_PI * (1.0 - uv.x);
    float sin_theta = sin(theta);
    return float3(sin_theta * cos(phi), cos(theta), sin_theta * sin(phi));
}

float2 env_direction_to_uv(float3 dir)
{
    float theta = acos(clamp(dir.y, -1.0, 1.0));
    float phi = atan2(dir.z, dir.x);
    float v = theta / PI;
    float u = 1.0 - (phi + PI) / TWO_PI;
    return float2(u, v);
}

// ---- Environment Sampling ----

// Sample environment map with importance sampling
// Returns sampled direction in world space and the PDF in solid angle measure
float3 sample_env_direction(float2 rand, out float pdf)
{
    int w = int(g_env_params.width);
    int h = int(g_env_params.height);

    // Binary search row using marginal CDF
    int row = binary_search_marginal_cdf(rand.y, h);
    float v = (float(row) + 0.5) / float(h);

    // Binary search column using conditional CDF for this row
    int col = binary_search_cdf_row(rand.x, row, w);
    float u = (float(col) + 0.5) / float(w);

    // Compute PDF in solid angle measure
    float lum_pdf = g_luminance_pdf.Load(int3(col, row, 0)).r;
    float theta = PI * v;
    float sin_theta = max(sin(theta), 1e-6);
    pdf = lum_pdf * float(w * h) / (TWO_PI * PI * sin_theta);

    return env_uv_to_direction(float2(u, v));
}

// Get environment PDF for a given direction (for MIS)
float get_env_pdf(float3 dir)
{
    float2 uv = env_direction_to_uv(dir);
    int w = int(g_env_params.width);
    int h = int(g_env_params.height);

    int col = clamp(int(uv.x * float(w)), 0, w - 1);
    int row = clamp(int(uv.y * float(h)), 0, h - 1);

    float lum_pdf = g_luminance_pdf.Load(int3(col, row, 0)).r;
    float theta = PI * uv.y;
    float sin_theta = max(sin(theta), 1e-6);
    return max(lum_pdf * float(w * h) / (TWO_PI * PI * sin_theta), 1e-6);
}

// Sample environment radiance for a given direction
float3 sample_env_radiance(float3 dir)
{
    return sample_env_radiance_mip(dir, 0.0);
}

float3 sample_env_radiance_mip(float3 dir, float mip)
{
    if (g_env_params.has_env_map == 0)
    {
        // Procedural sky fallback
        float t = 0.5 * (dir.y + 1.0);
        return lerp(float3(1.0), float3(0.5, 0.7, 1.0), t);
    }

    float2 uv = env_direction_to_uv(dir);
    return g_env_map.SampleLevel(g_env_sampler, uv, mip).xyz * g_env_params.intensity;
}

// ---- MIS Power Heuristic (beta=2) ----

float mis_weight(float pdf_a, float pdf_b)
{
    float a2 = pdf_a * pdf_a;
    float b2 = pdf_b * pdf_b;
    return a2 / (a2 + b2 + 1e-10);
}

#endif // ENVIRONMENT_SLANG
