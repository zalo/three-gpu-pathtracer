// Progressive accumulation and ACES tone mapping compute shader
// Uses storage buffers to avoid WebGPU storage texture read-write limitations

struct AccumulateUniforms
{
    uint sample_count; // 0 = reset
    uint width;
    uint height;
    uint _pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<float4>      g_input;       // current frame radiance
[[vk::binding(1, 0)]] RWStructuredBuffer<float4>     g_accumulator; // running sum (read+write)
[[vk::binding(2, 0)]] RWStructuredBuffer<float4>     g_display;     // tonemapped output
[[vk::binding(3, 0)]] ConstantBuffer<AccumulateUniforms> g_uniforms;

float3 aces_film(float3 x)
{
    x *= 0.6;
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

float3 linear_to_srgb(float3 linear)
{
    return pow(max(linear, float3(0.0)), float3(1.0 / 2.2));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    uint2 pixel = tid.xy;
    if (pixel.x >= g_uniforms.width || pixel.y >= g_uniforms.height) return;

    uint idx = pixel.y * g_uniforms.width + pixel.x;
    float4 current = g_input[idx];

    float4 accum;
    if (g_uniforms.sample_count == 0)
    {
        accum = float4(current.xyz, 1.0);
    }
    else
    {
        accum = g_accumulator[idx];
        accum += float4(current.xyz, 1.0);
    }

    g_accumulator[idx] = accum;

    // Normalize and tone map (matching SlangRenderer's aces_film)
    float3 color = accum.xyz / accum.w;
    color = aces_film(color);
    color = linear_to_srgb(color);

    g_display[idx] = float4(color, 1.0);
}
