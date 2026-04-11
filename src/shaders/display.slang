// Fullscreen blit: renders a fullscreen triangle, reading from display buffer

[[vk::binding(0, 0)]] StructuredBuffer<float4> g_display_buf;

struct DisplayUniforms
{
    uint width;
    uint height;
};

[[vk::binding(1, 0)]] ConstantBuffer<DisplayUniforms> g_uniforms;

struct VSOutput
{
    float4 position : SV_Position;
    float2 uv;
};

[shader("vertex")]
VSOutput vs_main(uint vid: SV_VertexID)
{
    VSOutput output;
    output.uv = float2((vid << 1) & 2, vid & 2);
    output.position = float4(output.uv * 2.0 - 1.0, 0.0, 1.0);
    output.uv.y = 1.0 - output.uv.y;
    return output;
}

[shader("fragment")]
float4 fs_main(VSOutput input) : SV_Target
{
    uint2 pixel = uint2(input.uv * float2(g_uniforms.width, g_uniforms.height));
    pixel = clamp(pixel, uint2(0), uint2(g_uniforms.width - 1, g_uniforms.height - 1));
    uint idx = pixel.y * g_uniforms.width + pixel.x;
    return g_display_buf[idx];
}
