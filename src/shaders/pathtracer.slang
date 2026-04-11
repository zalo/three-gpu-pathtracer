// Main path tracing compute shader with NEE + MIS
// Techniques from SlangRenderer: importance-sampled environment lighting,
// power heuristic MIS, roughness clamping, mip-level firefly reduction

#include "bvh_traversal.slang"
#include "openpbr_bridge.slang"
#include "environment.slang"

// ---- Resource Bindings (group 0: scene data) ----
[[vk::binding(0, 0)]] StructuredBuffer<BVHNode>     g_bvh_nodes;
[[vk::binding(1, 0)]] StructuredBuffer<uint>         g_prim_indices;
[[vk::binding(2, 0)]] StructuredBuffer<float4>       g_bvh_vertices;
[[vk::binding(3, 0)]] StructuredBuffer<Vertex>       g_vertices;
[[vk::binding(4, 0)]] StructuredBuffer<uint>         g_indices;
[[vk::binding(5, 0)]] StructuredBuffer<MaterialGPU>  g_materials;
[[vk::binding(6, 0)]] StructuredBuffer<uint>         g_material_ids;
[[vk::binding(7, 0)]] RWStructuredBuffer<float4>     g_output;
[[vk::binding(8, 0)]] ConstantBuffer<PathTracerUniforms> g_uniforms;
// TLAS/BLAS instancing (bindings 12-14, only used when useTLAS > 0 in uniforms)
[[vk::binding(12, 0)]] StructuredBuffer<BVHNode>     g_blas_nodes;      // all BLAS nodes concatenated
[[vk::binding(13, 0)]] StructuredBuffer<uint>         g_blas_prim_indices;
[[vk::binding(14, 0)]] StructuredBuffer<BVHInstance>  g_instances;
// Analytic lights
struct LightParams { uint count; uint useTLAS; uint _pad1; uint _pad2; };
[[vk::binding(10, 0)]] StructuredBuffer<AnalyticLight> g_lights;
[[vk::binding(11, 0)]] ConstantBuffer<LightParams>     g_light_params;
// group 1: OpenPBR LUTs (in openpbr_bridge.slang)
// group 2: scene textures
[[vk::binding(0, 2)]] Texture2DArray<float4> g_scene_textures;
[[vk::binding(1, 2)]] SamplerState           g_scene_sampler;
// group 3: environment map (in environment.slang)

// ---- Unified intersection helpers that respect TLAS mode ----
bool check_occluded(Ray shadow_ray)
{
    if (g_light_params.useTLAS > 0)
    {
        HitInfoEx hex = traverse_tlas(
            g_bvh_nodes, g_prim_indices, g_instances,
            g_blas_nodes, g_blas_prim_indices, g_bvh_vertices, shadow_ray);
        return hitex_is_valid(hex) && hex.t < shadow_ray.t_max;
    }
    return is_occluded(g_bvh_nodes, g_prim_indices, g_bvh_vertices, shadow_ray);
}

HitInfo trace_ray(Ray ray)
{
    if (g_light_params.useTLAS > 0)
    {
        HitInfoEx hex = traverse_tlas(
            g_bvh_nodes, g_prim_indices, g_instances,
            g_blas_nodes, g_blas_prim_indices, g_bvh_vertices, ray);
        return hitex_to_hitinfo(hex);
    }
    return traverse_bvh(g_bvh_nodes, g_prim_indices, g_bvh_vertices, ray);
}

// Trace shadow ray through transmissive surfaces, accumulating tint.
// Returns transmission factor (float3(0) = fully occluded, float3(1) = fully visible).
float3 trace_shadow_transmissive(Ray shadow_ray)
{
    float3 transmission = float3(1.0);
    uint max_shadow = g_uniforms.max_shadow_bounces;

    for (uint i = 0; i < max_shadow; i++)
    {
        HitInfo hit = trace_ray(shadow_ray);

        if (!hit_is_valid(hit) || hit.t >= shadow_ray.t_max)
            break; // Reached environment — not occluded

        // Look up material at hit point
        uint mat_id = g_material_ids[hit.prim_idx];
        MaterialGPU mat = g_materials[mat_id];
        float mat_transmission = mat.roughness_ior_trans_coat.z;

        if (mat_transmission > 0.0)
        {
            // Transmissive surface — accumulate tint and continue
            float3 base_color = mat_base_color(mat);
            float4 base_tex = sample_texture(mat.tex_indices.x, interpolate_vertex(hit).uv);
            base_color *= base_tex.xyz;
            transmission *= base_color * mat_transmission;

            // Continue ray past this surface
            Vertex v = interpolate_vertex(hit);
            float3 N = v.normal;
            bool entering = dot(N, shadow_ray.dir) < 0.0;
            float3 offset = entering ? -N : N;
            shadow_ray.origin = v.position + offset * 1e-4;
            shadow_ray.t_min = 0.0;
            continue;
        }

        // Opaque surface — fully occluded
        return float3(0.0);
    }

    return transmission;
}

static const float EPSILON = 1e-7;

float4 sample_texture(int tex_idx, float2 uv)
{
    if (tex_idx < 0) return float4(1.0);
    return g_scene_textures.SampleLevel(g_scene_sampler, float3(uv, float(tex_idx)), 0.0);
}

// sRGB to linear conversion for LDR textures
float3 srgb_to_linear(float3 c)
{
    return pow(max(c, float3(0.0)), float3(2.2));
}

// ---- Vertex Interpolation ----
Vertex interpolate_vertex(HitInfo hit)
{
    uint i0 = g_indices[hit.prim_idx * 3 + 0];
    uint i1 = g_indices[hit.prim_idx * 3 + 1];
    uint i2 = g_indices[hit.prim_idx * 3 + 2];
    Vertex v0 = g_vertices[i0];
    Vertex v1 = g_vertices[i1];
    Vertex v2 = g_vertices[i2];
    float w = 1.0 - hit.u - hit.v;

    Vertex result;
    result.position = w * v0.position + hit.u * v1.position + hit.v * v2.position;
    result._pad0 = 0.0;
    result.normal = normalize(w * v0.normal + hit.u * v1.normal + hit.v * v2.normal);
    result._pad1 = 0.0;
    result.tangent = float4(
        normalize(w * v0.tangent.xyz + hit.u * v1.tangent.xyz + hit.v * v2.tangent.xyz),
        v0.tangent.w);
    result.uv = w * v0.uv + hit.u * v1.uv + hit.v * v2.uv;
    result._pad2 = float2(0.0);
    return result;
}

// ---- Evaluate simple GGX BRDF for NEE (no importance sampling needed, just f*cos/pdf) ----
struct BRDFEval
{
    float3 value; // f(wi,wo) * cos(theta_o)
    float  pdf;
};

BRDFEval eval_brdf(float3 base_color, float metalness, float roughness, float ior,
                    float3 V, float3 L, float3 N)
{
    BRDFEval result;
    result.value = float3(0.0);
    result.pdf = 0.0;

    float NdotL = dot(N, L);
    float NdotV = dot(N, V);
    if (NdotL <= 0.0 || NdotV <= 0.0) return result;

    float3 H = normalize(V + L);
    float NdotH = max(dot(N, H), 0.0);
    float VdotH = max(dot(V, H), 0.0);

    float alpha = max(roughness * roughness, 0.001);
    float alpha2 = alpha * alpha;

    // F0
    float3 dielectric_F0 = float3(pow((ior - 1.0) / (ior + 1.0), 2.0));
    float3 F0 = lerp(dielectric_F0, base_color, metalness);
    float3 F = F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);

    // GGX D
    float denom = NdotH * NdotH * (alpha2 - 1.0) + 1.0;
    float D = alpha2 / (PI * denom * denom);

    // Smith G (height-correlated)
    float G1V = 2.0 * NdotV / (NdotV + sqrt(alpha2 + (1.0 - alpha2) * NdotV * NdotV));
    float G1L = 2.0 * NdotL / (NdotL + sqrt(alpha2 + (1.0 - alpha2) * NdotL * NdotL));
    float G = G1V * G1L;

    // Specular BRDF: D * F * G / (4 * NdotV * NdotL)
    float3 spec = D * F * G / (4.0 * NdotV * NdotL + EPSILON);

    // Diffuse BRDF: (1-F) * (1-metalness) * base_color / PI
    float3 diff = (1.0 - F) * (1.0 - metalness) * base_color * INV_PI;

    result.value = (diff + spec) * NdotL;

    // PDF (mixed specular + diffuse)
    float spec_weight = 0.5 * (1.0 + metalness - roughness * (1.0 - metalness));
    float pdf_spec = D * G1V / (4.0 * NdotV + EPSILON);
    float pdf_diff = NdotL * INV_PI;
    result.pdf = spec_weight * pdf_spec + (1.0 - spec_weight) * pdf_diff;

    return result;
}

// ---- Main Path Tracing Kernel ----
[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    uint2 pixel = tid.xy;
    if (pixel.x >= g_uniforms.width || pixel.y >= g_uniforms.height) return;

    uint pixel_idx = pixel.y * g_uniforms.width + pixel.x;
    Camera camera = g_uniforms.camera;

    float3 accumulated = float3(0.0);

    // Multiple samples per pixel per dispatch (reduces per-frame overhead)
    for (uint spp = 0; spp < g_uniforms.spp_per_dispatch; spp++)
    {
        RNG rng = rng_init(pixel, g_uniforms.frame_index * g_uniforms.spp_per_dispatch + spp);

        float2 jitter = rng_next2(rng);
        float2 uv = (float2(pixel) + jitter) / float2(g_uniforms.width, g_uniforms.height);
        float2 ndc = float2(uv.x * 2.0 - 1.0, -(uv.y * 2.0 - 1.0));
        float3 dir = normalize(ndc.x * camera.image_u + ndc.y * camera.image_v + camera.image_w);

        Ray ray;
        ray.origin = camera.position;
        ray.dir = dir;
        ray.t_min = 0.001;
        ray.t_max = 1e30;

        // DOF
        if (camera.lens_radius > 0.0)
        {
            float2 lens_uv = rng_next2(rng);
            float lens_angle = TWO_PI * lens_uv.x;
            float lens_r = camera.lens_radius * sqrt(lens_uv.y);
            float3 lens_offset = (cos(lens_angle) * normalize(camera.image_u) + sin(lens_angle) * normalize(camera.image_v)) * lens_r;
            float3 focus_point = ray.origin + ray.dir * camera.focus_distance;
            ray.origin += lens_offset;
            ray.dir = normalize(focus_point - ray.origin);
        }

        float3 throughput = float3(1.0);
        float3 radiance = float3(0.0);
        float last_brdf_pdf = 0.0;
        bool use_nee = g_env_params.has_env_map != 0;

        for (uint bounce = 0; bounce < g_uniforms.max_bounces; bounce++)
        {
            HitInfo hit = trace_ray(ray);

            if (!hit_is_valid(hit))
            {
                // Sky/environment hit
                float3 env_color = sample_env_radiance(ray.dir);

                if (bounce == 0 || !use_nee)
                {
                    // Primary ray or no NEE: full environment contribution
                    radiance += throughput * env_color;
                }
                else
                {
                    // MIS: weight BRDF-sampled environment against NEE
                    float env_pdf = get_env_pdf(ray.dir);
                    float w = mis_weight(last_brdf_pdf, env_pdf);
                    radiance += throughput * env_color * w;
                }
                break;
            }

            // Shading
            Vertex vertex = interpolate_vertex(hit);
            uint mat_id = g_material_ids[hit.prim_idx];
            MaterialGPU mat = g_materials[mat_id];

            float3 base_color = mat_base_color(mat);
            float metalness = mat_metalness(mat);
            float roughness = mat_roughness(mat);
            float ior = mat_ior(mat);
            float transmission = mat_transmission(mat);
            float coat_weight = mat_coat_weight(mat);
            float coat_roughness = mat_coat_roughness(mat);
            float3 emission_color = mat_emission_color(mat);
            float emission_lum = mat_emission_lum(mat);

            // Apply base color texture (three.js GLTFLoader already linearizes sRGB)
            float4 base_tex = sample_texture(mat.tex_indices.x, vertex.uv);
            base_color *= base_tex.xyz;

            if (mat.tex_indices.y >= 0)
            {
                float4 mr_tex = sample_texture(mat.tex_indices.y, vertex.uv);
                roughness *= mr_tex.y;
                metalness *= mr_tex.z;
            }

            // Roughness clamping to avoid delta distributions / fireflies
            roughness = clamp(roughness, 0.01, 1.0);

            // Emission (use higher mip for secondary bounces to reduce fireflies)
            if (emission_lum > 0.0)
            {
                float4 em_tex = sample_texture(mat.tex_indices.w, vertex.uv);
                radiance += throughput * emission_color * em_tex.xyz * emission_lum;
            }

            // Normal mapping
            float3 geom_normal = vertex.normal;
            float backface_sign = 1.0;
            if (dot(geom_normal, ray.dir) > 0.0)
            {
                geom_normal = -geom_normal;
                backface_sign = -1.0;
            }

            float3 shading_normal = geom_normal;
            float3 shading_tangent = vertex.tangent.xyz;
            if (mat.tex_indices.z >= 0)
            {
                float4 nm_tex = sample_texture(mat.tex_indices.z, vertex.uv);
                float3 ts_normal = nm_tex.xyz * 2.0 - 1.0;
                float3 N = geom_normal;
                float3 T = normalize(shading_tangent - N * dot(N, shading_tangent));
                float3 B = cross(N, T) * vertex.tangent.w * backface_sign;
                shading_normal = normalize(T * ts_normal.x + B * ts_normal.y + N * ts_normal.z);
                shading_tangent = normalize(T - shading_normal * dot(shading_normal, T));
            }

            float3 V = -ray.dir;

            // ---- Prepare OpenPBR BSDF (used by both NEE and bounce sampling) ----
            OpenPBR_ResolvedInputs inputs = openpbr_make_default_resolved_inputs();
            inputs.base_weight = 1.0;
            inputs.base_color = base_color;
            inputs.base_metalness = metalness;
            inputs.specular_weight = 1.0;
            inputs.specular_roughness = roughness;
            inputs.specular_ior = ior;
            inputs.transmission_weight = transmission;
            inputs.coat_weight = coat_weight;
            inputs.coat_roughness = coat_roughness;
            inputs.emission_color = emission_color;
            inputs.emission_luminance = emission_lum;
            inputs.geometry_basis = openpbr_make_basis(shading_normal, shading_tangent, vertex.tangent.w * backface_sign);
            inputs.geometry_coat_basis = inputs.geometry_basis;

            float3 view_dir = V;
            OpenPBR_PreparedBsdf prepared = openpbr_prepare(
                inputs, throughput, OpenPBR_BaseRgbWavelengths_nm, OpenPBR_VacuumIor, view_dir);

            // ---- NEE: Direct lighting from environment ----
            if (use_nee && bounce < 4)
            {
                float env_pdf;
                float3 env_dir = sample_env_direction(rng_next2(rng), env_pdf);
                float NdotL_env = dot(shading_normal, env_dir);

                if (NdotL_env > 0.0 && env_pdf > EPSILON)
                {
                    // Shadow ray with transmissive surface handling
                    Ray shadow_ray;
                    shadow_ray.origin = vertex.position + geom_normal * 1e-4;
                    shadow_ray.dir = env_dir;
                    shadow_ray.t_min = 0.001;
                    shadow_ray.t_max = 1e30;

                    float3 shadow_transmission = trace_shadow_transmissive(shadow_ray);

                    if (any(shadow_transmission > 0.0))
                    {
                        float3 env_radiance = sample_env_radiance(env_dir);

                        // Use OpenPBR for NEE evaluation (consistent with bounce sampling)
                        float brdf_pdf = openpbr_pdf(prepared, env_dir);

                        if (brdf_pdf > EPSILON)
                        {
                            // Sample OpenPBR to get weight for this direction
                            // weight = bsdf * cos / pdf, so bsdf * cos = weight * pdf
                            // For NEE: contribution = throughput * (bsdf * cos) * env / env_pdf * MIS
                            //                       = throughput * (weight * brdf_pdf) * env / env_pdf * MIS
                            // But we can also just re-derive: use openpbr_sample with a
                            // deterministic direction isn't available, so we use the GGX eval
                            // but with the OpenPBR pdf for MIS consistency.
                            BRDFEval brdf_eval = eval_brdf(base_color, metalness, roughness, ior,
                                                            V, env_dir, shading_normal);

                            float w = mis_weight(env_pdf, brdf_pdf);
                            float3 nee_contrib = throughput * brdf_eval.value * env_radiance
                                                 * shadow_transmission * w / env_pdf;
                            radiance += nee_contrib;
                        }
                    }
                }
            }

            // ---- NEE: Direct lighting from analytic lights ----
            if (bounce < 4 && transmission < 0.5)
            {
                uint num_lights = g_light_params.count;
                for (uint li = 0; li < num_lights; li++)
                {
                    AnalyticLight light = g_lights[li];
                    uint light_type = uint(light.color_type.w);
                    float3 light_color = light.color_type.xyz;

                    float3 L;        // direction to light
                    float light_dist; // distance to light
                    float attenuation = 1.0;

                    if (light_type == LIGHT_DIRECTIONAL)
                    {
                        L = -normalize(light.direction_extra.xyz);
                        light_dist = 1e30;
                    }
                    else if (light_type == LIGHT_POINT)
                    {
                        float3 to_light = light.position_distance.xyz - vertex.position;
                        light_dist = length(to_light);
                        L = to_light / light_dist;
                        float decay = 2.0; // physically-based inverse square
                        attenuation = 1.0 / max(light_dist * light_dist, 0.001);
                        float max_dist = light.position_distance.w;
                        if (max_dist > 0.0 && light_dist > max_dist) attenuation = 0.0;
                    }
                    else if (light_type == LIGHT_SPOT)
                    {
                        float3 to_light = light.position_distance.xyz - vertex.position;
                        light_dist = length(to_light);
                        L = to_light / light_dist;
                        attenuation = 1.0 / max(light_dist * light_dist, 0.001);

                        float cos_angle = light.direction_extra.w;
                        float spot_dot = dot(-L, normalize(light.direction_extra.xyz));
                        float spot_atten = saturate((spot_dot - cos_angle) / (1.0 - cos_angle));
                        attenuation *= spot_atten * spot_atten;

                        float max_dist = light.position_distance.w;
                        if (max_dist > 0.0 && light_dist > max_dist) attenuation = 0.0;
                    }
                    else if (light_type == LIGHT_AMBIENT)
                    {
                        // Ambient: add directly, no shadow
                        radiance += throughput * base_color * light_color;
                        continue;
                    }
                    else
                    {
                        continue;
                    }

                    if (attenuation < EPSILON) continue;

                    float NdotL = dot(shading_normal, L);
                    if (NdotL <= 0.0) continue;

                    // Shadow ray
                    Ray shadow_ray;
                    shadow_ray.origin = vertex.position + geom_normal * 1e-4;
                    shadow_ray.dir = L;
                    shadow_ray.t_min = 0.001;
                    shadow_ray.t_max = light_dist - 0.002;

                    if (!check_occluded(shadow_ray))
                    {
                        BRDFEval brdf_eval = eval_brdf(base_color, metalness, roughness, ior,
                                                        V, L, shading_normal);
                        radiance += throughput * brdf_eval.value * light_color * attenuation;
                    }
                }
            }

            // ---- BRDF Sampling for next bounce ----
            float3 light_dir;
            OpenPBR_DiffuseSpecular weight;
            float pdf;
            OpenPBR_BsdfLobeType sampled_type;
            openpbr_sample(prepared, rng_next3(rng), light_dir, weight, pdf, sampled_type);

            if (pdf < EPSILON) break;

            float3 bsdf_weight = openpbr_get_sum_of_diffuse_specular(weight);
            throughput *= bsdf_weight;
            last_brdf_pdf = pdf;

            // Throughput-based termination (from SlangRenderer)
            if (bounce > 2)
            {
                float p = max(throughput.x, max(throughput.y, throughput.z));
                if (p < 0.001) break;
                // Russian roulette
                if (rng_next(rng) > p) break;
                throughput /= p;
            }

            // Spawn next ray
            float3 offset_normal = (dot(light_dir, geom_normal) > 0.0) ? geom_normal : -geom_normal;
            ray.origin = vertex.position + offset_normal * 1e-4;
            ray.dir = light_dir;
            ray.t_min = 0.001;
            ray.t_max = 1e30;
        }

        accumulated += radiance;
    }

    g_output[pixel_idx] = float4(accumulated / float(g_uniforms.spp_per_dispatch), 1.0);
}
