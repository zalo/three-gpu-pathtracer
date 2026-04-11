#ifndef COMMON_SLANG
#define COMMON_SLANG

// Common types and utilities shared across all path tracer shaders

static const float PI = 3.14159265358979323846;
static const float TWO_PI = 6.28318530717958647692;
static const float INV_PI = 0.31830988618379067154;
static const float BVH_FAR = 1e30f;

// ---- BVH Node (Aila & Laine 64-byte layout, matches tinybvh BVH_GPU::BVHNode) ----
struct BVHNode
{
    float3 lmin;     // left child AABB min
    uint   left;     // left child index
    float3 lmax;     // left child AABB max
    uint   right;    // right child index
    float3 rmin;     // right child AABB min
    uint   triCount; // >0 indicates leaf node
    float3 rmax;     // right child AABB max
    uint   firstTri; // first triangle index in leaf
};

// ---- Ray ----
struct Ray
{
    float3 origin;
    float  t_min;
    float3 dir;
    float  t_max;
};

// ---- Hit Information ----
struct HitInfo
{
    float t;
    float u;
    float v;
    uint  prim_idx;
};

static const HitInfo NO_HIT = { BVH_FAR, 0.0, 0.0, 0xFFFFFFFF };

bool hit_is_valid(HitInfo hit) { return hit.prim_idx != 0xFFFFFFFF; }

// ---- Vertex (64 bytes, aligned for WebGPU) ----
struct Vertex
{
    float3 position;
    float  _pad0;
    float3 normal;
    float  _pad1;
    float4 tangent;  // xyz = tangent direction, w = handedness sign
    float2 uv;
    float2 _pad2;
};

// ---- GPU Material (explicit layout, no vec3 alignment surprises) ----
struct MaterialGPU
{
    float4 base_color_metalness;      // rgb=base_color, a=metalness
    float4 roughness_ior_trans_coat;  // r=roughness, g=ior, b=transmission, a=coat_weight
    float4 coat_rough_emission;       // r=coat_roughness, g/b/a=emission_color.rgb (SHIFTED)
    float4 emission_lum_tex01;        // r=emission_luminance, gba=padding
    int4   tex_indices;               // r=base_color, g=metalness_roughness, b=normal, a=emission
};

float3 mat_base_color(MaterialGPU m)       { return m.base_color_metalness.xyz; }
float  mat_metalness(MaterialGPU m)        { return m.base_color_metalness.w; }
float  mat_roughness(MaterialGPU m)        { return m.roughness_ior_trans_coat.x; }
float  mat_ior(MaterialGPU m)              { return m.roughness_ior_trans_coat.y; }
float  mat_transmission(MaterialGPU m)     { return m.roughness_ior_trans_coat.z; }
float  mat_coat_weight(MaterialGPU m)      { return m.roughness_ior_trans_coat.w; }
float  mat_coat_roughness(MaterialGPU m)   { return m.coat_rough_emission.x; }
float3 mat_emission_color(MaterialGPU m)   { return m.coat_rough_emission.yzw; }
float  mat_emission_lum(MaterialGPU m)     { return m.emission_lum_tex01.x; }

// ---- Camera ----
struct Camera
{
    float3 position;
    float  lens_radius;
    float3 image_u;   // right * tan(fov/2) * aspect
    float  focus_distance;
    float3 image_v;   // up * tan(fov/2)
    float  _pad0;
    float3 image_w;   // forward direction (normalized)
    float  _pad1;
};

// ---- Uniforms ----
struct PathTracerUniforms
{
    Camera camera;
    uint   frame_index;
    uint   max_bounces;
    uint   width;
    uint   height;
    uint   spp_per_dispatch;
    uint   _pad_uniforms0;
    uint   _pad_uniforms1;
    uint   _pad_uniforms2;
};

// ---- Analytic Light (packed as 3 x float4 = 48 bytes) ----
// type: 0=directional, 1=point, 2=spot, 3=ambient, 4=hemisphere
struct AnalyticLight
{
    float4 color_type;       // rgb=color*intensity, w=type (as float)
    float4 position_distance; // xyz=position, w=distance (0=infinite)
    float4 direction_extra;  // xyz=direction, w=extra (spot: cos(angle), area: width)
};

static const uint LIGHT_DIRECTIONAL = 0;
static const uint LIGHT_POINT = 1;
static const uint LIGHT_SPOT = 2;
static const uint LIGHT_AMBIENT = 3;

// ---- PCG Random Number Generator ----
struct RNG
{
    uint state;
};

uint pcg_hash(uint input)
{
    uint state = input * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

RNG rng_init(uint2 pixel, uint frame)
{
    RNG rng;
    rng.state = pcg_hash(pixel.x + pixel.y * 65536u + frame * 16777216u);
    return rng;
}

float rng_next(inout RNG rng)
{
    rng.state = pcg_hash(rng.state);
    return float(rng.state) / 4294967296.0;
}

float2 rng_next2(inout RNG rng)
{
    return float2(rng_next(rng), rng_next(rng));
}

float3 rng_next3(inout RNG rng)
{
    return float3(rng_next(rng), rng_next(rng), rng_next(rng));
}

// ---- Shading Frame ----
struct Frame
{
    float3 n; // normal
    float3 t; // tangent
    float3 b; // bitangent
};

Frame frame_from_normal(float3 n)
{
    Frame f;
    f.n = n;
    // Frisvad's method for building orthonormal basis
    if (n.z < -0.99999f)
    {
        f.t = float3(0.0, -1.0, 0.0);
        f.b = float3(-1.0, 0.0, 0.0);
    }
    else
    {
        float a = 1.0 / (1.0 + n.z);
        float b_val = -n.x * n.y * a;
        f.t = float3(1.0 - n.x * n.x * a, b_val, -n.x);
        f.b = float3(b_val, 1.0 - n.y * n.y * a, -n.y);
    }
    return f;
}

Frame frame_from_tangent(float3 n, float3 t, float handedness)
{
    Frame f;
    f.n = n;
    f.t = normalize(t - n * dot(n, t)); // Gram-Schmidt orthogonalize
    f.b = cross(n, f.t) * handedness;
    return f;
}

float3 frame_to_world(Frame f, float3 local)
{
    return local.x * f.t + local.y * f.b + local.z * f.n;
}

float3 frame_to_local(Frame f, float3 world)
{
    return float3(dot(f.t, world), dot(f.b, world), dot(f.n, world));
}

#endif // COMMON_SLANG
