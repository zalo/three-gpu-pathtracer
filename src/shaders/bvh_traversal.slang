#ifndef BVH_TRAVERSAL_SLANG
#define BVH_TRAVERSAL_SLANG

// BVH traversal with optional TLAS/BLAS two-level instancing
// Supports both single-level (flat) and two-level (instanced) scenes

#include "common.slang"

static const uint BVH_STACK_SIZE = 32;

// ---- Instance data (4 x float4 = 64 bytes) ----
struct BVHInstance
{
    float4 inv_transform_0;  // row 0 of inverse transform (4 cols)
    float4 inv_transform_1;  // row 1
    float4 inv_transform_2;  // row 2
    float4 offsets;          // x=blasNodeOff, y=blasPrimOff, z=blasTriOff, w=blasIdx (as uint bits)
};

float3 transform_point(BVHInstance inst, float3 p)
{
    return float3(
        dot(inst.inv_transform_0.xyz, p) + inst.inv_transform_0.w,
        dot(inst.inv_transform_1.xyz, p) + inst.inv_transform_1.w,
        dot(inst.inv_transform_2.xyz, p) + inst.inv_transform_2.w
    );
}

float3 transform_direction(BVHInstance inst, float3 d)
{
    return float3(
        dot(inst.inv_transform_0.xyz, d),
        dot(inst.inv_transform_1.xyz, d),
        dot(inst.inv_transform_2.xyz, d)
    );
}

// ---- Extended hit info for instanced traversal ----
struct HitInfoEx
{
    float t;
    float u;
    float v;
    uint  prim_idx;     // local primitive index within the BLAS
    uint  instance_idx; // which instance was hit
    uint  blas_tri_offset; // offset to add for global vertex lookup
};

static const HitInfoEx NO_HIT_EX = { BVH_FAR, 0.0, 0.0, 0xFFFFFFFF, 0xFFFFFFFF, 0 };
bool hitex_is_valid(HitInfoEx hit) { return hit.prim_idx != 0xFFFFFFFF; }

// Convert HitInfoEx to HitInfo for backward compatibility
HitInfo hitex_to_hitinfo(HitInfoEx hex)
{
    HitInfo h;
    h.t = hex.t;
    h.u = hex.u;
    h.v = hex.v;
    h.prim_idx = hex.prim_idx + hex.blas_tri_offset;
    return h;
}

// ---- Single AABB-Ray intersection helper ----
void intersect_bvh_node(
    BVHNode node, float3 inv_dir, float3 neg_o_inv_d, float t_max,
    out float dist_left, out float dist_right, out uint left, out uint right)
{
    float3 t1a = mad(node.lmin, inv_dir, neg_o_inv_d);
    float3 t2a = mad(node.lmax, inv_dir, neg_o_inv_d);
    float3 t1b = mad(node.rmin, inv_dir, neg_o_inv_d);
    float3 t2b = mad(node.rmax, inv_dir, neg_o_inv_d);

    float3 minta = min(t1a, t2a);
    float3 maxta = max(t1a, t2a);
    float3 mintb = min(t1b, t2b);
    float3 maxtb = max(t1b, t2b);

    float tmina = max(max(max(minta.x, minta.y), minta.z), 0.0);
    float tminb = max(max(max(mintb.x, mintb.y), mintb.z), 0.0);
    float tmaxa = min(min(min(maxta.x, maxta.y), maxta.z), t_max);
    float tmaxb = min(min(min(maxtb.x, maxtb.y), maxtb.z), t_max);

    dist_left  = (tmina > tmaxa) ? BVH_FAR : tmina;
    dist_right = (tminb > tmaxb) ? BVH_FAR : tminb;
    left = node.left;
    right = node.right;
}

// ---- Single-level BVH traversal (flat, no instancing) ----
HitInfo traverse_bvh(
    StructuredBuffer<BVHNode> bvh_nodes,
    StructuredBuffer<uint>    prim_indices,
    StructuredBuffer<float4>  bvh_vertices,
    Ray ray)
{
    HitInfo hit = NO_HIT;
    float3 inv_dir = 1.0 / ray.dir;
    float3 neg_o_inv_d = ray.origin * -inv_dir;

    uint node_idx = 0;
    uint stack[BVH_STACK_SIZE];
    uint stack_ptr = 0;

    while (true)
    {
        BVHNode node = bvh_nodes[node_idx];

        if (node.triCount > 0)
        {
            for (uint i = 0; i < node.triCount; i++)
            {
                uint tri_idx = prim_indices[node.firstTri + i];
                float4 v0 = bvh_vertices[tri_idx * 3 + 0];
                float4 v1 = bvh_vertices[tri_idx * 3 + 1];
                float4 v2 = bvh_vertices[tri_idx * 3 + 2];

                float3 edge1 = v1.xyz - v0.xyz;
                float3 edge2 = v2.xyz - v0.xyz;
                float3 h = cross(ray.dir, edge2);
                float a = dot(edge1, h);
                if (abs(a) < 1e-7f) continue;

                float f = 1.0 / a;
                float3 s = ray.origin - v0.xyz;
                float u = f * dot(s, h);
                if (u < 0.0 || u > 1.0) continue;

                float3 q = cross(s, edge1);
                float v = f * dot(ray.dir, q);
                if (v < 0.0 || u + v > 1.0) continue;

                float t = f * dot(edge2, q);
                if (t > ray.t_min && t < hit.t)
                {
                    hit.t = t;
                    hit.u = u;
                    hit.v = v;
                    hit.prim_idx = tri_idx;
                }
            }

            if (stack_ptr == 0) break;
            node_idx = stack[--stack_ptr];
            continue;
        }

        float dist_left, dist_right;
        uint left_child, right_child;
        intersect_bvh_node(node, inv_dir, neg_o_inv_d, hit.t, dist_left, dist_right, left_child, right_child);

        if (dist_left > dist_right)
        {
            float tmp = dist_left; dist_left = dist_right; dist_right = tmp;
            uint t = left_child; left_child = right_child; right_child = t;
        }

        if (dist_left == BVH_FAR)
        {
            if (stack_ptr == 0) break;
            node_idx = stack[--stack_ptr];
        }
        else
        {
            node_idx = left_child;
            if (dist_right != BVH_FAR) stack[stack_ptr++] = right_child;
        }
    }

    return hit;
}

// ---- Two-level TLAS/BLAS traversal ----
HitInfoEx traverse_tlas(
    StructuredBuffer<BVHNode>    tlas_nodes,
    StructuredBuffer<uint>       tlas_prim_indices,
    StructuredBuffer<BVHInstance> instances,
    StructuredBuffer<BVHNode>    blas_nodes,     // all BLASes concatenated
    StructuredBuffer<uint>       blas_prim_indices,
    StructuredBuffer<float4>     blas_vertices,  // all BLAS vertices concatenated
    Ray ray)
{
    HitInfoEx best = NO_HIT_EX;
    float3 inv_dir = 1.0 / ray.dir;
    float3 neg_o_inv_d = ray.origin * -inv_dir;

    // TLAS traversal
    uint tlas_stack[BVH_STACK_SIZE];
    uint tlas_sp = 0;
    uint tnode = 0;

    while (true)
    {
        BVHNode node = tlas_nodes[tnode];

        if (node.triCount > 0)
        {
            // Leaf: iterate instances
            for (uint i = 0; i < node.triCount; i++)
            {
                uint inst_idx = tlas_prim_indices[node.firstTri + i];
                BVHInstance inst = instances[inst_idx];

                // Transform ray into BLAS local space
                float3 local_origin = transform_point(inst, ray.origin);
                float3 local_dir = transform_direction(inst, ray.dir);

                float3 local_inv_dir = 1.0 / local_dir;
                float3 local_neg_o_inv_d = local_origin * -local_inv_dir;

                uint blas_node_off = asuint(inst.offsets.x);
                uint blas_prim_off = asuint(inst.offsets.y);
                uint blas_tri_off  = asuint(inst.offsets.z);

                // BLAS traversal
                uint blas_stack[BVH_STACK_SIZE];
                uint blas_sp = 0;
                uint bnode = blas_node_off;

                while (true)
                {
                    BVHNode blas_node = blas_nodes[bnode];

                    if (blas_node.triCount > 0)
                    {
                        for (uint j = 0; j < blas_node.triCount; j++)
                        {
                            uint local_tri = blas_prim_indices[blas_prim_off + blas_node.firstTri + j];
                            uint global_tri = blas_tri_off + local_tri;
                            float4 v0 = blas_vertices[global_tri * 3 + 0];
                            float4 v1 = blas_vertices[global_tri * 3 + 1];
                            float4 v2 = blas_vertices[global_tri * 3 + 2];

                            float3 edge1 = v1.xyz - v0.xyz;
                            float3 edge2 = v2.xyz - v0.xyz;
                            float3 h = cross(local_dir, edge2);
                            float a = dot(edge1, h);
                            if (abs(a) < 1e-7f) continue;

                            float f = 1.0 / a;
                            float3 s = local_origin - v0.xyz;
                            float u = f * dot(s, h);
                            if (u < 0.0 || u > 1.0) continue;

                            float3 q = cross(s, edge1);
                            float v = f * dot(local_dir, q);
                            if (v < 0.0 || u + v > 1.0) continue;

                            float t = f * dot(edge2, q);
                            if (t > ray.t_min && t < best.t)
                            {
                                best.t = t;
                                best.u = u;
                                best.v = v;
                                best.prim_idx = local_tri;
                                best.instance_idx = inst_idx;
                                best.blas_tri_offset = blas_tri_off;
                            }
                        }

                        if (blas_sp == 0) break;
                        bnode = blas_stack[--blas_sp];
                        continue;
                    }

                    float dl, dr;
                    uint bl, br;
                    intersect_bvh_node(blas_node, local_inv_dir, local_neg_o_inv_d, best.t, dl, dr, bl, br);

                    // Offset child indices to global BLAS node space
                    bl += blas_node_off;
                    br += blas_node_off;

                    if (dl > dr) { float tmp=dl; dl=dr; dr=tmp; uint t=bl; bl=br; br=t; }
                    if (dl == BVH_FAR) { if (blas_sp==0) break; bnode=blas_stack[--blas_sp]; }
                    else { bnode=bl; if (dr!=BVH_FAR) blas_stack[blas_sp++]=br; }
                }
            }

            if (tlas_sp == 0) break;
            tnode = tlas_stack[--tlas_sp];
            continue;
        }

        float dl, dr;
        uint tl, tr;
        intersect_bvh_node(node, inv_dir, neg_o_inv_d, best.t, dl, dr, tl, tr);

        if (dl > dr) { float tmp=dl; dl=dr; dr=tmp; uint t=tl; tl=tr; tr=t; }
        if (dl == BVH_FAR) { if (tlas_sp==0) break; tnode=tlas_stack[--tlas_sp]; }
        else { tnode=tl; if (dr!=BVH_FAR) tlas_stack[tlas_sp++]=tr; }
    }

    return best;
}

// ---- Shadow ray (single-level, for backward compat) ----
bool is_occluded(
    StructuredBuffer<BVHNode> bvh_nodes,
    StructuredBuffer<uint>    prim_indices,
    StructuredBuffer<float4>  bvh_vertices,
    Ray ray)
{
    float3 inv_dir = 1.0 / ray.dir;
    float3 neg_o_inv_d = ray.origin * -inv_dir;

    uint node_idx = 0;
    uint stack[BVH_STACK_SIZE];
    uint stack_ptr = 0;

    while (true)
    {
        BVHNode node = bvh_nodes[node_idx];

        if (node.triCount > 0)
        {
            for (uint i = 0; i < node.triCount; i++)
            {
                uint tri_idx = prim_indices[node.firstTri + i];
                float4 v0 = bvh_vertices[tri_idx * 3 + 0];
                float4 v1 = bvh_vertices[tri_idx * 3 + 1];
                float4 v2 = bvh_vertices[tri_idx * 3 + 2];

                float3 edge1 = v1.xyz - v0.xyz;
                float3 edge2 = v2.xyz - v0.xyz;
                float3 h = cross(ray.dir, edge2);
                float f = 1.0 / dot(edge1, h);
                float3 s = ray.origin - v0.xyz;
                float u = f * dot(s, h);
                float3 q = cross(s, edge1);
                float v = f * dot(ray.dir, q);

                if (u < 0.0 || v < 0.0 || u + v > 1.0) continue;
                float t = f * dot(edge2, q);
                if (t > ray.t_min && t < ray.t_max) return true;
            }

            if (stack_ptr == 0) break;
            node_idx = stack[--stack_ptr];
            continue;
        }

        float dl, dr;
        uint nl, nr;
        intersect_bvh_node(node, inv_dir, neg_o_inv_d, ray.t_max, dl, dr, nl, nr);

        if (dl > dr) { float tmp=dl; dl=dr; dr=tmp; uint t=nl; nl=nr; nr=t; }
        if (dl == BVH_FAR) { if (stack_ptr==0) break; node_idx=stack[--stack_ptr]; }
        else { node_idx=nl; if (dr!=BVH_FAR) stack[stack_ptr++]=nr; }
    }

    return false;
}

#endif // BVH_TRAVERSAL_SLANG
