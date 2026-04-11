// Emscripten: use aligned_alloc (define __linux__ for tinybvh's allocator path)
#ifdef __EMSCRIPTEN__
#ifndef __linux__
#define __linux__
#endif
#endif
#define TINYBVH_NO_SIMD
#define NO_THREADED_BUILDS
#define TINYBVH_IMPLEMENTATION
#include "tiny_bvh.h"
#include <emscripten/bind.h>
#include <vector>
#include <memory>

using namespace emscripten;
using namespace tinybvh;

// ---- Single-level builder (backward compatible) ----
class TinyBVHBuilder {
public:
	BVH_GPU bvhGpu;

	int buildStep1_BVH( uintptr_t vertexPtr, uint32_t triCount )
	{
		bvhvec4* verts = reinterpret_cast<bvhvec4*>( vertexPtr );
		if ( !verts || triCount == 0 ) return -1;
		bvhGpu.bvh.Build( verts, triCount );
		return (int)bvhGpu.bvh.usedNodes;
	}

	int buildStep2_Convert()
	{
		if ( bvhGpu.bvh.usedNodes == 0 ) return -1;
		bvhGpu.ConvertFrom( bvhGpu.bvh );
		return (int)bvhGpu.bvh.usedNodes;
	}

	int build( uintptr_t vertexPtr, uint32_t triCount )
	{
		int r = buildStep1_BVH( vertexPtr, triCount );
		if ( r < 0 ) return r;
		return buildStep2_Convert();
	}

	uintptr_t getNodesPtr() { return reinterpret_cast<uintptr_t>( bvhGpu.bvhNode ); }
	uint32_t getNodeCount() { return bvhGpu.bvh.usedNodes; }
	uintptr_t getPrimIdxPtr() { return reinterpret_cast<uintptr_t>( bvhGpu.bvh.primIdx ); }
	uint32_t getPrimIdxCount() { return bvhGpu.bvh.idxCount; }
	uint32_t getTriCount() { return bvhGpu.bvh.triCount; }
	float getSAHCost() { return bvhGpu.SAHCost(); }
};

// ---- Two-level scene builder (TLAS/BLAS instancing) ----
class TinyBVHScene {
public:
	// Per-BLAS data (use unique_ptr to avoid dangling pointers when vector reallocates)
	std::vector<std::unique_ptr<BVH_GPU>> blases;
	std::vector<BVHBase*> blasPtrs; // for TLAS build

	// Instance data
	std::vector<BLASInstance> instances;

	// TLAS
	BVH_GPU tlas;

	// Flattened output for GPU upload
	std::vector<float> flatNodes;      // all BLAS nodes concatenated (as floats)
	std::vector<uint32_t> flatPrimIdx; // all BLAS prim indices concatenated
	std::vector<float> flatInstances;  // per-instance: invTransform(12) + blasNodeOff(1) + blasPrimOff(1) + blasTriOff(1) + pad(1) = 16 floats

	uint32_t blasNodeOffsets_acc = 0;
	uint32_t blasPrimOffsets_acc = 0;
	uint32_t blasTriOffsets_acc = 0;

	// Add a BLAS from triangle vertices. Returns BLAS index.
	int addBLAS( uintptr_t vertexPtr, uint32_t triCount )
	{
		bvhvec4* verts = reinterpret_cast<bvhvec4*>( vertexPtr );
		if ( !verts || triCount == 0 ) return -1;

		int idx = (int)blases.size();
		blases.push_back( std::make_unique<BVH_GPU>() );
		BVH_GPU& blas = *blases.back();
		blas.Build( verts, triCount );

		return idx;
	}

	// Add an instance referencing a BLAS with a 4x4 transform (column-major, 16 floats)
	void addInstance( int blasIdx, uintptr_t transformPtr )
	{
		float* m = reinterpret_cast<float*>( transformPtr );
		BLASInstance inst( (uint32_t)blasIdx );

		// Copy 4x4 transform from three.js matrixWorld.elements (column-major)
		// bvhmat4 is row-major flat: cell[row*4+col]
		for ( int col = 0; col < 4; col++ )
			for ( int row = 0; row < 4; row++ )
				inst.transform[row * 4 + col] = m[col * 4 + row];

		inst.InvertTransform();
		inst.Update( &blases[blasIdx]->bvh );

		instances.push_back( inst );
	}

	// Build TLAS over all instances
	int buildTLAS()
	{
		if ( instances.empty() ) return -1;

		// Collect BLAS pointers
		blasPtrs.clear();
		for ( auto& b : blases ) blasPtrs.push_back( &b->bvh );

		tlas.Build( instances.data(), (uint32_t)instances.size(),
		            blasPtrs.data(), (uint32_t)blasPtrs.size() );

		// Flatten for GPU upload
		_flatten();

		return (int)tlas.bvh.usedNodes;
	}

	// ---- TLAS accessors ----
	uintptr_t getTLASNodesPtr() { return reinterpret_cast<uintptr_t>( tlas.bvhNode ); }
	uint32_t getTLASNodeCount() { return tlas.bvh.usedNodes; }
	uintptr_t getTLASPrimIdxPtr() { return reinterpret_cast<uintptr_t>( tlas.bvh.primIdx ); }
	uint32_t getTLASPrimIdxCount() { return tlas.bvh.idxCount; }
	uint32_t getInstanceCount() { return (uint32_t)instances.size(); }
	uint32_t getBLASCount() { return (uint32_t)blases.size(); }

	// ---- Flattened data accessors ----
	uintptr_t getFlatNodesPtr() { return reinterpret_cast<uintptr_t>( flatNodes.data() ); }
	uint32_t getFlatNodesSize() { return (uint32_t)flatNodes.size(); } // in floats
	uintptr_t getFlatPrimIdxPtr() { return reinterpret_cast<uintptr_t>( flatPrimIdx.data() ); }
	uint32_t getFlatPrimIdxSize() { return (uint32_t)flatPrimIdx.size(); } // in uint32s
	uintptr_t getFlatInstancesPtr() { return reinterpret_cast<uintptr_t>( flatInstances.data() ); }
	uint32_t getFlatInstancesSize() { return (uint32_t)flatInstances.size(); } // in floats

	// Per-BLAS accessors
	uint32_t getBLASNodeCount( int idx ) { return blases[idx]->bvh.usedNodes; }
	uint32_t getBLASTriCount( int idx ) { return blases[idx]->bvh.triCount; }

private:
	void _flatten()
	{
		flatNodes.clear();
		flatPrimIdx.clear();
		flatInstances.clear();

		// Concatenate all BLAS nodes and prim indices
		std::vector<uint32_t> blasNodeOffsets( blases.size() );
		std::vector<uint32_t> blasPrimOffsets( blases.size() );
		std::vector<uint32_t> blasTriOffsets( blases.size() );

		uint32_t nodeOff = 0, primOff = 0, triOff = 0;
		for ( size_t i = 0; i < blases.size(); i++ )
		{
			BVH_GPU& b = *blases[i];
			blasNodeOffsets[i] = nodeOff;
			blasPrimOffsets[i] = primOff;
			blasTriOffsets[i] = triOff;

			// Copy BLAS nodes as raw floats (16 floats per 64-byte node)
			uint32_t nc = b.bvh.usedNodes;
			const float* nodeData = reinterpret_cast<const float*>( b.bvhNode );
			flatNodes.insert( flatNodes.end(), nodeData, nodeData + nc * 16 );

			// Copy BLAS prim indices
			uint32_t ic = b.bvh.idxCount;
			flatPrimIdx.insert( flatPrimIdx.end(), b.bvh.primIdx, b.bvh.primIdx + ic );

			nodeOff += nc;
			primOff += ic;
			triOff += b.bvh.triCount;
		}

		// Build per-instance data: 4 x float4 = 16 floats per instance
		// [0-11]: inverse transform 4x3 (row-major, 3 rows of 4 columns)
		// [12]: blas node offset (as float, cast to uint in shader)
		// [13]: blas prim idx offset
		// [14]: blas tri offset (for vertex lookup)
		// [15]: blas index (for material offset lookup)
		for ( size_t i = 0; i < instances.size(); i++ )
		{
			const BLASInstance& inst = instances[i];
			uint32_t bi = inst.blasIdx;

			// Inverse transform as 4x3 row-major (12 floats)
			// bvhmat4 is flat: cell[row*4+col]
			for ( int row = 0; row < 3; row++ )
				for ( int col = 0; col < 4; col++ )
					flatInstances.push_back( inst.invTransform[row * 4 + col] );

			// Offsets (stored as float, reinterpreted as uint in shader)
			float nodeOffF, primOffF, triOffF, blasIdxF;
			uint32_t nodeOffU = blasNodeOffsets[bi];
			uint32_t primOffU = blasPrimOffsets[bi];
			uint32_t triOffU = blasTriOffsets[bi];
			memcpy( &nodeOffF, &nodeOffU, 4 );
			memcpy( &primOffF, &primOffU, 4 );
			memcpy( &triOffF, &triOffU, 4 );
			memcpy( &blasIdxF, &bi, 4 );

			flatInstances.push_back( nodeOffF );
			flatInstances.push_back( primOffF );
			flatInstances.push_back( triOffF );
			flatInstances.push_back( blasIdxF );
		}
	}
};

EMSCRIPTEN_BINDINGS( tinybvh )
{
	// Single-level builder (backward compatible)
	class_<TinyBVHBuilder>( "TinyBVHBuilder" )
		.constructor()
		.function( "build", &TinyBVHBuilder::build )
		.function( "buildStep1_BVH", &TinyBVHBuilder::buildStep1_BVH )
		.function( "buildStep2_Convert", &TinyBVHBuilder::buildStep2_Convert )
		.function( "getNodesPtr", &TinyBVHBuilder::getNodesPtr )
		.function( "getNodeCount", &TinyBVHBuilder::getNodeCount )
		.function( "getPrimIdxPtr", &TinyBVHBuilder::getPrimIdxPtr )
		.function( "getPrimIdxCount", &TinyBVHBuilder::getPrimIdxCount )
		.function( "getTriCount", &TinyBVHBuilder::getTriCount )
		.function( "getSAHCost", &TinyBVHBuilder::getSAHCost );

	// Two-level scene builder (TLAS/BLAS instancing)
	class_<TinyBVHScene>( "TinyBVHScene" )
		.constructor()
		.function( "addBLAS", &TinyBVHScene::addBLAS )
		.function( "addInstance", &TinyBVHScene::addInstance )
		.function( "buildTLAS", &TinyBVHScene::buildTLAS )
		.function( "getTLASNodesPtr", &TinyBVHScene::getTLASNodesPtr )
		.function( "getTLASNodeCount", &TinyBVHScene::getTLASNodeCount )
		.function( "getTLASPrimIdxPtr", &TinyBVHScene::getTLASPrimIdxPtr )
		.function( "getTLASPrimIdxCount", &TinyBVHScene::getTLASPrimIdxCount )
		.function( "getInstanceCount", &TinyBVHScene::getInstanceCount )
		.function( "getBLASCount", &TinyBVHScene::getBLASCount )
		.function( "getFlatNodesPtr", &TinyBVHScene::getFlatNodesPtr )
		.function( "getFlatNodesSize", &TinyBVHScene::getFlatNodesSize )
		.function( "getFlatPrimIdxPtr", &TinyBVHScene::getFlatPrimIdxPtr )
		.function( "getFlatPrimIdxSize", &TinyBVHScene::getFlatPrimIdxSize )
		.function( "getFlatInstancesPtr", &TinyBVHScene::getFlatInstancesPtr )
		.function( "getFlatInstancesSize", &TinyBVHScene::getFlatInstancesSize )
		.function( "getBLASNodeCount", &TinyBVHScene::getBLASNodeCount )
		.function( "getBLASTriCount", &TinyBVHScene::getBLASTriCount );
}
