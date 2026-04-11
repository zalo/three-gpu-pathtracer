/**
 * Extracts geometry, materials, and texture data from a three.js scene
 * and builds a BVH using tinybvh WASM for GPU upload.
 */
import { Matrix3, Matrix4, Vector3 } from 'three';

export class SceneProcessor {

	constructor() {

		this.tinybvh = null;
		this.builder = null;

	}

	async init( tinybvhModule ) {

		this.tinybvh = tinybvhModule;

	}

	/**
	 * Process a three.js scene into flat typed arrays ready for GPU upload.
	 * @param {THREE.Scene} scene
	 * @param {Function} [onProgress] - Callback: onProgress(label, fraction 0-1)
	 * @returns {Promise<Object>} Scene data for GPU buffers
	 */
	async process( scene, onProgress ) {

		const progress = onProgress || ( () => {} );

		// Collect all meshes with world transforms
		const meshes = [];
		scene.traverse( ( obj ) => {

			if ( obj.isMesh && obj.geometry && obj.material ) {

				obj.updateWorldMatrix( true, false );
				meshes.push( obj );

			}

		} );

		if ( meshes.length === 0 ) {

			throw new Error( 'No meshes found in scene' );

		}

		// Collect unique materials and assign indices
		const materialMap = new Map();
		let materialIndex = 0;

		for ( const mesh of meshes ) {

			const materials = Array.isArray( mesh.material ) ? mesh.material : [ mesh.material ];
			for ( const mat of materials ) {

				if ( ! mat ) continue;
				if ( ! materialMap.has( mat.uuid ) ) {

					materialMap.set( mat.uuid, { index: materialIndex ++, material: mat } );

				}

			}

		}

		// Count totals
		let totalVertices = 0;
		let totalTriangles = 0;

		for ( const mesh of meshes ) {

			const geo = mesh.geometry;
			totalVertices += geo.attributes.position.count;
			totalTriangles += geo.index
				? geo.index.count / 3
				: geo.attributes.position.count / 3;

		}

		// Allocate arrays
		// Vertex struct: 16 floats per vertex (64 bytes, padded for WebGPU alignment)
		const vertexData = new Float32Array( totalVertices * 16 );
		const indices = new Uint32Array( totalTriangles * 3 );
		const materialIds = new Uint32Array( totalTriangles );
		// BVH vertices: 3 bvhvec4 per triangle (3 * 4 floats = 12 floats per tri)
		const bvhVertices = new Float32Array( totalTriangles * 3 * 4 ); // world-space (for flat BVH)
		const bvhVerticesLocal = new Float32Array( totalTriangles * 3 * 4 ); // local-space (for TLAS BLASes)

		let vertexOffset = 0;
		let triOffset = 0;

		const _pos = new Vector3();
		const _norm = new Vector3();
		const _normalMatrix = new Matrix3();

		for ( const mesh of meshes ) {

			const geo = mesh.geometry;
			const posAttr = geo.attributes.position;
			const normAttr = geo.attributes.normal;
			const tangentAttr = geo.attributes.tangent;
			const uvAttr = geo.attributes.uv;
			const vertCount = posAttr.count;

			const worldMatrix = mesh.matrixWorld;
			_normalMatrix.getNormalMatrix( worldMatrix );

			// Determine material index for this mesh
			const materials = Array.isArray( mesh.material ) ? mesh.material : [ mesh.material ];
			const groups = geo.groups.length > 0 ? geo.groups : [ { start: 0, count: Infinity, materialIndex: 0 } ];

			// Write vertex data (position + normal + tangent + uv, padded)
			for ( let i = 0; i < vertCount; i ++ ) {

				// Position (world space)
				_pos.fromBufferAttribute( posAttr, i );
				_pos.applyMatrix4( worldMatrix );
				const vOff = ( vertexOffset + i ) * 16;
				vertexData[ vOff + 0 ] = _pos.x;
				vertexData[ vOff + 1 ] = _pos.y;
				vertexData[ vOff + 2 ] = _pos.z;
				vertexData[ vOff + 3 ] = 0; // pad

				// Normal (world space)
				if ( normAttr ) {

					_norm.fromBufferAttribute( normAttr, i );
					_norm.applyMatrix3( _normalMatrix ).normalize();

				} else {

					_norm.set( 0, 1, 0 );

				}

				vertexData[ vOff + 4 ] = _norm.x;
				vertexData[ vOff + 5 ] = _norm.y;
				vertexData[ vOff + 6 ] = _norm.z;
				vertexData[ vOff + 7 ] = 0; // pad

				// Tangent
				if ( tangentAttr ) {

					vertexData[ vOff + 8 ] = tangentAttr.getX( i );
					vertexData[ vOff + 9 ] = tangentAttr.getY( i );
					vertexData[ vOff + 10 ] = tangentAttr.getZ( i );
					vertexData[ vOff + 11 ] = tangentAttr.getW( i );

				} else {

					// Default tangent along X axis
					vertexData[ vOff + 8 ] = 1;
					vertexData[ vOff + 9 ] = 0;
					vertexData[ vOff + 10 ] = 0;
					vertexData[ vOff + 11 ] = 1;

				}

				// UV
				if ( uvAttr ) {

					vertexData[ vOff + 12 ] = uvAttr.getX( i );
					vertexData[ vOff + 13 ] = uvAttr.getY( i );

				} else {

					vertexData[ vOff + 12 ] = 0;
					vertexData[ vOff + 13 ] = 0;

				}

				vertexData[ vOff + 14 ] = 0; // pad
				vertexData[ vOff + 15 ] = 0; // pad

			}

			// Write index data and BVH vertices
			const indexAttr = geo.index;
			const meshTriCount = indexAttr ? indexAttr.count / 3 : vertCount / 3;

			for ( let t = 0; t < meshTriCount; t ++ ) {

				let i0, i1, i2;
				if ( indexAttr ) {

					i0 = indexAttr.getX( t * 3 + 0 );
					i1 = indexAttr.getX( t * 3 + 1 );
					i2 = indexAttr.getX( t * 3 + 2 );

				} else {

					i0 = t * 3 + 0;
					i1 = t * 3 + 1;
					i2 = t * 3 + 2;

				}

				// Global index (offset by accumulated vertex count)
				const gi0 = vertexOffset + i0;
				const gi1 = vertexOffset + i1;
				const gi2 = vertexOffset + i2;

				indices[ ( triOffset + t ) * 3 + 0 ] = gi0;
				indices[ ( triOffset + t ) * 3 + 1 ] = gi1;
				indices[ ( triOffset + t ) * 3 + 2 ] = gi2;

				// BVH vertices — world-space (for flat BVH)
				const bOff = ( triOffset + t ) * 12;
				bvhVertices[ bOff + 0 ] = vertexData[ gi0 * 16 + 0 ];
				bvhVertices[ bOff + 1 ] = vertexData[ gi0 * 16 + 1 ];
				bvhVertices[ bOff + 2 ] = vertexData[ gi0 * 16 + 2 ];
				bvhVertices[ bOff + 3 ] = 0;
				bvhVertices[ bOff + 4 ] = vertexData[ gi1 * 16 + 0 ];
				bvhVertices[ bOff + 5 ] = vertexData[ gi1 * 16 + 1 ];
				bvhVertices[ bOff + 6 ] = vertexData[ gi1 * 16 + 2 ];
				bvhVertices[ bOff + 7 ] = 0;
				bvhVertices[ bOff + 8 ] = vertexData[ gi2 * 16 + 0 ];
				bvhVertices[ bOff + 9 ] = vertexData[ gi2 * 16 + 1 ];
				bvhVertices[ bOff + 10 ] = vertexData[ gi2 * 16 + 2 ];
				bvhVertices[ bOff + 11 ] = 0;

				// BVH vertices — local-space (for TLAS BLASes)
				const lp0 = new Vector3().fromBufferAttribute( posAttr, i0 );
				const lp1 = new Vector3().fromBufferAttribute( posAttr, i1 );
				const lp2 = new Vector3().fromBufferAttribute( posAttr, i2 );
				bvhVerticesLocal[ bOff + 0 ] = lp0.x;
				bvhVerticesLocal[ bOff + 1 ] = lp0.y;
				bvhVerticesLocal[ bOff + 2 ] = lp0.z;
				bvhVerticesLocal[ bOff + 3 ] = 0;
				bvhVerticesLocal[ bOff + 4 ] = lp1.x;
				bvhVerticesLocal[ bOff + 5 ] = lp1.y;
				bvhVerticesLocal[ bOff + 6 ] = lp1.z;
				bvhVerticesLocal[ bOff + 7 ] = 0;
				bvhVerticesLocal[ bOff + 8 ] = lp2.x;
				bvhVerticesLocal[ bOff + 9 ] = lp2.y;
				bvhVerticesLocal[ bOff + 10 ] = lp2.z;
				bvhVerticesLocal[ bOff + 11 ] = 0;

				// Determine material for this triangle
				let matIdx = 0;
				const triStart = t * 3;
				for ( const group of groups ) {

					const groupEnd = group.start + ( group.count === Infinity ? meshTriCount * 3 : group.count );
					if ( triStart >= group.start && triStart < groupEnd ) {

						const mat = materials[ group.materialIndex || 0 ];
						if ( mat && materialMap.has( mat.uuid ) ) matIdx = materialMap.get( mat.uuid ).index;
						break;

					}

				}

				materialIds[ triOffset + t ] = matIdx;

			}

			vertexOffset += vertCount;
			triOffset += meshTriCount;

		}

		// Track per-mesh info for TLAS instancing
		const meshInfos = [];
		let meshOff = 0;
		for ( const mesh of meshes ) {

			const geo = mesh.geometry;
			const tc = geo.index ? geo.index.count / 3 : geo.attributes.position.count / 3;
			meshInfos.push( {
				triOffset: meshOff,
				triCount: tc,
				transform: mesh.matrixWorld.elements.slice(),
				materialIdx: 0, // filled later if needed
			} );
			meshOff += tc;

		}

		// Helper to yield to browser for UI updates
		const yieldFrame = () => new Promise( r => requestAnimationFrame( r ) );

		progress( 'Extracting geometry...', 0.3 );
		await yieldFrame();

		// Build flat BVH (always, used for primary traversal and shadow rays)
		progress( 'Allocating BVH memory...', 0.4 );
		await yieldFrame();

		const bvhResult = this._buildBVH( bvhVertices, totalTriangles, progress );

		// Also build TLAS/BLAS structure if available (for future dynamic scene support)
		if ( this.tinybvh.TinyBVHScene && meshInfos.length > 1 ) {

			try {

				const tlasResult = this._buildTLAS( bvhVertices, meshInfos, progress );
				bvhResult.useTLAS = true;
				bvhResult.tlasNodes = tlasResult.tlasNodes;
				bvhResult.tlasPrimIndices = tlasResult.tlasPrimIndices;
				bvhResult.blasNodes = tlasResult.blasNodes;
				bvhResult.blasPrimIndices = tlasResult.blasPrimIndices;
				bvhResult.blasVertices = tlasResult.blasVertices;
				bvhResult.instances = tlasResult.instances;
				bvhResult.instanceCount = tlasResult.instanceCount;

			} catch ( e ) {

				console.warn( 'TLAS build failed, using flat BVH:', e.message );

			}

		}

		// Extract textures and pack materials
		progress( 'Packing materials...', 0.9 );
		await yieldFrame();

		const { textureMap, textureList } = this._extractTextures( materialMap );
		const packedMaterials = this._packMaterials( materialMap, textureMap );

		progress( 'Done', 1.0 );

		const result = {
			bvhNodes: bvhResult.nodes,
			primIndices: bvhResult.primIndices,
			bvhVertices,
			vertexData,
			indices,
			materials: packedMaterials,
			materialIds,
			triCount: totalTriangles,
			vertexCount: totalVertices,
			materialCount: materialMap.size,
			textures: textureList,
		};

		// Include TLAS data if available
		if ( bvhResult.useTLAS ) {

			result.useTLAS = true;
			result.tlasNodes = bvhResult.tlasNodes;
			result.tlasPrimIndices = bvhResult.tlasPrimIndices;
			result.blasNodes = bvhResult.blasNodes;
			result.blasPrimIndices = bvhResult.blasPrimIndices;
			result.blasVertices = bvhResult.blasVertices;
			result.instanceData = bvhResult.instances;
			result.instanceCount = bvhResult.instanceCount;

		}

		return result;

	}

	_buildBVH( bvhVertices, triCount, progress ) {

		const tinybvh = this.tinybvh;

		// Allocate WASM heap memory and copy vertex data
		const byteSize = bvhVertices.byteLength;
		const ptr = tinybvh._malloc( byteSize );
		tinybvh.HEAPF32.set( bvhVertices, ptr / 4 );

		const builder = new tinybvh.TinyBVHBuilder();

		// Step 1: SAH BVH build (~70% of BVH time)
		progress( `Building BVH (${( triCount / 1000 ).toFixed( 0 )}K triangles)...`, 0.5 );
		builder.buildStep1_BVH( ptr, triCount );

		// Step 2: Convert to GPU layout (~30% of BVH time)
		progress( 'Converting BVH to GPU format...', 0.75 );
		builder.buildStep2_Convert();

		// Extract BVH nodes (64 bytes per node = 16 floats)
		// Re-access heap buffers in case build caused growth
		const heapBuffer = tinybvh.HEAPF32.buffer;
		const nodeCount = builder.getNodeCount();
		const nodesPtr = builder.getNodesPtr();
		const nodes = new Float32Array( heapBuffer.slice( nodesPtr, nodesPtr + nodeCount * 64 ) );

		// Extract primitive index remapping
		const idxCount = builder.getPrimIdxCount();
		const idxPtr = builder.getPrimIdxPtr();
		const primIndices = new Uint32Array( heapBuffer.slice( idxPtr, idxPtr + idxCount * 4 ) );

		const sahCost = builder.getSAHCost();
		console.log( `BVH built: ${nodeCount} nodes, ${idxCount} prim indices, SAH cost: ${sahCost.toFixed( 2 )}` );

		// Clean up
		tinybvh._free( ptr );
		builder.delete();

		return { nodes, primIndices };

	}

	_buildTLAS( bvhVerticesWorld, meshInfos, progress ) {

		const tinybvh = this.tinybvh;
		const scene = new tinybvh.TinyBVHScene();

		// Upload WORLD-space vertex data — instances use identity transforms
		// This avoids local→world transform issues; TLAS provides spatial partitioning
		const byteSize = bvhVerticesWorld.byteLength;
		const ptr = tinybvh._malloc( byteSize );
		tinybvh.HEAPF32.set( bvhVerticesWorld, ptr / 4 );

		// Build one BLAS per mesh
		progress( `Building ${meshInfos.length} BLASes...`, 0.5 );
		for ( let i = 0; i < meshInfos.length; i ++ ) {

			const info = meshInfos[ i ];
			// Pointer to this mesh's vertices within the heap
			const meshPtr = ptr + info.triOffset * 3 * 16; // 3 verts * 16 bytes per bvhvec4
			scene.addBLAS( meshPtr, info.triCount );

		}

		// Add instances with identity transforms (vertices already in world space)
		progress( 'Creating instances...', 0.65 );
		const identity = new Float32Array( [ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 ] );
		const tPtr = tinybvh._malloc( 64 );
		tinybvh.HEAPF32.set( identity, tPtr / 4 );

		for ( let i = 0; i < meshInfos.length; i ++ ) {

			scene.addInstance( i, tPtr );

		}

		tinybvh._free( tPtr );

		// Log BLAS info
		for ( let i = 0; i < meshInfos.length; i ++ ) {

			const nc = scene.getBLASNodeCount( i );
			const tc = scene.getBLASTriCount( i );
			console.log( `BLAS ${i}: ${nc} nodes, ${tc} tris, transform: [${meshInfos[ i ].transform.slice( 12, 15 ).map( v => v.toFixed( 2 ) ).join( ',' )}]` );

		}

		// Build TLAS
		progress( 'Building TLAS...', 0.75 );
		const tlasNodeCount = scene.buildTLAS();

		if ( tlasNodeCount < 0 ) {

			console.error( 'TLAS build failed' );
			scene.delete();
			tinybvh._free( ptr );
			// Fallback to single-level
			return this._buildBVH( bvhVertices, meshInfos.reduce( ( s, m ) => s + m.triCount, 0 ), progress );

		}

		progress( 'Extracting TLAS data...', 0.85 );

		const heap = tinybvh.HEAPF32.buffer;
		const heapU32 = tinybvh.HEAPU32.buffer;

		// Extract TLAS nodes
		const tlasNodesPtr = scene.getTLASNodesPtr();
		const tlasNodes = new Float32Array( heap.slice( tlasNodesPtr, tlasNodesPtr + tlasNodeCount * 64 ) );

		// Extract TLAS prim indices
		const tlasPrimCount = scene.getTLASPrimIdxCount();
		const tlasPrimPtr = scene.getTLASPrimIdxPtr();
		const tlasPrimIndices = new Uint32Array( heapU32.slice( tlasPrimPtr, tlasPrimPtr + tlasPrimCount * 4 ) );

		// Extract flattened BLAS nodes
		const flatNodesPtr = scene.getFlatNodesPtr();
		const flatNodesSize = scene.getFlatNodesSize(); // in floats
		const flatNodes = new Float32Array( heap.slice( flatNodesPtr, flatNodesPtr + flatNodesSize * 4 ) );

		// Extract flattened BLAS prim indices
		const flatPrimPtr = scene.getFlatPrimIdxPtr();
		const flatPrimSize = scene.getFlatPrimIdxSize();
		const flatPrimIndices = new Uint32Array( heapU32.slice( flatPrimPtr, flatPrimPtr + flatPrimSize * 4 ) );

		// Extract instance data
		const flatInstPtr = scene.getFlatInstancesPtr();
		const flatInstSize = scene.getFlatInstancesSize();
		const flatInstances = new Float32Array( heap.slice( flatInstPtr, flatInstPtr + flatInstSize * 4 ) );

		const instanceCount = scene.getInstanceCount();
		const blasCount = scene.getBLASCount();

		console.log( `TLAS built: ${tlasNodeCount} TLAS nodes, ${instanceCount} instances, ${blasCount} BLASes, ${flatNodesSize / 16} total BLAS nodes` );

		scene.delete();
		tinybvh._free( ptr );

		return {
			useTLAS: true,
			tlasNodes,
			tlasPrimIndices,
			blasNodes: flatNodes,
			blasPrimIndices: flatPrimIndices,
			blasVertices: bvhVerticesWorld, // world-space vertices (identity instance transforms)
			instances: flatInstances,
			instanceCount,
			blasCount,
		};

	}

	/**
	 * Extract unique textures from materials and assign indices.
	 */
	_extractTextures( materialMap ) {

		const textureMap = new Map(); // texture.uuid → index
		const textureList = []; // ordered list of three.js Texture objects
		let texIndex = 0;

		const addTexture = ( tex ) => {

			if ( ! tex || ! tex.image ) return - 1;
			if ( textureMap.has( tex.uuid ) ) return textureMap.get( tex.uuid );
			textureMap.set( tex.uuid, texIndex );
			textureList.push( tex );
			return texIndex ++;

		};

		for ( const { material } of materialMap.values() ) {

			addTexture( material.map );
			addTexture( material.metalnessMap || material.roughnessMap );
			addTexture( material.normalMap );
			addTexture( material.emissiveMap );

		}

		console.log( `Extracted ${textureList.length} unique textures` );
		return { textureMap, textureList };

	}

	/**
	 * Pack materials into flat Float32Array matching MaterialGPU struct.
	 * Layout: 5 x float4 = 20 floats = 80 bytes per material (no vec3 alignment issues).
	 *   [0-3]   base_color.rgb + metalness        (float4)
	 *   [4-7]   roughness, ior, transmission, coat (float4)
	 *   [8-11]  coat_roughness, emission.rgb       (float4)
	 *   [12-15] emission_luminance, pad, pad, pad  (float4)
	 *   [16-19] tex indices (as int4)              (int4)
	 */
	_packMaterials( materialMap, textureMap ) {

		const getTexIdx = ( tex ) => {

			if ( ! tex || ! tex.image ) return - 1;
			return textureMap.has( tex.uuid ) ? textureMap.get( tex.uuid ) : - 1;

		};

		const FLOATS_PER_MAT = 20;
		const data = new Float32Array( materialMap.size * FLOATS_PER_MAT );
		const intView = new Int32Array( data.buffer );

		for ( const { index, material } of materialMap.values() ) {

			const off = index * FLOATS_PER_MAT;
			const color = material.color || { r: 1, g: 1, b: 1 };
			const emissive = material.emissive || { r: 0, g: 0, b: 0 };
			const emissiveIntensity = material.emissiveIntensity !== undefined ? material.emissiveIntensity : 1;

			// float4: base_color.rgb + metalness
			data[ off + 0 ] = color.r;
			data[ off + 1 ] = color.g;
			data[ off + 2 ] = color.b;
			data[ off + 3 ] = material.metalness !== undefined ? material.metalness : 0;

			// float4: roughness, ior, transmission, coat_weight
			data[ off + 4 ] = material.roughness !== undefined ? material.roughness : 0.5;
			data[ off + 5 ] = material.ior !== undefined ? material.ior : 1.5;
			data[ off + 6 ] = material.transmission !== undefined ? material.transmission : 0;
			data[ off + 7 ] = material.clearcoat !== undefined ? material.clearcoat : 0;

			// float4: coat_roughness, emission_color.rgb
			data[ off + 8 ] = material.clearcoatRoughness !== undefined ? material.clearcoatRoughness : 0;
			data[ off + 9 ] = emissive.r;
			data[ off + 10 ] = emissive.g;
			data[ off + 11 ] = emissive.b;

			// float4: emission_luminance, pad, pad, pad
			data[ off + 12 ] = emissiveIntensity;
			data[ off + 13 ] = 0;
			data[ off + 14 ] = 0;
			data[ off + 15 ] = 0;

			// int4: texture indices
			const baseTexIdx = getTexIdx( material.map );
			const mrTexIdx = getTexIdx( material.metalnessMap || material.roughnessMap );
			const normTexIdx = getTexIdx( material.normalMap );
			const emTexIdx = getTexIdx( material.emissiveMap );
			intView[ off + 16 ] = baseTexIdx;
			intView[ off + 17 ] = mrTexIdx;
			intView[ off + 18 ] = normTexIdx;
			intView[ off + 19 ] = emTexIdx;

			console.log( `Material ${index} "${material.name}": color=(${color.r.toFixed( 2 )},${color.g.toFixed( 2 )},${color.b.toFixed( 2 )}) metal=${data[ off + 3 ].toFixed( 2 )} rough=${data[ off + 4 ].toFixed( 2 )} tex=[${baseTexIdx},${mrTexIdx},${normTexIdx},${emTexIdx}]` );

		}

		return data;

	}

}
