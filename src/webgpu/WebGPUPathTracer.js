/**
 * Main WebGPU path tracer orchestrator.
 * Uses storage buffers for all intermediate data to avoid WebGPU texture limitations.
 */
import { GPUBufferManager } from './GPUBufferManager.js';
import { PipelineManager } from './PipelineManager.js';
import { createOpenPBRLUTs } from './OpenPBRLUTData.js';
import { EnvironmentMap } from './EnvironmentMap.js';

export class WebGPUPathTracer {

	constructor() {

		this.device = null;
		this.context = null;
		this.canvasFormat = null;

		this.bufferManager = null;
		this.pipelineManager = null;

		this.pathTraceBindGroup0 = null;
		this.pathTraceBindGroup1 = null;
		this.accumulateBindGroup = null;
		this.displayBindGroup = null;

		this.width = 0;
		this.height = 0;
		this.sampleCount = 0;
		this.hasScene = false;

		// Configurable rendering parameters
		this.maxBounces = 8;
		this.sppPerDispatch = 4;

		// Performance tracking
		this._perfLastTime = 0;
		this._perfFrames = 0;
		this._perfMraysPerSec = 0;

		this._uniformData = null;
		this._openpbrLUTs = null;
		this._envMap = null;
		this.pathTraceBindGroup3 = null;

	}

	async init( canvas ) {

		if ( ! navigator.gpu ) throw new Error( 'WebGPU is not supported in this browser' );

		const adapter = await navigator.gpu.requestAdapter( { powerPreference: 'high-performance' } );
		if ( ! adapter ) throw new Error( 'No WebGPU adapter found' );

		this.device = await adapter.requestDevice( {
			requiredLimits: {
				maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
				maxBufferSize: adapter.limits.maxBufferSize,
				maxStorageBuffersPerShaderStage: Math.min( 16, adapter.limits.maxStorageBuffersPerShaderStage ),
			},
		} );

		this.device.lost.then( ( info ) => console.error( 'WebGPU device lost:', info.message ) );

		this.canvasFormat = navigator.gpu.getPreferredCanvasFormat();
		this.context = canvas.getContext( 'webgpu' );
		this.context.configure( {
			device: this.device,
			format: this.canvasFormat,
			alphaMode: 'opaque',
		} );

		this.width = canvas.width;
		this.height = canvas.height;

		this.bufferManager = new GPUBufferManager( this.device );
		this.pipelineManager = new PipelineManager( this.device, this.canvasFormat );
		this.pipelineManager.createPipelines();

		// Create render buffers
		this._createRenderBuffers();

		// Create OpenPBR LUT textures
		this._openpbrLUTs = createOpenPBRLUTs( this.device );
		this._createLUTBindGroup();

		// Create display bind group
		this._createDisplayBindGroup();

	}

	_createRenderBuffers() {

		const pixelCount = this.width * this.height;
		const byteSize = pixelCount * 16; // 4 floats * 4 bytes per pixel

		// Destroy old buffers if they exist
		const existing = [ 'renderOutput', 'accumulator', 'displayBuf', 'accumUniforms', 'displayUniforms' ];
		for ( const name of existing ) {

			const buf = this.bufferManager.getBuffer( name );
			if ( buf ) buf.destroy();

		}

		// Output from path tracer (written each frame)
		this.bufferManager.buffers.renderOutput = this.device.createBuffer( {
			size: byteSize,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		} );

		// Accumulator (read + write)
		this.bufferManager.buffers.accumulator = this.device.createBuffer( {
			size: byteSize,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		} );

		// Display buffer (written by accumulate, read by display)
		this.bufferManager.buffers.displayBuf = this.device.createBuffer( {
			size: byteSize,
			usage: GPUBufferUsage.STORAGE,
		} );

		// Accumulate uniforms
		this.bufferManager.createUniformBuffer( 'accumUniforms', 16 );

		// Display uniforms (16 bytes minimum for uniform buffer alignment)
		this.bufferManager.createUniformBuffer( 'displayUniforms', 16 );
		this.device.queue.writeBuffer(
			this.bufferManager.getBuffer( 'displayUniforms' ),
			0,
			new Uint32Array( [ this.width, this.height ] )
		);

	}

	_createLUTBindGroup() {

		const luts = this._openpbrLUTs;
		this.pathTraceBindGroup1 = this.device.createBindGroup( {
			layout: this.pipelineManager.pathTracePipeline.getBindGroupLayout( 1 ),
			entries: [
				{ binding: 0, resource: { buffer: luts.lut2D } },
				{ binding: 1, resource: { buffer: luts.lut3D_0 } },
				{ binding: 2, resource: { buffer: luts.lut3D_1 } },
			],
		} );

	}

	_uploadTextures( textures ) {

		// Destroy previous texture array
		if ( this._sceneTextureArray ) {

			this._sceneTextureArray.destroy();

		}

		if ( textures.length === 0 ) {

			// Create a 1x1 dummy texture so the bind group is valid
			this._sceneTextureArray = this.device.createTexture( {
				size: [ 1, 1, 1 ],
				format: 'rgba8unorm',
				usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
			} );
			this._sceneTextureSampler = this.device.createSampler( {
				magFilter: 'linear',
				minFilter: 'linear',
				addressModeU: 'repeat',
				addressModeV: 'repeat',
			} );
			this._createTextureBindGroup();
			return;

		}

		// Find max texture dimensions (all textures resized to match for texture array)
		let maxW = 0, maxH = 0;
		for ( const tex of textures ) {

			const img = tex.image;
			maxW = Math.max( maxW, img.width || img.naturalWidth || 1 );
			maxH = Math.max( maxH, img.height || img.naturalHeight || 1 );

		}

		// Cap at 2048 to avoid excessive memory
		maxW = Math.min( maxW, 2048 );
		maxH = Math.min( maxH, 2048 );

		this._sceneTextureArray = this.device.createTexture( {
			size: [ maxW, maxH, textures.length ],
			format: 'rgba8unorm',
			usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
		} );

		// Upload each texture layer using a canvas for format conversion
		const canvas = new OffscreenCanvas( maxW, maxH );
		const ctx = canvas.getContext( '2d' );

		for ( let i = 0; i < textures.length; i ++ ) {

			const img = textures[ i ].image;
			ctx.clearRect( 0, 0, maxW, maxH );
			ctx.drawImage( img, 0, 0, maxW, maxH );
			const imageData = ctx.getImageData( 0, 0, maxW, maxH );

			this.device.queue.writeTexture(
				{ texture: this._sceneTextureArray, origin: [ 0, 0, i ] },
				imageData.data,
				{ bytesPerRow: maxW * 4, rowsPerImage: maxH },
				[ maxW, maxH, 1 ],
			);

		}

		this._sceneTextureSampler = this.device.createSampler( {
			magFilter: 'linear',
			minFilter: 'linear',
			addressModeU: 'repeat',
			addressModeV: 'repeat',
		} );

		this._createTextureBindGroup();
		console.log( `Uploaded ${textures.length} textures (${maxW}x${maxH})` );

	}

	_createTextureBindGroup() {

		this.pathTraceBindGroup2 = this.device.createBindGroup( {
			layout: this.pipelineManager.pathTracePipeline.getBindGroupLayout( 2 ),
			entries: [
				{ binding: 0, resource: this._sceneTextureArray.createView( { dimension: '2d-array' } ) },
				{ binding: 1, resource: this._sceneTextureSampler },
			],
		} );

	}

	_uploadLights( lights ) {

		const bm = this.bufferManager;

		// Pack lights: 3 x float4 (12 floats = 48 bytes) per light
		const FLOATS_PER_LIGHT = 12;
		const lightCount = lights.length;
		// Ensure at least 1 element for valid buffer (even if no lights)
		const data = new Float32Array( Math.max( 1, lightCount ) * FLOATS_PER_LIGHT );

		const typeMap = { directional: 0, point: 1, spot: 2, ambient: 3, hemisphere: 4, area: 5 };

		for ( let i = 0; i < lightCount; i ++ ) {

			const l = lights[ i ];
			const off = i * FLOATS_PER_LIGHT;
			const intensity = l.intensity;

			// color_type: rgb = color * intensity, w = type
			data[ off + 0 ] = l.color[ 0 ] * intensity;
			data[ off + 1 ] = l.color[ 1 ] * intensity;
			data[ off + 2 ] = l.color[ 2 ] * intensity;
			data[ off + 3 ] = typeMap[ l.type ] || 0;

			// position_distance: xyz = position, w = distance
			data[ off + 4 ] = l.position[ 0 ];
			data[ off + 5 ] = l.position[ 1 ];
			data[ off + 6 ] = l.position[ 2 ];
			data[ off + 7 ] = l.distance || 0;

			// direction_extra: xyz = direction, w = extra
			data[ off + 8 ] = l.direction ? l.direction[ 0 ] : 0;
			data[ off + 9 ] = l.direction ? l.direction[ 1 ] : - 1;
			data[ off + 10 ] = l.direction ? l.direction[ 2 ] : 0;
			data[ off + 11 ] = l.type === 'spot' ? Math.cos( l.angle || Math.PI / 3 ) : 0;

		}

		bm.createStorageBuffer( 'lights', data );

		// Light params uniform: count + useTLAS (16 bytes minimum for uniform alignment)
		bm.createUniformBuffer( 'lightCount', 16 );
		this.device.queue.writeBuffer(
			bm.getBuffer( 'lightCount' ), 0,
			new Uint32Array( [ lightCount, this._useTLAS ? 1 : 0, 0, 0 ] )
		);

		console.log( `Uploaded ${lightCount} analytic lights` );

	}

	async loadEnvironment( url ) {

		this._envMap = new EnvironmentMap( this.device );
		await this._envMap.load( url );

		// Create env params uniform buffer (16 bytes: width, height, intensity, has_env_map)
		const envParamsBuf = this.device.createBuffer( {
			size: 16,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		} );

		this.bufferManager.buffers.envParams = envParamsBuf;

		const envData = new ArrayBuffer( 16 );
		const envUints = new Uint32Array( envData );
		const envFloats = new Float32Array( envData );
		envUints[ 0 ] = this._envMap.width;
		envUints[ 1 ] = this._envMap.height;
		envFloats[ 2 ] = 1.0; // intensity
		envUints[ 3 ] = 1;    // has_env_map = true
		this.device.queue.writeBuffer( envParamsBuf, 0, envData );

		this._createEnvBindGroup();
		this.sampleCount = 0;

	}

	_createEnvBindGroup() {

		if ( ! this._envMap || ! this._envMap.ready ) return;

		this.pathTraceBindGroup3 = this._envMap.createBindGroup(
			this.pipelineManager.pathTracePipeline.getBindGroupLayout( 3 )
		);

	}

	_createAccumulateBindGroup() {

		const bm = this.bufferManager;
		this.accumulateBindGroup = this.device.createBindGroup( {
			layout: this.pipelineManager.accumulatePipeline.getBindGroupLayout( 0 ),
			entries: [
				{ binding: 0, resource: { buffer: bm.getBuffer( 'renderOutput' ) } },
				{ binding: 1, resource: { buffer: bm.getBuffer( 'accumulator' ) } },
				{ binding: 2, resource: { buffer: bm.getBuffer( 'displayBuf' ) } },
				{ binding: 3, resource: { buffer: bm.getBuffer( 'accumUniforms' ) } },
			],
		} );

	}

	_createDisplayBindGroup() {

		const bm = this.bufferManager;
		this.displayBindGroup = this.device.createBindGroup( {
			layout: this.pipelineManager.displayPipeline.getBindGroupLayout( 0 ),
			entries: [
				{ binding: 0, resource: { buffer: bm.getBuffer( 'displayBuf' ) } },
				{ binding: 1, resource: { buffer: bm.getBuffer( 'displayUniforms' ) } },
			],
		} );

	}

	resize( width, height ) {

		if ( width === this.width && height === this.height ) return;

		this.width = width;
		this.height = height;

		this._createRenderBuffers();
		this._createDisplayBindGroup();

		if ( this.hasScene ) {

			this._createAccumulateBindGroup();

		}

		this.sampleCount = 0;

	}

	/**
	 * Upload scene data to GPU and create bind groups.
	 */
	setScene( sceneData ) {

		// Destroy previous scene buffers
		const oldSceneBuffers = [ 'bvhNodes', 'primIndices', 'bvhVertices', 'vertices', 'indices', 'materials', 'materialIds', 'ptUniforms' ];
		for ( const name of oldSceneBuffers ) {

			const buf = this.bufferManager.getBuffer( name );
			if ( buf ) buf.destroy();

		}

		const bm = this.bufferManager;

		bm.createStorageBuffer( 'bvhNodes', sceneData.bvhNodes );
		bm.createStorageBuffer( 'primIndices', sceneData.primIndices );
		bm.createStorageBuffer( 'bvhVertices', sceneData.bvhVertices );
		bm.createStorageBuffer( 'vertices', sceneData.vertexData );
		bm.createStorageBuffer( 'indices', sceneData.indices );
		bm.createStorageBuffer( 'materials', sceneData.materials );
		bm.createStorageBuffer( 'materialIds', sceneData.materialIds );

		// PathTracerUniforms: Camera (64 bytes) + 8 uints (32 bytes) = 96 bytes
		bm.createUniformBuffer( 'ptUniforms', 96 );
		this._uniformData = new ArrayBuffer( 96 );

		// EnvMapParams uniform (16 bytes) if not already created
		if ( ! bm.getBuffer( 'envParams' ) ) {

			bm.createUniformBuffer( 'envParams', 16 );
			// Default: no env map
			this.device.queue.writeBuffer( bm.getBuffer( 'envParams' ), 0, new Uint32Array( [ 0, 0, 0, 0 ] ) );

		}

		// Upload TLAS/BLAS data if available
		this._useTLAS = this.enableTLAS && !! sceneData.useTLAS;
		if ( sceneData.useTLAS ) {

			// In TLAS mode: swap bindings 0-2 to use TLAS data
			bm.createStorageBuffer( 'tlasNodes', sceneData.tlasNodes );
			bm.createStorageBuffer( 'tlasPrimIndices', sceneData.tlasPrimIndices );
			bm.createStorageBuffer( 'blasVertices', sceneData.blasVertices );
			bm.createStorageBuffer( 'blasNodes', sceneData.blasNodes );
			bm.createStorageBuffer( 'blasPrimIndices', sceneData.blasPrimIndices );
			bm.createStorageBuffer( 'instanceData', sceneData.instanceData );

		} else {

			// Create dummy buffers so bind group is valid
			bm.createStorageBuffer( 'blasNodes', new Float32Array( 16 ) );
			bm.createStorageBuffer( 'blasPrimIndices', new Uint32Array( [ 0 ] ) );
			bm.createStorageBuffer( 'instanceData', new Float32Array( 16 ) );

		}

		// Pack and upload analytic lights
		this._uploadLights( sceneData.lights || [] );

		// Recreate render buffers (in case size changed)
		this._createRenderBuffers();

		// Create path trace bind group (group 0)
		// In TLAS mode: binding 0=TLAS nodes, binding 1=TLAS prim idx, binding 2=local vertices
		const nodesBuffer = this._useTLAS ? bm.getBuffer( 'tlasNodes' ) : bm.getBuffer( 'bvhNodes' );
		const primIdxBuffer = this._useTLAS ? bm.getBuffer( 'tlasPrimIndices' ) : bm.getBuffer( 'primIndices' );
		const verticesBuffer = this._useTLAS ? bm.getBuffer( 'blasVertices' ) : bm.getBuffer( 'bvhVertices' );

		this.pathTraceBindGroup0 = this.device.createBindGroup( {
			layout: this.pipelineManager.pathTracePipeline.getBindGroupLayout( 0 ),
			entries: [
				{ binding: 0, resource: { buffer: nodesBuffer } },
				{ binding: 1, resource: { buffer: primIdxBuffer } },
				{ binding: 2, resource: { buffer: verticesBuffer } },
				{ binding: 3, resource: { buffer: bm.getBuffer( 'vertices' ) } },
				{ binding: 4, resource: { buffer: bm.getBuffer( 'indices' ) } },
				{ binding: 5, resource: { buffer: bm.getBuffer( 'materials' ) } },
				{ binding: 6, resource: { buffer: bm.getBuffer( 'materialIds' ) } },
				{ binding: 7, resource: { buffer: bm.getBuffer( 'renderOutput' ) } },
				{ binding: 8, resource: { buffer: bm.getBuffer( 'ptUniforms' ) } },
				{ binding: 9, resource: { buffer: bm.getBuffer( 'envParams' ) } },
				{ binding: 10, resource: { buffer: bm.getBuffer( 'lights' ) } },
				{ binding: 11, resource: { buffer: bm.getBuffer( 'lightCount' ) } },
				{ binding: 12, resource: { buffer: bm.getBuffer( 'blasNodes' ) } },
				{ binding: 13, resource: { buffer: bm.getBuffer( 'blasPrimIndices' ) } },
				{ binding: 14, resource: { buffer: bm.getBuffer( 'instanceData' ) } },
			],
		} );

		// Upload textures and create bind group 2
		this._uploadTextures( sceneData.textures );

		// Create accumulate bind group
		this._createAccumulateBindGroup();

		// Create display bind group
		this._createDisplayBindGroup();

		this.hasScene = true;
		this.sampleCount = 0;

		console.log( `Scene uploaded: ${sceneData.triCount} triangles, ${sceneData.materialCount} materials, ${sceneData.textures.length} textures` );

	}

	/**
	 * Update camera uniforms.
	 */
	updateCamera( cameraData ) {

		if ( ! this._uniformData ) return;

		const floats = new Float32Array( this._uniformData );
		const uints = new Uint32Array( this._uniformData );

		// Camera (64 bytes = 16 floats)
		floats[ 0 ] = cameraData.position[ 0 ];
		floats[ 1 ] = cameraData.position[ 1 ];
		floats[ 2 ] = cameraData.position[ 2 ];
		floats[ 3 ] = cameraData.lens_radius || 0;
		floats[ 4 ] = cameraData.image_u[ 0 ];
		floats[ 5 ] = cameraData.image_u[ 1 ];
		floats[ 6 ] = cameraData.image_u[ 2 ];
		floats[ 7 ] = cameraData.focus_distance || 10;
		floats[ 8 ] = cameraData.image_v[ 0 ];
		floats[ 9 ] = cameraData.image_v[ 1 ];
		floats[ 10 ] = cameraData.image_v[ 2 ];
		floats[ 11 ] = 0;
		floats[ 12 ] = cameraData.image_w[ 0 ];
		floats[ 13 ] = cameraData.image_w[ 1 ];
		floats[ 14 ] = cameraData.image_w[ 2 ];
		floats[ 15 ] = 0;

		uints[ 16 ] = this.sampleCount;
		uints[ 17 ] = this.maxBounces;
		uints[ 18 ] = this.width;
		uints[ 19 ] = this.height;
		uints[ 20 ] = this.sppPerDispatch;

		this.device.queue.writeBuffer( this.bufferManager.getBuffer( 'ptUniforms' ), 0, this._uniformData );
		this.sampleCount = 0;

	}

	renderSample() {

		if ( ! this.hasScene ) return;

		// Update frame index and render params
		const uints = new Uint32Array( this._uniformData );
		uints[ 16 ] = this.sampleCount;
		uints[ 17 ] = this.maxBounces;
		uints[ 20 ] = this.sppPerDispatch;
		this.device.queue.writeBuffer( this.bufferManager.getBuffer( 'ptUniforms' ), 0, this._uniformData );

		// Update accumulate uniforms
		this.device.queue.writeBuffer(
			this.bufferManager.getBuffer( 'accumUniforms' ), 0,
			new Uint32Array( [ this.sampleCount, this.width, this.height, 0 ] )
		);

		const encoder = this.device.createCommandEncoder();

		// Pass 1: Path trace
		const ptPass = encoder.beginComputePass();
		ptPass.setPipeline( this.pipelineManager.pathTracePipeline );
		ptPass.setBindGroup( 0, this.pathTraceBindGroup0 );
		ptPass.setBindGroup( 1, this.pathTraceBindGroup1 );
		ptPass.setBindGroup( 2, this.pathTraceBindGroup2 );
		if ( this.pathTraceBindGroup3 ) ptPass.setBindGroup( 3, this.pathTraceBindGroup3 );
		ptPass.dispatchWorkgroups( Math.ceil( this.width / 8 ), Math.ceil( this.height / 8 ) );
		ptPass.end();

		// Pass 2: Accumulate + tone map
		const accumPass = encoder.beginComputePass();
		accumPass.setPipeline( this.pipelineManager.accumulatePipeline );
		accumPass.setBindGroup( 0, this.accumulateBindGroup );
		accumPass.dispatchWorkgroups( Math.ceil( this.width / 8 ), Math.ceil( this.height / 8 ) );
		accumPass.end();

		// Pass 3: Display blit
		const renderPass = encoder.beginRenderPass( {
			colorAttachments: [ {
				view: this.context.getCurrentTexture().createView(),
				loadOp: 'clear',
				storeOp: 'store',
				clearValue: { r: 0, g: 0, b: 0, a: 1 },
			} ],
		} );
		renderPass.setPipeline( this.pipelineManager.displayPipeline );
		renderPass.setBindGroup( 0, this.displayBindGroup );
		renderPass.draw( 3 );
		renderPass.end();

		this.device.queue.submit( [ encoder.finish() ] );
		this.sampleCount ++;

		// Measure GPU throughput every ~0.5s
		this._perfFrames ++;
		const now = performance.now();
		if ( this._perfLastTime === 0 ) this._perfLastTime = now;
		const elapsed = now - this._perfLastTime;
		if ( elapsed >= 500 ) {

			const totalRays = this._perfFrames * this.width * this.height * this.sppPerDispatch;
			this._perfMraysPerSec = totalRays / ( elapsed * 1000 ); // elapsed is ms, want Mrays
			this._perfFrames = 0;
			this._perfLastTime = now;

		}

	}

	get mraysPerSec() {

		return this._perfMraysPerSec;

	}

	resetAccumulation() {

		this.sampleCount = 0;

	}

	get samples() {

		return this.sampleCount * this.sppPerDispatch;

	}

}
