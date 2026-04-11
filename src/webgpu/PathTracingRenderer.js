/**
 * Drop-in replacement for three.js WebGPURenderer / WebGLRenderer.
 *
 * Usage:
 *   const renderer = new PathTracingRenderer({ canvas, antialias: true });
 *   renderer.setSize(window.innerWidth, window.innerHeight);
 *   renderer.setPixelRatio(window.devicePixelRatio);
 *   renderer.toneMapping = ACESFilmicToneMapping;
 *   renderer.toneMappingExposure = 0.5;
 *
 *   function animate() {
 *       renderer.render(scene, camera);
 *       requestAnimationFrame(animate);
 *   }
 *
 * The renderer automatically:
 * - Detects scene changes and rebuilds BVH
 * - Extracts materials, textures, and lights from the three.js scene
 * - Loads environment maps from scene.environment
 * - Accumulates samples progressively (resets on camera/scene change)
 */

import { WebGPUPathTracer } from './WebGPUPathTracer.js';
import { SceneProcessor } from './SceneProcessor.js';
import { Vector3 } from 'three';

// Tone mapping constants (matching three.js)
const NoToneMapping = 0;
const LinearToneMapping = 1;
const ReinhardToneMapping = 2;
const CineonToneMapping = 3;
const ACESFilmicToneMapping = 4;

export class PathTracingRenderer {

	constructor( params = {} ) {

		// Create or use provided canvas
		this.domElement = params.canvas || document.createElement( 'canvas' );

		// Public properties matching three.js Renderer API
		this.toneMapping = ACESFilmicToneMapping;
		this.toneMappingExposure = 0.5;
		this.autoClear = true;

		// Path tracer settings
		this.renderScale = params.renderScale || 1;
		this.environmentURL = null;
		this.environmentIntensity = 1;
		this._tinybvhURL = params.tinybvhURL || null;

		// Internal state
		this._pathTracer = new WebGPUPathTracer();
		this._sceneProcessor = null;
		this._initialized = false;
		this._initPromise = null;
		this._pixelRatio = 1;
		this._width = 0;
		this._height = 0;

		// Scene change tracking
		this._sceneVersion = - 1;
		this._cameraVersion = - 1;
		this._lastCameraMatrix = new Float32Array( 16 );
		this._sceneNeedsRebuild = true;

		// Animation loop
		this._animationLoop = null;
		this._animationFrameId = null;

		// Auto-initialize
		this._initPromise = this._init();

	}

	async _init() {

		try {

			await this._pathTracer.init( this.domElement );

			// Init scene processor with tinybvh
			this._sceneProcessor = new SceneProcessor();
			try {

				const tinybvhUrl = this._tinybvhURL
					|| new URL( './libs/tinybvh.js', window.location.href ).href;
				const { default: TinyBVH } = await import( /* @vite-ignore */ tinybvhUrl );
				const mod = await TinyBVH( {
					locateFile: ( p ) => {

						const base = tinybvhUrl.substring( 0, tinybvhUrl.lastIndexOf( '/' ) + 1 );
						return base + p;

					},
				} );
				await this._sceneProcessor.init( mod );

			} catch ( e ) {

				console.warn( 'PathTracingRenderer: tinybvh WASM not found, scene loading will fail.', e.message );

			}

			// Auto-load environment if URL was set before init completed
			if ( this.environmentURL ) {

				try {

					await this._pathTracer.loadEnvironment( this.environmentURL );

				} catch ( e ) {

					console.warn( 'PathTracingRenderer: environment load failed:', e.message );

				}

			}

			this._initialized = true;

		} catch ( e ) {

			console.error( 'PathTracingRenderer: WebGPU initialization failed:', e.message );

		}

	}

	// ---- three.js Renderer API ----

	get enableTLAS() { return this._pathTracer.enableTLAS; }
	set enableTLAS( v ) { this._pathTracer.enableTLAS = v; }

	get samples() {

		return this._pathTracer.samples;

	}

	setSize( width, height, updateStyle = true ) {

		this._width = width;
		this._height = height;

		const w = Math.floor( width * this._pixelRatio * this.renderScale );
		const h = Math.floor( height * this._pixelRatio * this.renderScale );

		this.domElement.width = w;
		this.domElement.height = h;

		if ( updateStyle ) {

			this.domElement.style.width = width + 'px';
			this.domElement.style.height = height + 'px';

		}

		if ( this._initialized ) {

			this._pathTracer.resize( w, h );

		}

	}

	setPixelRatio( value ) {

		this._pixelRatio = value;
		if ( this._width > 0 ) {

			this.setSize( this._width, this._height );

		}

	}

	async render( scene, camera ) {

		if ( this._rendering ) return;
		this._rendering = true;

		try {

			if ( ! this._initialized ) {

				await this._initPromise;
				if ( ! this._initialized ) return;

			}

			// Check if scene needs rebuild
			const sceneVer = this._computeSceneVersion( scene );
			if ( sceneVer !== this._sceneVersion ) {

				this._sceneVersion = sceneVer;
				await this._rebuildScene( scene );

			}

			// Check if camera changed
			if ( this._cameraChanged( camera ) ) {

				this._updateCamera( camera );

			}

			// Render a sample
			this._pathTracer.renderSample();

		} finally {

			this._rendering = false;

		}

	}

	setAnimationLoop( callback ) {

		if ( this._animationFrameId !== null ) {

			cancelAnimationFrame( this._animationFrameId );
			this._animationFrameId = null;

		}

		this._animationLoop = callback;

		if ( callback !== null ) {

			const loop = ( time ) => {

				this._animationFrameId = requestAnimationFrame( loop );
				callback( time );

			};

			this._animationFrameId = requestAnimationFrame( loop );

		}

	}

	/**
	 * Load an HDR environment map. Call before render() or set scene.environment.
	 */
	async setEnvironment( url, intensity = 1 ) {

		this.environmentURL = url;
		this.environmentIntensity = intensity;

		if ( this._initialized ) {

			await this._pathTracer.loadEnvironment( url );

		}

	}

	resetAccumulation() {

		this._pathTracer.resetAccumulation();

	}

	dispose() {

		if ( this._animationFrameId !== null ) {

			cancelAnimationFrame( this._animationFrameId );

		}

	}

	// ---- Internal: Scene Processing ----

	_computeSceneVersion( scene ) {

		// Simple version: count meshes + sum material IDs
		// A real implementation would track object add/remove/material changes
		let version = 0;
		scene.traverse( ( obj ) => {

			if ( obj.isMesh && obj.visible ) {

				version ++;
				if ( obj.material ) version += obj.material.id;

			}

			if ( obj.isLight ) version += obj.id * 1000;

		} );

		return version;

	}

	async _rebuildScene( scene ) {

		if ( ! this._sceneProcessor ) return;

		// Load environment from scene if set
		if ( scene.environment && scene.environment.image && ! this._pathTracer._envMap ) {

			// scene.environment is a three.js Texture — for HDR we need the URL
			// Users should call setEnvironment() directly for HDR env maps
			console.log( 'PathTracingRenderer: scene.environment detected (use setEnvironment() for HDR)' );

		}

		// Process scene
		try {

			const sceneData = await this._sceneProcessor.process( scene );

			// Extract lights
			sceneData.lights = this._extractLights( scene );

			this._pathTracer.setScene( sceneData );
			this._sceneNeedsRebuild = false;

		} catch ( e ) {

			console.error( 'PathTracingRenderer: scene processing failed:', e.message );

		}

	}

	_extractLights( scene ) {

		const lights = [];

		scene.traverse( ( obj ) => {

			if ( ! obj.isLight ) return;

			const light = {
				type: 'unknown',
				color: [ 1, 1, 1 ],
				intensity: obj.intensity || 1,
				position: [ 0, 0, 0 ],
				direction: [ 0, - 1, 0 ],
			};

			if ( obj.color ) {

				light.color = [ obj.color.r, obj.color.g, obj.color.b ];

			}

			obj.updateMatrixWorld( true );
			const pos = new Vector3();
			pos.setFromMatrixPosition( obj.matrixWorld );
			light.position = [ pos.x, pos.y, pos.z ];

			if ( obj.isDirectionalLight ) {

				light.type = 'directional';
				const targetPos = new Vector3();
				if ( obj.target ) {

					obj.target.updateMatrixWorld( true );
					targetPos.setFromMatrixPosition( obj.target.matrixWorld );

				}

				const dir = targetPos.sub( pos ).normalize();
				light.direction = [ dir.x, dir.y, dir.z ];

			} else if ( obj.isPointLight ) {

				light.type = 'point';
				light.distance = obj.distance || 0;
				light.decay = obj.decay !== undefined ? obj.decay : 2;

			} else if ( obj.isSpotLight ) {

				light.type = 'spot';
				light.distance = obj.distance || 0;
				light.decay = obj.decay !== undefined ? obj.decay : 2;
				light.angle = obj.angle || Math.PI / 3;
				light.penumbra = obj.penumbra || 0;

				const targetPos = new Vector3();
				if ( obj.target ) {

					obj.target.updateMatrixWorld( true );
					targetPos.setFromMatrixPosition( obj.target.matrixWorld );

				}

				const dir = targetPos.sub( pos ).normalize();
				light.direction = [ dir.x, dir.y, dir.z ];

			} else if ( obj.isAmbientLight ) {

				light.type = 'ambient';

			} else if ( obj.isHemisphereLight ) {

				light.type = 'hemisphere';
				light.groundColor = obj.groundColor
					? [ obj.groundColor.r, obj.groundColor.g, obj.groundColor.b ]
					: [ 0.2, 0.2, 0.2 ];

			} else if ( obj.isRectAreaLight ) {

				light.type = 'area';
				light.width = obj.width || 10;
				light.height = obj.height || 10;

			}

			lights.push( light );

		} );

		return lights;

	}

	// ---- Internal: Camera ----

	_cameraChanged( camera ) {

		camera.updateMatrixWorld();
		const m = camera.matrixWorld.elements;

		for ( let i = 0; i < 16; i ++ ) {

			if ( Math.abs( m[ i ] - this._lastCameraMatrix[ i ] ) > 1e-6 ) {

				return true;

			}

		}

		return false;

	}

	_updateCamera( camera ) {

		camera.updateMatrixWorld();
		const m = camera.matrixWorld.elements;

		for ( let i = 0; i < 16; i ++ ) {

			this._lastCameraMatrix[ i ] = m[ i ];

		}

		const aspect = camera.aspect || ( this._width / this._height );
		const fov = ( camera.fov || 50 ) * Math.PI / 180;
		const halfH = Math.tan( fov / 2 );
		const halfW = halfH * aspect;

		const right = new Vector3( m[ 0 ], m[ 1 ], m[ 2 ] ).normalize();
		const up = new Vector3( m[ 4 ], m[ 5 ], m[ 6 ] ).normalize();
		const forward = new Vector3( - m[ 8 ], - m[ 9 ], - m[ 10 ] ).normalize();

		this._pathTracer.updateCamera( {
			position: [ camera.position.x, camera.position.y, camera.position.z ],
			image_u: [ right.x * halfW, right.y * halfW, right.z * halfW ],
			image_v: [ up.x * halfH, up.y * halfH, up.z * halfH ],
			image_w: [ forward.x, forward.y, forward.z ],
			lens_radius: camera.bokehSize || 0,
			focus_distance: camera.focusDistance || 10,
		} );

	}

}
