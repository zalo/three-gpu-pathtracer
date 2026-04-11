import {
	Scene,
	PerspectiveCamera,
	Box3,
	Vector3,
	Group,
	LoadingManager,
} from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/examples/jsm/loaders/DRACOLoader.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { WebGPUPathTracer } from '../src/webgpu/WebGPUPathTracer.js';
import { SceneProcessor } from '../src/webgpu/SceneProcessor.js';

const DEFAULT_MODEL_URL = 'https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/DragonDispersion/glTF-Binary/DragonDispersion.glb';
const ENV_URL = 'https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/citrus_orchard_puresky_1k.hdr';

const isMobile = /Android|iPhone|iPad|iPod|Mobile/i.test( navigator.userAgent ) || ( 'ontouchstart' in window );
const renderScale = isMobile ? 1 / 8 : 1;

let pathTracer, sceneProcessor;
let controls, camera;
let isModelLoaded = false;
let cameraChanged = true;

const dropZone = document.getElementById( 'drop-zone' );
const infoEl = document.getElementById( 'info' );
const canvas = document.getElementById( 'canvas' );
const progressBar = document.getElementById( 'progress-bar' );
const progressLabel = document.getElementById( 'progress-label' );
const progressFill = document.getElementById( 'progress-fill' );

function showProgress( label, pct ) {

	progressBar.classList.add( 'visible' );
	progressLabel.textContent = label;
	progressFill.style.width = pct + '%';

}

function hideProgress() {

	progressBar.classList.remove( 'visible' );

}

init();

async function init() {

	canvas.width = Math.floor( window.innerWidth * window.devicePixelRatio * renderScale );
	canvas.height = Math.floor( window.innerHeight * window.devicePixelRatio * renderScale );

	showProgress( 'Initializing WebGPU...', 5 );

	pathTracer = new WebGPUPathTracer();
	pathTracer.enableTLAS = true;
	if ( isMobile ) {

		pathTracer.sppPerDispatch = 1;
		pathTracer.maxBounces = 4;
		pathTracer.maxShadowBounces = 4;

	}

	try {

		await pathTracer.init( canvas );

	} catch ( e ) {

		progressLabel.textContent = `WebGPU Error: ${e.message}`;
		progressFill.style.background = '#f44';
		progressFill.style.width = '100%';
		console.error( e );
		return;

	}

	showProgress( 'Loading tinybvh WASM...', 15 );

	sceneProcessor = new SceneProcessor();

	try {

		const tinybvhUrl = new URL( './libs/tinybvh.js', import.meta.url ).href;
		const { default: TinyBVH } = await import( /* @vite-ignore */ tinybvhUrl );
		const tinybvhModule = await TinyBVH( {
			locateFile: ( path ) => new URL( `./libs/${path}`, import.meta.url ).href,
		} );
		await sceneProcessor.init( tinybvhModule );

	} catch ( e ) {

		console.warn( 'tinybvh WASM load failed:', e.message );

	}

	showProgress( 'Loading environment map...', 25 );

	try {

		await pathTracer.loadEnvironment( ENV_URL );

	} catch ( e ) {

		console.warn( 'Environment map failed to load:', e.message );

	}

	camera = new PerspectiveCamera( 50, window.innerWidth / window.innerHeight, 0.025, 500 );
	camera.position.set( 0, 0, 4 );

	controls = new OrbitControls( camera, canvas );
	controls.addEventListener( 'change', () => {

		cameraChanged = true;

	} );

	controls.update();

	window.addEventListener( 'resize', onResize );
	setupDragDrop();

	// Load default model
	showProgress( 'Downloading dragon model...', 35 );

	try {

		const resp = await fetch( DEFAULT_MODEL_URL );
		const blob = await resp.blob();
		const arrayBuffer = await blob.arrayBuffer();

		showProgress( 'Parsing GLTF...', 55 );

		const gltfLoader = new GLTFLoader();
		const dracoLoader = new DRACOLoader();
		dracoLoader.setDecoderPath( 'https://www.gstatic.com/draco/versioned/decoders/1.5.7/' );
		gltfLoader.setDRACOLoader( dracoLoader );

		gltfLoader.parse( arrayBuffer, '', async ( gltf ) => {

			await processGLTF( gltf );

		} );

	} catch ( e ) {

		console.warn( 'Default model failed to load:', e.message );
		hideProgress();

	}

	animate();

}

function setupDragDrop() {

	window.addEventListener( 'dragover', ( e ) => {

		e.preventDefault();
		if ( ! isModelLoaded ) {

			dropZone.classList.add( 'drag-over' );

		}

	} );

	window.addEventListener( 'dragleave', ( e ) => {

		if ( e.relatedTarget === null || e.relatedTarget === document.documentElement ) {

			dropZone.classList.remove( 'drag-over' );

		}

	} );

	window.addEventListener( 'drop', ( e ) => {

		e.preventDefault();
		dropZone.classList.remove( 'drag-over' );

		const files = e.dataTransfer.files;
		if ( files.length === 0 ) return;

		showProgress( 'Loading model...', 10 );
		loadGLTFFiles( files );

	} );

}

function loadGLTFFiles( files ) {

	const fileMap = new Map();
	let rootUrl = null;

	for ( const file of files ) {

		const url = URL.createObjectURL( file );
		fileMap.set( file.name, url );

		if ( file.name.match( /\.gltf$/i ) ) {

			rootUrl = url;

		}

	}

	const loadingManager = new LoadingManager();
	loadingManager.setURLModifier( ( url ) =>
		fileMap.get( url.split( '/' ).pop() ) || url
	);

	const dracoLoader = new DRACOLoader();
	dracoLoader.setDecoderPath( 'https://www.gstatic.com/draco/versioned/decoders/1.5.7/' );

	const gltfLoader = new GLTFLoader( loadingManager );
	gltfLoader.setDRACOLoader( dracoLoader );

	const onLoad = async ( gltf ) => {

		try {

			await processGLTF( gltf );

		} catch ( err ) {

			progressLabel.textContent = `Error: ${err.message}`;
			console.error( err );

		}

		fileMap.forEach( ( url ) => URL.revokeObjectURL( url ) );

	};

	if ( rootUrl ) {

		gltfLoader.load( rootUrl, onLoad );

	} else {

		const file = files[ 0 ];
		const reader = new FileReader();
		reader.onload = ( e ) => {

			gltfLoader.parse( e.target.result, '', onLoad );

		};

		reader.readAsArrayBuffer( file );

	}

}

async function processGLTF( gltf ) {

	const tempScene = new Scene();
	const container = new Group();
	container.add( gltf.scene );
	tempScene.add( container );

	// Auto-fit camera
	const box = new Box3().setFromObject( gltf.scene );
	const center = box.getCenter( new Vector3() );
	const size = box.getSize( new Vector3() );

	gltf.scene.position.sub( center );
	container.rotation.y = Math.PI;

	const maxDim = Math.max( size.x, size.y, size.z );
	const fov = camera.fov * ( Math.PI / 180 );
	const distance = ( maxDim / ( 2 * Math.tan( fov / 2 ) ) ) * 1.5;
	camera.position.set( 0, 0, - distance );

	camera.near = maxDim / 100;
	camera.far = maxDim * 10;
	camera.updateProjectionMatrix();

	controls.target.set( 0, 0, 0 );
	controls.update();

	// Process scene for GPU (async with progress reporting)
	const sceneData = await sceneProcessor.process( tempScene, ( label, frac ) => {

		showProgress( label, 55 + frac * 40 ); // map [0,1] → [55%, 95%]

	} );

	showProgress( 'Uploading to GPU...', 95 );
	await new Promise( r => requestAnimationFrame( r ) );

	pathTracer.setScene( sceneData );
	cameraChanged = true;

	showProgress( 'Ready', 100 );

	setTimeout( () => {

		hideProgress();
		dropZone.innerText = 'Drop GLTF/GLB file here';
		dropZone.classList.add( 'hidden' );
		isModelLoaded = true;

	}, 400 );

}

function buildCameraData() {

	camera.updateMatrixWorld();

	const aspect = camera.aspect;
	const fov = camera.fov * Math.PI / 180;
	const halfH = Math.tan( fov / 2 );
	const halfW = halfH * aspect;

	const m = camera.matrixWorld.elements;
	const right = new Vector3( m[ 0 ], m[ 1 ], m[ 2 ] ).normalize();
	const up = new Vector3( m[ 4 ], m[ 5 ], m[ 6 ] ).normalize();
	const forward = new Vector3( - m[ 8 ], - m[ 9 ], - m[ 10 ] ).normalize();

	return {
		position: [ camera.position.x, camera.position.y, camera.position.z ],
		image_u: [ right.x * halfW, right.y * halfW, right.z * halfW ],
		image_v: [ up.x * halfH, up.y * halfH, up.z * halfH ],
		image_w: [ forward.x, forward.y, forward.z ],
		lens_radius: 0,
		focus_distance: controls.target.distanceTo( camera.position ),
	};

}

function onResize() {

	const w = window.innerWidth;
	const h = window.innerHeight;
	const dpr = window.devicePixelRatio;

	canvas.width = Math.floor( w * dpr * renderScale );
	canvas.height = Math.floor( h * dpr * renderScale );
	canvas.style.width = w + 'px';
	canvas.style.height = h + 'px';

	camera.aspect = w / h;
	camera.updateProjectionMatrix();

	pathTracer.resize( canvas.width, canvas.height );
	cameraChanged = true;

}

function animate() {

	requestAnimationFrame( animate );

	if ( cameraChanged && isModelLoaded ) {

		pathTracer.updateCamera( buildCameraData() );
		cameraChanged = false;

	}

	if ( isModelLoaded ) {

		pathTracer.renderSample();

		const spp = pathTracer.samples;
		const mrays = pathTracer.mraysPerSec;
		infoEl.textContent = `${spp} spp | ${mrays.toFixed( 1 )} Mrays/s | ${canvas.width}x${canvas.height}`;

	}

}
