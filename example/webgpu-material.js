import {
	Scene,
	PerspectiveCamera,
	Box3,
	Vector3,
	Group,
} from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/examples/jsm/loaders/DRACOLoader.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { WebGPUPathTracer } from '../src/webgpu/WebGPUPathTracer.js';
import { SceneProcessor } from '../src/webgpu/SceneProcessor.js';

const ENV_URL = 'https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/wooden_studio_02_1k.hdr';

const MODELS = {
	'Dragon Dispersion': {
		url: 'https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/DragonDispersion/glTF-Binary/DragonDispersion.glb',
		credit: 'Khronos glTF Sample Assets',
	},
	'Damaged Helmet': {
		url: 'https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/DamagedHelmet/glTF/DamagedHelmet.gltf',
		credit: 'glTF Sample Model',
	},
	'Flight Helmet': {
		url: 'https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/FlightHelmet/glTF/FlightHelmet.gltf',
		credit: 'glTF Sample Model',
	},
	'Terrarium Robots': {
		url: 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/main/models/terrarium-robots/scene.gltf',
		credit: 'Model by "nyancube" on Sketchfab',
	},
	'Japanese Bridge': {
		url: 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/main/models/japanese-bridge-garden/scene.glb',
		credit: 'Model by "kristenlee" on Sketchfab',
	},
};

const isMobile = /Android|iPhone|iPad|iPod|Mobile/i.test( navigator.userAgent ) || ( 'ontouchstart' in window );

let pathTracer, sceneProcessor;
let controls, camera;
let cameraChanged = true;
let isModelLoaded = false;

const canvas = document.getElementById( 'canvas' );
const infoEl = document.getElementById( 'info' );
const creditEl = document.getElementById( 'credit' );
const selectEl = document.getElementById( 'model-select' );
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

// Populate model selector
for ( const name in MODELS ) {

	const opt = document.createElement( 'option' );
	opt.value = name;
	opt.textContent = name;
	selectEl.appendChild( opt );

}

selectEl.addEventListener( 'change', () => loadModel( selectEl.value ) );

init();

async function init() {

	canvas.width = Math.floor( window.innerWidth * window.devicePixelRatio * ( isMobile ? 0.25 : 1 ) );
	canvas.height = Math.floor( window.innerHeight * window.devicePixelRatio * ( isMobile ? 0.25 : 1 ) );

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
		return;

	}

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

	try {

		await pathTracer.loadEnvironment( ENV_URL );

	} catch ( e ) {

		console.warn( 'Environment map failed:', e.message );

	}

	camera = new PerspectiveCamera( 50, window.innerWidth / window.innerHeight, 0.025, 500 );
	controls = new OrbitControls( camera, canvas );
	controls.addEventListener( 'change', () => {

		cameraChanged = true;

	} );

	window.addEventListener( 'resize', onResize );

	// Load first model
	await loadModel( selectEl.value );

	animate();

}

async function loadModel( name ) {

	const model = MODELS[ name ];
	if ( ! model ) return;

	isModelLoaded = false;
	showProgress( 'Downloading model...', 10 );
	creditEl.textContent = model.credit || '';

	const gltfLoader = new GLTFLoader();
	const dracoLoader = new DRACOLoader();
	dracoLoader.setDecoderPath( 'https://www.gstatic.com/draco/versioned/decoders/1.5.7/' );
	gltfLoader.setDRACOLoader( dracoLoader );

	try {

		const gltf = await gltfLoader.loadAsync( model.url );

		showProgress( 'Processing scene...', 50 );

		const tempScene = new Scene();
		const container = new Group();
		container.add( gltf.scene );
		tempScene.add( container );

		const box = new Box3().setFromObject( gltf.scene );
		const center = box.getCenter( new Vector3() );
		const size = box.getSize( new Vector3() );
		gltf.scene.position.sub( center );

		const maxDim = Math.max( size.x, size.y, size.z );
		const fov = camera.fov * ( Math.PI / 180 );
		const distance = ( maxDim / ( 2 * Math.tan( fov / 2 ) ) ) * 1.5;
		camera.position.set( 0, 0, distance );
		camera.near = maxDim / 100;
		camera.far = maxDim * 10;
		camera.updateProjectionMatrix();
		controls.target.set( 0, 0, 0 );
		controls.update();

		const sceneData = await sceneProcessor.process( tempScene, ( label, frac ) => {

			showProgress( label, 50 + frac * 45 );

		} );

		pathTracer.setScene( sceneData );
		cameraChanged = true;
		isModelLoaded = true;
		hideProgress();

	} catch ( e ) {

		progressLabel.textContent = `Error: ${e.message}`;
		console.error( e );

	}

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
	const scale = isMobile ? 0.25 : 1;

	canvas.width = Math.floor( w * window.devicePixelRatio * scale );
	canvas.height = Math.floor( h * window.devicePixelRatio * scale );
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
