import {
	Scene,
	PerspectiveCamera,
	MeshPhysicalMaterial,
	BoxGeometry,
	PlaneGeometry,
	CylinderGeometry,
	Mesh,
	PointLight,
	AmbientLight,
	DoubleSide,
	Color,
} from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { PathTracingRenderer } from '../src/webgpu/PathTracingRenderer.js';

const isMobile = /Android|iPhone|iPad|iPod|Mobile/i.test( navigator.userAgent ) || ( 'ontouchstart' in window );

let renderer, controls, camera, scene;
const infoEl = document.getElementById( 'info' );

init();

async function init() {

	const canvas = document.getElementById( 'canvas' );
	canvas.width = Math.floor( window.innerWidth * window.devicePixelRatio );
	canvas.height = Math.floor( window.innerHeight * window.devicePixelRatio );

	renderer = new PathTracingRenderer( {
		canvas,
		tinybvhURL: new URL( './libs/tinybvh.js', import.meta.url ).href,
	} );
	renderer.setSize( window.innerWidth, window.innerHeight );
	renderer.setPixelRatio( isMobile ? 0.5 : window.devicePixelRatio );
	renderer.environmentURL = 'https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/studio_small_09_1k.hdr';
	renderer.enableTLAS = true;

	if ( isMobile ) {

		renderer._pathTracer.sppPerDispatch = 1;
		renderer._pathTracer.maxBounces = 4;
		renderer._pathTracer.maxShadowBounces = 4;

	}

	// Camera — matching the three.js SSGI Cornell Box example
	camera = new PerspectiveCamera( 40, window.innerWidth / window.innerHeight, 0.1, 100 );
	camera.position.set( 0, 10, 30 );

	controls = new OrbitControls( camera, canvas );
	controls.target.set( 0, 7, 0 );
	controls.update();

	// Scene
	scene = new Scene();

	const wallGeometry = new PlaneGeometry( 1, 1 );

	// Left wall — red
	const leftWall = new Mesh(
		wallGeometry,
		new MeshPhysicalMaterial( { color: 0xff0000, side: DoubleSide } )
	);
	leftWall.scale.set( 20, 15, 1 );
	leftWall.rotation.y = Math.PI * 0.5;
	leftWall.position.set( - 10, 7.5, 0 );
	scene.add( leftWall );

	// Right wall — green
	const rightWall = new Mesh(
		wallGeometry,
		new MeshPhysicalMaterial( { color: 0x00ff00, side: DoubleSide } )
	);
	rightWall.scale.set( 20, 15, 1 );
	rightWall.rotation.y = Math.PI * - 0.5;
	rightWall.position.set( 10, 7.5, 0 );
	scene.add( rightWall );

	// White material for floor, back wall, ceiling, boxes
	const whiteMaterial = new MeshPhysicalMaterial( { color: 0xffffff, side: DoubleSide } );

	// Floor
	const floor = new Mesh( wallGeometry, whiteMaterial );
	floor.scale.set( 20, 20, 1 );
	floor.rotation.x = Math.PI * - 0.5;
	scene.add( floor );

	// Back wall
	const backWall = new Mesh( wallGeometry, whiteMaterial );
	backWall.scale.set( 15, 20, 1 );
	backWall.rotation.z = Math.PI * - 0.5;
	backWall.position.set( 0, 7.5, - 10 );
	scene.add( backWall );

	// Ceiling
	const ceiling = new Mesh( wallGeometry, whiteMaterial );
	ceiling.scale.set( 20, 20, 1 );
	ceiling.rotation.x = Math.PI * 0.5;
	ceiling.position.set( 0, 15, 0 );
	scene.add( ceiling );

	// Box material (separate, single-sided)
	const boxMaterial = new MeshPhysicalMaterial( { color: 0xffffff } );

	// Tall box
	const tallBox = new Mesh(
		new BoxGeometry( 5, 7, 5 ),
		boxMaterial
	);
	tallBox.rotation.y = Math.PI * 0.25;
	tallBox.position.set( - 3, 3.5, - 2 );
	scene.add( tallBox );

	// Short box
	const shortBox = new Mesh(
		new BoxGeometry( 4, 4, 4 ),
		boxMaterial
	);
	shortBox.rotation.y = Math.PI * - 0.1;
	shortBox.position.set( 4, 2, 4 );
	scene.add( shortBox );

	// Light source geometry (emissive cylinder on the ceiling)
	const lightSource = new Mesh(
		new CylinderGeometry( 2.5, 2.5, 0.1, 64 ),
		new MeshPhysicalMaterial( {
			color: 0x000000,
			emissive: 0xffffff,
			emissiveIntensity: 10,
		} )
	);
	lightSource.position.set( 0, 14.95, 0 );
	scene.add( lightSource );

	// Point light
	const pointLight = new PointLight( 0xffffff, 100 );
	pointLight.position.set( 0, 13, 0 );
	pointLight.distance = 100;
	scene.add( pointLight );

	// Ambient light
	const ambientLight = new AmbientLight( 0x0c0c0c );
	scene.add( ambientLight );

	window.addEventListener( 'resize', onResize );
	animate();

}

function onResize() {

	const w = window.innerWidth;
	const h = window.innerHeight;

	renderer.setSize( w, h );
	camera.aspect = w / h;
	camera.updateProjectionMatrix();

}

function animate() {

	requestAnimationFrame( animate );

	renderer.render( scene, camera );

	const pt = renderer._pathTracer;
	if ( pt ) {

		const spp = pt.samples;
		const mrays = pt.mraysPerSec;
		infoEl.textContent = `${spp} spp | ${mrays.toFixed( 1 )} Mrays/s`;

	}

}
