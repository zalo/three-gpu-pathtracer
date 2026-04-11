import {
	Scene,
	PerspectiveCamera,
	MeshPhysicalMaterial,
	SphereGeometry,
	BoxGeometry,
	PlaneGeometry,
	Mesh,
	Color,
	DoubleSide,
} from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { PathTracingRenderer } from '../src/webgpu/PathTracingRenderer.js';

const ENV_URL = 'https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/wooden_studio_02_1k.hdr';

const isMobile = /Android|iPhone|iPad|iPod|Mobile/i.test( navigator.userAgent ) || ( 'ontouchstart' in window );

let renderer, controls, camera, scene;
const infoEl = document.getElementById( 'info' );

init();

async function init() {

	const canvas = document.getElementById( 'canvas' );
	canvas.width = Math.floor( window.innerWidth * window.devicePixelRatio );
	canvas.height = Math.floor( window.innerHeight * window.devicePixelRatio );

	// Create path tracing renderer (drop-in three.js replacement)
	renderer = new PathTracingRenderer( { canvas } );
	renderer.setSize( window.innerWidth, window.innerHeight );
	renderer.setPixelRatio( isMobile ? 0.5 : window.devicePixelRatio );
	renderer.environmentURL = ENV_URL;
	renderer.enableTLAS = true;

	if ( isMobile ) {

		renderer._pathTracer.sppPerDispatch = 1;
		renderer._pathTracer.maxBounces = 4;
		renderer._pathTracer.maxShadowBounces = 4;

	}

	// Camera
	camera = new PerspectiveCamera( 50, window.innerWidth / window.innerHeight, 0.1, 100 );
	camera.position.set( 3, 2.5, 4 );

	// Controls
	controls = new OrbitControls( camera, canvas );
	controls.target.set( 0, 0.5, 0 );
	controls.update();

	// Scene
	scene = new Scene();

	// Ground plane
	const ground = new Mesh(
		new PlaneGeometry( 10, 10 ),
		new MeshPhysicalMaterial( {
			color: 0xcccccc,
			roughness: 0.8,
			metalness: 0.0,
			side: DoubleSide,
		} )
	);
	ground.rotation.x = - Math.PI / 2;
	scene.add( ground );

	// Glass sphere
	const glassSphere = new Mesh(
		new SphereGeometry( 0.6, 64, 64 ),
		new MeshPhysicalMaterial( {
			color: 0xffffff,
			roughness: 0.0,
			metalness: 0.0,
			transmission: 1.0,
			ior: 1.5,
			thickness: 1.2,
		} )
	);
	glassSphere.position.set( 0, 0.6, 0 );
	scene.add( glassSphere );

	// Metal sphere
	const metalSphere = new Mesh(
		new SphereGeometry( 0.4, 64, 64 ),
		new MeshPhysicalMaterial( {
			color: 0xddaa44,
			roughness: 0.1,
			metalness: 1.0,
		} )
	);
	metalSphere.position.set( - 1.2, 0.4, 0.5 );
	scene.add( metalSphere );

	// Matte box
	const matteBox = new Mesh(
		new BoxGeometry( 0.8, 0.8, 0.8 ),
		new MeshPhysicalMaterial( {
			color: 0xcc4444,
			roughness: 0.9,
			metalness: 0.0,
		} )
	);
	matteBox.position.set( 1.0, 0.4, - 0.3 );
	matteBox.rotation.y = 0.5;
	scene.add( matteBox );

	// Glossy sphere
	const glossySphere = new Mesh(
		new SphereGeometry( 0.35, 64, 64 ),
		new MeshPhysicalMaterial( {
			color: 0x4488cc,
			roughness: 0.3,
			metalness: 0.0,
		} )
	);
	glossySphere.position.set( 0.3, 0.35, 1.2 );
	scene.add( glossySphere );

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
