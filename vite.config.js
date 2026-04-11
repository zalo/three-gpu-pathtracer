import { searchForWorkspaceRoot } from 'vite';
import fs from 'fs';
import viteSlang from './third_party/vite-slang/src/index.js';

const keyPath = '/tmp/certs/key.pem';
const certPath = '/tmp/certs/cert.pem';
const hasLocalCerts = fs.existsSync( keyPath ) && fs.existsSync( certPath );

export default {

	root: './example/',
	base: './',
	plugins: [
		viteSlang( { target: 'WGSL' } ),
	],
	build: {
		outDir: '../dist/',
		sourcemap: true,
		rollupOptions: {
			input: fs
				.readdirSync( './example/' )
				.filter( p => /\.html$/.test( p ) )
				.map( p => `./example/${ p }` ),
		},
	},
	server: {
		...( hasLocalCerts ? {
			https: {
				key: fs.readFileSync( keyPath ),
				cert: fs.readFileSync( certPath ),
			},
		} : {} ),
		fs: {
			allow: [
				// search up for workspace root
				searchForWorkspaceRoot( process.cwd() ),
			],
		},
	},
	assetsInclude: [ '**/*.wasm' ],
	optimizeDeps: {
    	exclude: [ 'three-mesh-bvh' ],
  	},
};
