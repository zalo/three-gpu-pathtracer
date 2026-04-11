/**
 * Creates and manages WebGPU compute and render pipelines from Slang shaders.
 */

// vite-slang compiles these to WGSL at build time
import { code as pathTracerCode, reflection as pathTracerReflection } from '../shaders/pathtracer.slang';
import { code as accumulateCode, reflection as accumulateReflection } from '../shaders/accumulate.slang';
import { code as displayCode } from '../shaders/display.slang';

export class PipelineManager {

	constructor( device, canvasFormat ) {

		this.device = device;
		this.canvasFormat = canvasFormat;

		this.pathTracePipeline = null;
		this.accumulatePipeline = null;
		this.displayPipeline = null;

		// Expose reflection data for bind group layout inspection
		this.pathTracerReflection = pathTracerReflection;
		this.accumulateReflection = accumulateReflection;

	}

	createPipelines() {

		// Path tracer compute pipeline
		const ptModule = this.device.createShaderModule( {
			code: pathTracerCode,
			label: 'pathtracer.slang',
		} );

		this.pathTracePipeline = this.device.createComputePipeline( {
			layout: 'auto',
			compute: {
				module: ptModule,
				entryPoint: 'main',
			},
		} );

		// Accumulate compute pipeline
		const accumModule = this.device.createShaderModule( {
			code: accumulateCode,
			label: 'accumulate.slang',
		} );

		this.accumulatePipeline = this.device.createComputePipeline( {
			layout: 'auto',
			compute: {
				module: accumModule,
				entryPoint: 'main',
			},
		} );

		// Display render pipeline (fullscreen triangle blit)
		const displayModule = this.device.createShaderModule( {
			code: displayCode,
			label: 'display.slang',
		} );

		this.displayPipeline = this.device.createRenderPipeline( {
			layout: 'auto',
			vertex: {
				module: displayModule,
				entryPoint: 'vs_main',
			},
			fragment: {
				module: displayModule,
				entryPoint: 'fs_main',
				targets: [ { format: this.canvasFormat } ],
			},
			primitive: {
				topology: 'triangle-list',
			},
		} );

	}

}
