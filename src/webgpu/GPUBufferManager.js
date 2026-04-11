/**
 * Manages WebGPU buffer and texture creation for the path tracer.
 */
export class GPUBufferManager {

	constructor( device ) {

		this.device = device;
		this.buffers = {};
		this.textures = {};

	}

	createStorageBuffer( name, data ) {

		const buffer = this.device.createBuffer( {
			size: Math.max( data.byteLength, 4 ), // WebGPU requires size > 0
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
			mappedAtCreation: true,
		} );

		new Uint8Array( buffer.getMappedRange() ).set( new Uint8Array( data.buffer, data.byteOffset, data.byteLength ) );
		buffer.unmap();

		this.buffers[ name ] = buffer;
		return buffer;

	}

	createUniformBuffer( name, size ) {

		const buffer = this.device.createBuffer( {
			size,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		} );

		this.buffers[ name ] = buffer;
		return buffer;

	}

	updateBuffer( name, data ) {

		this.device.queue.writeBuffer( this.buffers[ name ], 0, data );

	}

	createStorageTexture( name, width, height, format = 'rgba16float' ) {

		const texture = this.device.createTexture( {
			size: [ width, height ],
			format,
			usage:
				GPUTextureUsage.STORAGE_BINDING |
				GPUTextureUsage.TEXTURE_BINDING |
				GPUTextureUsage.COPY_SRC,
		} );

		this.textures[ name ] = texture;
		return texture;

	}

	getBuffer( name ) {

		return this.buffers[ name ];

	}

	getTexture( name ) {

		return this.textures[ name ];

	}

	destroy() {

		for ( const buffer of Object.values( this.buffers ) ) {

			buffer.destroy();

		}

		for ( const texture of Object.values( this.textures ) ) {

			texture.destroy();

		}

		this.buffers = {};
		this.textures = {};

	}

}
