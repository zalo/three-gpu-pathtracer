/**
 * Loads an equirectangular HDR environment map and precomputes
 * luminance-weighted CDF tables for importance sampling.
 *
 * Based on the technique from SlangRenderer: hierarchical marginal + conditional CDFs
 * with sin(theta) weighting for correct spherical area measure.
 */
import { RGBELoader } from 'three/examples/jsm/loaders/RGBELoader.js';

export class EnvironmentMap {

	constructor( device ) {

		this.device = device;
		this.envTexture = null;
		this.marginalCDF = null;
		this.conditionalCDF = null;
		this.luminancePDF = null;
		this.sampler = null;
		this.width = 0;
		this.height = 0;
		this.totalLuminance = 0;
		this.ready = false;

	}

	async load( url ) {

		// Load HDR using three.js RGBELoader
		const loader = new RGBELoader();
		const texture = await loader.loadAsync( url );
		const data = texture.image.data; // Float32Array or Uint16Array (half float)
		const width = texture.image.width;
		const height = texture.image.height;

		this.width = width;
		this.height = height;

		// Convert to float32 RGBA
		let floatData;
		if ( data instanceof Float32Array ) {

			floatData = data;

		} else {

			// Uint16Array of float16 values → Float32
			floatData = new Float32Array( data.length );
			for ( let i = 0; i < data.length; i ++ ) {

				floatData[ i ] = this._float16ToFloat32( data[ i ] );

			}

		}

		// Compute luminance and CDF tables
		this._buildImportanceSamplingTables( floatData, width, height );

		// Upload environment texture
		this._uploadEnvTexture( floatData, width, height );

		this.ready = true;
		console.log( `Environment map loaded: ${width}x${height}, total luminance: ${this.totalLuminance.toFixed( 2 )}` );

	}

	_buildImportanceSamplingTables( data, width, height ) {

		// Compute luminance for each pixel, weighted by sin(theta) for spherical area
		const luminance = new Float32Array( width * height );
		const EPSILON = 1e-10;

		for ( let y = 0; y < height; y ++ ) {

			// sin(theta) correction for equirectangular projection
			const theta = Math.PI * ( y + 0.5 ) / height;
			const sinTheta = Math.sin( theta );

			for ( let x = 0; x < width; x ++ ) {

				const idx = ( y * width + x ) * 4;
				const r = data[ idx + 0 ];
				const g = data[ idx + 1 ];
				const b = data[ idx + 2 ];
				// Rec. 709 luminance
				const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
				luminance[ y * width + x ] = Math.max( lum * sinTheta, EPSILON );

			}

		}

		// Row sums for marginal distribution
		const rowSums = new Float32Array( height );
		let totalLum = 0;
		for ( let y = 0; y < height; y ++ ) {

			let sum = 0;
			for ( let x = 0; x < width; x ++ ) {

				sum += luminance[ y * width + x ];

			}

			rowSums[ y ] = sum;
			totalLum += sum;

		}

		this.totalLuminance = totalLum;

		// Marginal CDF (row selection)
		const marginalCDF = new Float32Array( height );
		let cumSum = 0;
		for ( let y = 0; y < height; y ++ ) {

			cumSum += rowSums[ y ] / totalLum;
			marginalCDF[ y ] = cumSum;

		}

		marginalCDF[ height - 1 ] = 1.0; // ensure last is exactly 1

		// Conditional CDF (column selection per row)
		const conditionalCDF = new Float32Array( width * height );
		for ( let y = 0; y < height; y ++ ) {

			let rowCum = 0;
			const rowTotal = Math.max( rowSums[ y ], EPSILON );
			for ( let x = 0; x < width; x ++ ) {

				rowCum += luminance[ y * width + x ] / rowTotal;
				conditionalCDF[ y * width + x ] = rowCum;

			}

			conditionalCDF[ y * width + ( width - 1 ) ] = 1.0;

		}

		// Luminance PDF (for MIS weighting)
		const luminancePDF = new Float32Array( width * height );
		for ( let i = 0; i < width * height; i ++ ) {

			luminancePDF[ i ] = luminance[ i ] / totalLum;

		}

		// Upload CDF textures to GPU
		this._uploadCDFTextures( marginalCDF, conditionalCDF, luminancePDF, width, height );

	}

	_uploadCDFTextures( marginalCDF, conditionalCDF, luminancePDF, width, height ) {

		const device = this.device;

		// Marginal CDF: 1D texture (1 x height), r32float
		this.marginalCDF = device.createTexture( {
			size: [ 1, height ],
			format: 'r32float',
			usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
		} );

		device.queue.writeTexture(
			{ texture: this.marginalCDF },
			marginalCDF,
			{ bytesPerRow: 4 },
			[ 1, height ],
		);

		// Conditional CDF: 2D texture (width x height), r32float
		this.conditionalCDF = device.createTexture( {
			size: [ width, height ],
			format: 'r32float',
			usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
		} );

		device.queue.writeTexture(
			{ texture: this.conditionalCDF },
			conditionalCDF,
			{ bytesPerRow: width * 4, rowsPerImage: height },
			[ width, height ],
		);

		// Luminance PDF: 2D texture (width x height), r32float
		this.luminancePDF = device.createTexture( {
			size: [ width, height ],
			format: 'r32float',
			usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
		} );

		device.queue.writeTexture(
			{ texture: this.luminancePDF },
			luminancePDF,
			{ bytesPerRow: width * 4, rowsPerImage: height },
			[ width, height ],
		);

	}

	_uploadEnvTexture( data, width, height ) {

		const device = this.device;

		// Create env map texture with mipmaps
		const mipLevelCount = Math.floor( Math.log2( Math.max( width, height ) ) ) + 1;

		this.envTexture = device.createTexture( {
			size: [ width, height ],
			format: 'rgba16float',
			usage:
				GPUTextureUsage.TEXTURE_BINDING |
				GPUTextureUsage.COPY_DST,
			mipLevelCount,
		} );

		// Convert float32 to float16 for upload
		const halfData = this._float32ToFloat16( data );

		// Upload base mip
		device.queue.writeTexture(
			{ texture: this.envTexture, mipLevel: 0 },
			halfData,
			{ bytesPerRow: width * 8, rowsPerImage: height },
			[ width, height ],
		);

		// Generate mipmaps via averaging (simple box filter)
		this._generateMipmaps( data, width, height, mipLevelCount );

		// Create sampler (nearest for CDF lookups, linear for env map)
		this.sampler = device.createSampler( {
			magFilter: 'linear',
			minFilter: 'linear',
			mipmapFilter: 'linear',
		} );

		this.cdfSampler = device.createSampler( {
			magFilter: 'nearest',
			minFilter: 'nearest',
		} );

	}

	_generateMipmaps( data, width, height, mipLevelCount ) {

		let srcData = data; // float32
		let srcW = width;
		let srcH = height;

		for ( let mip = 1; mip < mipLevelCount; mip ++ ) {

			const dstW = Math.max( 1, srcW >> 1 );
			const dstH = Math.max( 1, srcH >> 1 );
			const dstData = new Float32Array( dstW * dstH * 4 );

			for ( let y = 0; y < dstH; y ++ ) {

				for ( let x = 0; x < dstW; x ++ ) {

					const sx = x * 2;
					const sy = y * 2;
					const sx1 = Math.min( sx + 1, srcW - 1 );
					const sy1 = Math.min( sy + 1, srcH - 1 );

					const i00 = ( sy * srcW + sx ) * 4;
					const i10 = ( sy * srcW + sx1 ) * 4;
					const i01 = ( sy1 * srcW + sx ) * 4;
					const i11 = ( sy1 * srcW + sx1 ) * 4;

					const di = ( y * dstW + x ) * 4;
					for ( let c = 0; c < 4; c ++ ) {

						dstData[ di + c ] = ( srcData[ i00 + c ] + srcData[ i10 + c ] + srcData[ i01 + c ] + srcData[ i11 + c ] ) * 0.25;

					}

				}

			}

			const halfData = this._float32ToFloat16( dstData );
			this.device.queue.writeTexture(
				{ texture: this.envTexture, mipLevel: mip },
				halfData,
				{ bytesPerRow: dstW * 8, rowsPerImage: dstH },
				[ dstW, dstH ],
			);

			srcData = dstData;
			srcW = dstW;
			srcH = dstH;

		}

	}

	_float16ToFloat32( h ) {

		const sign = ( h >> 15 ) & 0x1;
		const exponent = ( h >> 10 ) & 0x1F;
		const mantissa = h & 0x3FF;

		if ( exponent === 0 ) {

			// Denormalized or zero
			return ( sign ? - 1 : 1 ) * ( mantissa / 1024 ) * Math.pow( 2, - 14 );

		} else if ( exponent === 31 ) {

			// Inf or NaN
			return mantissa === 0 ? ( sign ? - Infinity : Infinity ) : NaN;

		}

		return ( sign ? - 1 : 1 ) * Math.pow( 2, exponent - 15 ) * ( 1 + mantissa / 1024 );

	}

	// Convert Float32Array (RGBA) to Uint16Array (float16 RGBA)
	_float32ToFloat16( float32Data ) {

		const len = float32Data.length;
		const uint16Data = new Uint16Array( len );
		const buf = new ArrayBuffer( 4 );
		const f32 = new Float32Array( buf );
		const u32 = new Uint32Array( buf );

		for ( let i = 0; i < len; i ++ ) {

			f32[ 0 ] = float32Data[ i ];
			const bits = u32[ 0 ];
			const sign = ( bits >> 16 ) & 0x8000;
			const exponent = ( ( bits >> 23 ) & 0xFF ) - 127 + 15;
			const mantissa = bits & 0x7FFFFF;

			if ( exponent <= 0 ) {

				// Denormalized or zero
				uint16Data[ i ] = sign;

			} else if ( exponent >= 31 ) {

				// Overflow → infinity
				uint16Data[ i ] = sign | 0x7C00;

			} else {

				uint16Data[ i ] = sign | ( exponent << 10 ) | ( mantissa >> 13 );

			}

		}

		return uint16Data;

	}

	/**
	 * Create a bind group with all environment map resources.
	 * Layout: binding 0 = env texture, 1 = sampler, 2 = marginal CDF,
	 *         3 = conditional CDF, 4 = luminance PDF, 5 = CDF sampler
	 */
	createBindGroup( layout ) {

		return this.device.createBindGroup( {
			layout,
			entries: [
				{ binding: 0, resource: this.envTexture.createView() },
				{ binding: 1, resource: this.sampler },
				{ binding: 2, resource: this.marginalCDF.createView() },
				{ binding: 3, resource: this.conditionalCDF.createView() },
				{ binding: 4, resource: this.luminancePDF.createView() },
			],
		} );

	}

}
