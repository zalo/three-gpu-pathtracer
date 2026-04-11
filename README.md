# three-gpu-pathtracer

Path tracing renderer for [three.js](https://threejs.org/) with two backends:

- **WebGL** (original) -- uses [three-mesh-bvh](https://github.com/gkjohnson/three-mesh-bvh) with GLSL shaders
- **WebGPU** (new) -- uses [Slang](https://shader-slang.com/) shaders compiled to WGSL, [tinybvh](https://github.com/jbikker/tinybvh) for BVH construction via WASM, and the [OpenPBR](https://github.com/AcademySoftwareFoundation/OpenPBR) BSDF

## Live Demos

| Demo | Description |
|------|-------------|
| [WebGPU Viewer](https://zalo.github.io/three-gpu-pathtracer/webgpu-viewer.html) | Drag-and-drop GLTF/GLB viewer with Slang path tracer |
| [WebGPU Basic](https://zalo.github.io/three-gpu-pathtracer/webgpu-basic.html) | Minimal three.js integration example |
| [WebGL Viewer](https://zalo.github.io/three-gpu-pathtracer/viewer.html) | Original WebGL drag-and-drop viewer |
| [Material Gallery](https://zalo.github.io/three-gpu-pathtracer/index.html) | Showcase of physically-based materials |

## Architecture

```
example/
  webgpu-viewer.js     Standalone WebGPU viewer (drag-and-drop GLTF)
  webgpu-basic.js      Minimal three.js integration via PathTracingRenderer
  libs/
    tinybvh.js/.wasm   BVH construction (Emscripten build of tinybvh)

src/
  webgpu/
    WebGPUPathTracer.js    Low-level WebGPU orchestrator (buffers, pipelines, dispatch)
    PathTracingRenderer.js Drop-in three.js renderer (scene -> GPU, camera sync, accumulation)
    SceneProcessor.js      Extracts geometry, materials, textures from three.js scene graph
    PipelineManager.js     Compiles Slang shaders to WGSL via vite-slang
    EnvironmentMap.js      HDR environment loading + importance sampling CDF
    GPUBufferManager.js    WebGPU buffer lifecycle management
    OpenPBRLUTData.js      Pre-baked OpenPBR lookup tables

  shaders/
    pathtracer.slang       Main path tracing compute kernel (NEE + MIS + OpenPBR)
    bvh_traversal.slang    BVH traversal (single-level flat + two-level TLAS/BLAS)
    common.slang           Shared types: BVHNode, Ray, HitInfo, Camera, Material, RNG
    environment.slang      Environment map sampling + importance-weighted direction sampling
    openpbr_bridge.slang   OpenPBR BSDF integration (texture LUT mode for WebGPU)
    accumulate.slang       Progressive accumulation + ACES tone mapping
    display.slang          Final blit to screen

wasm/
  tinybvh_bindings.cpp   Emscripten bindings for tinybvh (BVH_GPU + TLAS/BLAS scene builder)
  CMakeLists.txt         Build config (requires Emscripten SDK)
  build.sh               One-step WASM build script

third_party/
  tinybvh/               BVH construction library (Aila & Laine GPU format)
  openpbr-bsdf/          OpenPBR BSDF reference implementation
  vite-slang/            Vite plugin: compiles .slang -> WGSL at dev/build time
  slangpy/               Slang Python bindings (for tooling)
```

## How it works

### Rendering pipeline

Each frame dispatches three compute/render passes:

1. **Path trace** -- `pathtracer.slang` traces rays through the scene. Each thread handles one pixel, tracing `sppPerDispatch` samples with up to `maxBounces` bounces. Uses Next Event Estimation (NEE) with MIS for direct environment lighting, and OpenPBR BSDF for material sampling.

2. **Accumulate** -- `accumulate.slang` blends the new samples into a running average and applies ACES tone mapping.

3. **Display** -- `display.slang` blits the tone-mapped result to the canvas.

### BVH acceleration

Geometry is accelerated with [tinybvh](https://github.com/jbikker/tinybvh) compiled to WebAssembly:

- **Single-level (flat)** -- all triangles in one BVH. Simple, fast for static scenes.
- **Two-level (TLAS/BLAS)** -- one BLAS per mesh, TLAS over instances. Enables future support for dynamic/instanced scenes. The BLAS nodes are concatenated into a single GPU buffer with per-instance offset metadata.

Both use the Aila & Laine 64-byte node format, traversed in compute shaders with a manual stack.

### Materials

The OpenPBR BSDF handles:
- Metallic/dielectric base layer (GGX microfacet)
- Specular reflection with Fresnel
- Transmission/refraction (glass, liquids)
- Clear coat
- Emission

Shadow rays trace through transmissive surfaces (up to `maxShadowBounces`), accumulating base-color tint for colored glass shadows and caustics.

## Quick start

### Using PathTracingRenderer (three.js drop-in)

```js
import { PathTracingRenderer } from './src/webgpu/PathTracingRenderer.js';

const renderer = new PathTracingRenderer({ canvas });
renderer.setSize(window.innerWidth, window.innerHeight);

// Load a scene with three.js as usual
const scene = new THREE.Scene();
scene.add(gltf.scene);

// Set environment
renderer.environmentURL = 'environment.hdr';

function animate() {
    renderer.render(scene, camera);
    requestAnimationFrame(animate);
}
animate();
```

### Using WebGPUPathTracer (low-level)

```js
import { WebGPUPathTracer } from './src/webgpu/WebGPUPathTracer.js';
import { SceneProcessor } from './src/webgpu/SceneProcessor.js';

const pt = new WebGPUPathTracer();
pt.enableTLAS = true;
pt.maxBounces = 8;
pt.maxShadowBounces = 8;
pt.sppPerDispatch = 4;

await pt.init(canvas);
await pt.loadEnvironment('environment.hdr');

const processor = new SceneProcessor();
await processor.init(tinybvhModule);
const sceneData = await processor.process(threeScene);
pt.setScene(sceneData);

function animate() {
    pt.updateCamera(cameraData);
    pt.renderSample();
    requestAnimationFrame(animate);
}
```

### Configurable parameters

| Property | Default | Description |
|----------|---------|-------------|
| `maxBounces` | 8 | Maximum path depth |
| `maxShadowBounces` | 8 | Maximum shadow ray bounces through transmissive surfaces |
| `sppPerDispatch` | 4 | Samples per pixel per compute dispatch |
| `enableTLAS` | false | Use two-level acceleration structure |

Mobile defaults: `sppPerDispatch=1`, `maxBounces=4`, `maxShadowBounces=4`.

## Building the WASM module

Requires the [Emscripten SDK](https://emscripten.org/docs/getting_started/downloads.html):

```bash
source ~/emsdk/emsdk_env.sh
bash wasm/build.sh
```

## Development

```bash
npm install
npx vite --host        # dev server with hot reload + Slang compilation
npx vite build         # production build to dist/
```

## Credits

- Original three-gpu-pathtracer by [Garrett Johnson](https://github.com/gkjohnson)
- [tinybvh](https://github.com/jbikker/tinybvh) by Jacco Bikker
- [OpenPBR](https://github.com/AcademySoftwareFoundation/OpenPBR) by Academy Software Foundation
- [Slang](https://shader-slang.com/) by Shader Slang team
- Environment maps from [Poly Haven](https://polyhaven.com/)

## License

MIT
