const COMPUTE_WORKGROUP_SIZE = 64; // keep in sync with decode.wgsl @workgroup_size

import { setupRayTracePipeline } from './rayTraceSetup.js';
import { setupDisplayPipeline } from './displaySetup.js';
import { Camera, calculateAllBrickLODs } from './camera.js';
import { OrbitController, computeDatasetBounds } from './orbitController.js';
import { initCache } from './cacheManager.js';
import { createLogger } from './logger.js';

const log = createLogger('WebGPU');

const defaultLightingOptions = {
    lightAngle: 25,  // degrees, -90 (west) to +90 (east)
    ambient: 0.5,
    shadowAlphaThreshold: 0.2,
    aoBlend: 0.1,
    aoStrength: 0.45,
    diffuseStrength: 0.8
};

// GPU globals
let camera;
let cameraUniformBuffer;
let device;
let context;
let canvas;
let computePipeline;
let computeBindGroup;
let outVoxelBuffer;
let voxelBufferSize;
let brickCache; // Cache state manager
let cameraController;
let lightingUniformBuffer;
let lightingState = { ...defaultLightingOptions };
let updateLighting = true;  // Mark for initial upload
let activeDataset = null;
let labelColorsBufferRef = null;
let baseLabelColorsData = null;
let currentHighlightLabel = null;

export function getDatasetLabels() {
    return activeDataset?.__cachedData?.transferFunction?.labels ?? [];
}

// ==================== Utility Functions ====================

async function loadShader(url) {
    const response = await fetch(url);
    return await response.text();
}

// ==================== GPU Setup Functions ====================

async function initializeGPU() {
    // Check for WebGPU support
    if (!navigator.gpu) {
        console.error("WebGPU not supported!");
        return false;
    }

    // Get GPU adapter and device
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        console.error("No GPU adapter found. Check your browser and hardware.");
        return false;
    }

    const requiredLimits = {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize
    };

    // Request timestamp query feature for GPU timing inside passes
    const requiredFeatures = [];
    if (adapter.features.has('chromium-experimental-timestamp-query-inside-passes')) {
        requiredFeatures.push('chromium-experimental-timestamp-query-inside-passes');
    } else if (adapter.features.has('timestamp-query')) {
        requiredFeatures.push('timestamp-query');
    }

    device = await adapter.requestDevice({
        requiredLimits,
        requiredFeatures
    });
    console.log("WebGPU device ready!", device);

    // Get canvas and configure context
    canvas = document.getElementById("gpuCanvas");
    context = canvas.getContext("webgpu");

    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
        device: device,
        format: format,
        alphaMode: 'opaque'
    });

    return true;
}

function computeBrickOffsets(dataset, brickLODs) {
    // Compute offsets and totals for all bricks using their individual LODs
    let streamOffset = 0;
    let paletteOffset = 0;
    let outputOffset = 0;

    let totalU32Count = 0;
    let totalPaletteEntries = 0;
    let totalVoxels = 0;

    for (let i = 0; i < dataset.bricks.length; i++) {
        const b = dataset.bricks[i];
        const lod = brickLODs[i];  // Use individual brick LOD

        // streamOffset in u32 array indices (not bytes)
        b.streamOffset = streamOffset;
        totalU32Count += Math.ceil(b.encodedData.length / 4);

        // Debug logging removed after validation

        streamOffset += Math.ceil(b.encodedData.length / 4);

        b.paletteOffset = paletteOffset;
        totalPaletteEntries += b.palette.length;
        paletteOffset += b.palette.length;

        b.outputOffset = outputOffset;
        b.targetLOD = lod;  // Set per-brick LOD
        // Calculate voxels for this brick: 2^(3*lod) since each LOD halves per dimension
        // Math.floor ensures integer (though powers of 2 should already be integers)
        const voxelsPerBrick = Math.floor(Math.pow(2, 3 * lod));
        outputOffset += voxelsPerBrick;
        totalVoxels += voxelsPerBrick;
    }

    return {
        totalU32Count,
        totalPaletteEntries,
        totalVoxels: Math.floor(totalVoxels)  // Ensure final sum is integer for WebGPU
    };
}

function createVoxelBuffers(totalVoxels) {
    // Buffer size = totalVoxels * 4 bytes (each voxel is a u32 storing packed RGBA8)
    // Math.floor ensures integer value required by WebGPU createBuffer
    voxelBufferSize = Math.floor(totalVoxels * 4);

    // Validate buffer size
    if (!Number.isFinite(voxelBufferSize) || voxelBufferSize <= 0) {
        console.error(`Invalid voxelBufferSize: ${voxelBufferSize}, totalVoxels: ${totalVoxels}`);
        throw new Error(`Invalid buffer size calculated: ${voxelBufferSize}`);
    }

    // Storage buffer for compute shader output
    outVoxelBuffer = device.createBuffer({
        size: voxelBufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });
}

function writeLightingUniform(options) {
    if (!lightingUniformBuffer || !device) return;

    // Only update if options changed
    let changed = false;
    for (const key in options) {
        if (lightingState[key] !== options[key]) {
            changed = true;
            break;
        }
    }

    if (!changed && !updateLighting) {
        return;  // No changes, skip GPU write
    }

    lightingState = { ...lightingState, ...options };

    // Compute light direction from angle (west to east arc)
    const angleRad = (lightingState.lightAngle * Math.PI) / 180;
    const lightDir = [
        Math.sin(angleRad),   // X: -1 (west) to +1 (east)
        0.6,                   // Y: fixed elevation
        -Math.cos(angleRad)   // Z: completes the arc
    ];

    const ambient = lightingState.ambient;
    const diffuseStrength = lightingState.diffuseStrength;
    const lightingData = new Float32Array([
        lightDir[0], lightDir[1], lightDir[2], 0.0,
        ambient, 0.0, 0.0, 0.0,
        0.0, diffuseStrength, 0.0, 0.0
    ]);
    device.queue.writeBuffer(lightingUniformBuffer, 0, lightingData);
    updateLighting = false;
}

function buildStreamAndPaletteData(dataset, totalU32Count, totalPaletteEntries) {
    const nibbleStreamData = new Uint32Array(totalU32Count);
    const totalPaletteU32s = totalPaletteEntries * 2;
    const paletteData = new Uint32Array(totalPaletteU32s);

    let writeOffset = 0;
    let pOff = 0;

    for (const b of dataset.bricks) {
        const brickIdx = dataset.bricks.indexOf(b);
        const expectedOffset = b.streamOffset;

        // Verify offset matches
        if (writeOffset !== expectedOffset) {
            console.error(`OFFSET MISMATCH for brick ${brickIdx}: expected=${expectedOffset} u32s, actual=${writeOffset} u32s`);
        }

        // Populate nibble stream (big-endian packing for WGSL bit extraction)
        // Preserve the exact slice for this brick (important when encodedData is a view)
        const src = new Uint8Array(b.encodedData.buffer, b.encodedData.byteOffset, b.encodedData.byteLength);
        for (let i = 0; i < src.length; i += 4) {
            nibbleStreamData[writeOffset++] =
                ((src[i + 0] ?? 0) << 24) |  // byte 0 at bits 24-31
                ((src[i + 1] ?? 0) << 16) |  // byte 1 at bits 16-23
                ((src[i + 2] ?? 0) << 8) |  // byte 2 at bits 8-15
                (src[i + 3] ?? 0);           // byte 3 at bits 0-7
        }

        // console.log(`Brick ${dataset.bricks.indexOf(b)} encodedData:`, Array.from(src).map(x => x.toString(2).padStart(8, '0')));
        // const start = b.streamOffset;
        // const end = start + Math.ceil(src.length / 4);
        // console.log(`Brick ${dataset.bricks.indexOf(b)} nibbleStreamData:`, Array.from(nibbleStreamData.slice(start, end)).map(x => x.toString(2).padStart(32, '0')));

        // Populate palette - split each u64 into two u32 values
        for (const v of b.palette) {
            const bigVal = BigInt(v);
            const low = Number(bigVal & 0xFFFFFFFFn);
            const high = Number(bigVal >> 32n);
            paletteData[pOff++] = low;
            paletteData[pOff++] = high;
        }
    }

    return { nibbleStreamData, paletteData };
}

// Cached version: reuse stream/palette data if dataset already processed
function populateStreamAndPaletteData(dataset, totalU32Count, totalPaletteEntries) {
    if (!dataset.__cachedData) {
        dataset.__cachedData = {};
    }

    if (!dataset.__cachedData.streamPalette) {
        dataset.__cachedData.streamPalette = buildStreamAndPaletteData(dataset, totalU32Count, totalPaletteEntries);
        log.log(`Cached stream/palette data (${(dataset.__cachedData.streamPalette.nibbleStreamData.byteLength / 1024 / 1024).toFixed(2)} MB)`);
    }

    return dataset.__cachedData.streamPalette;
}

function createBrickInfoBuffer(dataset) {
    const brickCount = dataset.bricks.length;

    // Build Morton-to-index lookup table
    // The brick arrays are indexed 0..brickCount-1 (sparse)
    // But the shader needs to look up bricks by Morton code (0..511 for 8x8x8)
    // This table maps: mortonCode -> actualBrickIndex (or 0xFFFFFFFF if no brick exists)
    const maxMortonCode = brickCount > 0 ? Math.max(...dataset.bricks.map(b => b.ID)) : 0;
    const lookupSize = maxMortonCode + 1;
    const mortonLookup = new Uint32Array(lookupSize).fill(0xFFFFFFFF);  // 0xFFFFFFFF = no brick

    for (let i = 0; i < brickCount; i++) {
        const mortonCode = dataset.bricks[i].ID;
        mortonLookup[mortonCode] = i;  // Map Morton code to brick array index
    }

    // Static buffer: aligned to 32 bytes per brick (8 u32s)
    // Fields: nSymbols, paletteSize, streamOffset, paletteOffset, flags, encodedSizeBytes, mortonId, pad2
    const STATIC_STRIDE_U32 = 8;
    const staticBrickData = new Uint32Array(brickCount * STATIC_STRIDE_U32);

    // Dynamic buffer: 4 u32s per brick (16 bytes)
    // Fields: outputOffset (u32), targetLOD (u32), lodScale (f32), pad (u32)
    // Must match stride used in cacheProcessing.js and renderLoop.js
    const DYNAMIC_STRIDE_U32 = 4;
    const dynamicBrickData = new Uint32Array(brickCount * DYNAMIC_STRIDE_U32);
    const dynamicBrickDataF32 = new Float32Array(dynamicBrickData.buffer);  // Float32 view for lodScale

    for (let i = 0; i < brickCount; i++) {
        const b = dataset.bricks[i];

        // Check if brick is empty (all palette entries are 0)
        let isEmpty = true;
        for (const paletteValue of b.palette) {
            if (paletteValue !== 0n) {
                isEmpty = false;
                break;
            }
        }

        // Static data (write once) - aligned to 32 bytes
        const staticBase = i * STATIC_STRIDE_U32;
        staticBrickData[staticBase + 0] = b.nSymbols;     // Number of RANS symbols
        staticBrickData[staticBase + 1] = b.paletteSize;
        staticBrickData[staticBase + 2] = b.streamOffset;
        staticBrickData[staticBase + 3] = b.paletteOffset;
        staticBrickData[staticBase + 4] = isEmpty ? 1 : 0;  // flags: bit 0 = isEmpty
        staticBrickData[staticBase + 5] = b.encodedSize;    // encoded size in bytes (bounds check)
        staticBrickData[staticBase + 6] = b.ID;             // mortonId (for debug validation in shader)
        staticBrickData[staticBase + 7] = 0;  // pad2

        // Debug output trimmed now that validation is done

        // Dynamic data (will be updated per frame after cache allocation)
        const dynamicBase = i * DYNAMIC_STRIDE_U32;
        const lod = b.targetLOD || 0;
        // LOD 0 = coarsest (1 voxel), LOD 6 = finest (64 voxels)
        const lodSize = Math.pow(2, lod);
        const lodScale = lodSize / dataset.header.brickSize;

        // Initialize with sentinel values - actual offsets will be filled after cache allocation
        dynamicBrickData[dynamicBase + 0] = 0xFFFFFFFF;  // outputOffset - invalid until cached
        dynamicBrickData[dynamicBase + 1] = lod;         // targetLOD
        dynamicBrickDataF32[dynamicBase + 2] = lodScale;  // lodScale (as float)
        dynamicBrickData[dynamicBase + 3] = 0;           // pad
    }

    const staticBricksBuffer = device.createBuffer({
        size: staticBrickData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(staticBricksBuffer, 0, staticBrickData);

    const dynamicBricksBuffer = device.createBuffer({
        size: dynamicBrickData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(dynamicBricksBuffer, 0, dynamicBrickData);

    // Create Morton lookup buffer
    const mortonLookupBuffer = device.createBuffer({
        size: mortonLookup.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(mortonLookupBuffer, 0, mortonLookup);

    return { staticBricksBuffer, dynamicBricksBuffer, mortonLookupBuffer };
}

function createGPUBuffers(nibbleStreamData, paletteData) {
    const nibbleStreamBuffer = device.createBuffer({
        size: nibbleStreamData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(nibbleStreamBuffer, 0, nibbleStreamData);

    const paletteBuffer = device.createBuffer({
        size: paletteData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(paletteBuffer, 0, paletteData);

    return { nibbleStreamBuffer, paletteBuffer };
}

function buildTransferFunctionBuffers(dataset) {
    // Automatically extract all unique labels from dataset brick palettes
    const uniqueLabels = new Set();

    for (const brick of dataset.bricks) {
        for (const label of brick.palette) {
            if (label !== 0n) {  // Skip empty label (0)
                uniqueLabels.add(label);
            }
        }
    }

    log.log(`Transfer function: Found ${uniqueLabels.size} unique labels in dataset`);

    // Generate transfer function entries with varied colors and opacities
    const transferFunction = [];
    const labels = Array.from(uniqueLabels).sort((a, b) => a < b ? -1 : 1);

    // Create label->RGBA map for validation
    const transferFunctionMap = new Map();

    // Generate colors using different strategies based on label count
    for (let i = 0; i < labels.length; i++) {
        const label = labels[i];

        // Strategy: Spread colors across hue spectrum with midrange brightness so lighting remains visible
        // Use narrower saturation/lightness to avoid neon-bright colors that overpower shading
        const hue = (i * 137.5) % 360;              // Golden angle for good distribution
        const saturation = 0.55 + (i % 3) * 0.08;   // 0.55–0.71
        const lightness = 0.35 + (i % 4) * 0.05;    // 0.35–0.50
        const opacity = 0.25 + (i % 5) * 0.08;      // 0.25–0.57

        // Convert HSL to RGB
        // Clamp channels to keep headroom for lighting modulation
        const { r: rIn, g: gIn, b: bIn } = hslToRgb(hue, saturation, lightness);
        const r = Math.min(Math.floor(rIn * 0.92), 255);
        const g = Math.min(Math.floor(gIn * 0.92), 255);
        const b = Math.min(Math.floor(bIn * 0.92), 255);
        const alpha = Math.floor(opacity * 255);

        const rgba = (r << 0) | (g << 8) | (b << 16) | (alpha << 24);
        transferFunction.push({ label, rgba });
        transferFunctionMap.set(label, rgba);  // Store for validation
    }

    // If no unique labels found, use dummy entry
    const numEntries = Math.max(transferFunction.length, 1);

    // labelKeys: array of vec2<u32> (low 32 bits, high 32 bits per label)
    const labelKeysData = new Uint32Array(numEntries * 2);
    const labelColorsData = new Uint32Array(numEntries);

    if (transferFunction.length === 0) {
        // No mappings: use dummy entry (label 0 -> transparent)
        labelKeysData[0] = 0;
        labelKeysData[1] = 0;
        labelColorsData[0] = 0x00000000;
    } else {
        // Populate from transfer function
        transferFunction.forEach((entry, i) => {
            labelKeysData[i * 2] = Number(entry.label & 0xFFFFFFFFn);     // low u32
            labelKeysData[i * 2 + 1] = Number(entry.label >> 32n);        // high u32
            labelColorsData[i] = entry.rgba;
        });
    }

    const baseLabelColorsData = new Uint32Array(labelColorsData);

    return { labelKeysData, labelColorsData, baseLabelColorsData, transferFunctionMap, labels };
}

// Cached version: reuse transfer function data if dataset already processed
function createTransferFunctionBuffers(dataset) {
    if (!dataset.__cachedData) {
        dataset.__cachedData = {};
    }

    if (!dataset.__cachedData.transferFunction) {
        dataset.__cachedData.transferFunction = buildTransferFunctionBuffers(dataset);
    }

    const { labelKeysData, labelColorsData, baseLabelColorsData: baseColorsCopy, transferFunctionMap, labels } = dataset.__cachedData.transferFunction;
    const labelKeysBuffer = device.createBuffer({
        size: labelKeysData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(labelKeysBuffer, 0, labelKeysData);

    const labelColorsBuffer = device.createBuffer({
        size: labelColorsData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(labelColorsBuffer, 0, labelColorsData);

    // Persist references for UI-driven updates
    baseLabelColorsData = baseColorsCopy;
    labelColorsBufferRef = labelColorsBuffer;
    return { labelKeysBuffer, labelColorsBuffer, transferFunctionMap, baseLabelColorsData: baseColorsCopy };
}

// HSL to RGB conversion helper
function hslToRgb(h, s, l) {
    const c = (1 - Math.abs(2 * l - 1)) * s;
    const x = c * (1 - Math.abs((h / 60) % 2 - 1));
    const m = l - c / 2;

    let r, g, b;
    if (h < 60) {
        [r, g, b] = [c, x, 0];
    } else if (h < 120) {
        [r, g, b] = [x, c, 0];
    } else if (h < 180) {
        [r, g, b] = [0, c, x];
    } else if (h < 240) {
        [r, g, b] = [0, x, c];
    } else if (h < 300) {
        [r, g, b] = [x, 0, c];
    } else {
        [r, g, b] = [c, 0, x];
    }

    return {
        r: Math.round((r + m) * 255),
        g: Math.round((g + m) * 255),
        b: Math.round((b + m) * 255)
    };
}

async function setupComputePipeline(staticBricksBuffer, dynamicBricksBuffer, nibbleStreamBuffer, paletteBuffer, workQueueBuffer, workCountBuffer, labelKeysBuffer, labelColorsBuffer) {
    const computeCode = await loadShader("shaders/decode.wgsl");
    const computeModule = device.createShaderModule({ code: computeCode });

    computePipeline = device.createComputePipeline({
        layout: "auto",
        compute: { module: computeModule, entryPoint: "decodeBrick" }
    });

    computeBindGroup = device.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: staticBricksBuffer } },
            { binding: 1, resource: { buffer: dynamicBricksBuffer } },
            { binding: 2, resource: { buffer: nibbleStreamBuffer } },
            { binding: 3, resource: { buffer: paletteBuffer } },
            { binding: 4, resource: { buffer: outVoxelBuffer } },
            { binding: 5, resource: { buffer: workQueueBuffer } },
            { binding: 6, resource: { buffer: workCountBuffer } },
            { binding: 7, resource: { buffer: labelKeysBuffer } },
            { binding: 8, resource: { buffer: labelColorsBuffer } }
        ]
    });
}

export async function initWebGPU(dataset) {
    // Initialize GPU hardware
    if (!await initializeGPU()) {
        return;
    }

    activeDataset = dataset;

    // Initialize camera system centered on the dataset
    const bounds = computeDatasetBounds(dataset);
    const diag = Math.hypot(bounds.size.x, bounds.size.y, bounds.size.z);
    const boundingRadius = diag * 0.5;
    // Keep camera outside the dataset: pad by 10% of the radius
    const minDistance = boundingRadius * 1.2;
    const maxDistance = boundingRadius * 15.0;
    // Start a bit further back to keep the initial view well outside the volume
    const initialRadius = boundingRadius * 3.0;
    const initialYaw = Math.PI * 0.25;   // 45 degrees
    const initialPitch = Math.PI * 0.15;

    camera = new Camera({ x: bounds.center.x, y: bounds.center.y, z: bounds.center.z + initialRadius }, Math.PI / 3);
    camera.lookAt(bounds.center);

    cameraController = new OrbitController({
        canvas,
        camera,
        target: bounds.center,
        minDistance,
        maxDistance,
        maxPitch: Math.PI / 3,
        rotateSpeed: 0.0035,
        zoomSpeed: 0.2,
        initialYaw,
        initialPitch,
        initialRadius
    });

    // Create camera uniform buffer (64 bytes: position/tanHalfFov + basis)
    cameraUniformBuffer = device.createBuffer({
        size: 64,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(cameraUniformBuffer, 0, camera.getUniformData());
    camera.markUploaded();
    if (cameraController && cameraController.markUploaded) {
        cameraController.markUploaded();
    }

    // Calculate maximum LOD from brick resolution (LOD = log2(brickSize) => full-res decode)
    const maxLOD = Math.floor(Math.log2(dataset.header.brickSize));

    // Calculate LOD for each brick BEFORE decoding using the true maximum
    const brickLODsResult = await calculateAllBrickLODs(
        dataset.bricks,
        camera,
        Math.min(canvas.width, canvas.height),  // Use smaller dimension for conservative LOD
        maxLOD,                                  // Decode up to full brick resolution
        dataset.header.brickSize                 // Pass brick size from dataset header
    );

    // Extract lodArray from result (calculateAllBrickLODs returns {lodArray, offsetArray})
    const brickLODs = brickLODsResult.lodArray || brickLODsResult;

    // Log LOD distribution across all bricks
    const lodDistribution = new Array(maxLOD + 1).fill(0);
    let totalVoxelsNeeded = 0;
    for (let i = 0; i < brickLODs.length; i++) {
        const lod = brickLODs[i];
        lodDistribution[lod]++;
        totalVoxelsNeeded += Math.pow(2, 3 * lod);
    }

    let lodDistributionMsg = "Brick LOD Distribution (0=coarsest, " + maxLOD + "=finest):\n";
    for (let lod = 0; lod <= maxLOD; lod++) {
        const count = lodDistribution[lod];
        const voxelsPerBrick = Math.pow(2, 3 * lod);
        const totalForLod = count * voxelsPerBrick;
        const percentage = (count / brickLODs.length * 100).toFixed(1);
        lodDistributionMsg += `  LOD ${lod}: ${count} bricks (${percentage}%) - ${totalForLod.toLocaleString()} voxels total\n`;
    }
    lodDistributionMsg += `Total voxels needed: ${totalVoxelsNeeded.toLocaleString()}`;
    log.log(lodDistributionMsg);

    // Define fixed cache pool size (independent of current visibility)
    // Allocate either 2GB or the maximum device supported size for a storage buffer
    const MAX_CACHE_BYTES = 2 * 1024 * 1024 * 1024; // 2GB
    const deviceMaxStorage = device.limits.maxStorageBufferBindingSize;
    const cacheBufferBytes = Math.min(MAX_CACHE_BYTES, deviceMaxStorage);

    // Maximum voxels per brick is at full resolution (LOD maxLOD), which is brickSize^3
    const maxVoxelsPerBrick = Math.floor(Math.pow(dataset.header.brickSize, 3)); // 262,144 voxels at 64³
    const cachePoolVoxels = Math.floor(cacheBufferBytes / 4); // Each voxel is 4 bytes (u32)
    const maxBricksInCache = Math.floor(cachePoolVoxels / maxVoxelsPerBrick); // Max bricks if all at highest LOD

    log.log(`brickSize=${dataset.header.brickSize}, maxLOD=${maxLOD}, maxVoxelsPerBrick=${maxVoxelsPerBrick}`);
    log.log(`Fixed cache pool: ${cachePoolVoxels} voxels (${maxBricksInCache} bricks at full LOD)`);
    log.log(`Cache buffer size: ${cachePoolVoxels * 4} bytes (${(cachePoolVoxels * 4 / 1024 / 1024).toFixed(2)} MB)`);

    // Initialize brick cache manager with fixed pool
    brickCache = initCache(cachePoolVoxels, maxLOD);
    log.log("Cache stats:", brickCache.getStats());

    // Create voxel output buffers with fixed size
    createVoxelBuffers(cachePoolVoxels);

    // Derive brick grid dimensions from brick positions
    let maxBx = 0, maxBy = 0, maxBz = 0;
    for (const b of dataset.bricks) {
        maxBx = Math.max(maxBx, Math.floor(b.position.x / dataset.header.brickSize));
        maxBy = Math.max(maxBy, Math.floor(b.position.y / dataset.header.brickSize));
        maxBz = Math.max(maxBz, Math.floor(b.position.z / dataset.header.brickSize));
    }
    const bricksPerAxis = Math.max(maxBx, maxBy, maxBz) + 1; // assume cubic grid coverage
    const gridSize = bricksPerAxis * dataset.header.brickSize;

    // Compute initial brick offsets for first frame (still needed for encoded data layout)
    const offsets = computeBrickOffsets(dataset, brickLODs);

    // Populate stream and palette data
    const { nibbleStreamData, paletteData } = populateStreamAndPaletteData(
        dataset,
        offsets.totalU32Count,
        offsets.totalPaletteEntries
    );

    // Create brick info buffers (static and dynamic)
    const brickInfoBuffers = createBrickInfoBuffer(dataset);
    staticBricksBuffer = brickInfoBuffers.staticBricksBuffer;
    dynamicBricksBuffer = brickInfoBuffers.dynamicBricksBuffer;
    const mortonLookupBuffer = brickInfoBuffers.mortonLookupBuffer;

    // Create stream and palette GPU buffers
    const { nibbleStreamBuffer, paletteBuffer } = createGPUBuffers(
        nibbleStreamData,
        paletteData
    );

    // Create transfer function buffers (automatically extracts labels from dataset)
    const { labelKeysBuffer, labelColorsBuffer, transferFunctionMap } = createTransferFunctionBuffers(dataset);

    // Create work queue buffer (max capacity = total brick count)
    workQueueBuffer = device.createBuffer({
        size: dataset.bricks.length * 4,  // u32 per brick index
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });

    // Create work count buffer (single u32)
    workCountBuffer = device.createBuffer({
        size: 4,  // one u32
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    // Create brick request buffer for GPU-driven visibility (one u32 per brick)
    // 0 = not accessed, 1 = accessed (marked during ray marching)
    const brickRequestBuffer = device.createBuffer({
        size: dataset.bricks.length * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });

    // Initialize request buffer to 0 (all bricks initially not accessed)
    const initialRequests = new Uint32Array(dataset.bricks.length).fill(0);
    device.queue.writeBuffer(brickRequestBuffer, 0, initialRequests);

    // Create readback buffer for CPU processing of GPU requests
    const brickRequestReadback = device.createBuffer({
        size: dataset.bricks.length * 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    // Setup compute pipeline
    await setupComputePipeline(staticBricksBuffer, dynamicBricksBuffer, nibbleStreamBuffer, paletteBuffer, workQueueBuffer, workCountBuffer, labelKeysBuffer, labelColorsBuffer);

    // Scene uniform for ray tracing (brickSize)
    const sceneUniformBuffer = device.createBuffer({
        size: 32, // gridSize, brickSize, bricksPerAxis, padding + invGridSize, invBrickSize, padding
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    const sceneData = new Float32Array([
        gridSize,                      // u32 -> f32
        dataset.header.brickSize,      // u32 -> f32
        bricksPerAxis,                 // u32 -> f32
        0,
        1.0 / gridSize,
        1.0 / dataset.header.brickSize,
        0,
        0
    ]);
    device.queue.writeBuffer(sceneUniformBuffer, 0, sceneData);

    // Morton codes are now computed on-the-fly in WGSL (bit interleaving is cheap ALU)
    // This saves massive memory (264MB for 256³ grids) and avoids memory bandwidth
    // ALU cost (~20 ops) is negligible vs ray marching

    // Lighting/AO defaults for ray tracing (can be updated via UI)
    const lightingOptions = { ...defaultLightingOptions };

    // Setup ray tracing pipeline (needs decoded voxels + brick metadata + scene info + camera)
    const rayMarchState = await setupRayTracePipeline(
        device,
        outVoxelBuffer,
        dynamicBricksBuffer,
        staticBricksBuffer,
        sceneUniformBuffer,
        cameraUniformBuffer,
        voxelBufferSize,
        brickRequestBuffer,
        paletteBuffer,       // Palette for LOD 0 bricks
        lightingOptions      // Lighting + AO params
    );

    // Keep a handle to the lighting uniform for live updates
    lightingUniformBuffer = rayMarchState.lightingUniformBuffer;
    lightingState = { ...lightingOptions };
    writeLightingUniform({});

    // Setup display pipeline to render raytraced output to canvas
    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    const displayState = await setupDisplayPipeline(
        device,
        rayMarchState.rayTraceOutputBuffer,
        rayMarchState.screenUniformBuffer,
        canvasFormat
    );

    return {
        computePipeline,
        computeBindGroup,
        outVoxelBuffer,
        voxelBufferSize,
        device,
        COMPUTE_WORKGROUP_SIZE,
        camera,
        cameraController,
        cameraUniformBuffer,
        calculateAllBrickLODs,
        brickCache,
        staticBricksBuffer,
        dynamicBricksBuffer,
        mortonLookupBuffer,      // Morton code to brick index lookup
        workQueueBuffer,
        workCountBuffer,
        brickRequestBuffer,      // GPU request buffer
        brickRequestReadback,    // CPU readback buffer
        maxLOD,                  // Maximum LOD for reference
        initialBrickLODs: brickLODs,  // Return initial LOD array
        transferFunctionMap,     // Transfer function for validation
        canvas,  // Add canvas for access in renderLoop
        ...rayMarchState,
        ...displayState,
        context  // Add canvas context for rendering
    };
}

let staticBricksBuffer;
let dynamicBricksBuffer;

let workQueueBuffer;
let workCountBuffer;

export function getGPUState() {
    return {
        device,
        computePipeline,
        computeBindGroup,
        outVoxelBuffer,
        voxelBufferSize,
        COMPUTE_WORKGROUP_SIZE,
        camera,
        cameraUniformBuffer,
        calculateAllBrickLODs,
        computeBrickOffsets,
        staticBricksBuffer,
        dynamicBricksBuffer,
        brickCache,
        workQueueBuffer,
        workCountBuffer,
        aoHistoryBuffer,
        lightingUniformBuffer
    };
}

export function updateLightingOptions(options) {
    writeLightingUniform(options);
}

function rebuildLabelColors(hiddenSet, highlightLabel) {
    const tf = activeDataset.__cachedData?.transferFunction;
    if (!tf || !tf.labels) return null;

    const updated = new Uint32Array(baseLabelColorsData);

    tf.labels.forEach((label, idx) => {
        const isHidden = hiddenSet?.has(label);
        if (isHidden) {
            updated[idx] = 0x00000000;
            return;
        }

        if (highlightLabel !== null && label === highlightLabel) {
            const rgba = baseLabelColorsData[idx];
            // brighten RGB by ~80% with clamp, keep alpha
            const r = Math.min(255, ((rgba >> 0) & 0xFF) * 1.8);
            const g = Math.min(255, ((rgba >> 8) & 0xFF) * 1.8);
            const b = Math.min(255, ((rgba >> 16) & 0xFF) * 1.8);
            const a = (rgba >> 24) & 0xFF;
            updated[idx] = (Math.floor(r) << 0) | (Math.floor(g) << 8) | (Math.floor(b) << 16) | (a << 24);
        }
    });

    return updated;
}

function writeLabelColors(hiddenSet, highlightLabel) {
    if (!device || !labelColorsBufferRef || !baseLabelColorsData || !activeDataset) return { applied: false, reason: "GPU not ready" };
    const tf = activeDataset.__cachedData?.transferFunction;
    if (!tf || !tf.labels) return { applied: false, reason: "Transfer function missing" };

    const updatedColors = rebuildLabelColors(hiddenSet, highlightLabel);
    if (!updatedColors) return { applied: false, reason: "Transfer function missing" };

    device.queue.writeBuffer(labelColorsBufferRef, 0, updatedColors);
    tf.labelColorsData = updatedColors;
    return { applied: true, hiddenCount: hiddenSet?.size ?? 0, total: tf.labels.length };
}

export function applyLabelVisibility(hiddenLabelList) {
    if (!activeDataset) {
        return { applied: false, reason: "GPU not ready" };
    }

    const hiddenSet = new Set(hiddenLabelList.map((l) => BigInt(l)));
    activeDataset.hiddenLabels = hiddenSet;
    return writeLabelColors(hiddenSet, currentHighlightLabel);
}

export function applyLabelHover(labelOrNull) {
    if (!activeDataset) return;
    currentHighlightLabel = labelOrNull === null ? null : BigInt(labelOrNull);
    const hiddenSet = activeDataset.hiddenLabels || new Set();
    writeLabelColors(hiddenSet, currentHighlightLabel);
}
