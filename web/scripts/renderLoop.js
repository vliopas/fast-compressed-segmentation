// ==================== Imports ====================

import { createLogger } from './logger.js';
import { BufferPoolManager } from './bufferManager.js';
import { checkCameraMovement } from './culling.js';
import { runComputePass, runRayTracePass, runDisplayPass, runClearRayOutputPass } from './gpuPasses.js';
import {
    initializeCache as initCacheInternal,
    processCache,
    updateBrickInfoBuffer,
    updateWorkQueue
} from './cacheProcessing.js';

const log = createLogger('RenderLoop');

// ==================== Configuration Constants ====================

/**
 * Render loop configuration
 */
export const RENDER_CONFIG = {
    LOD_UPDATE_INTERVAL: 1,        // Update LODs every 1 frame to keep near-camera LODs fresh
    FRAME_SAMPLE_COUNT: 60,         // Number of frames to average for performance stats
    PERF_UPDATE_INTERVAL: 10        // Update performance display every N frames
};

// ==================== State Management ====================

/**
 * Centralized render loop state
 */
class RenderLoopState {
    constructor() {
        // Core references
        this.gpuState = null;
        this.dataset = null;
        this.maxLOD = 0;
        this.cameraController = null;

        // Buffer management
        this.bufferManager = new BufferPoolManager();

        // Readback state
        this.isRequestReadbackMapped = false;
        this.pendingGPURequests = null;
        this.gpuRequestsReady = false;

        // LOD update state
        this.lodUpdateInProgress = false;
        this.framesSinceLastLODUpdate = 0;

        // Camera tracking
        this.lastCameraPosition = null;
        this.lastCameraDirection = null;

        // Cache tracking
        this.lastAllocatedBrickCount = -1;
        this.lastAllocatedOutputOffsets = [];

        // Performance tracking
        this.lastFrameTime = performance.now();
        this.frameTimes = [];

        // GPU timing
        this.querySet = null;
        this.stagingBuffer = null;
        this.gpuTimings = { decode: 0, raytrace: 0, overall: 0 };
        this.isGPUResultsMapped = false;
    }

    /**
     * Reset state for full cache rebuild
     */
    reset() {
        this.lodUpdateInProgress = false;
        this.framesSinceLastLODUpdate = RENDER_CONFIG.LOD_UPDATE_INTERVAL;
        this.lastAllocatedBrickCount = -1;
        this.lastAllocatedOutputOffsets = [];
        this.lastCameraPosition = null;
        this.lastCameraDirection = null;
    }
}

// Global state instance
const state = new RenderLoopState();

// ==================== Initialization ====================

/**
 * Initialize the render loop with GPU state and dataset
 * @param {Object} initialGpuState - WebGPU state object
 * @param {Object} initialDataset - Dataset with bricks and header
 */
export function initRenderLoop(initialGpuState, initialDataset) {
    state.gpuState = initialGpuState;
    state.dataset = initialDataset;

    if (!state.dataset.hiddenLabels) {
        state.dataset.hiddenLabels = new Set();
    }
    state.cameraController = initialGpuState.cameraController ?? null;

    // Calculate max LOD based on brick size
    // At LOD k, voxel count = brickSize / 2^k
    // Max useful LOD is when we reach 1 voxel: brickSize / 2^maxLOD = 1
    // Therefore: maxLOD = log2(brickSize)
    state.maxLOD = Math.floor(Math.log2(state.dataset.header.brickSize));

    // Initialize GPU timing infrastructure (3 timestamps: decode start/end, raytrace end)
    const device = initialGpuState.device;
    const hasTimestamps = device.features.has('chromium-experimental-timestamp-query-inside-passes');
    console.log(`GPU timestamp queries: ${hasTimestamps ? 'SUPPORTED ✓' : 'NOT SUPPORTED (timing will show --)'}`);

    state.querySet = device.createQuerySet({
        type: 'timestamp',
        count: 3
    });
    state.stagingBuffer = device.createBuffer({
        size: 24,  // 3 timestamps × 8 bytes each
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    // Initialize cache and work queue with pre-calculated LODs from webgpuSetup
    initializeCache(initialGpuState.initialBrickLODs);

    // Start the frame loop
    requestAnimationFrame(frame);
}

/**
 * Initialize cache with visible bricks
 * @param {Array<number>} initialLODs - Pre-calculated LOD array
 */
function initializeCache(initialLODs) {
    // On first frame, allocate ALL bricks regardless of visibility
    // This ensures the full dataset is immediately available when camera moves
    // After first frame, subsequent updates use normal frustum culling
    const forceAll = true;  // Always force all bricks on initialization

    // Use the calculated LODs, not forced full resolution
    const result = initCacheInternal(
        state.gpuState,
        state.dataset,
        initialLODs,
        state.dataset.header.brickSize,
        forceAll
    );

    const { allocatedBricks } = result;

    // Ensure pooled buffers are allocated before filling work queues
    state.bufferManager.ensureBufferPools(state.dataset.bricks.length, allocatedBricks.length);

    // Prepare and write buffers for initial frame
    const { data: dynamicBrickData } = updateBrickInfoBuffer({
        lodArray: initialLODs,
        allocatedBricks,
        cache: state.gpuState.brickCache,
        dataset: state.dataset,
        brickSize: state.dataset.header.brickSize
    });

    const { workQueue, workCount, count } = updateWorkQueue({
        allocatedBricks,
        workQueuePool: state.bufferManager.workQueuePool,
        workCountBuffer: state.bufferManager.workCountBuffer
    });

    // Ensure the CPU-side count is set so the first compute pass dispatches
    state.gpuState.workCount = count;

    // Batch GPU writes for initialization
    const device = state.gpuState.device;
    device.queue.writeBuffer(state.gpuState.dynamicBricksBuffer, 0, dynamicBrickData);
    if (workQueue) {
        device.queue.writeBuffer(state.gpuState.workQueueBuffer, 0, workQueue);
    }
    device.queue.writeBuffer(state.gpuState.workCountBuffer, 0, new Uint32Array([count]));

    state.lastAllocatedBrickCount = allocatedBricks.length;
}

/**
 * Update camera uniform buffer if needed
 */
function updateCameraUniformIfNeeded() {
    if (!state.gpuState.camera || !state.gpuState.cameraUniformBuffer) return;

    // If controller reports a pending upload, ensure camera uniforms update before GPU work
    const needsUpload = (state.cameraController && state.cameraController.needsUpload()) ||
        state.gpuState.camera.needsUpload?.();
    if (!needsUpload) return;

    state.gpuState.device.queue.writeBuffer(
        state.gpuState.cameraUniformBuffer,
        0,
        state.gpuState.camera.getUniformData()
    );

    if (state.cameraController && state.cameraController.markUploaded) {
        state.cameraController.markUploaded();
    } else if (state.gpuState.camera.markUploaded) {
        state.gpuState.camera.markUploaded();
    }
}

// ==================== Core Frame Loop ====================

/**
 * Main render loop frame function
 */
function frame() {
    // Performance tracking
    const currentTime = performance.now();
    const frameTime = currentTime - state.lastFrameTime;
    state.lastFrameTime = currentTime;

    state.frameTimes.push(frameTime);
    if (state.frameTimes.length > RENDER_CONFIG.FRAME_SAMPLE_COUNT) {
        state.frameTimes.shift();
    }

    const hasTimestampFeature = state.gpuState.device.features.has('chromium-experimental-timestamp-query-inside-passes');

    // Update performance display every N frames
    if (state.frameTimes.length % RENDER_CONFIG.PERF_UPDATE_INTERVAL === 0) {
        const decodeElem = document.getElementById('perf-decode');
        const raytraceElem = document.getElementById('perf-raytrace');
        const overallElem = document.getElementById('perf-overall');

        if (hasTimestampFeature) {
            if (decodeElem) decodeElem.textContent = `Decode ${state.gpuTimings.decode.toFixed(2)} ms`;
            if (raytraceElem) raytraceElem.textContent = `Raytrace ${state.gpuTimings.raytrace.toFixed(2)} ms`;
            if (overallElem) overallElem.textContent = `Overall ${state.gpuTimings.overall.toFixed(2)} ms`;
        } else {
            if (decodeElem) decodeElem.textContent = `Decode --`;
            if (raytraceElem) raytraceElem.textContent = `Raytrace --`;
            if (overallElem) overallElem.textContent = `Overall --`;
        }
    }

    updateCameraUniformIfNeeded();

    // Skip frame if request readback buffers are currently mapped
    if (state.isRequestReadbackMapped) {
        requestAnimationFrame(frame);
        return;
    }

    // Only recalculate LODs periodically AND when camera has moved
    state.framesSinceLastLODUpdate++;
    if (state.framesSinceLastLODUpdate >= RENDER_CONFIG.LOD_UPDATE_INTERVAL &&
        !state.lodUpdateInProgress &&
        cameraHasMoved()) {
        state.framesSinceLastLODUpdate = 0;
        updateBrickLODs();
    }

    // Pre-submission: Clear brick request buffer BEFORE creating encoder
    // This must happen in GPU order before ray marching writes to it
    if (state.gpuState.brickRequestBuffer && state.dataset.bricks.length > 0) {
        const clearValue = new Uint32Array(state.dataset.bricks.length).fill(0);  // 0 = not accessed
        state.gpuState.device.queue.writeBuffer(state.gpuState.brickRequestBuffer, 0, clearValue);
    }

    const encoder = state.gpuState.device.createCommandEncoder();

    // GPU timestamp 0: decode start (if feature available)
    if (hasTimestampFeature) {
        let tsPass = encoder.beginComputePass();
        tsPass.writeTimestamp(state.querySet, 0);
        tsPass.end();
    }

    // Run compute pass to decode bricks
    runComputePass(encoder, state.gpuState);

    // GPU timestamp 1: decode end / raytrace start
    if (hasTimestampFeature) {
        let tsPass = encoder.beginComputePass();
        tsPass.writeTimestamp(state.querySet, 1);
        tsPass.end();
    }

    // CRITICAL: Clear ray trace output buffer using compute shader
    // This prevents stale pixel data artifacts when voxels become transparent (hidden labels)
    // Using a compute shader ensures proper synchronization and complete clearing
    runClearRayOutputPass(encoder, state.gpuState);

    // Run ray tracing pass
    runRayTracePass(encoder, state.gpuState);

    // GPU timestamp 2: raytrace end / overall end
    if (hasTimestampFeature) {
        let tsPass = encoder.beginComputePass();
        tsPass.writeTimestamp(state.querySet, 2);
        tsPass.end();
    }

    // Run display pass to show results on canvas
    runDisplayPass(encoder, state.gpuState);

    // Copy GPU request buffer for readback (track which bricks were accessed)
    if (state.gpuState.brickRequestBuffer && state.gpuState.brickRequestReadback && !state.isRequestReadbackMapped) {
        encoder.copyBufferToBuffer(
            state.gpuState.brickRequestBuffer,
            0,
            state.gpuState.brickRequestReadback,
            0,
            state.dataset.bricks.length * 4
        );
    }

    // Copy timestamp results to staging buffer for readback (only if feature is available)
    if (hasTimestampFeature) {
        encoder.copyQuerySetResultsToBuffer(state.querySet, 0, 3, state.stagingBuffer, 0);
    }

    state.gpuState.device.queue.submit([encoder.finish()]);
    // Readback GPU timestamps asynchronously (only if feature is available)
    if (hasTimestampFeature && !state.isGPUResultsMapped) {
        state.isGPUResultsMapped = true;
        state.stagingBuffer.mapAsync(GPUMapMode.READ).then(() => {
            const mapped = state.stagingBuffer.getMappedRange();
            const timestamps = new BigUint64Array(mapped);
            const ts0 = Number(timestamps[0]);  // decode start
            const ts1 = Number(timestamps[1]);  // decode end / raytrace start
            const ts2 = Number(timestamps[2]);  // raytrace end

            state.gpuTimings.decode = (ts1 - ts0) / 1_000_000;    // ns to ms
            state.gpuTimings.raytrace = (ts2 - ts1) / 1_000_000;  // ns to ms
            state.gpuTimings.overall = (ts2 - ts0) / 1_000_000;   // ns to ms

            state.stagingBuffer.unmap();
            state.isGPUResultsMapped = false;
        }).catch((err) => {
            console.error('GPU timestamp readback failed:', err);
            state.isGPUResultsMapped = false;
        });
    }

    // Process GPU request readback asynchronously (non-blocking)
    if (state.gpuState.brickRequestReadback && !state.isRequestReadbackMapped) {
        state.isRequestReadbackMapped = true;
        state.gpuState.brickRequestReadback.mapAsync(GPUMapMode.READ).then(() => {
            const mapped = state.gpuState.brickRequestReadback.getMappedRange();
            // Copy data out before unmapping
            state.pendingGPURequests = new Uint32Array(mapped).slice();
            state.gpuState.brickRequestReadback.unmap();
            state.isRequestReadbackMapped = false;
            state.gpuRequestsReady = true;  // Signal that new requests are available
        }).catch((err) => {
            console.error('Error during GPU request readback:', err);
            state.isRequestReadbackMapped = false;
        });
    }

    requestAnimationFrame(frame);
}

// ==================== LOD Management & Cache Processing ====================

/**
 * Update brick LODs based on camera position
 */
function updateBrickLODs() {
    const { calculateAllBrickLODs, device, brickCache, canvas } = state.gpuState;

    if (!state.gpuState.camera || !brickCache) return;

    state.lodUpdateInProgress = true;

    // Recalculate LODs based on current camera position (async with Workers)
    calculateAllBrickLODs(
        state.dataset.bricks,
        state.gpuState.camera,
        Math.min(canvas.width, canvas.height),  // Use smaller dimension for conservative LOD
        state.maxLOD,
        state.dataset.header.brickSize
    ).then((result) => {
        const { lodArray, offsetArray } = result;

        // Cache management pipeline
        processCachePipeline(lodArray, offsetArray);
    }).catch((err) => {
        console.error("Error calculating brick LODs:", err);
    }).finally(() => {
        state.lodUpdateInProgress = false;
    });
}

/**
 * Process cache based on LOD calculations
 * @param {Array<number>} lodArray - LOD values for all bricks
 * @param {Array<number>} offsetArray - Offset values
 */
function processCachePipeline(lodArray, offsetArray) {
    const { allocatedBricks, evictedCount, stats, candidateBricks } = processCache({
        lodArray,
        offsetArray,
        gpuState: state.gpuState,
        dataset: state.dataset,
        pendingGPURequests: state.pendingGPURequests,
        gpuRequestsReady: state.gpuRequestsReady
    });

    // Clear GPU requests flag after processing
    if (state.gpuRequestsReady) {
        state.gpuRequestsReady = false;
    }

    // --- Improved Patch: Track and clear both old and new output buffer offsets for bricks whose cache position changes ---
    const brickCount = state.dataset.bricks.length;
    const brickSize = state.dataset.header.brickSize;
    const voxelsPerBrick = Math.pow(brickSize, 3);
    const device = state.gpuState.device;
    const outVoxelBuffer = state.gpuState.outVoxelBuffer;

    // Track previous offsets for each brick
    if (!state._prevBrickOffsets) {
        state._prevBrickOffsets = new Array(brickCount).fill(0xFFFFFFFF);
    }

    // Initialize tracking arrays
    if (!state._prevBrickOffsets) {
        state._prevBrickOffsets = new Array(brickCount).fill(0xFFFFFFFF);
    }
    if (!state._prevBrickLods) {
        state._prevBrickLods = new Array(brickCount).fill(-1);
    }

    // Update previous tracking for next frame
    for (let i = 0; i < brickCount; i++) {
        const brick = state.dataset.bricks[i];
        const currOffset = brick.outputOffset;
        const currCacheInfo = state.gpuState.brickCache.getCacheInfo(i);

        state._prevBrickOffsets[i] = currOffset;
        state._prevBrickLods[i] = currCacheInfo ? currCacheInfo.lod : -1;
    }

    // Always refresh dynamic brick info buffer to avoid stale offsets after reallocations/LOD changes
    state.bufferManager.ensureBufferPools(
        state.dataset.bricks.length,
        allocatedBricks.length
    );

    const { data: dynamicBrickData } = updateBrickInfoBuffer({
        lodArray,
        allocatedBricks,
        cache: state.gpuState.brickCache,
        dataset: state.dataset,
        brickSize: state.dataset.header.brickSize
    });

    const { workQueue, workCount, count } = updateWorkQueue({
        allocatedBricks,
        workQueuePool: state.bufferManager.workQueuePool,
        workCountBuffer: state.bufferManager.workCountBuffer
    });

    // Batch all GPU writes together for efficiency
    device.queue.writeBuffer(state.gpuState.dynamicBricksBuffer, 0, dynamicBrickData);

    if (workQueue) {
        device.queue.writeBuffer(state.gpuState.workQueueBuffer, 0, workQueue);
    }
    device.queue.writeBuffer(state.gpuState.workCountBuffer, 0, new Uint32Array([count]));

    state.gpuState.workCount = count;
    state.lastAllocatedBrickCount = allocatedBricks.length;
}

// ==================== Camera Movement Detection ====================

/**
 * Check if camera has moved significantly
 * @returns {boolean} True if camera moved beyond threshold
 */
function cameraHasMoved() {
    if (!state.gpuState.camera) return false;

    const result = checkCameraMovement(
        state.gpuState.camera,
        state.lastCameraPosition,
        state.lastCameraDirection
    );

    if (result.hasMoved) {
        state.lastCameraPosition = result.position;
        state.lastCameraDirection = result.direction;
    }

    return result.hasMoved;
}

// ==================== Exported Utility Functions ====================

/**
 * Force a full brick cache refresh and reload
 */
export function forceFullBrickRefresh() {
    if (!state.gpuState || !state.gpuState.brickCache) return;

    // Capture currently resident bricks so we can immediately re-decode them
    const cachedIndices = Array.from(state.gpuState.brickCache.residency.keys());

    state.reset();

    // CRITICAL: Write EMPTY_VALUE (0xFFFFFFFF) to entire voxel buffer
    // This ensures hidden labels are truly gone, not just zeroed
    if (state.gpuState.outVoxelBuffer && Number.isFinite(state.gpuState.voxelBufferSize)) {
        const voxelCount = state.gpuState.voxelBufferSize / 4; // u32 elements
        const EMPTY_VALUE = 0xFFFFFFFF;

        // Write in chunks to avoid memory issues
        const chunkSize = 256 * 1024; // 1MB chunks (256K u32 values)
        for (let offset = 0; offset < voxelCount; offset += chunkSize) {
            const count = Math.min(chunkSize, voxelCount - offset);
            const chunk = new Uint32Array(count).fill(EMPTY_VALUE);
            state.gpuState.device.queue.writeBuffer(
                state.gpuState.outVoxelBuffer,
                offset * 4,
                chunk
            );
        }
    }

    // Immediately re-decode all cached bricks so the next frame has data
    if (cachedIndices.length > 0 && state.gpuState.workQueueBuffer && state.gpuState.workCountBuffer) {
        cachedIndices.sort((a, b) => a - b);

        state.bufferManager.ensureBufferPools(state.dataset.bricks.length, cachedIndices.length);
        for (let i = 0; i < cachedIndices.length; i++) {
            state.bufferManager.workQueuePool[i] = cachedIndices[i];
        }

        state.bufferManager.workCountBuffer[0] = cachedIndices.length;
        state.gpuState.workCount = cachedIndices.length;

        state.gpuState.device.queue.writeBuffer(
            state.gpuState.workQueueBuffer,
            0,
            state.bufferManager.workQueuePool,
            0,
            cachedIndices.length
        );
        state.gpuState.device.queue.writeBuffer(
            state.gpuState.workCountBuffer,
            0,
            state.bufferManager.workCountBuffer
        );
    } else {
        // No cached bricks; ensure the work counter is zeroed
        state.gpuState.workCount = 0;
        if (state.gpuState.workCountBuffer) {
            state.gpuState.device.queue.writeBuffer(state.gpuState.workCountBuffer, 0, new Uint32Array([0]));
        }
    }
}

/**
 * Re-decode all currently cached bricks using the latest transfer function/colors
 */
export function reDecodeCachedBricks() {
    if (!state.gpuState || !state.gpuState.brickCache) return;

    const cachedIndices = Array.from(state.gpuState.brickCache.residency.keys());
    if (cachedIndices.length === 0) return;

    cachedIndices.sort((a, b) => a - b);

    state.bufferManager.ensureBufferPools(state.dataset.bricks.length, cachedIndices.length);
    for (let i = 0; i < cachedIndices.length; i++) {
        state.bufferManager.workQueuePool[i] = cachedIndices[i];
    }

    state.bufferManager.workCountBuffer[0] = cachedIndices.length;
    state.gpuState.workCount = cachedIndices.length;

    state.gpuState.device.queue.writeBuffer(
        state.gpuState.workQueueBuffer,
        0,
        state.bufferManager.workQueuePool,
        0,
        cachedIndices.length
    );
    state.gpuState.device.queue.writeBuffer(
        state.gpuState.workCountBuffer,
        0,
        state.bufferManager.workCountBuffer
    );
}

