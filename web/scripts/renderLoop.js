// ==================== State & Initialization ====================

import { createLogger } from './logger.js';
import DecoderValidator from './decoderValidation.js';

const log = createLogger('Cache');

let isReadbackMapped = false;
let gpuState;
let dataset;
let maxLOD;
let hasRequestedReadback = false; // ensure we read back after first completed frame
let validationRunning = false;
let lodUpdateInProgress = false; // prevent concurrent LOD updates
let framesSinceLastLODUpdate = 0;
const LOD_UPDATE_INTERVAL = 60; // Only update LODs every 60 frames (~1 second at 60fps)
const BUFFER_GROWTH_FACTOR = 1.5; // Grow pools with headroom to avoid churn
const MIN_DYNAMIC_CAP = 1024;     // Reasonable floor for dynamic brick data pool
const MIN_WORKQUEUE_CAP = 256;    // Reasonable floor for work queue pool
let lastCameraPosition = null;
let lastCameraDirection = null;
let runValidationOnce = false; // trigger flag from app.js

// Buffer reuse pool to reduce GC pressure
let dynamicBrickDataPool = null;
let workQueuePool = null;
let workCountBuffer = new Uint32Array(1);
let lastAllocatedBrickCount = -1; // Track if cache changed
let lastAllocatedOutputOffsets = []; // Track which regions were written
let clearRegionsBuffer = null; // GPU buffer for clear regions (chunked)
let clearRegionsData = null; // CPU array to accumulate regions
let clearRegionCountBuffer = null; // Persistent uniform for total region count
let clearRegionBaseBuffer = null; // Uniform for chunk base index
let cachedClearBindGroup = null; // Cache bind group to avoid recreating each frame
let cameraController = null;

// GPU request tracking state
let isRequestReadbackMapped = false;  // Track if request buffer is being read
let pendingGPURequests = null;        // Store readback results for next frame
let gpuRequestsReady = false;         // Flag when new GPU requests are available

// Performance tracking
let lastFrameTime = performance.now();
let frameTimes = [];
const FRAME_SAMPLE_COUNT = 60;

function growCapacity(currentLength, requiredLength, minimumLength) {
    if (currentLength >= requiredLength) return currentLength;
    const base = currentLength === 0 ? requiredLength : currentLength;
    return Math.max(Math.ceil(base * BUFFER_GROWTH_FACTOR), requiredLength, minimumLength);
}

function ensureBufferPools(brickCount, maxAllocated) {
    const requiredDynamic = brickCount * 2;
    const requiredWorkQueue = Math.max(maxAllocated, 1);
    const dynamicCap = growCapacity(dynamicBrickDataPool?.length ?? 0, requiredDynamic, MIN_DYNAMIC_CAP);
    const workCap = growCapacity(workQueuePool?.length ?? 0, requiredWorkQueue, MIN_WORKQUEUE_CAP);

    if (!dynamicBrickDataPool || dynamicBrickDataPool.length < requiredDynamic) {
        dynamicBrickDataPool = new Uint32Array(dynamicCap);
    }
    if (!workQueuePool || workQueuePool.length < requiredWorkQueue) {
        workQueuePool = new Uint32Array(workCap);
    }
}

export function initRenderLoop(initialGpuState, initialDataset) {
    gpuState = initialGpuState;
    dataset = initialDataset;
    cameraController = initialGpuState.cameraController ?? null;

    // Calculate max LOD based on brick size
    // At LOD k, voxel count = brickSize / 2^k
    // Max useful LOD is when we reach 1 voxel: brickSize / 2^maxLOD = 1
    // Therefore: maxLOD = log2(brickSize)
    maxLOD = Math.floor(Math.log2(dataset.header.brickSize));

    // Initialize cache and work queue with pre-calculated LODs from webgpuSetup
    initializeCache(initialGpuState.initialBrickLODs);

    // Start the frame loop
    requestAnimationFrame(frame);
}

// Trigger a one-time validation readback of outVoxelBuffer
export function requestValidationOnce() {
    runValidationOnce = true;
}

function initializeCache(initialLODs) {
    const cache = gpuState.brickCache;
    const allocatedBricks = [];

    // Allocate cache entries only for bricks that are initially visible
    for (let i = 0; i < dataset.bricks.length; i++) {
        const lod = initialLODs[i];
        const brick = dataset.bricks[i];

        // Reuse the same visibility tests used in later frames
        const inFrustum = isBrickInFrustum(brick, dataset.header.brickSize);
        const hasVisible = inFrustum ? brickHasVisibleVoxels(brick) : false;
        const isVisible = inFrustum && hasVisible;

        if (!isVisible) {
            continue; // Skip invisible bricks on the first frame
        }

        cache.markBrickNeeded(i, lod);
        
        const blockStart = cache.allocateBlock(lod);
        if (blockStart !== null) {
            allocatedBricks.push({ index: i, lod, blockStart });
            cache.setCached(i, lod, blockStart);
        } else {
            console.warn(`Failed to allocate cache for brick ${i} at LOD ${lod}`);
        }
    }

    // Prepare and write buffers for initial frame
    const dynamicBrickData = updateBrickInfoBuffer(initialLODs, allocatedBricks, cache);
    const { workQueue, workCount } = updateWorkQueue(allocatedBricks);
    
    // Batch GPU writes for initialization
    const device = gpuState.device;
    device.queue.writeBuffer(gpuState.dynamicBricksBuffer, 0, dynamicBrickData);
    if (workQueue) {
        device.queue.writeBuffer(gpuState.workQueueBuffer, 0, workQueue);
    }
    device.queue.writeBuffer(gpuState.workCountBuffer, 0, workCount);
    
    cache.clearMarkedBricks();
    lastAllocatedBrickCount = allocatedBricks.length;
    
    log.log(`Initialized: ${allocatedBricks.length}/${dataset.bricks.length} bricks allocated for first frame`);
}

function updateCameraUniformIfNeeded() {
    if (!gpuState.camera || !gpuState.cameraUniformBuffer) return;

    // If controller reports a pending upload, ensure camera uniforms update before GPU work
    const needsUpload = (cameraController && cameraController.needsUpload()) || gpuState.camera.needsUpload?.();
    if (!needsUpload) return;

    gpuState.device.queue.writeBuffer(
        gpuState.cameraUniformBuffer,
        0,
        gpuState.camera.getUniformData()
    );

    if (cameraController && cameraController.markUploaded) {
        cameraController.markUploaded();
    } else if (gpuState.camera.markUploaded) {
        gpuState.camera.markUploaded();
    }
}

// ==================== Core Frame Loop ====================

function frame() {
    // Performance tracking
    const currentTime = performance.now();
    const frameTime = currentTime - lastFrameTime;
    lastFrameTime = currentTime;
    
    frameTimes.push(frameTime);
    if (frameTimes.length > FRAME_SAMPLE_COUNT) {
        frameTimes.shift();
    }
    
    // Update performance display every 10 frames
    if (frameTimes.length % 10 === 0) {
        const avgFrameTime = frameTimes.reduce((a, b) => a + b, 0) / frameTimes.length;
        const fps = 1000 / avgFrameTime;
        const fpsElem = document.getElementById('perf-fps');
        const msElem = document.getElementById('perf-ms');
        if (fpsElem) fpsElem.textContent = `${Math.round(fps)} FPS`;
        if (msElem) msElem.textContent = `${avgFrameTime.toFixed(1)} ms`;
    }
    
    updateCameraUniformIfNeeded();

    // Skip frame if readback buffers are currently mapped
    if (isReadbackMapped || isRequestReadbackMapped) {
        requestAnimationFrame(frame);
        return;
    }

    // Only recalculate LODs periodically AND when camera has moved
    framesSinceLastLODUpdate++;
    if (framesSinceLastLODUpdate >= LOD_UPDATE_INTERVAL && !lodUpdateInProgress && cameraHasMoved()) {
        framesSinceLastLODUpdate = 0;
        updateBrickLODs();
    }

    // Pre-submission: Clear brick request buffer BEFORE creating encoder
    // This must happen in GPU order before ray marching writes to it
    if (gpuState.brickRequestBuffer && dataset.bricks.length > 0) {
        const clearValue = new Uint32Array(dataset.bricks.length).fill(0);  // 0 = not accessed
        gpuState.device.queue.writeBuffer(gpuState.brickRequestBuffer, 0, clearValue);
    }

    const encoder = gpuState.device.createCommandEncoder();

    // Clear ray trace output buffer to black
    encoder.clearBuffer(gpuState.rayTraceOutputBuffer);

    // Clear voxel buffer at the start of each frame
    runClearPass(encoder);

    // Run compute pass to decode bricks
    runComputePass(encoder);

    // Run ray tracing pass
    runRayTracePass(encoder);
    
    // Run display pass to show results on canvas
    runDisplayPass(encoder);

    // Copy GPU request buffer for readback (track which bricks were accessed)
    if (gpuState.brickRequestBuffer && gpuState.brickRequestReadback && !isRequestReadbackMapped) {
        encoder.copyBufferToBuffer(
            gpuState.brickRequestBuffer,
            0,
            gpuState.brickRequestReadback,
            0,
            dataset.bricks.length * 4
        );
    }

    // If validation requested, copy voxel buffer to readback buffer once
    if (runValidationOnce && !validationRunning) {
        encoder.copyBufferToBuffer(
            gpuState.outVoxelBuffer,
            0,
            gpuState.readbackBuffer,
            0,
            gpuState.voxelBufferSize
        );
        validationRunning = true;
        runValidationOnce = false;
    }

    gpuState.device.queue.submit([encoder.finish()]);

    // Process GPU request readback asynchronously (non-blocking)
    if (gpuState.brickRequestReadback && !isRequestReadbackMapped) {
        isRequestReadbackMapped = true;
        gpuState.brickRequestReadback.mapAsync(GPUMapMode.READ).then(() => {
            const mapped = gpuState.brickRequestReadback.getMappedRange();
            // Copy data out before unmapping
            pendingGPURequests = new Uint32Array(mapped).slice();
            gpuState.brickRequestReadback.unmap();
            isRequestReadbackMapped = false;
            gpuRequestsReady = true;  // Signal that new requests are available
        }).catch((err) => {
            console.error('Error during GPU request readback:', err);
            isRequestReadbackMapped = false;
        });
    }

    // Kick off async readback + validation once the copy completes
    if (validationRunning && !isReadbackMapped) {
        isReadbackMapped = true;
        gpuState.readbackBuffer.mapAsync(GPUMapMode.READ).then(async () => {
            const mapped = gpuState.readbackBuffer.getMappedRange();
            const gpuData = new Uint32Array(mapped).slice(); // copy out
            gpuState.readbackBuffer.unmap();
            isReadbackMapped = false;

            // Slice to the first brick using its cache bloczk offset
            const cacheInfo = gpuState.brickCache.getCacheInfo(0);
            const start = cacheInfo ? cacheInfo.blockStart : 0;
            const voxelsPerBrick = Math.pow(dataset.header.brickSize, 3);
            const gpuSlice = gpuData.slice(start, start + voxelsPerBrick);

            const validator = new DecoderValidator();
            
            // Set transfer function if available (for matching GPU label->RGBA conversion)
            if (gpuState.transferFunctionMap) {
                validator.setTransferFunction(gpuState.transferFunctionMap);
            }
            
            await validator.loadCPUReference();
            validator.setGPUOutput(gpuSlice);
            validator.printResults();
            validationRunning = false;
        }).catch((err) => {
            console.error('Error during GPU readback for validation:', err);
            isReadbackMapped = false;
            validationRunning = false;
        });
    }

    requestAnimationFrame(frame);
}

// ==================== LOD Management & Cache Processing ====================

function updateBrickLODs() {
    const { calculateAllBrickLODs, device, brickCache, canvas } = gpuState;
    
    if (!gpuState.camera || !brickCache) return;
    
    lodUpdateInProgress = true;
    
    // Recalculate LODs based on current camera position (async with Workers)
    calculateAllBrickLODs(
        dataset.bricks,
        gpuState.camera,
        Math.min(canvas.width, canvas.height),  // Use smaller dimension for conservative LOD
        maxLOD,
        dataset.header.brickSize
    ).then((result) => {
        const { lodArray, offsetArray } = result;
        
        // Cache management pipeline
        processCache(lodArray, offsetArray);
    }).catch((err) => {
        console.error("Error calculating brick LODs:", err);
    }).finally(() => {
        lodUpdateInProgress = false;
    });
}

function processCache(lodArray, offsetArray) {
    const cache = gpuState.brickCache;
    const device = gpuState.device;
    const brickCount = dataset.bricks.length;
    
    // PHASE 1: CPU Pre-filtering (Frustum + Empty brick culling)
    // Build candidate set - don't cache yet, wait for GPU requests
    const candidateBricks = new Set();
    let frustumCulled = 0;
    let tfCulled = 0;
    
    for (let i = 0; i < brickCount; i++) {
        const brick = dataset.bricks[i];
        
        // Pre-filter: frustum culling (eliminate bricks outside view)
        const inFrustum = isBrickInFrustum(brick, dataset.header.brickSize);
        if (!inFrustum) {
            frustumCulled++;
            continue;
        }
        
        // Pre-filter: empty brick culling (eliminate transparent bricks)
        const hasVisible = brickHasVisibleVoxels(brick);
        if (!hasVisible) {
            tfCulled++;
            continue;
        }
        
        // Passed pre-filters: mark as candidate (not cached yet!)
        candidateBricks.add(i);
    }
    
    // PHASE 2: GPU Request Processing (Occlusion culling via ray marching)
    // Use GPU requests from previous frame to determine which candidates to actually cache
    const bricksToDecompress = [];
    let gpuRequestedCount = 0;
    
    if (gpuRequestsReady && pendingGPURequests && pendingGPURequests.length >= brickCount) {
        // Process GPU requests: only cache bricks that were actually accessed by rays
        for (const brickIndex of candidateBricks) {
            const isAccessed = pendingGPURequests[brickIndex];  // 1 = accessed, 0 = not accessed
            
            // Check if GPU actually accessed this brick
            if (isAccessed === 1) {
                gpuRequestedCount++;
                const lod = lodArray[brickIndex];  // Use CPU-calculated LOD for accessed bricks
                const { isNew, lodChanged } = cache.markBrickNeeded(brickIndex, lod);
                
                // If LOD changed, free the old block first
                if (lodChanged && !isNew) {
                    const oldCacheInfo = cache.getCacheInfo(brickIndex);
                    if (oldCacheInfo) {
                        cache.freeBlock(oldCacheInfo.lod, oldCacheInfo.blockStart);
                        cache.residency.delete(brickIndex);
                    }
                }
                
                // If not cached OR LOD changed, need to decompress
                if (isNew || lodChanged) {
                    bricksToDecompress.push({ index: brickIndex, lod });
                }
            }
            // If requestedLOD === 0xFF, brick was not accessed by GPU (occluded) - don't cache
        }
        
        // Clear flag after processing
        gpuRequestsReady = false;
    } else {
        // Fallback: no GPU requests yet (first few frames) - use CPU-only filtering
        for (const brickIndex of candidateBricks) {
            const lod = lodArray[brickIndex];
            const { isNew, lodChanged } = cache.markBrickNeeded(brickIndex, lod);
            
            if (lodChanged && !isNew) {
                const oldCacheInfo = cache.getCacheInfo(brickIndex);
                if (oldCacheInfo) {
                    cache.freeBlock(oldCacheInfo.lod, oldCacheInfo.blockStart);
                    cache.residency.delete(brickIndex);
                }
            }
            
            if (isNew || lodChanged) {
                bricksToDecompress.push({ index: brickIndex, lod });
            }
        }
    }
    
    // Step 3: Evict unmarked bricks from previous frame
    const evicted = cache.getEvictedBricks();
    for (const { brickIndex, lod, blockStart } of evicted) {
        cache.freeBlock(lod, blockStart);
        cache.residency.delete(brickIndex);
    }
    
    // Step 3: Allocate blocks for new bricks
    const allocatedBricks = [];
    let needsRebuild = false;
    
    for (const { index, lod } of bricksToDecompress) {
        let blockStart = cache.allocateBlock(lod);
        
        if (blockStart === null && !needsRebuild) {
            // First allocation failure - trigger rebuild and retry
            console.warn("Cache allocation failed - rebuilding");
            cache.rebuild();
            
            // Rebuild clears residency, so re-mark visible bricks as new
            for (const { index: idx, lod: l } of bricksToDecompress) {
                cache.markBrickNeeded(idx, l);
            }
            
            needsRebuild = true;
            blockStart = cache.allocateBlock(lod); // Retry after rebuild
        }
        
        if (blockStart === null) {
            // Still failed after rebuild - give up
            console.error(`Cannot allocate cache for brick ${index} LOD ${lod} even after rebuild`);
            continue;
        }
        
        allocatedBricks.push({ index, lod, blockStart });
        cache.setCached(index, lod, blockStart);
    }
    
    // Step 5: Update brick info buffer with cache locations (only if cache changed)
    if (allocatedBricks.length > 0 || lastAllocatedBrickCount > 0) {
        const dynamicBrickData = updateBrickInfoBuffer(lodArray, allocatedBricks, cache);
        const { workQueue, workCount } = updateWorkQueue(allocatedBricks);
        
        // Batch all GPU writes together for efficiency (minimize separate queue submissions)
        const device = gpuState.device;
        const encoder = device.createCommandEncoder();
        
        // Use a single encoder to batch buffer writes
        device.queue.writeBuffer(gpuState.dynamicBricksBuffer, 0, dynamicBrickData);
        if (workQueue) {
            device.queue.writeBuffer(gpuState.workQueueBuffer, 0, workQueue);
        }
        device.queue.writeBuffer(gpuState.workCountBuffer, 0, workCount);
        
        lastAllocatedBrickCount = allocatedBricks.length;
    } else {
        // No bricks need decompression this frame; ensure workCount is zeroed so compute pass is skipped
        gpuState.workCount = 0;
        device.queue.writeBuffer(gpuState.workCountBuffer, 0, new Uint32Array([0]));
    }
    
    // Clear marked bricks for next frame
    cache.clearMarkedBricks();
    
    // Log stats periodically
    if (Math.random() < 0.01) { // 1% of frames
        log.log("Stats:", cache.getStats());
        log.log(`CPU Pre-filtering: ${brickCount} total | ${candidateBricks.size} candidates | ${frustumCulled} frustum culled | ${tfCulled} TF culled`);
        log.log(`GPU Visibility: ${gpuRequestedCount} accessed (${candidateBricks.size - gpuRequestedCount} occluded)`);
        log.log(`Cache: ${bricksToDecompress.length} to decompress | ${evicted.length} evicted`);
    }
}

// ==================== GPU Buffer Updates ====================

function updateBrickInfoBuffer(lodArray, allocatedBricks, cache) {
    const device = gpuState.device;
    const brickCount = dataset.bricks.length;
    const DYNAMIC_STRIDE_U32 = 3; // outputOffset, targetLOD, lodScale (as float32)
    const brickSize = dataset.header.brickSize;
    
    // Reuse buffer pool
    ensureBufferPools(brickCount, allocatedBricks.length);
    const dynamicBrickData = new Uint32Array(brickCount * DYNAMIC_STRIDE_U32);  // Can't reuse pool - need Float32 view
    const dynamicBrickDataF32 = new Float32Array(dynamicBrickData.buffer);     // Float32 view for lodScale
    
    // Track output offsets for this frame (used for partial clear)
    lastAllocatedOutputOffsets = [];
    
    for (let i = 0; i < brickCount; i++) {
        const b = dataset.bricks[i];
        const base = i * DYNAMIC_STRIDE_U32;
        const cacheInfo = cache.getCacheInfo(i);
        const lod = cacheInfo ? cacheInfo.lod : lodArray[i];
        
        // Precompute lodScale: (2^lod) / brickSize
        const lodSize = Math.pow(2, lod);
        const lodScale = lodSize / brickSize;
        
        // Dynamic data only (static data never changes, already in GPU)
        if (cacheInfo) {
            dynamicBrickData[base + 0] = cacheInfo.blockStart;  // outputOffset (u32)
            dynamicBrickData[base + 1] = cacheInfo.lod;         // targetLOD (u32)
            dynamicBrickDataF32[base + 2] = lodScale;           // lodScale (f32)
            lastAllocatedOutputOffsets.push(cacheInfo.blockStart);
        } else {
            dynamicBrickData[base + 0] = 0xFFFFFFFF;            // Not cached (invalid sentinel)
            dynamicBrickData[base + 1] = lodArray[i];           // targetLOD (u32)
            dynamicBrickDataF32[base + 2] = lodScale;           // lodScale (f32)
        }
        
        b.targetLOD = lodArray[i];
        b.outputOffset = cacheInfo ? cacheInfo.blockStart : 0xFFFFFFFF;
    }
    
    // Data prepared, GPU write will be batched later
    // Only upload the portion that corresponds to actual bricks
    return dynamicBrickData.subarray(0, brickCount * DYNAMIC_STRIDE_U32);
}

function updateWorkQueue(allocatedBricks) {
    // Brick IDs are Morton indices, so they're already spatially ordered
    // Sort by brick index to preserve Morton order (optimal 3D locality)
    allocatedBricks.sort((a, b) => a.index - b.index);
    
    // Reuse buffer pool
    ensureBufferPools(dataset.bricks.length, allocatedBricks.length);
    const workQueue = workQueuePool;
    
    // Populate work queue only with active bricks
    for (let i = 0; i < allocatedBricks.length; i++) {
        workQueue[i] = allocatedBricks[i].index;
    }
    
    // Prepare work count
    workCountBuffer[0] = allocatedBricks.length;
    
    // Store work count in gpuState for dispatch
    gpuState.workCount = allocatedBricks.length;
    
    // Return prepared data for batched GPU write
    return {
        workQueue: allocatedBricks.length > 0 ? workQueue.subarray(0, allocatedBricks.length) : null,
        workCount: workCountBuffer
    };
}

// ==================== GPU Render Passes ====================

function runClearPass(encoder) {
    // Only clear regions that will be written to by the compute pass
    // If no bricks are being decompressed, skip the clear entirely
    if (!gpuState.workCount || gpuState.workCount === 0) {
        return;  // No decompression = no new data = skip clear
    }
    
    // Collect clear regions from allocated bricks
    const clearRegions = [];
    const brickSize = dataset.header.brickSize;
    
    // Get the current visible bricks with their cache info
    const cache = gpuState.brickCache;
    const brickCount = dataset.bricks.length;
    
    for (let i = 0; i < brickCount; i++) {
        const cacheInfo = cache.getCacheInfo(i);
        if (cacheInfo) {
            const lod = cacheInfo.lod;
            const outputOffset = cacheInfo.blockStart;
            
            // Calculate voxel count for this LOD: 2^(3*lod)
            const voxelCount = Math.floor(Math.pow(2, 3 * lod));
            
            clearRegions.push({
                offset: outputOffset,
                voxelCount: voxelCount
            });
        }
    }
    
    const totalRegionCount = clearRegions.length;

    // Skip if no regions to clear
    if (totalRegionCount === 0) {
        return;
    }
    
    // Persistent buffers
    if (!clearRegionCountBuffer) {
        clearRegionCountBuffer = gpuState.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
    }
    if (!clearRegionBaseBuffer) {
        clearRegionBaseBuffer = gpuState.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
    }

    // Write total region count once
    gpuState.device.queue.writeBuffer(clearRegionCountBuffer, 0, new Uint32Array([totalRegionCount]));

    // Process regions in chunks of 256 (shader cap)
    const MAX_CHUNK = 256;
    const workgroupSize = 256;

    // Ensure GPU region buffer is allocated to max chunk size once
    const maxChunkBytes = MAX_CHUNK * 2 * 4; // offset + count per region, 4 bytes each
    if (!clearRegionsBuffer || clearRegionsBuffer.size < maxChunkBytes) {
        clearRegionsBuffer = gpuState.device.createBuffer({
            size: Math.max(maxChunkBytes, 2048),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: false
        });
    }

    for (let base = 0; base < totalRegionCount; base += MAX_CHUNK) {
        const chunkCount = Math.min(MAX_CHUNK, totalRegionCount - base);

        // Ensure CPU staging buffer is large enough for this chunk (offset + count per region)
        if (!clearRegionsData || clearRegionsData.length < chunkCount * 2) {
            clearRegionsData = new Uint32Array(Math.max(chunkCount * 2, 512));
        }

        // Fill chunk slice
        for (let i = 0; i < chunkCount; i++) {
            const r = clearRegions[base + i];
            clearRegionsData[i * 2] = r.offset;
            clearRegionsData[i * 2 + 1] = r.voxelCount;
        }

        gpuState.device.queue.writeBuffer(clearRegionsBuffer, 0, clearRegionsData, 0, chunkCount * 2 * 4);
        gpuState.device.queue.writeBuffer(clearRegionBaseBuffer, 0, new Uint32Array([base]));

        // Create bind group once and cache it (buffers don't change between chunks)
        if (!cachedClearBindGroup) {
            cachedClearBindGroup = gpuState.device.createBindGroup({
                layout: gpuState.clearPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: gpuState.outVoxelBuffer } },
                    { binding: 1, resource: { buffer: clearRegionCountBuffer } },
                    { binding: 2, resource: { buffer: clearRegionsBuffer } },
                    { binding: 3, resource: { buffer: clearRegionBaseBuffer } }
                ]
            });
        }

        const clearPass = encoder.beginComputePass();
        clearPass.setPipeline(gpuState.clearPipeline);
        clearPass.setBindGroup(0, cachedClearBindGroup);  // Reuse cached bind group
        clearPass.dispatchWorkgroups(chunkCount); // one workgroup per region in this chunk
        clearPass.end();
    }
}

function runComputePass(encoder) {
    // Skip if no bricks need decompression
    const workCount = gpuState.workCount >>> 0;
    if (!workCount) { return; }
    
    const computePass = encoder.beginComputePass();
    computePass.setPipeline(gpuState.computePipeline);
    computePass.setBindGroup(0, gpuState.computeBindGroup);
    // Dispatch one workgroup per brick (workgroup cooperatively fills sub-blocks)
    computePass.dispatchWorkgroups(workCount);
    computePass.end();
    // After dispatching, clear CPU-side counter so we don't re-dispatch the same work next frame.
    // Leave GPU buffer untouched; it will be overwritten next time bricks are queued.
    gpuState.workCount = 0;
}

function runRayTracePass(encoder) {
    if (!gpuState.rayTracePipeline) return; // Skip if not initialized
    
    const rayTracePass = encoder.beginComputePass();
    rayTracePass.setPipeline(gpuState.rayTracePipeline);
    rayTracePass.setBindGroup(0, gpuState.rayTraceBindGroup);
    
    // Dispatch with workgroups based on output resolution
    const threadsPerGroup = 8; // 8x8 threads per group
    const groupsX = Math.ceil(gpuState.rayTraceOutputSize.width / threadsPerGroup) >>> 0;
    const groupsY = Math.ceil(gpuState.rayTraceOutputSize.height / threadsPerGroup) >>> 0;
    
    rayTracePass.dispatchWorkgroups(groupsX, groupsY);
    rayTracePass.end();
}

function runDisplayPass(encoder) {
    if (!gpuState.displayPipeline || !gpuState.context) return; // Skip if not initialized
    
    const renderPass = encoder.beginRenderPass({
        colorAttachments: [{
            view: gpuState.context.getCurrentTexture().createView(),
            clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
            loadOp: 'clear',
            storeOp: 'store',
        }],
    });
    
    renderPass.setPipeline(gpuState.displayPipeline);
    renderPass.setBindGroup(0, gpuState.displayBindGroup);
    renderPass.draw(3); // Draw fullscreen triangle
    renderPass.end();
}

// ==================== Visibility & Culling ====================

/**
 * Check if camera has moved significantly since last LOD update
 */
function cameraHasMoved() {
    if (!gpuState.camera) return false;
    
    const camera = gpuState.camera;
    const MOVEMENT_THRESHOLD = 0.5;      // Position threshold (world units)
    const ROTATION_THRESHOLD = 0.05;     // Direction threshold (radians)
    const MOVEMENT_THRESHOLD_SQ = MOVEMENT_THRESHOLD * MOVEMENT_THRESHOLD;
    const ROT_DOT_THRESHOLD = Math.cos(ROTATION_THRESHOLD); // avoid acos per frame
    
    if (!lastCameraPosition || !lastCameraDirection) {
        // First check - consider camera as moved
        lastCameraPosition = { ...camera.position };
        lastCameraDirection = { ...camera.direction };
        return true;
    }
    
    // Check position change (squared distance to avoid sqrt)
    const dx = camera.position.x - lastCameraPosition.x;
    const dy = camera.position.y - lastCameraPosition.y;
    const dz = camera.position.z - lastCameraPosition.z;
    const posChangeSq = dx * dx + dy * dy + dz * dz;
    
    // Check direction change using dot product vs cosine threshold
    const dot = (camera.direction.x * lastCameraDirection.x) +
                (camera.direction.y * lastCameraDirection.y) +
                (camera.direction.z * lastCameraDirection.z);
    const hasRotated = dot < ROT_DOT_THRESHOLD;
    const hasMoved = posChangeSq > MOVEMENT_THRESHOLD_SQ || hasRotated;
    
    if (hasMoved) {
        lastCameraPosition = { ...camera.position };
        lastCameraDirection = { ...camera.direction };
    }
    
    return hasMoved;
}

/**
 * Check if a brick is visible based on frustum and transfer function
 */
function isBrickVisible(brick, brickSize) {
    // Frustum culling: check if brick intersects camera frustum
    if (!isBrickInFrustum(brick, brickSize)) {
        return false;
    }
    
    // Transfer function culling: check if brick has any visible voxels
    if (!brickHasVisibleVoxels(brick)) {
        return false;
    }
    
    return true;
}

/**
 * Frustum culling: Check if brick AABB intersects camera frustum
 */
function isBrickInFrustum(brick, brickSize) {
    const camera = gpuState.camera;
    if (!camera) return true; // No camera = assume visible
    
    // Brick AABB (axis-aligned bounding box)
    const brickMin = brick.position;
    const brickMax = {
        x: brick.position.x + brickSize,
        y: brick.position.y + brickSize,
        z: brick.position.z + brickSize
    };
    
    // Brick center
    const center = {
        x: (brickMin.x + brickMax.x) / 2,
        y: (brickMin.y + brickMax.y) / 2,
        z: (brickMin.z + brickMax.z) / 2
    };
    
    // console.log(`Brick ${brick.ID}: center=(${center.x}, ${center.y}, ${center.z}), min=(${brickMin.x}, ${brickMin.y}, ${brickMin.z}), max=(${brickMax.x}, ${brickMax.y}, ${brickMax.z})`);
    
    // Vector from camera to brick center
    const toCenter = {
        x: center.x - camera.position.x,
        y: center.y - camera.position.y,
        z: center.z - camera.position.z
    };
    
    // Distance to brick center
    const distance = Math.sqrt(toCenter.x * toCenter.x + toCenter.y * toCenter.y + toCenter.z * toCenter.z);
    
    // Normalize direction to brick
    const dirToBrick = {
        x: toCenter.x / distance,
        y: toCenter.y / distance,
        z: toCenter.z / distance
    };
    
    // Dot product with camera direction (how aligned is brick with view direction)
    const dot = dirToBrick.x * camera.direction.x +
                dirToBrick.y * camera.direction.y +
                dirToBrick.z * camera.direction.z;
    
    // Brick diagonal radius (maximum distance from center to corner)
    const brickRadius = Math.sqrt(3) * (brickSize / 2);
    
    // FOV-based culling: brick must be within FOV cone + radius margin
    // cos(fov/2) gives the minimum dot product for objects in view
    const cosHalfFov = Math.cos(camera.fov / 2);
    
    // Approximate frustum test: dot > cos(fov/2) - margin
    // Allow bricks slightly outside FOV to account for brick size
    const margin = brickRadius / distance; // Angular size of brick
    
    if (dot < cosHalfFov - margin) {
        return false; // Outside FOV
    }
    
    // Behind camera check (with margin for partially visible bricks)
    if (distance < -brickRadius) {
        return false;
    }
    
    return true;
}

/**
 * Transfer function culling: Check if brick palette has visible voxels
 * Empty label is 0 (background from MICrONS dataset)
 * Visible = has any non-zero label in palette
 */
function brickHasVisibleVoxels(brick) {
    // Check if ANY palette entry is non-zero (non-empty)
    for (const paletteValue of brick.palette) {
        if (paletteValue !== 0n) {  // 0 = empty/background
            return true; // Has at least one non-empty voxel (neuron)
        }
    }
    
    return false; // All palette entries are 0 (empty)
}

// ==================== GPU Readback & Analysis ====================

async function readBackOnce() {
    // Wait until GPU finishes the submitted work
    await gpuState.device.queue.onSubmittedWorkDone();

    // Mark buffer as being read to prevent frame() from using it
    isReadbackMapped = true;

    try {
        await gpuState.readbackBuffer.mapAsync(GPUMapMode.READ);
        const mappedRange = gpuState.readbackBuffer.getMappedRange();
        const data = new Uint32Array(mappedRange);

        // Convert and analyze voxel data - must do this while mapped
        const voxels = convertVoxelData(data);
        
        // Unmap BEFORE logging to free the buffer
        gpuState.readbackBuffer.unmap();
        
        // Now safe to log
        logVoxelStatistics(voxels);
    } catch (err) {
        console.error("Readback failed:", err);
        if (gpuState.readbackBuffer && gpuState.readbackBuffer.getMappedRange) {
            try {
                gpuState.readbackBuffer.unmap();
            } catch (e) {
                // Already unmapped
            }
        }
    } finally {
        // Always unmark the flag when done
        isReadbackMapped = false;
    }
}

function convertVoxelData(data) {
    const voxels = [];
    // Now data contains u32 RGBA values directly (4 bytes per voxel)
    for (let i = 0; i < data.length; i++) {
        voxels.push(data[i]);
    }
    return voxels;
}

function logVoxelStatistics(voxels) {
    const EMPTY_VALUE = 0x00000000;  // Transparent RGBA
    const emptyCount = voxels.filter(v => v === EMPTY_VALUE).length;
    const nonEmptyCount = voxels.length - emptyCount;
    
    console.log(`Buffer size: ${voxels.length} voxels`);
    console.log(`Non-empty voxels: ${nonEmptyCount}`);
    console.log(`Empty voxels (0x00000000): ${emptyCount}`);
    console.log("First 128 decoded voxels (RGBA u32, decimal):", voxels.slice(0, 128).map(v => v.toString()));
    
    // Avoid spread operator with large arrays - use reduce instead
    let minVal = voxels[0];
    let maxVal = voxels[0];
    for (let i = 1; i < voxels.length; i++) {
        if (voxels[i] < minVal) minVal = voxels[i];
        if (voxels[i] > maxVal) maxVal = voxels[i];
    }
    console.log(`Full buffer statistics - min: 0x${minVal.toString(16)}, max: 0x${maxVal.toString(16)}`);
}
