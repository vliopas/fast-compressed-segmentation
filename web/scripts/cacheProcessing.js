/**
 * Cache Processing Functions
 * Handles cache initialization, LOD updates, and brick processing
 */

import { createLogger } from './logger.js';
import { isBrickInFrustum, brickHasVisibleVoxels } from './culling.js';

const log = createLogger('CacheProcessing');

/**
 * Initialize cache with visible bricks at calculated LODs
 * @param {Object} gpuState - GPU state object
 * @param {Object} dataset - Dataset with bricks
 * @param {Array<number>} initialLODs - Pre-calculated LOD array
 * @param {number} brickSize - Size of bricks
 * @param {boolean} forceAllBricks - If true, allocate all bricks regardless of visibility (for validation)
 * @returns {{allocatedBricks: Array, allocatedCount: number}} Initialization stats
 */
export function initializeCache(gpuState, dataset, initialLODs, brickSize, forceAllBricks = false) {
    const cache = gpuState.brickCache;
    const allocatedBricks = [];

    // Log LOD distribution
    const maxLOD = Math.max(...initialLODs);
    const lodDistribution = new Array(maxLOD + 1).fill(0);
    let totalVoxelsNeeded = 0;
    for (let i = 0; i < initialLODs.length; i++) {
        const lod = initialLODs[i];
        lodDistribution[lod]++;
        totalVoxelsNeeded += Math.pow(2, 3 * lod);
    }

    let lodDistributionMsg = `Brick LOD Distribution (${initialLODs.length} bricks, 0=coarsest, ${maxLOD}=finest):\n`;
    for (let lod = 0; lod <= maxLOD; lod++) {
        const count = lodDistribution[lod];
        const voxelsPerBrick = Math.pow(2, 3 * lod);
        const totalForLod = count * voxelsPerBrick;
        const percentage = (count / initialLODs.length * 100).toFixed(1);
        lodDistributionMsg += `  LOD ${lod}: ${count} bricks (${percentage}%) - ${totalForLod.toLocaleString()} voxels total\n`;
    }
    lodDistributionMsg += `Total voxels needed: ${totalVoxelsNeeded.toLocaleString()} / ${cache.totalBaseElements.toLocaleString()} available`;
    log.log(lodDistributionMsg);

    // Allocate cache entries for bricks
    for (let i = 0; i < dataset.bricks.length; i++) {
        const lod = initialLODs[i];
        const brick = dataset.bricks[i];

        let isVisible;
        if (forceAllBricks) {
            // For validation, allocate all bricks regardless of visibility
            isVisible = true;
        } else {
            // Normal mode: only allocate visible bricks
            const inFrustum = isBrickInFrustum(brick, brickSize, gpuState.camera);
            const hasVisible = inFrustum ? brickHasVisibleVoxels(brick, dataset.hiddenLabels) : false;
            isVisible = inFrustum && hasVisible;
        }

        if (!isVisible) {
            continue; // Skip invisible bricks on the first frame
        }

        cache.markBrickNeeded(i, lod);

        const blockStart = cache.allocateBlock(lod);
        if (blockStart !== null) {
            allocatedBricks.push({ index: i, lod, blockStart });
            cache.setCached(i, lod, blockStart);
        } else {
            console.warn(`Failed to allocate brick ${i} at LOD ${lod}`);
        }
    }

    cache.clearMarkedBricks();

    if (allocatedBricks.length < dataset.bricks.length) {
        console.warn(`Only ${allocatedBricks.length} of ${dataset.bricks.length} bricks allocated`);
    }

    return {
        allocatedBricks,
        allocatedCount: allocatedBricks.length
    };
}

/**
 * Process cache updates based on visibility and LOD changes
 * @param {Object} params - Parameters object
 * @param {Array<number>} params.lodArray - LOD array for all bricks
 * @param {Array<number>} params.offsetArray - Offset array (unused but kept for compatibility)
 * @param {Object} params.gpuState - GPU state object
 * @param {Object} params.dataset - Dataset with bricks
 * @param {Uint32Array} params.pendingGPURequests - GPU request buffer
 * @param {boolean} params.gpuRequestsReady - Whether GPU requests are available
 * @returns {Object} Processing results with allocatedBricks and stats
 */
export function processCache({ lodArray, offsetArray, gpuState, dataset, pendingGPURequests, gpuRequestsReady }) {
    const cache = gpuState.brickCache;
    const brickCount = dataset.bricks.length;
    const brickSize = dataset.header.brickSize;

    // CPU Pre-filtering (Frustum + Empty brick culling)
    const candidateBricks = new Set();
    let frustumCulled = 0;
    let tfCulled = 0;

    for (let i = 0; i < brickCount; i++) {
        const brick = dataset.bricks[i];

        // Pre-filter: frustum culling (eliminate bricks outside view)
        const inFrustum = isBrickInFrustum(brick, brickSize, gpuState.camera);
        if (!inFrustum) {
            frustumCulled++;
            continue;
        }

        // Pre-filter: empty brick culling (eliminate transparent bricks)
        const hasVisible = brickHasVisibleVoxels(brick, dataset.hiddenLabels);
        if (!hasVisible) {
            tfCulled++;
            continue;
        }

        // Passed pre-filters: mark as candidate (not cached yet!)
        candidateBricks.add(i);
    }

    // GPU Request Processing (Occlusion culling via ray marching)
    const bricksToDecompress = [];
    let gpuRequestedCount = 0;

    // TEMPORARY: Disable GPU occlusion culling - keep all visible bricks in cache
    // This will help us debug the 50% match rate issue
    const useGPUCulling = false;

    if (useGPUCulling && gpuRequestsReady && pendingGPURequests && pendingGPURequests.length >= brickCount) {
        // Process GPU requests: only cache bricks that were actually accessed by rays
        for (const brickIndex of candidateBricks) {
            const isAccessed = pendingGPURequests[brickIndex];  // 1 = accessed, 0 = not accessed

            // Check if GPU actually accessed this brick
            if (isAccessed === 1) {
                gpuRequestedCount++;
                const lod = lodArray[brickIndex];
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
        }
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

    // Evict unmarked bricks from previous frame
    const evicted = cache.getEvictedBricks();
    for (const { brickIndex, lod, blockStart } of evicted) {
        cache.freeBlock(lod, blockStart);
        cache.residency.delete(brickIndex);
    }

    // Allocate blocks for new bricks
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

    // Clear marked bricks for next frame
    cache.clearMarkedBricks();

    return {
        allocatedBricks,
        evictedCount: evicted.length,
        candidateCount: candidateBricks.size,
        candidateBricks: Array.from(candidateBricks),
        stats: { frustumCulled, tfCulled, gpuRequestedCount }
    };
}

/**
 * Update brick info buffer with cache locations
 * @param {Object} params - Parameters object
 * @param {Array<number>} params.lodArray - LOD array for all bricks
 * @param {Array} params.allocatedBricks - Allocated bricks this frame
 * @param {Object} params.cache - Brick cache object
 * @param {Object} params.dataset - Dataset with bricks
 * @param {number} params.brickSize - Size of bricks
 * @returns {{data: Uint32Array, offsets: Array<number>}} Buffer data and offsets
 */
export function updateBrickInfoBuffer({ lodArray, allocatedBricks, cache, dataset, brickSize }) {
    const brickCount = dataset.bricks.length;
    const DYNAMIC_STRIDE_U32 = 4; // outputOffset, targetLOD, lodScale (as float32), pad

    const dynamicBrickData = new Uint32Array(brickCount * DYNAMIC_STRIDE_U32);
    const dynamicBrickDataF32 = new Float32Array(dynamicBrickData.buffer);

    // Track output offsets for this frame (used for partial clear)
    const lastAllocatedOutputOffsets = [];
    let cachedBrickCount = 0;

    for (let i = 0; i < brickCount; i++) {
        const b = dataset.bricks[i];
        const base = i * DYNAMIC_STRIDE_U32;
        const cacheInfo = cache.getCacheInfo(i);
        const lod = cacheInfo ? cacheInfo.lod : lodArray[i];

        // Precompute lodScale: LOD 0 = coarsest (1 voxel), LOD k = 2^k voxels
        const lodSize = Math.pow(2, lod);
        const lodScale = lodSize / brickSize;

        // Dynamic data only (static data never changes, already in GPU)
        if (cacheInfo) {
            dynamicBrickData[base + 0] = cacheInfo.blockStart;  // outputOffset (u32)
            dynamicBrickData[base + 1] = cacheInfo.lod;         // targetLOD (u32)
            dynamicBrickDataF32[base + 2] = lodScale;           // lodScale (f32)
            dynamicBrickData[base + 3] = 0;                    // pad
            lastAllocatedOutputOffsets.push(cacheInfo.blockStart);
            cachedBrickCount++;
        } else {
            dynamicBrickData[base + 0] = 0xFFFFFFFF;            // Not cached (invalid sentinel)
            dynamicBrickData[base + 1] = lodArray[i];           // targetLOD (u32)
            dynamicBrickDataF32[base + 2] = lodScale;           // lodScale (f32)
            dynamicBrickData[base + 3] = 0;                    // pad
        }

        b.targetLOD = lodArray[i];
        b.outputOffset = cacheInfo ? cacheInfo.blockStart : 0xFFFFFFFF;
    }

    // Upload the full buffer for all bricks
    if (cachedBrickCount < brickCount) {
        console.warn(`Brick info: ${cachedBrickCount} / ${brickCount} bricks have cache allocations (${((cachedBrickCount / brickCount) * 100).toFixed(1)}%)`);
    }

    return {
        data: dynamicBrickData,
        offsets: lastAllocatedOutputOffsets
    };
}

/**
 * Update work queue with bricks to decompress
 * @param {Object} params - Parameters object
 * @param {Array} params.allocatedBricks - Allocated bricks this frame
 * @param {Uint32Array} params.workQueuePool - Reusable work queue buffer
 * @param {Uint32Array} params.workCountBuffer - Work count buffer
 * @returns {{workQueue: Uint32Array|null, workCount: Uint32Array, count: number}} Work queue data
 */
export function updateWorkQueue({ allocatedBricks, workQueuePool, workCountBuffer }) {
    // Brick IDs are Morton indices, so they're already spatially ordered
    // Sort by brick index to preserve Morton order (optimal 3D locality)
    allocatedBricks.sort((a, b) => a.index - b.index);

    // Populate work queue only with active bricks
    for (let i = 0; i < allocatedBricks.length; i++) {
        workQueuePool[i] = allocatedBricks[i].index;
    }

    // Prepare work count
    workCountBuffer[0] = allocatedBricks.length;

    // Return prepared data for batched GPU write
    return {
        workQueue: allocatedBricks.length > 0 ? workQueuePool.subarray(0, allocatedBricks.length) : null,
        workCount: workCountBuffer,
        count: allocatedBricks.length
    };
}
