/**
 * Cache Manager - SAS-inspired brick cache for CSV decompression
 * Manages per-LOD stacks of free blocks and residency tracking
 */

import { createLogger } from './logger.js';

const log = createLogger('Cache');

export class CacheState {
    constructor(totalVoxels, maxLOD) {
        // We track offsets directly in voxel units to match GPU buffers
        this.totalBaseElements = totalVoxels;
        this.maxLOD = maxLOD;

        // Allocation state
        this.cacheTop = 0; // Current allocation cursor
        this.baseElementsAllocated = 0; // Total allocated so far

        // Per-LOD stacks: stack[l] = array of free block start indices for LOD l
        this.stacks = new Array(maxLOD + 1);
        for (let l = 0; l <= maxLOD; l++) {
            this.stacks[l] = [];
        }

        // Residency map: brick index -> {lod, cacheBlockStart, size}
        // Null if brick not in cache
        this.residency = new Map();

        // Sequence counters per LOD (for marking visible bricks)
        this.sequences = new Array(maxLOD + 1);
        for (let l = 0; l <= maxLOD; l++) {
            this.sequences[l] = 0;
        }

        // Marked bricks for current frame (to track which stay/evict)
        this.markedBricks = new Map(); // brick index -> sequence number
    }

    /**
     * Get block size (in voxels) for a given LOD
     * LOD l contains 2^(3*l) voxels (e.g., LOD 6 for a 64³ brick = 262,144 voxels)
     */
    getBlockSize(lod) {
        return Math.pow(2, 3 * lod);
    }

    /**
     * Allocate a free block of the given LOD size
     * Returns: start index of block, or null if allocation fails
     */
    allocateBlock(lod) {
        // LOD 0 (coarsest, 1 voxel) doesn't need cache - read directly from palette
        if (lod === 0) {
            console.error("LOD 0 bricks should not be allocated in cache - use palette directly");
            return null;
        }

        const blockSize = this.getBlockSize(lod);
        const stack = this.stacks[lod];

        // Try to get from stack first
        if (stack.length > 0) {
            return stack.pop();
        }

        // Try to allocate new block from top
        if (this.baseElementsAllocated + blockSize <= this.totalBaseElements) {
            const blockStart = this.cacheTop;
            this.cacheTop += blockSize;
            this.baseElementsAllocated += blockSize;
            return blockStart;
        }

        // Allocation failed - let caller rebuild and re-decode bricks
        console.warn(`Allocation failed for LOD ${lod} (${blockSize} elements needed). No space left in cache.`);
        return null;
    }

    /**
     * Free a block back to its LOD stack
     */
    freeBlock(lod, blockStart) {
        // LOD 0 bricks don't participate in cache
        if (lod === 0) return;

        this.stacks[lod].push(blockStart);
    }

    /**
     * Mark a brick as needed for this frame
     * Detects and handles LOD changes (evicts old version if LOD differs)
     * Returns: { isNew, lodChanged }
     *   - isNew: true if brick wasn't marked before
     *   - lodChanged: true if brick exists in cache but at different LOD
     */
    markBrickNeeded(brickIndex, lod) {
        this.sequences[lod]++;
        const seqNum = this.sequences[lod];

        // Check if brick already exists in cache
        const cached = this.residency.get(brickIndex);
        let lodChanged = false;

        // If LOD changed, evict the old version
        if (cached && cached.lod !== lod) {
            lodChanged = true;
            this.freeBlock(cached.lod, cached.blockStart);
            this.residency.delete(brickIndex);
        }

        // New = not currently resident
        const isNew = !this.residency.has(brickIndex);

        // Mark for this frame (used for eviction detection)
        this.markedBricks.set(brickIndex, seqNum);

        return { isNew, lodChanged };
    }

    /**
     * Get bricks to evict (marked in previous frame but not in current)
     * Returns array of {brickIndex, lod, blockStart}
     */
    getEvictedBricks() {
        const evicted = [];

        for (const [brickIndex, residencyInfo] of this.residency.entries()) {
            if (!this.markedBricks.has(brickIndex)) {
                evicted.push({
                    brickIndex,
                    lod: residencyInfo.lod,
                    blockStart: residencyInfo.blockStart
                });
            }
        }

        return evicted;
    }

    /**
     * Register a brick as now cached
     */
    setCached(brickIndex, lod, blockStart) {
        this.residency.set(brickIndex, {
            lod,
            blockStart,
            size: this.getBlockSize(lod)
        });
    }

    /**
     * Check if a brick is cached
     */
    isCached(brickIndex) {
        return this.residency.has(brickIndex);
    }

    /**
     * Get cache info for a brick
     */
    getCacheInfo(brickIndex) {
        return this.residency.get(brickIndex) || null;
    }

    /**
     * Compact cache to remove fragmentation while preserving cached bricks
     * Groups blocks by LOD and rewrites residency pointers to contiguous positions
     */
    compact() {
        console.warn("Compacting cache to remove fragmentation...");

        // Group blocks by LOD
        const blocksByLOD = {};
        for (let l = 0; l <= this.maxLOD; l++) {
            blocksByLOD[l] = [];
        }

        // Collect all cached blocks
        for (const [brickIndex, info] of this.residency.entries()) {
            blocksByLOD[info.lod].push({
                brickIndex,
                oldBlockStart: info.blockStart,
                size: info.size
            });
        }

        // Rewrite positions starting from offset 0
        let newOffset = 0;

        for (let l = 1; l <= this.maxLOD; l++) {  // Skip LOD 0 (not cached)
            for (const block of blocksByLOD[l]) {
                // Update residency with new position
                this.residency.get(block.brickIndex).blockStart = newOffset;
                newOffset += block.size;
            }
        }

        // Update cache state
        this.cacheTop = newOffset;
        this.baseElementsAllocated = newOffset;

        // Rebuild stacks from new compacted layout
        for (let l = 0; l <= this.maxLOD; l++) {
            this.stacks[l] = [];
        }

        // Note: We don't populate stacks with freed blocks because we just compacted
        // Fresh allocations will come from cacheTop as needed
        log.log(`Compacted: ${this.residency.size} bricks preserved, new top: ${this.cacheTop}`);
    }

    /**
     * Clear cache residency and reset (for rebuild)
     */
    rebuild() {
        console.warn("Cache rebuild triggered - clearing all residency");
        this.residency.clear();
        this.cacheTop = 0;
        this.baseElementsAllocated = 0;

        for (let l = 0; l <= this.maxLOD; l++) {
            this.stacks[l] = [];
        }
    }

    /**
     * Clear marked bricks for next frame
     */
    clearMarkedBricks() {
        this.markedBricks.clear();
    }

    /**
     * Get stats for debugging
     */
    getStats() {
        return {
            totalBaseElements: this.totalBaseElements,
            allocated: this.baseElementsAllocated,
            free: this.totalBaseElements - this.baseElementsAllocated,
            cachedBricks: this.residency.size,
            stackSizes: this.stacks.map((s, l) => ({ lod: l, free: s.length }))
        };
    }
}

/**
 * Initialize cache with given total voxels and max LOD
 */
export function initCache(totalVoxels, maxLOD) {
    const cache = new CacheState(totalVoxels, maxLOD);

    log.log(`Initialized: ${totalVoxels} voxels available, LOD 0-${maxLOD}`);
    return cache;
}
