/**
 * Buffer Pool Manager
 * Manages reusable GPU buffer pools to reduce GC pressure
 */

export const BUFFER_CONFIG = {
    GROWTH_FACTOR: 1.5,
    MIN_DYNAMIC_CAP: 1024,
    MIN_WORKQUEUE_CAP: 256
};

export class BufferPoolManager {
    constructor() {
        this.dynamicBrickDataPool = null;
        this.workQueuePool = null;
        this.workCountBuffer = new Uint32Array(1);
    }

    /**
     * Calculate new capacity with growth headroom
     * @param {number} currentLength - Current buffer length
     * @param {number} requiredLength - Required buffer length
     * @param {number} minimumLength - Minimum allowed length
     * @returns {number} New capacity
     */
    growCapacity(currentLength, requiredLength, minimumLength) {
        if (currentLength >= requiredLength) return currentLength;
        const base = currentLength === 0 ? requiredLength : currentLength;
        return Math.max(Math.ceil(base * BUFFER_CONFIG.GROWTH_FACTOR), requiredLength, minimumLength);
    }

    /**
     * Ensure buffer pools are large enough for current frame
     * @param {number} brickCount - Total number of bricks
     * @param {number} maxAllocated - Maximum allocated bricks
     */
    ensureBufferPools(brickCount, maxAllocated) {
        const requiredDynamic = brickCount * 2;
        const requiredWorkQueue = Math.max(maxAllocated, 1);
        const dynamicCap = this.growCapacity(
            this.dynamicBrickDataPool?.length ?? 0,
            requiredDynamic,
            BUFFER_CONFIG.MIN_DYNAMIC_CAP
        );
        const workCap = this.growCapacity(
            this.workQueuePool?.length ?? 0,
            requiredWorkQueue,
            BUFFER_CONFIG.MIN_WORKQUEUE_CAP
        );

        if (!this.dynamicBrickDataPool || this.dynamicBrickDataPool.length < requiredDynamic) {
            this.dynamicBrickDataPool = new Uint32Array(dynamicCap);
        }
        if (!this.workQueuePool || this.workQueuePool.length < requiredWorkQueue) {
            this.workQueuePool = new Uint32Array(workCap);
        }
    }

}
